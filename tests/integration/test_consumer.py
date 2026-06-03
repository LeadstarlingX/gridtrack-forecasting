"""
Full consumer round-trip integration tests.

Each test:
  .NET side (simulated) → fanout exchange → consumer → result queue → assertion

A real RabbitMQ container is started once per session (see tests/conftest.py).
The consumer runs in a background asyncio Task, connected to that container.

Run with:
    pytest tests/integration/test_consumer.py -v
(Docker must be running.)
"""
import asyncio
import json
from datetime import datetime, timezone
from uuid import uuid4

import aio_pika
import pytest

from app.messaging.consumer import start_consumer
from app.models import DeliveryAnomalyIntegrationEvent, DriverPositionIntegrationEvent


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
async def running_consumer(rabbitmq_url, mocker):
    """Start the real consumer against the test RabbitMQ container.

    Steps
    -----
    1. Patch settings.rabbitmq_url (via object.__setattr__ to bypass pydantic).
    2. Stub _groq_note so tests run offline.
    3. Pre-declare + purge both result queues *before* the consumer starts —
       this eliminates the race where the consumer publishes before the queue
       exists (RabbitMQ would silently drop the message).
    4. Start the consumer task and sleep 2 s for connect + queue binding.
    5. Yield (setup_channel, urgency_q, forecast_q) to the test.
    6. Cancel the task and restore the original URL on teardown.
    """
    from app.config import settings as cfg

    original_url = cfg.rabbitmq_url
    object.__setattr__(cfg, "rabbitmq_url", rabbitmq_url)
    mocker.patch("app.services.anomaly._groq_note", return_value="act immediately")

    setup_conn = await aio_pika.connect_robust(rabbitmq_url)
    setup_channel = await setup_conn.channel()

    urgency_q = await setup_channel.declare_queue(
        "gridtrack.urgency-results", durable=True
    )
    forecast_q = await setup_channel.declare_queue(
        "gridtrack.forecast-results", durable=True
    )
    await urgency_q.purge()
    await forecast_q.purge()

    task = asyncio.create_task(start_consumer())
    await asyncio.sleep(2.0)  # allow consumer to connect + bind fanout queues

    yield setup_channel, urgency_q, forecast_q

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await setup_conn.close()
    object.__setattr__(cfg, "rabbitmq_url", original_url)


async def _publish(url: str, exchange_name: str, body: bytes, content_type: str = "application/json"):
    """Open a dedicated publish connection, send one message, close."""
    conn = await aio_pika.connect_robust(url)
    ch = await conn.channel()
    exchange = await ch.declare_exchange(
        exchange_name, aio_pika.ExchangeType.FANOUT, durable=True
    )
    await exchange.publish(
        aio_pika.Message(body=body, content_type=content_type),
        routing_key="",
    )
    await conn.close()


async def _wait_for_message(
    queue: aio_pika.abc.AbstractQueue, timeout: float = 10.0
) -> aio_pika.abc.AbstractIncomingMessage:
    """Poll queue until a message is available or the timeout expires.

    aio_pika's Queue.get() uses AMQP basic.get (a non-blocking pull): it raises
    QueueEmpty immediately if no message is ready. asyncio.wait_for() doesn't
    help because the coroutine completes instantly rather than blocking. This
    helper retries every 200 ms so tests don't race against consumer processing.
    """
    start = asyncio.get_event_loop().time()
    while True:
        try:
            return await queue.get(no_ack=True)
        except aio_pika.exceptions.QueueEmpty:
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed >= timeout:
                raise TimeoutError(f"No message arrived after {timeout:.0f} s")
            await asyncio.sleep(0.2)


# ── tests ─────────────────────────────────────────────────────────────────────


async def test_anomaly_event_produces_urgency_result(running_consumer, rabbitmq_url):
    """
    .NET publishes DeliveryAnomalyIntegrationEvent to gridtrack.anomaly fanout.
    Consumer must score it and publish UrgencyResultMessage to gridtrack.urgency-results.
    """
    _, urgency_q, _ = running_consumer

    delivery_id = uuid4()
    event = DeliveryAnomalyIntegrationEvent(
        deliveryId=delivery_id,
        districtId="mezzeh",
        anomalyType="StalePosition",
        reason="No GPS ping for 20 minutes",
        driverLat=33.505,
        driverLng=36.243,
        occurredAt=datetime.now(timezone.utc),
    )
    await _publish(rabbitmq_url, "gridtrack.anomaly", event.model_dump_json().encode())

    msg = await _wait_for_message(urgency_q)
    payload = json.loads(msg.body)

    assert payload["deliveryId"] == str(delivery_id)
    assert 0 <= payload["urgencyScore"] <= 10
    assert payload["aiNote"] == "act immediately"


async def test_urgency_result_matches_dotnet_contract(running_consumer, rabbitmq_url):
    """
    Field names and types must exactly match what Wolverine deserializes as
    UrgencyResultMessage. A rename or type change here silently breaks .NET.
    """
    _, urgency_q, _ = running_consumer

    event = DeliveryAnomalyIntegrationEvent(
        deliveryId=uuid4(),
        districtId="kafrsousa",
        anomalyType="RouteDeviation",
        reason="Driver off route by 500 m",
        driverLat=33.497,
        driverLng=36.272,
        occurredAt=datetime.now(timezone.utc),
    )
    await _publish(rabbitmq_url, "gridtrack.anomaly", event.model_dump_json().encode())

    msg = await _wait_for_message(urgency_q)
    payload = json.loads(msg.body)

    assert set(payload.keys()) == {"deliveryId", "urgencyScore", "aiNote"}
    assert isinstance(payload["deliveryId"], str)
    assert isinstance(payload["urgencyScore"], int)
    assert isinstance(payload["aiNote"], str)
    assert 0 <= payload["urgencyScore"] <= 10


async def test_position_event_produces_forecast_result(running_consumer, rabbitmq_url):
    """
    .NET publishes DriverPositionIntegrationEvent to gridtrack.positions fanout.
    First event for a district has no throttle so a ForecastResultMessage must appear.
    """
    _, _, forecast_q = running_consumer

    district = "babtouma"
    event = DriverPositionIntegrationEvent(
        driverId=uuid4(),
        districtId=district,
        lat=33.522,
        lng=36.307,
        deliveryStatus="InTransit",
        timestamp=datetime.now(timezone.utc),
    )
    await _publish(rabbitmq_url, "gridtrack.positions", event.model_dump_json().encode())

    msg = await _wait_for_message(forecast_q)
    payload = json.loads(msg.body)

    assert payload["districtId"] == district


async def test_forecast_result_matches_dotnet_contract(running_consumer, rabbitmq_url):
    """
    Field names, types, and enum values must exactly match what Wolverine
    deserializes as ForecastResultMessage.
    """
    _, _, forecast_q = running_consumer

    event = DriverPositionIntegrationEvent(
        driverId=uuid4(),
        districtId="malki",
        lat=33.517,
        lng=36.281,
        deliveryStatus="InTransit",
        timestamp=datetime.now(timezone.utc),
    )
    await _publish(rabbitmq_url, "gridtrack.positions", event.model_dump_json().encode())

    msg = await _wait_for_message(forecast_q)
    payload = json.loads(msg.body)

    assert set(payload.keys()) == {
        "districtId", "expectedDeliveries", "staffingRatio", "label", "color", "generatedAt"
    }
    assert isinstance(payload["districtId"], str)
    assert isinstance(payload["expectedDeliveries"], int)
    assert isinstance(payload["staffingRatio"], float)
    assert payload["label"] in ("Critical", "Moderate", "Low demand")
    assert payload["color"] in ("#f87171", "#fbbf24", "#34d399")
    assert payload["generatedAt"]  # non-empty ISO string


async def test_malformed_message_is_skipped_consumer_survives(running_consumer, rabbitmq_url):
    """
    Consumer must ack and log malformed messages without crashing.
    The valid event published immediately after must still be processed.
    """
    _, urgency_q, _ = running_consumer

    conn = await aio_pika.connect_robust(rabbitmq_url)
    ch = await conn.channel()
    exchange = await ch.declare_exchange(
        "gridtrack.anomaly", aio_pika.ExchangeType.FANOUT, durable=True
    )

    # Publish garbage — will trigger the `except Exception` path in on_message
    await exchange.publish(
        aio_pika.Message(body=b"{{not valid json!!!"),
        routing_key="",
    )
    await asyncio.sleep(0.5)  # give consumer time to log the error

    # Valid event right after — consumer must still be running
    delivery_id = uuid4()
    valid_event = DeliveryAnomalyIntegrationEvent(
        deliveryId=delivery_id,
        districtId="mezzeh",
        anomalyType="EtaExceeded",
        reason="45 minutes overdue",
        driverLat=33.505,
        driverLng=36.243,
        occurredAt=datetime.now(timezone.utc),
    )
    await exchange.publish(
        aio_pika.Message(
            body=valid_event.model_dump_json().encode(),
            content_type="application/json",
        ),
        routing_key="",
    )
    await conn.close()

    msg = await _wait_for_message(urgency_q)
    payload = json.loads(msg.body)
    assert payload["deliveryId"] == str(delivery_id)


async def test_burst_of_anomaly_events_all_processed(running_consumer, rabbitmq_url):
    """
    Three distinct anomaly events published back-to-back.
    All three UrgencyResultMessages must appear on the output queue.
    """
    _, urgency_q, _ = running_consumer

    delivery_ids = [uuid4() for _ in range(3)]

    conn = await aio_pika.connect_robust(rabbitmq_url)
    ch = await conn.channel()
    exchange = await ch.declare_exchange(
        "gridtrack.anomaly", aio_pika.ExchangeType.FANOUT, durable=True
    )
    for delivery_id in delivery_ids:
        event = DeliveryAnomalyIntegrationEvent(
            deliveryId=delivery_id,
            districtId="kafrsousa",
            anomalyType="UnexpectedStop",
            reason="Driver stopped for 12 minutes",
            driverLat=33.497,
            driverLng=36.272,
            occurredAt=datetime.now(timezone.utc),
        )
        await exchange.publish(
            aio_pika.Message(
                body=event.model_dump_json().encode(),
                content_type="application/json",
            ),
            routing_key="",
        )
    await conn.close()

    received = set()
    for _ in range(3):
        msg = await _wait_for_message(urgency_q)
        received.add(json.loads(msg.body)["deliveryId"])

    assert received == {str(d) for d in delivery_ids}
