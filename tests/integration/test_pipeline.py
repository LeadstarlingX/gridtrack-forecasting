import asyncio
import json
from datetime import datetime, timezone
from uuid import uuid4

import aio_pika
import pytest

from app.models import DeliveryAnomalyIntegrationEvent, DriverPositionIntegrationEvent
from app.services.anomaly import score_anomaly
from app.services.forecast import update_forecast
from app.messaging.publisher import publish


async def _get_channel(url: str):
    conn = await aio_pika.connect_robust(url)
    return conn, await conn.channel()


async def test_anomaly_pipeline_publishes_urgency_result(rabbitmq_url, mocker):
    """
    Simulate: .NET publishes anomaly event → Python scores → result lands on urgency queue.
    """
    mocker.patch("app.services.anomaly._groq_note", return_value="check driver now")

    conn, channel = await _get_channel(rabbitmq_url)

    result_queue = await channel.declare_queue("gridtrack.urgency-results", durable=False)

    event = DeliveryAnomalyIntegrationEvent(
        deliveryId=uuid4(),
        districtId="mezzeh",
        anomalyType="StalePosition",
        reason="No movement for 25 min",
        driverLat=33.505,
        driverLng=36.243,
        occurredAt=datetime.now(timezone.utc),
    )
    result = await score_anomaly(event)
    assert result is not None

    await publish(channel, result)

    message = await asyncio.wait_for(result_queue.get(no_ack=True), timeout=5.0)
    payload = json.loads(message.body)

    assert payload["deliveryId"] == str(event.deliveryId)
    assert 0 <= payload["urgencyScore"] <= 10
    assert isinstance(payload["aiNote"], str)

    await conn.close()


async def test_publisher_routes_forecast_to_correct_queue(rabbitmq_url):
    """
    Verify ForecastResultMessage lands on gridtrack.forecast-results, not urgency queue.
    """
    from app.models import ForecastResultMessage

    conn, channel = await _get_channel(rabbitmq_url)
    queue = await channel.declare_queue("gridtrack.forecast-results", durable=False)

    msg = ForecastResultMessage(
        districtId="babtouma",
        expectedDeliveries=12,
        staffingRatio=0.75,
        label="Moderate",
        color="#fbbf24",
        generatedAt=datetime.now(timezone.utc).isoformat(),
    )
    await publish(channel, msg)

    message = await asyncio.wait_for(queue.get(no_ack=True), timeout=5.0)
    payload = json.loads(message.body)

    assert payload["districtId"] == "babtouma"
    assert payload["label"] == "Moderate"

    await conn.close()
