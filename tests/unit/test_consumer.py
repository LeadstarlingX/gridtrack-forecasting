import asyncio
import json
import logging
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import app.messaging.consumer as consumer_mod
from app.messaging.consumer import start_consumer, _run_consumer
from app.models import DeliveryAnomalyIntegrationEvent


def make_mock_infra():
    """Return (connection, channel, exchange, queue) — all async-safe mocks."""
    mock_queue = AsyncMock()
    mock_exchange = AsyncMock()

    mock_channel = AsyncMock()
    mock_channel.declare_exchange.return_value = mock_exchange
    mock_channel.declare_queue.return_value = mock_queue

    mock_conn = MagicMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)
    mock_conn.channel = AsyncMock(return_value=mock_channel)

    return mock_conn, mock_channel, mock_exchange, mock_queue


def make_anomaly_body(**overrides) -> bytes:
    data = {
        "deliveryId": str(uuid4()),
        "districtId": "mezzeh",
        "anomalyType": "EtaExceeded",
        "reason": "driver stalled",
        "driverLat": 33.5,
        "driverLng": 36.2,
        "occurredAt": datetime.now(timezone.utc).isoformat(),
        **overrides,
    }
    return json.dumps(data).encode()


def make_mock_message(body: bytes) -> MagicMock:
    msg = MagicMock()
    msg.body = body
    msg.process.return_value.__aenter__ = AsyncMock(return_value=None)
    msg.process.return_value.__aexit__ = AsyncMock(return_value=False)
    return msg


async def _cancel(task):
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


# ── start_consumer: retry logic ───────────────────────────────────────────────

async def test_start_consumer_propagates_cancelled_error(mocker):
    mocker.patch(
        "app.messaging.consumer._run_consumer",
        side_effect=asyncio.CancelledError,
    )
    with pytest.raises(asyncio.CancelledError):
        await start_consumer()


async def test_start_consumer_retries_after_exception(mocker):
    call_count = 0

    async def run_once():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("RabbitMQ unavailable")
        raise asyncio.CancelledError()

    mocker.patch("app.messaging.consumer._run_consumer", side_effect=run_once)
    mocker.patch("asyncio.sleep", new=AsyncMock())

    with pytest.raises(asyncio.CancelledError):
        await start_consumer()

    assert call_count == 2


async def test_start_consumer_logs_error_before_retry(mocker, caplog):
    call_count = 0

    async def run_once():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("connection refused")
        raise asyncio.CancelledError()

    mocker.patch("app.messaging.consumer._run_consumer", side_effect=run_once)
    mocker.patch("asyncio.sleep", new=AsyncMock())

    with caplog.at_level(logging.ERROR, logger="app.messaging.consumer"):
        with pytest.raises(asyncio.CancelledError):
            await start_consumer()

    assert "Consumer crashed" in caplog.text


# ── _run_consumer: setup helpers ─────────────────────────────────────────────

async def _start_consumer_task(mocker, mock_conn):
    """Patch infra, start _run_consumer, wait for ready, return task."""
    mocker.patch("aio_pika.connect_robust", return_value=mock_conn)
    consumer_mod.ready = asyncio.Event()
    task = asyncio.create_task(_run_consumer())
    try:
        await asyncio.wait_for(consumer_mod.ready.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        pass
    return task


# ── _run_consumer: structural assertions ─────────────────────────────────────

async def test_run_consumer_sets_ready_event(mocker):
    mock_conn, *_ = make_mock_infra()
    task = await _start_consumer_task(mocker, mock_conn)
    try:
        assert consumer_mod.ready.is_set()
    finally:
        await _cancel(task)


async def test_run_consumer_subscribes_to_both_exchanges(mocker):
    mock_conn, mock_channel, _, __ = make_mock_infra()
    task = await _start_consumer_task(mocker, mock_conn)
    try:
        declared = [c.args[0] for c in mock_channel.declare_exchange.call_args_list]
        assert "gridtrack.anomaly" in declared
        assert "gridtrack.positions" in declared
    finally:
        await _cancel(task)


async def test_run_consumer_registers_consume_callback_per_exchange(mocker):
    mock_conn, _, __, mock_queue = make_mock_infra()
    task = await _start_consumer_task(mocker, mock_conn)
    try:
        assert mock_queue.consume.call_count == 2
    finally:
        await _cancel(task)


async def test_run_consumer_sets_qos_prefetch(mocker):
    mock_conn, mock_channel, _, __ = make_mock_infra()
    task = await _start_consumer_task(mocker, mock_conn)
    try:
        mock_channel.set_qos.assert_awaited_once_with(prefetch_count=10)
    finally:
        await _cancel(task)


# ── on_message: happy path ────────────────────────────────────────────────────
# EXCHANGE_MAP binds handlers as default args at loop-iteration time, so we
# patch.dict the map before starting the task — not the module-level name.

async def test_on_message_dispatches_anomaly_to_score_anomaly(mocker):
    mock_conn, _, __, mock_queue = make_mock_infra()
    mocker.patch("aio_pika.connect_robust", return_value=mock_conn)

    processed = []

    async def fake_score(event):
        processed.append(event)
        return None

    mocker.patch.dict(
        consumer_mod.EXCHANGE_MAP,
        {"gridtrack.anomaly": (DeliveryAnomalyIntegrationEvent, fake_score)},
    )
    consumer_mod.ready = asyncio.Event()
    task = asyncio.create_task(_run_consumer())
    await asyncio.wait_for(consumer_mod.ready.wait(), timeout=1.0)

    on_message = mock_queue.consume.call_args_list[0].args[0]
    await on_message(make_mock_message(make_anomaly_body(districtId="kafrsousa")))
    await _cancel(task)

    assert len(processed) == 1
    assert processed[0].districtId == "kafrsousa"


async def test_on_message_publishes_result_when_handler_returns_value(mocker):
    from app.models import UrgencyResultMessage
    mock_conn, _, __, mock_queue = make_mock_infra()
    mocker.patch("aio_pika.connect_robust", return_value=mock_conn)

    urgency = UrgencyResultMessage(deliveryId="d1", urgencyScore=8, aiNote="critical")

    async def fake_score(_):
        return urgency

    mock_publish = AsyncMock()
    mocker.patch("app.messaging.consumer.publish", new=mock_publish)
    mocker.patch.dict(
        consumer_mod.EXCHANGE_MAP,
        {"gridtrack.anomaly": (DeliveryAnomalyIntegrationEvent, fake_score)},
    )
    consumer_mod.ready = asyncio.Event()
    task = asyncio.create_task(_run_consumer())
    await asyncio.wait_for(consumer_mod.ready.wait(), timeout=1.0)

    on_message = mock_queue.consume.call_args_list[0].args[0]
    await on_message(make_mock_message(make_anomaly_body()))
    await _cancel(task)

    mock_publish.assert_awaited_once()


async def test_on_message_skips_publish_when_handler_returns_none(mocker):
    mock_conn, _, __, mock_queue = make_mock_infra()
    mocker.patch("aio_pika.connect_robust", return_value=mock_conn)

    async def fake_score(_):
        return None

    mock_publish = AsyncMock()
    mocker.patch("app.messaging.consumer.publish", new=mock_publish)
    mocker.patch.dict(
        consumer_mod.EXCHANGE_MAP,
        {"gridtrack.anomaly": (DeliveryAnomalyIntegrationEvent, fake_score)},
    )
    consumer_mod.ready = asyncio.Event()
    task = asyncio.create_task(_run_consumer())
    await asyncio.wait_for(consumer_mod.ready.wait(), timeout=1.0)

    on_message = mock_queue.consume.call_args_list[0].args[0]
    await on_message(make_mock_message(make_anomaly_body()))
    await _cancel(task)

    mock_publish.assert_not_awaited()


async def test_on_message_handles_invalid_json_without_raising(mocker, caplog):
    mock_conn, _, __, mock_queue = make_mock_infra()
    mocker.patch("aio_pika.connect_robust", return_value=mock_conn)

    async def fake_score(_):
        return None

    mocker.patch.dict(
        consumer_mod.EXCHANGE_MAP,
        {"gridtrack.anomaly": (DeliveryAnomalyIntegrationEvent, fake_score)},
    )
    consumer_mod.ready = asyncio.Event()
    task = asyncio.create_task(_run_consumer())
    await asyncio.wait_for(consumer_mod.ready.wait(), timeout=1.0)

    on_message = mock_queue.consume.call_args_list[0].args[0]
    with caplog.at_level(logging.ERROR, logger="app.messaging.consumer"):
        await on_message(make_mock_message(b"not valid json at all"))
    await _cancel(task)

    assert "Error processing message" in caplog.text
