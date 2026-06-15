import asyncio
import logging
from datetime import datetime, timezone

import aio_pika
from pydantic import BaseModel

from app.config import settings
from app.messaging.publisher import publish
from app.models import (
    DeliveryAnomalyIntegrationEvent,
    DeliveryCompletedIntegrationEvent,
    DriverPositionIntegrationEvent,
)
from app.services.anomaly import score_anomaly
from app.services.completion import handle_completion
from app.services.forecast import update_forecast
from app.services.incident import check_incident
from app.services.surge import check_surge

# Set once all exchanges are declared and queues are bound.
# Polled by the /ready endpoint so the test factory waits before sending events.
ready = asyncio.Event()

logger = logging.getLogger(__name__)


# ── Composite handlers ───────────────────────────────────────────────────────

async def handle_position(event: DriverPositionIntegrationEvent) -> list[BaseModel]:
    """Runs forecast + surge detection; returns 0-2 results."""
    results: list[BaseModel] = []
    forecast = await update_forecast(event)
    if forecast is not None:
        results.append(forecast)
        now   = datetime.now(timezone.utc)
        surge = check_surge(event.districtId, forecast.expectedDeliveries, now)
        if surge is not None:
            results.append(surge)
    return results


async def handle_anomaly(event: DeliveryAnomalyIntegrationEvent) -> list[BaseModel]:
    """Scores urgency + checks for incident threshold; returns 1-2 results."""
    urgency  = await score_anomaly(event)
    incident = await check_incident(event)
    results: list[BaseModel] = [urgency]
    if incident is not None:
        results.append(incident)
    return results


EXCHANGE_MAP = {
    "gridtrack.anomaly":      (DeliveryAnomalyIntegrationEvent,   handle_anomaly),
    "gridtrack.positions":    (DriverPositionIntegrationEvent,     handle_position),
    "gridtrack.completions":  (DeliveryCompletedIntegrationEvent,  handle_completion),
}


async def start_consumer() -> None:
    """Connect to RabbitMQ and consume all configured exchanges.
    Retries with backoff on connection failure (handles Render cold starts)."""
    while True:
        try:
            await _run_consumer()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Consumer crashed: %s — retrying in 5s", exc)
            await asyncio.sleep(5)


async def _run_consumer() -> None:
    connection = await aio_pika.connect_robust(settings.rabbitmq_url)
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=10)

        for exchange_name, (schema, handler) in EXCHANGE_MAP.items():
            exchange = await channel.declare_exchange(
                exchange_name,
                aio_pika.ExchangeType.FANOUT,
                durable=True,
            )
            queue = await channel.declare_queue("", exclusive=True)
            await queue.bind(exchange)

            async def on_message(
                msg: aio_pika.IncomingMessage,
                _schema=schema,
                _handler=handler,
            ) -> None:
                async with msg.process():
                    try:
                        event = _schema.model_validate_json(msg.body)
                        logger.info("Received %s", _schema.__name__)
                        result = await _handler(event)
                        # Handlers may return None, a single BaseModel, or a list.
                        if result is None:
                            return
                        items = result if isinstance(result, list) else [result]
                        for item in items:
                            if item is not None:
                                logger.info("Publishing %s", type(item).__name__)
                                await publish(channel, item)
                    except Exception as exc:
                        logger.error("Error processing message: %s", exc)

            await queue.consume(on_message)
            logger.info("Subscribed to exchange: %s", exchange_name)

        logger.info("Consumer ready — waiting for messages")
        ready.set()
        await asyncio.Future()
