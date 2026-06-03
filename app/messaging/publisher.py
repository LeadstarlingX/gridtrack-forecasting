import logging

import aio_pika
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_QUEUE_MAP: dict[str, str] = {
    "UrgencyResultMessage":  "gridtrack.urgency-results",
    "ForecastResultMessage": "gridtrack.forecast-results",
}


async def publish(channel: aio_pika.Channel, message: BaseModel) -> None:
    type_name = type(message).__name__
    queue_name = _QUEUE_MAP.get(type_name)
    if not queue_name:
        logger.warning("No queue mapped for message type: %s", type_name)
        return

    body = message.model_dump_json().encode()
    await channel.default_exchange.publish(
        aio_pika.Message(
            body=body,
            content_type="application/json",
        ),
        routing_key=queue_name,
    )
    logger.debug("Published %s to %s", type_name, queue_name)
