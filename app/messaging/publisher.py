import logging

import aio_pika
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_QUEUE_MAP: dict[str, str] = {
    "UrgencyResultMessage":  "gridtrack.urgency-results",
    "ForecastResultMessage": "gridtrack.forecast-results",
}

# Wolverine identifies message types by their full .NET type name, lowercased.
# Format: "{namespace}.{classname}" — must match the C# type exactly.
# See: GridTrack.Application.IntegrationEvents in the .NET backend.
_MESSAGE_TYPE_HEADERS: dict[str, str] = {
    "UrgencyResultMessage":  "gridtrack.application.integrationevents.urgencyresultmessage",
    "ForecastResultMessage": "gridtrack.application.integrationevents.forecastresultmessage",
}


async def publish(channel: aio_pika.Channel, message: BaseModel) -> None:
    type_name = type(message).__name__
    queue_name = _QUEUE_MAP.get(type_name)
    if not queue_name:
        logger.warning("No queue mapped for message type: %s", type_name)
        return

    message_type = _MESSAGE_TYPE_HEADERS.get(type_name)
    if not message_type:
        logger.warning("No message-type header mapped for: %s", type_name)
        return

    body = message.model_dump_json().encode()
    await channel.default_exchange.publish(
        aio_pika.Message(
            body=body,
            content_type="application/json",
            headers={"message-type": message_type},
        ),
        routing_key=queue_name,
    )
    logger.debug("Published %s → %s (type=%s)", type_name, queue_name, message_type)
