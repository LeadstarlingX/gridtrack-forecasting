import logging

from app.models import DeliveryCompletedIntegrationEvent
from app.services.forecast import release_driver

logger = logging.getLogger(__name__)


async def handle_completion(event: DeliveryCompletedIntegrationEvent) -> None:
    release_driver(event.districtId, str(event.driverId))
    logger.info(
        "Delivery %s completed in district %s — driver %s released",
        event.deliveryId,
        event.districtId,
        event.driverId,
    )
    return None
