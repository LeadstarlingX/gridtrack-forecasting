from uuid import uuid4

import app.services.forecast as forecast_svc
from app.models import DeliveryCompletedIntegrationEvent
from app.services.completion import handle_completion

from datetime import datetime, timezone


def make_completed(district="mezzeh", driver_id=None) -> DeliveryCompletedIntegrationEvent:
    now = datetime.now(timezone.utc)
    did = driver_id or uuid4()
    return DeliveryCompletedIntegrationEvent(
        deliveryId=uuid4(),
        driverId=did,
        districtId=district,
        pickedUpAt=now,
        deliveredAt=now,
        actualDurationSeconds=1200.0,
        expectedDurationSeconds=1500.0,
    )


import pytest


@pytest.fixture(autouse=True)
def reset_state():
    forecast_svc._active_drivers.clear()
    yield
    forecast_svc._active_drivers.clear()


async def test_handle_completion_removes_driver_from_active_set():
    driver_id = uuid4()
    district = "mezzeh"
    forecast_svc._active_drivers[district].add(str(driver_id))
    assert str(driver_id) in forecast_svc._active_drivers[district]

    event = make_completed(district=district, driver_id=driver_id)
    result = await handle_completion(event)

    assert result is None
    assert str(driver_id) not in forecast_svc._active_drivers[district]


async def test_handle_completion_is_idempotent_when_driver_not_in_set():
    event = make_completed(district="kafrsousa")
    result = await handle_completion(event)
    assert result is None


async def test_handle_completion_does_not_affect_other_districts():
    driver_id = uuid4()
    forecast_svc._active_drivers["mezzeh"].add(str(driver_id))
    forecast_svc._active_drivers["malki"].add(str(driver_id))

    event = make_completed(district="mezzeh", driver_id=driver_id)
    await handle_completion(event)

    assert str(driver_id) not in forecast_svc._active_drivers["mezzeh"]
    assert str(driver_id) in forecast_svc._active_drivers["malki"]
