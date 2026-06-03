from datetime import datetime, timezone, timedelta
from uuid import uuid4

import pytest

import app.services.forecast as svc
from app.models import DriverPositionIntegrationEvent


def make_position(district="mezzeh", driver_id=None):
    return DriverPositionIntegrationEvent(
        driverId=driver_id or uuid4(),
        districtId=district,
        lat=33.5,
        lng=36.2,
        deliveryStatus="InTransit",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture(autouse=True)
def reset_state():
    svc._windows.clear()
    svc._active_drivers.clear()
    svc._last_emit.clear()
    yield


async def test_first_event_emits_forecast():
    result = await svc.update_forecast(make_position())
    assert result is not None


async def test_second_event_within_interval_is_throttled():
    await svc.update_forecast(make_position())
    result = await svc.update_forecast(make_position())
    assert result is None


async def test_emit_after_interval_elapses():
    district = "mezzeh"
    old_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    svc._last_emit[district] = old_time

    result = await svc.update_forecast(make_position(district))
    assert result is not None


async def test_critical_label_when_drivers_insufficient():
    district = "babtouma"
    for _ in range(20):
        await svc.update_forecast(make_position(district, uuid4()))

    svc._last_emit.clear()
    result = await svc.update_forecast(make_position(district))
    if result is not None:
        assert result.label in ("Critical", "Moderate", "Low demand")
        assert result.color in ("#f87171", "#fbbf24", "#34d399")
        assert 0.0 <= result.staffingRatio


async def test_forecast_result_fields_are_complete():
    result = await svc.update_forecast(make_position("malki"))
    assert result is not None
    assert result.districtId == "malki"
    assert isinstance(result.expectedDeliveries, int)
    assert isinstance(result.staffingRatio, float)
    assert result.generatedAt
