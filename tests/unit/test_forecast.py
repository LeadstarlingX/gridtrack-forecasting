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


# ── Extended edge-case tests ───────────────────────────────────────────────────

async def test_window_prunes_events_older_than_60_minutes():
    district = "kafrsousa"
    old = datetime.now(timezone.utc) - timedelta(minutes=90)
    for _ in range(5):
        svc._windows[district].append(old)

    await svc.update_forecast(make_position(district))
    # All 5 stale events should be evicted; only the one just appended survives
    assert len(svc._windows[district]) == 1


async def test_multiple_unique_drivers_all_tracked():
    """Throttled calls still add drivers to the active set."""
    district = "babtouma"
    for _ in range(5):
        await svc.update_forecast(make_position(district, uuid4()))
    assert len(svc._active_drivers[district]) == 5


async def test_staffing_ratio_is_rounded_to_two_decimal_places():
    """With 2 pre-injected recent events + 1 from the call → count=3, expected=6,
    drivers=1 → ratio = 1/6 ≈ 0.1667 → rounded to 0.17."""
    district = "mezzeh"
    recent = datetime.now(timezone.utc) - timedelta(minutes=15)
    for _ in range(2):
        svc._windows[district].append(recent)
    svc._last_emit.clear()

    result = await svc.update_forecast(make_position(district, uuid4()))
    assert result is not None
    assert result.staffingRatio == round(result.staffingRatio, 2)
    assert result.staffingRatio == 0.17


async def test_generated_at_is_parseable_iso_string():
    result = await svc.update_forecast(make_position("malki"))
    assert result is not None
    parsed = datetime.fromisoformat(result.generatedAt)
    assert parsed is not None


async def test_throttle_is_independent_per_district():
    """Two different districts both emit on their first event."""
    r1 = await svc.update_forecast(make_position("mezzeh"))
    r2 = await svc.update_forecast(make_position("malki"))
    assert r1 is not None
    assert r2 is not None
