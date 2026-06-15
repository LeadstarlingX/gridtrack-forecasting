import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

from app.models import DriverPositionIntegrationEvent, ForecastResultMessage

logger = logging.getLogger(__name__)

_windows: dict[str, deque[datetime]] = defaultdict(deque)
_active_drivers: dict[str, set[str]] = defaultdict(set)
_last_emit: dict[str, datetime] = {}

WINDOW = timedelta(minutes=60)
EMIT_INTERVAL = timedelta(minutes=5)
CRITICAL_RATIO = 0.70
MODERATE_RATIO = 0.85


async def update_forecast(
    event: DriverPositionIntegrationEvent,
) -> ForecastResultMessage | None:
    now = datetime.now(timezone.utc)
    district = event.districtId

    _active_drivers[district].add(str(event.driverId))
    _windows[district].append(now)

    cutoff = now - WINDOW
    while _windows[district] and _windows[district][0] < cutoff:
        _windows[district].popleft()

    if not _should_emit(district, now):
        return None

    half_cutoff = now - timedelta(minutes=30)
    count_last_30 = sum(1 for t in _windows[district] if t >= half_cutoff)
    expected = count_last_30 * 2

    driver_count = len(_active_drivers[district])
    ratio = driver_count / expected if expected > 0 else 1.0

    if ratio < CRITICAL_RATIO:
        label, color = "Critical", "#f87171"
    elif ratio < MODERATE_RATIO:
        label, color = "Moderate", "#fbbf24"
    else:
        label, color = "Low demand", "#34d399"

    logger.info(
        "Forecast %s: expected=%d drivers=%d ratio=%.2f label=%s",
        district, expected, driver_count, ratio, label,
    )

    return ForecastResultMessage(
        districtId=district,
        expectedDeliveries=expected,
        staffingRatio=round(ratio, 2),
        label=label,
        color=color,
        generatedAt=now.isoformat(),
    )


def release_driver(district: str, driver_id: str) -> None:
    _active_drivers[district].discard(driver_id)


def _should_emit(district: str, now: datetime) -> bool:
    last = _last_emit.get(district)
    if last is None or (now - last) >= EMIT_INTERVAL:
        _last_emit[district] = now
        return True
    return False
