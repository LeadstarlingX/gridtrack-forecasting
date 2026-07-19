"""Delivery demand forecasting.

Real-time per-district staffing ratio via a SARIMA model fitted on
historical hourly delivery counts from Postgres (cache refreshed hourly).
Falls back to the naive count×2 extrapolation when history is too sparse.

Global (system-wide) SARIMA forecast powers the KPI delivery trend chart.
"""

import asyncio
import logging
import os
from collections import defaultdict, deque
from datetime import datetime, date, timedelta, timezone

from app.db import get_pool
from app.models import DriverPositionIntegrationEvent, ForecastResultMessage

logger = logging.getLogger(__name__)

# ── Real-time sliding window (per-district) ───────────────────────────────────
_windows: dict[str, deque[datetime]] = defaultdict(deque)
_active_drivers: dict[str, set[str]] = defaultdict(set)
_last_emit: dict[str, datetime] = {}

WINDOW = timedelta(minutes=60)
EMIT_INTERVAL = timedelta(minutes=5)
CRITICAL_RATIO = 0.70
MODERATE_RATIO = 0.85

# ── SARIMA cache ──────────────────────────────────────────────────────────────
# Keyed by district → (expected_count, fitted_at).
# Re-fitted at most once per hour to avoid blocking the consumer loop.
_sarima_cache: dict[str, tuple[float, datetime]] = {}
SARIMA_TTL = timedelta(hours=1)
SARIMA_MIN_POINTS = 48  # 2 full seasonal periods (s=24 hours)
# 0 = no cap; set SARIMA_DAILY_CAP env var to limit sim-inflated predictions
SARIMA_DAILY_CAP: float = float(os.getenv("SARIMA_DAILY_CAP", "0"))


# ── Public API ────────────────────────────────────────────────────────────────

async def update_forecast(event: DriverPositionIntegrationEvent) -> ForecastResultMessage | None:
    now = datetime.now(timezone.utc)
    district = event.districtId

    _active_drivers[district].add(str(event.driverId))
    _windows[district].append(now)

    cutoff = now - WINDOW
    while _windows[district] and _windows[district][0] < cutoff:
        _windows[district].popleft()

    if not _should_emit(district, now):
        return None

    expected = await _get_expected(district)
    driver_count = len(_active_drivers[district])
    ratio = driver_count / expected if expected > 0 else 1.0

    if ratio < CRITICAL_RATIO:
        label, color = "Critical", "#f87171"
    elif ratio < MODERATE_RATIO:
        label, color = "Moderate", "#fbbf24"
    else:
        label, color = "Low demand", "#34d399"

    logger.info(
        "Forecast %s: expected=%.1f drivers=%d ratio=%.2f label=%s",
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


async def sarima_forecast_global(days: int = 3) -> list[dict]:
    """System-wide delivery demand forecast aggregated to daily buckets.

    Uses a seasonal-naive baseline (last 7-day daily average) anchored by the
    week-over-week growth trend, capped at ±8 % per day.  This is intentionally
    simpler than SARIMA: our seeded training data is too regular (fixed random
    seed, identical daily distributions) for SARIMAX to produce stable multi-day
    forecasts — it either goes numerically explosive or decays to zero.  For real
    operational data with natural variance, swap this back to _fit_and_predict.
    """
    try:
        pool = await get_pool()
        rows = await pool.fetch(
            """
            SELECT date_trunc('hour', "CreatedAt" AT TIME ZONE 'UTC') AS hr,
                   COUNT(*)::float                                      AS cnt
            FROM   "Deliveries"
            WHERE  "CreatedAt" >= NOW() - INTERVAL '14 days'
            GROUP  BY 1
            ORDER  BY 1
            """,
        )
        if not rows:
            return []

        counts = [float(r["cnt"]) for r in rows]
        last_hr: datetime = rows[-1]["hr"]
        if last_hr.tzinfo is None:
            last_hr = last_hr.replace(tzinfo=timezone.utc)
        base_day: date = (last_hr + timedelta(hours=1)).date()

        # Daily average for the most recent half of the data — the forecast baseline.
        # Use the second half vs first half so the trend is always computable
        # regardless of how many hourly buckets are present (some hours drop out of
        # GROUP BY when they have zero deliveries, so len(counts) < 14*24 is common).
        half        = max(len(counts) // 2, 1)
        curr_window = len(counts) - half
        curr_daily  = sum(counts[half:]) / max(1, curr_window / 24)
        prev_daily  = sum(counts[:half]) / max(1, half / 24)

        # Day-over-day growth rate derived from the two halves; clipped to ±8 %/day.
        days_in_half = curr_window / 24
        daily_trend  = (curr_daily - prev_daily) / max(prev_daily, 1) / max(days_in_half, 1)
        daily_trend  = max(-0.08, min(0.08, daily_trend))

        return [
            {
                "bucket": (base_day + timedelta(days=d)).isoformat(),
                "value":  round(curr_daily * (1.0 + daily_trend * (d + 1)), 1),
            }
            for d in range(days)
        ]

    except Exception as exc:
        logger.warning("Global delivery trend forecast failed: %s", exc)
        return []


# ── Private helpers ───────────────────────────────────────────────────────────

def _should_emit(district: str, now: datetime) -> bool:
    last = _last_emit.get(district)
    if last is None or (now - last) >= EMIT_INTERVAL:
        _last_emit[district] = now
        return True
    return False


async def _get_expected(district: str) -> float:
    """Return SARIMA-predicted expected deliveries for the next hour (cached hourly)."""
    cached = _sarima_cache.get(district)
    if cached and (datetime.now(timezone.utc) - cached[1]) < SARIMA_TTL:
        return cached[0]
    return await _refit_district(district)


async def _refit_district(district: str) -> float:
    try:
        pool = await get_pool()
        rows = await pool.fetch(
            """
            SELECT date_trunc('hour', "CreatedAt" AT TIME ZONE 'UTC') AS hr,
                   COUNT(*)::float                                      AS cnt
            FROM   "Deliveries"
            WHERE  "DistrictId" = $1
              AND  "CreatedAt" >= NOW() - INTERVAL '14 days'
            GROUP  BY 1
            ORDER  BY 1
            """,
            district,
        )
        counts = [float(r["cnt"]) for r in rows]

        if len(counts) < SARIMA_MIN_POINTS:
            fallback = (counts[-1] * 2) if counts else 4.0
            _sarima_cache[district] = (fallback, datetime.now(timezone.utc))
            return fallback

        expected = await asyncio.get_running_loop().run_in_executor(
            None, _fit_and_predict, counts, 1
        )
        value = expected[0]
        _sarima_cache[district] = (value, datetime.now(timezone.utc))
        logger.info("SARIMA refit district=%s expected=%.1f", district, value)
        return value

    except Exception as exc:
        logger.warning("SARIMA district refit failed (%s): %s", district, exc)
        stale = _sarima_cache.get(district)
        return stale[0] if stale else 4.0


def _fit_and_predict(counts: list[float], steps: int) -> list[float]:
    """Synchronous SARIMA fit — must be called via run_in_executor.

    Uses SARIMA(1,0,1)(1,0,1,24) — no differencing.  Double-differencing
    (d=1,D=1) drives near-stationary data to near-zero after transformation,
    causing the scale anchor to amplify predictions to astronomically large
    values.  With d=0,D=0 the model fits on the raw level directly and
    forecasts stay in the same magnitude as the training data.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX  # lazy import
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            counts,
            order=(2, 0, 0),
            seasonal_order=(1, 0, 0, 24),
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        fit = model.fit(disp=False, maxiter=50)
    return [max(0.0, float(v)) for v in fit.forecast(steps=steps)]
