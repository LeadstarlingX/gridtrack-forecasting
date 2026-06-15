"""Demand-surge detector.

Called every time forecast.py emits a result (once per EMIT_INTERVAL per district).
Maintains a rolling history of position-event counts and fires a DemandSurgeMessage
when the current count exceeds the historical baseline by SURGE_Z_THRESHOLD standard
deviations, subject to a cooldown so the same district cannot spam surge alerts.
"""

import statistics
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

from app.models import DemandSurgeMessage

logger = logging.getLogger(__name__)

SURGE_Z_THRESHOLD = 2.0      # sigma above baseline to trigger
SURGE_MIN_SAMPLES = 5        # need at least this many historical readings
SURGE_COOLDOWN    = timedelta(minutes=15)

_history:    dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=24))
_last_surge: dict[str, datetime]  = {}


def check_surge(district: str, current_count: int, now: datetime) -> DemandSurgeMessage | None:
    """Check whether current_count is a statistical surge vs the rolling history.

    The current count is appended to history AFTER the check so it acts as the
    observation, not part of the baseline.
    """
    baseline = list(_history[district])

    surge: DemandSurgeMessage | None = None

    if len(baseline) >= SURGE_MIN_SAMPLES:
        mean  = statistics.mean(baseline)
        stdev = statistics.stdev(baseline) if len(baseline) >= 2 else 0.0

        if stdev == 0:
            # All historical values identical; flag surge if current is materially higher.
            stdev = max(1.0, mean * 0.1)

        z = (current_count - mean) / stdev
        if z >= SURGE_Z_THRESHOLD:
                last = _last_surge.get(district)
                if last is None or (now - last) >= SURGE_COOLDOWN:
                    _last_surge[district] = now
                    surge = DemandSurgeMessage(
                        districtId=district,
                        currentCount=current_count,
                        historicalMean=round(mean, 1),
                        deviations=round(z, 2),
                        detectedAt=now.isoformat(),
                    )
                    logger.info(
                        "Demand surge in %s: count=%d mean=%.1f z=%.2f",
                        district, current_count, mean, z,
                    )

    _history[district].append(current_count)
    return surge
