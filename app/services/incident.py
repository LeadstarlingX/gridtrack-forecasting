"""Anomaly incident clustering.

Tracks anomaly timestamps per district in a rolling 30-minute window.
When INCIDENT_THRESHOLD anomalies accumulate, calls Groq for a one-line
incident summary and emits an AnomalyIncidentMessage — once per INCIDENT_COOLDOWN.
"""

import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

from groq import AsyncGroq

from app.config import settings
from app.models import AnomalyIncidentMessage, DeliveryAnomalyIntegrationEvent

logger = logging.getLogger(__name__)

INCIDENT_THRESHOLD = 3
INCIDENT_WINDOW    = timedelta(minutes=30)
INCIDENT_COOLDOWN  = timedelta(minutes=30)

_anomaly_window: dict[str, deque[tuple[datetime, str]]] = defaultdict(deque)
_last_incident:  dict[str, datetime] = {}

_groq = AsyncGroq(api_key=settings.groq_api_key)
_MODEL = "llama-3.3-70b-versatile"


async def check_incident(
    event: DeliveryAnomalyIntegrationEvent,
    now: datetime | None = None,
) -> AnomalyIncidentMessage | None:
    """Record anomaly and return an incident message if the threshold is crossed."""
    if now is None:
        now = datetime.now(timezone.utc)

    district = event.districtId
    window   = _anomaly_window[district]
    window.append((now, event.anomalyType))

    cutoff = now - INCIDENT_WINDOW
    while window and window[0][0] < cutoff:
        window.popleft()

    if len(window) < INCIDENT_THRESHOLD:
        return None

    last = _last_incident.get(district)
    if last is not None and (now - last) < INCIDENT_COOLDOWN:
        return None

    _last_incident[district] = now
    count  = len(window)
    types  = [t for _, t in window]
    try:
        summary = await _groq_summary(district, count, types)
    except Exception as exc:
        logger.warning("_groq_summary raised unexpectedly: %s", exc)
        summary = f"{count} anomalies in {district} — manual review needed"

    logger.info("Incident in %s: %d anomalies — %s", district, count, summary)
    return AnomalyIncidentMessage(
        districtId=district,
        anomalyCount=count,
        windowMinutes=30,
        summary=summary,
        detectedAt=now.isoformat(),
    )


async def _groq_summary(district: str, count: int, types: list[str]) -> str:
    type_summary = ", ".join(
        f"{v}×{k}" for k, v in
        sorted(
            {t: types.count(t) for t in set(types)}.items(),
            key=lambda x: -x[1],
        )
    )
    prompt = (
        f"Delivery operations incident in district '{district}', Damascus. "
        f"{count} anomalies detected in the last 30 minutes: {type_summary}. "
        "Write a single concise dispatcher alert in under 20 words. "
        "Do not use punctuation at the end."
    )
    try:
        resp = await _groq.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("Groq incident summary failed: %s", exc)
        return f"{count} anomalies detected in {district} — manual review needed"
