import logging

from groq import AsyncGroq

from app.config import settings
from app.models import DeliveryAnomalyIntegrationEvent, UrgencyResultMessage

logger = logging.getLogger(__name__)

URGENCY_BASE: dict[str, int] = {
    "StalePosition":  5,
    "EtaExceeded":    3,
    "RouteDeviation": 4,
    "UnexpectedStop": 4,
}

DISTRICT_BOOST: dict[str, int] = {
    "kafrsousa": 2,
    "babtouma":  1,
    "malki":     0,
    "mezzeh":    0,
}

_groq = AsyncGroq(api_key=settings.groq_api_key)


async def score_anomaly(event: DeliveryAnomalyIntegrationEvent) -> UrgencyResultMessage:
    score = URGENCY_BASE.get(event.anomalyType, 2)
    score += DISTRICT_BOOST.get(event.districtId, 0)
    score = min(10, score)

    try:
        note = await _groq_note(event, score)
    except Exception as exc:
        logger.warning("Groq call failed (%s), using fallback note", exc)
        note = _fallback_note(event, score)

    return UrgencyResultMessage(
        deliveryId=str(event.deliveryId),
        urgencyScore=score,
        aiNote=note,
    )


async def _groq_note(event: DeliveryAnomalyIntegrationEvent, score: int) -> str:
    prompt = (
        f"Delivery anomaly: type={event.anomalyType}, "
        f"reason='{event.reason}', district={event.districtId}, "
        f"urgency={score}/10. "
        "Write a single concise action note for a dispatcher (max 15 words)."
    )
    resp = await _groq.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=40,
    )
    return resp.choices[0].message.content.strip()


def _fallback_note(event: DeliveryAnomalyIntegrationEvent, score: int) -> str:
    level = "Critical" if score >= 8 else "High" if score >= 6 else "Moderate"
    return f"{level} — {event.reason.lower()}. Manual check recommended."
