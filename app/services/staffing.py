"""Shift staffing assistant.

Receives aggregated historical stats from .NET and asks Groq to recommend
how many drivers to schedule for a given district / hour combination.
Returns structured JSON validated by StaffingResponse.
"""

import json
import logging
import re

from groq import AsyncGroq

from app.config import settings
from app.models import StaffingRequest, StaffingResponse

logger = logging.getLogger(__name__)

_groq  = AsyncGroq(api_key=settings.groq_api_key)
_MODEL = "llama-3.3-70b-versatile"

_SYSTEM = """\
You are a logistics operations planner for a Damascus courier service.
Respond ONLY with valid JSON — no explanation, no markdown, no extra text.
Required schema:
{
  "recommended_drivers": <integer ≥ 1>,
  "confidence": "high" | "medium" | "low",
  "reasoning": "<one concise sentence>"
}
Rules:
- recommended_drivers must be a positive integer.
- confidence is "high" when historical data is consistent, "medium" for moderate variance, "low" when data is sparse or a surge was recently detected.
- reasoning must be a single sentence under 25 words.
"""


async def get_staffing(req: StaffingRequest) -> StaffingResponse:
    prompt = _build_prompt(req)
    try:
        raw = await _call_groq(prompt)
        return _parse(raw)
    except Exception as exc:
        logger.warning("Staffing LLM call failed: %s", exc)
        return _safe_default(req)


def _build_prompt(req: StaffingRequest) -> str:
    day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day = day_names[req.day_of_week] if 0 <= req.day_of_week <= 6 else "Unknown"
    surge_note = " A demand surge was recently detected in this district." if req.recent_surge_detected else ""
    return (
        f"{_SYSTEM}\n"
        f"District: {req.district}\n"
        f"Target slot: {day} at {req.target_hour:02d}:00\n"
        f"Historical average deliveries at this slot (last 4 weeks): {req.historical_avg_deliveries:.1f}\n"
        f"{surge_note}\n"
        "How many drivers should be scheduled? Respond with JSON:"
    )


async def _call_groq(prompt: str) -> str:
    resp = await _groq.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
    )
    return resp.choices[0].message.content.strip()


def _parse(raw: str) -> StaffingResponse:
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    match   = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return _safe_default_response()
    try:
        data       = json.loads(match.group())
        drivers    = max(1, int(data.get("recommended_drivers", 2)))
        confidence = data.get("confidence", "low")
        if confidence not in ("high", "medium", "low"):
            confidence = "low"
        reasoning  = str(data.get("reasoning", "Insufficient data for high confidence."))[:200]
        return StaffingResponse(
            recommended_drivers=drivers,
            confidence=confidence,
            reasoning=reasoning,
        )
    except Exception as exc:
        logger.warning("Failed to parse staffing JSON: %s raw=%r", exc, raw[:200])
        return _safe_default_response()


def _safe_default(req: StaffingRequest) -> StaffingResponse:
    estimate = max(1, round(req.historical_avg_deliveries / 3))
    return StaffingResponse(
        recommended_drivers=estimate,
        confidence="low",
        reasoning="AI unavailable — estimate based on historical delivery average.",
    )


def _safe_default_response() -> StaffingResponse:
    return StaffingResponse(
        recommended_drivers=2,
        confidence="low",
        reasoning="Could not parse AI response — manual review advised.",
    )
