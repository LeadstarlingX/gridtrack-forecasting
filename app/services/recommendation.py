"""Structured AI recommendations for delivery operations."""

import json
import logging
import re

from app.models import RecommendationRequest, RecommendationResponse
from app.services.chatbot import call_llm

logger = logging.getLogger(__name__)

_VALID_ACTIONS = frozenset({"Reassign", "Contact", "Cancel", "Monitor"})

_SYSTEM_PROMPT = """\
You are a delivery operations assistant for a Damascus courier service.
Respond ONLY with valid JSON — no explanation, no markdown, no extra text.
Required schema:
{
  "recommended_action": "Reassign" | "Contact" | "Cancel" | "Monitor",
  "candidate_rank": 1 | 2 | 3 | null,
  "reason": "<one concise sentence>",
  "urgency_score": <integer 1-10>
}
Rules:
- candidate_rank must be null when recommended_action is Contact, Cancel, or Monitor.
- candidate_rank is 1, 2, or 3 when recommended_action is Reassign.
- urgency_score reflects how urgent the situation is (10 = critical).
"""


def _build_prompt(req: RecommendationRequest) -> str:
    lines = [
        _SYSTEM_PROMPT,
        f"\nDelivery {req.delivery_id} in district '{req.district_id}'.",
    ]
    if req.anomaly_type:
        lines.append(f"Anomaly detected: {req.anomaly_type}. Reason: {req.anomaly_reason or 'unknown'}.")
    else:
        lines.append("No anomaly — proactive assignment recommendation requested.")

    if req.candidates:
        lines.append("\nAvailable drivers (ranked by composite score):")
        for c in req.candidates:
            rate = f"{c.on_time_rate_pct:.0%}" if c.on_time_rate_pct is not None else "no history"
            lines.append(
                f"  Rank {c.rank}: {c.name} — {c.distance_m:.0f} m away, "
                f"on-time rate {rate}, score {c.score:.3f}"
            )
    else:
        lines.append("\nNo available drivers nearby.")

    lines.append("\nProvide your recommendation as JSON:")
    return "\n".join(lines)


def _parse_response(raw: str) -> RecommendationResponse:
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    # Extract the first JSON object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        logger.warning("LLM returned no JSON block; using safe default. raw=%r", raw[:200])
        return _safe_default()
    try:
        data = json.loads(match.group())
        action = data.get("recommended_action", "Monitor")
        if action not in _VALID_ACTIONS:
            action = "Monitor"
        rank = data.get("candidate_rank")
        if not isinstance(rank, int) or rank not in (1, 2, 3):
            rank = None
        if action == "Reassign" and rank is None:
            rank = 1
        reason = str(data.get("reason", "No reason provided."))[:300]
        urgency = int(data.get("urgency_score", 5))
        urgency = max(1, min(10, urgency))
        return RecommendationResponse(
            recommended_action=action,
            candidate_rank=rank,
            reason=reason,
            urgency_score=urgency,
        )
    except Exception as exc:
        logger.warning("Failed to parse LLM recommendation JSON: %s. raw=%r", exc, raw[:200])
        return _safe_default()


def _safe_default() -> RecommendationResponse:
    return RecommendationResponse(
        recommended_action="Monitor",
        candidate_rank=None,
        reason="AI recommendation unavailable; manual review advised.",
        urgency_score=5,
    )


async def get_recommendation(req: RecommendationRequest) -> RecommendationResponse:
    prompt = _build_prompt(req)
    try:
        raw = await call_llm(prompt)
        return _parse_response(raw)
    except Exception as exc:
        logger.warning("Recommendation LLM call failed: %s", exc)
        return _safe_default()
