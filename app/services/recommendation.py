"""Structured AI recommendations for delivery operations."""

import json
import logging
import re

from app.models import RecommendationRequest, RecommendationResponse
from app.services.chatbot import call_llm

logger = logging.getLogger(__name__)

_VALID_ACTIONS = frozenset({"Reassign", "Contact", "Cancel", "Monitor"})

def _build_prompt(req: RecommendationRequest) -> str:
    lines = [f"Delivery {req.delivery_id}, district '{req.district_id}'."]

    if req.anomaly_type:
        lines.append(f"Anomaly: {req.anomaly_type}. {req.anomaly_reason or 'unknown'}.")
    else:
        lines.append("No anomaly — proactive assignment recommendation.")

    if req.candidates:
        lines.append("Available drivers (ranked by composite score):")
        for c in req.candidates:
            rate = f"{c.on_time_rate_pct:.0%}" if c.on_time_rate_pct is not None else "no history"
            lines.append(
                f"  Rank {c.rank}: {c.name} — {c.distance_m:.0f} m away, "
                f"on-time rate {rate}, score {c.score:.3f}"
            )
    else:
        lines.append("No available drivers nearby.")

    lines.append(
        '\nOutput a single JSON object — no other text:\n'
        '{"recommended_action":"Reassign"|"Contact"|"Cancel"|"Monitor",'
        '"candidate_rank":1|2|3|null,"reason":"one sentence","urgency_score":1-10}\n'
        'candidate_rank is 1-3 only when Reassign, else null.\n'
        'Output:'
    )
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
        raw = await call_llm(prompt, response_format={"type": "json_object"})
        return _parse_response(raw)
    except Exception as exc:
        logger.warning("Recommendation LLM call failed: %s", exc)
        return _safe_default()
