"""Unit tests for recommendation service: prompt building and response parsing."""

import pytest
from app.models import CandidateContext, RecommendationRequest
from app.services.recommendation import _build_prompt, _parse_response


def _make_req(**kwargs) -> RecommendationRequest:
    defaults = {
        "delivery_id": "abc-123",
        "district_id": "mezzeh",
        "anomaly_type": "EtaExceeded",
        "anomaly_reason": "Stalled 20 min",
        "candidates": [
            CandidateContext(rank=1, name="Ahmad", distance_m=300, on_time_rate_pct=0.8, score=0.75),
            CandidateContext(rank=2, name="Ziad",  distance_m=900, on_time_rate_pct=None, score=0.55),
        ],
    }
    return RecommendationRequest(**{**defaults, **kwargs})


# ── prompt building ────────────────────────────────────────────────────────────

def test_prompt_includes_district_and_anomaly():
    req = _make_req()
    prompt = _build_prompt(req)
    assert "mezzeh" in prompt
    assert "EtaExceeded" in prompt
    assert "Stalled 20 min" in prompt


def test_prompt_includes_candidate_names_and_ranks():
    req = _make_req()
    prompt = _build_prompt(req)
    assert "Rank 1" in prompt
    assert "Ahmad" in prompt
    assert "Rank 2" in prompt
    assert "Ziad" in prompt


def test_prompt_shows_no_history_when_on_time_rate_is_none():
    req = _make_req()
    prompt = _build_prompt(req)
    assert "no history" in prompt


def test_prompt_labels_proactive_when_no_anomaly():
    req = _make_req(anomaly_type=None, anomaly_reason=None)
    prompt = _build_prompt(req)
    assert "proactive" in prompt.lower()


# ── response parsing ───────────────────────────────────────────────────────────

def test_parse_valid_json():
    raw = '{"recommended_action": "Reassign", "candidate_rank": 2, "reason": "Test.", "urgency_score": 8}'
    result = _parse_response(raw)
    assert result.recommended_action == "Reassign"
    assert result.candidate_rank == 2
    assert result.urgency_score == 8


def test_parse_strips_markdown_fences():
    raw = "```json\n{\"recommended_action\": \"Monitor\", \"candidate_rank\": null, \"reason\": \"ok\", \"urgency_score\": 3}\n```"
    result = _parse_response(raw)
    assert result.recommended_action == "Monitor"
    assert result.candidate_rank is None


def test_parse_clamps_urgency_to_valid_range():
    raw = '{"recommended_action": "Cancel", "candidate_rank": null, "reason": "x", "urgency_score": 99}'
    result = _parse_response(raw)
    assert result.urgency_score == 10

    raw2 = '{"recommended_action": "Cancel", "candidate_rank": null, "reason": "x", "urgency_score": -5}'
    result2 = _parse_response(raw2)
    assert result2.urgency_score == 1


def test_parse_invalid_action_falls_back_to_monitor():
    raw = '{"recommended_action": "Destroy", "candidate_rank": null, "reason": "x", "urgency_score": 5}'
    result = _parse_response(raw)
    assert result.recommended_action == "Monitor"


def test_parse_garbage_returns_safe_default():
    result = _parse_response("This is not JSON at all.")
    assert result.recommended_action == "Monitor"
    assert result.urgency_score == 5


def test_parse_reassign_with_missing_rank_defaults_to_1():
    raw = '{"recommended_action": "Reassign", "candidate_rank": null, "reason": "x", "urgency_score": 5}'
    result = _parse_response(raw)
    assert result.recommended_action == "Reassign"
    assert result.candidate_rank == 1
