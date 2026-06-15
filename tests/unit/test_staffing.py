from unittest.mock import AsyncMock, patch
import pytest

import app.services.staffing as svc
from app.models import StaffingRequest, StaffingResponse


def make_request(**kwargs) -> StaffingRequest:
    defaults = dict(
        district="mezzeh",
        target_datetime="2026-06-15T09:00:00Z",
        day_of_week=6,
        target_hour=9,
        historical_avg_deliveries=12.0,
        recent_surge_detected=False,
    )
    defaults.update(kwargs)
    return StaffingRequest(**defaults)


# ── Happy path ────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_returns_valid_response_when_groq_succeeds(monkeypatch):
    groq_json = '{"recommended_drivers": 4, "confidence": "high", "reasoning": "Historical pattern is consistent."}'
    monkeypatch.setattr(svc, "_call_groq", AsyncMock(return_value=groq_json))

    result = await svc.get_staffing(make_request())

    assert isinstance(result, StaffingResponse)
    assert result.recommended_drivers == 4
    assert result.confidence == "high"
    assert "consistent" in result.reasoning


@pytest.mark.anyio
async def test_surge_flag_affects_prompt_but_not_parsing(monkeypatch):
    prompt_captured = []

    async def capture_prompt(prompt: str) -> str:
        prompt_captured.append(prompt)
        return '{"recommended_drivers": 5, "confidence": "medium", "reasoning": "Surge adjustment."}'

    monkeypatch.setattr(svc, "_call_groq", capture_prompt)

    await svc.get_staffing(make_request(recent_surge_detected=True))
    assert "surge" in prompt_captured[0].lower()


# ── Edge cases ────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_clamps_drivers_to_minimum_one(monkeypatch):
    monkeypatch.setattr(svc, "_call_groq", AsyncMock(return_value='{"recommended_drivers": 0, "confidence": "low", "reasoning": "Zero."}'))
    result = await svc.get_staffing(make_request())
    assert result.recommended_drivers >= 1


@pytest.mark.anyio
async def test_invalid_confidence_coerced_to_low(monkeypatch):
    monkeypatch.setattr(svc, "_call_groq", AsyncMock(return_value='{"recommended_drivers": 3, "confidence": "extreme", "reasoning": "Ok."}'))
    result = await svc.get_staffing(make_request())
    assert result.confidence == "low"


@pytest.mark.anyio
async def test_non_json_response_returns_safe_default(monkeypatch):
    monkeypatch.setattr(svc, "_call_groq", AsyncMock(return_value="Sorry, I cannot answer."))
    result = await svc.get_staffing(make_request())
    assert result.recommended_drivers >= 1
    assert result.confidence == "low"


# ── Groq failure / fallback ───────────────────────────────────────────────────

@pytest.mark.anyio
async def test_groq_failure_returns_estimate_based_on_history(monkeypatch):
    monkeypatch.setattr(svc, "_call_groq", AsyncMock(side_effect=Exception("timeout")))

    result = await svc.get_staffing(make_request(historical_avg_deliveries=9.0))
    assert isinstance(result, StaffingResponse)
    assert result.confidence == "low"
    # Estimate is max(1, round(9 / 3)) = 3
    assert result.recommended_drivers == 3


@pytest.mark.anyio
async def test_minimum_one_driver_even_with_zero_history(monkeypatch):
    monkeypatch.setattr(svc, "_call_groq", AsyncMock(side_effect=Exception("timeout")))
    result = await svc.get_staffing(make_request(historical_avg_deliveries=0.0))
    assert result.recommended_drivers >= 1
