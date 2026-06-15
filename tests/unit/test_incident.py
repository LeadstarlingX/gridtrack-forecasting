from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

import app.services.incident as svc
from app.models import AnomalyIncidentMessage, DeliveryAnomalyIntegrationEvent


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def make_anomaly(district: str = "mezzeh", anomaly_type: str = "EtaExceeded") -> DeliveryAnomalyIntegrationEvent:
    return DeliveryAnomalyIntegrationEvent(
        deliveryId=uuid4(),
        districtId=district,
        anomalyType=anomaly_type,
        reason="Test reason",
        driverLat=33.5,
        driverLng=36.2,
        occurredAt=now_utc(),
    )


@pytest.fixture(autouse=True)
def reset_state():
    svc._anomaly_window.clear()
    svc._last_incident.clear()
    yield


# ── Below threshold ───────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_no_incident_below_threshold():
    now = now_utc()
    for _ in range(svc.INCIDENT_THRESHOLD - 1):
        result = await svc.check_incident(make_anomaly(), now)
    assert result is None


@pytest.mark.anyio
async def test_no_incident_single_anomaly():
    result = await svc.check_incident(make_anomaly(), now_utc())
    assert result is None


# ── Threshold crossed ─────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_incident_fires_at_threshold(monkeypatch):
    monkeypatch.setattr(svc, "_groq_summary", AsyncMock(return_value="3 stalls in mezzeh"))

    now = now_utc()
    result = None
    for _ in range(svc.INCIDENT_THRESHOLD):
        result = await svc.check_incident(make_anomaly(), now)

    assert isinstance(result, AnomalyIncidentMessage)
    assert result.districtId == "mezzeh"
    assert result.anomalyCount == svc.INCIDENT_THRESHOLD
    assert result.summary == "3 stalls in mezzeh"


@pytest.mark.anyio
async def test_incident_contains_30_minute_window(monkeypatch):
    monkeypatch.setattr(svc, "_groq_summary", AsyncMock(return_value="Incident"))

    now = now_utc()
    for _ in range(svc.INCIDENT_THRESHOLD):
        result = await svc.check_incident(make_anomaly(), now)

    assert result is not None
    assert result.windowMinutes == 30


# ── Old anomalies are evicted ─────────────────────────────────────────────────

@pytest.mark.anyio
async def test_old_anomalies_outside_window_do_not_count():
    old = now_utc() - timedelta(minutes=40)   # outside 30-min window
    for _ in range(svc.INCIDENT_THRESHOLD):
        svc._anomaly_window["mezzeh"].append((old, "EtaExceeded"))

    result = await svc.check_incident(make_anomaly(), now_utc())
    assert result is None                    # old entries evicted, only 1 new


# ── Cooldown ──────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_no_second_incident_within_cooldown(monkeypatch):
    monkeypatch.setattr(svc, "_groq_summary", AsyncMock(return_value="Incident"))

    now = now_utc()
    for _ in range(svc.INCIDENT_THRESHOLD):
        await svc.check_incident(make_anomaly(), now)

    # Another batch immediately after
    for _ in range(svc.INCIDENT_THRESHOLD):
        result = await svc.check_incident(make_anomaly(), now + timedelta(minutes=1))
    assert result is None


@pytest.mark.anyio
async def test_incident_fires_again_after_cooldown(monkeypatch):
    monkeypatch.setattr(svc, "_groq_summary", AsyncMock(return_value="Incident"))

    now = now_utc()
    for _ in range(svc.INCIDENT_THRESHOLD):
        await svc.check_incident(make_anomaly(), now)

    after_cooldown = now + svc.INCIDENT_COOLDOWN + timedelta(seconds=1)
    result = None
    for _ in range(svc.INCIDENT_THRESHOLD):
        result = await svc.check_incident(make_anomaly(), after_cooldown)

    assert result is not None


# ── District isolation ────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_incident_is_per_district(monkeypatch):
    monkeypatch.setattr(svc, "_groq_summary", AsyncMock(return_value="Incident"))

    now = now_utc()
    # Max out mezzeh cooldown
    svc._last_incident["mezzeh"] = now

    # babtouma should fire independently
    result = None
    for _ in range(svc.INCIDENT_THRESHOLD):
        result = await svc.check_incident(make_anomaly("babtouma"), now)

    assert result is not None
    assert result.districtId == "babtouma"


# ── Groq fallback ────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_groq_failure_returns_incident_with_fallback_summary(monkeypatch):
    """check_incident must not propagate _groq_summary failures; it falls back to a static summary."""
    monkeypatch.setattr(svc, "_groq_summary", AsyncMock(side_effect=Exception("Groq down")))

    now = now_utc()
    result = None
    for _ in range(svc.INCIDENT_THRESHOLD):
        result = await svc.check_incident(make_anomaly(), now)

    assert isinstance(result, AnomalyIncidentMessage)
    assert result.anomalyCount == svc.INCIDENT_THRESHOLD
    assert "manual review" in result.summary.lower()
