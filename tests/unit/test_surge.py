from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

import app.services.surge as svc
from app.models import DemandSurgeMessage


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture(autouse=True)
def reset_state():
    svc._history.clear()
    svc._last_surge.clear()
    yield


def _prime_history(district: str, counts: list[int]) -> None:
    for c in counts:
        svc._history[district].append(c)


# ── No surge cases ────────────────────────────────────────────────────────────

def test_no_surge_when_insufficient_history():
    result = svc.check_surge("mezzeh", 100, now_utc())
    assert result is None


def test_no_surge_when_exactly_at_threshold_boundary():
    _prime_history("mezzeh", [10, 10, 10, 10])          # 4 samples — one short
    result = svc.check_surge("mezzeh", 100, now_utc())
    assert result is None


def test_no_surge_when_count_is_normal():
    # mean=10, stdev≈0.7 → z = (11-10)/0.7 ≈ 1.4 which is below SURGE_Z_THRESHOLD=2.0
    _prime_history("mezzeh", [10, 11, 9, 10, 10])
    result = svc.check_surge("mezzeh", 11, now_utc())
    assert result is None


# ── Surge detected ────────────────────────────────────────────────────────────

def test_surge_detected_when_count_is_extreme_outlier():
    _prime_history("mezzeh", [5, 5, 5, 5, 5])
    result = svc.check_surge("mezzeh", 50, now_utc())    # way above baseline
    assert isinstance(result, DemandSurgeMessage)
    assert result.districtId == "mezzeh"
    assert result.currentCount == 50
    assert result.deviations >= svc.SURGE_Z_THRESHOLD


def test_surge_message_contains_correct_district():
    _prime_history("babtouma", [3, 3, 3, 3, 3])
    result = svc.check_surge("babtouma", 30, now_utc())
    assert result is not None
    assert result.districtId == "babtouma"


def test_surge_historical_mean_matches_baseline():
    baseline = [10, 10, 10, 10, 10]
    _prime_history("mezzeh", baseline)
    result = svc.check_surge("mezzeh", 100, now_utc())
    assert result is not None
    assert result.historicalMean == 10.0


# ── Cooldown ─────────────────────────────────────────────────────────────────

def test_no_second_surge_within_cooldown():
    _prime_history("mezzeh", [5, 5, 5, 5, 5])
    now = now_utc()
    first  = svc.check_surge("mezzeh", 50, now)
    assert first is not None

    # Append the outlier to history so next call has enough samples
    _prime_history("mezzeh", [5, 5, 5, 5])
    second = svc.check_surge("mezzeh", 50, now + timedelta(minutes=5))
    assert second is None   # still within cooldown


def test_surge_fires_again_after_cooldown_expires():
    _prime_history("mezzeh", [5, 5, 5, 5, 5])
    now = now_utc()
    svc.check_surge("mezzeh", 50, now)

    _prime_history("mezzeh", [5, 5, 5, 5])
    result = svc.check_surge("mezzeh", 50, now + svc.SURGE_COOLDOWN + timedelta(seconds=1))
    assert result is not None


# ── History accumulation ──────────────────────────────────────────────────────

def test_current_count_is_appended_to_history_after_check():
    _prime_history("mezzeh", [5, 5, 5, 5])
    before = len(svc._history["mezzeh"])
    svc.check_surge("mezzeh", 5, now_utc())
    assert len(svc._history["mezzeh"]) == before + 1


def test_district_isolation():
    _prime_history("mezzeh",   [5, 5, 5, 5, 5])
    _prime_history("babtouma", [5, 5, 5, 5, 5])

    svc._last_surge["mezzeh"] = now_utc()           # mezzeh is in cooldown

    result = svc.check_surge("babtouma", 50, now_utc())
    assert result is not None   # babtouma unaffected by mezzeh cooldown
