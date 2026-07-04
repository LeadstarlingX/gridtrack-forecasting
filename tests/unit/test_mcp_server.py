"""Tests for app/mcp_server.py — MCP tools and _BearerAuth middleware."""
import json
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

from app.mcp_server import (
    _BearerAuth,
    get_active_drivers,
    get_anomalies,
    get_deliveries_summary,
    get_district_status,
    get_stalled_drivers,
    get_activity_trend,
    get_peak_hours,
)


# ── _BearerAuth ───────────────────────────────────────────────────────────────

async def test_bearer_auth_allows_valid_token():
    inner = AsyncMock()
    auth = _BearerAuth(inner, "mykey")
    scope = {"type": "http", "headers": [(b"authorization", b"Bearer mykey")]}
    await auth(scope, AsyncMock(), AsyncMock())
    inner.assert_awaited_once()


async def test_bearer_auth_rejects_wrong_token():
    inner = AsyncMock()
    auth = _BearerAuth(inner, "mykey")
    send = AsyncMock()
    scope = {"type": "http", "headers": [(b"authorization", b"Bearer wrong")]}
    await auth(scope, AsyncMock(), send)
    inner.assert_not_awaited()
    first_msg = send.call_args_list[0][0][0]
    assert first_msg["status"] == 401


async def test_bearer_auth_rejects_missing_token():
    inner = AsyncMock()
    auth = _BearerAuth(inner, "secret")
    send = AsyncMock()
    await auth({"type": "http", "headers": []}, AsyncMock(), send)
    inner.assert_not_awaited()
    assert send.call_args_list[0][0][0]["status"] == 401


async def test_bearer_auth_passes_non_http_scope():
    inner = AsyncMock()
    auth = _BearerAuth(inner, "mykey")
    # lifespan events bypass auth
    await auth({"type": "lifespan"}, AsyncMock(), AsyncMock())
    inner.assert_awaited_once()


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_pool(*rows_per_call):
    """Return a mock asyncpg pool whose fetch/fetchval calls return successive rows."""
    pool = AsyncMock()
    pool.fetch.side_effect = list(rows_per_call)
    pool.fetchval.return_value = 3
    return pool


def _row(**kw):
    """A dict that supports dict(r) like an asyncpg Record."""
    return kw


# ── get_active_drivers ────────────────────────────────────────────────────────

async def test_get_active_drivers_returns_all_when_no_filter(mocker):
    rows = [_row(DriverId="uid1", Name="Ali", DistrictId="d1", lat=33.5, lng=36.2)]
    mocker.patch("app.mcp_server.get_pool", new=AsyncMock(return_value=_make_pool(rows)))
    result = json.loads(await get_active_drivers())
    assert len(result) == 1
    assert result[0]["Name"] == "Ali"


async def test_get_active_drivers_filters_by_district(mocker):
    rows = [_row(DriverId="uid2", Name="Omar", DistrictId="d2", lat=33.6, lng=36.3)]
    mocker.patch("app.mcp_server.get_pool", new=AsyncMock(return_value=_make_pool(rows)))
    result = json.loads(await get_active_drivers(district_id="d2"))
    assert result[0]["DistrictId"] == "d2"


# ── get_anomalies ─────────────────────────────────────────────────────────────

async def test_get_anomalies_no_filter(mocker):
    rows = [_row(DeliveryId="del1", Status="InTransit", DistrictId="d1",
                 CreatedAt="2026-07-01", ExpectedEta="2026-07-01")]
    mocker.patch("app.mcp_server.get_pool", new=AsyncMock(return_value=_make_pool(rows)))
    result = json.loads(await get_anomalies())
    assert result[0]["DeliveryId"] == "del1"


async def test_get_anomalies_with_district(mocker):
    rows = []
    mocker.patch("app.mcp_server.get_pool", new=AsyncMock(return_value=_make_pool(rows)))
    result = json.loads(await get_anomalies(district_id="d1", hours=2))
    assert result == []


# ── get_deliveries_summary ───────────────────────────────────────────────────

async def test_get_deliveries_summary_groups_by_status(mocker):
    rows = [_row(Status="Delivered", DistrictId="d1", count=5)]
    mocker.patch("app.mcp_server.get_pool", new=AsyncMock(return_value=_make_pool(rows)))
    result = json.loads(await get_deliveries_summary())
    assert result[0]["count"] == 5


# ── get_district_status ──────────────────────────────────────────────────────

async def test_get_district_status_combines_drivers_and_deliveries(mocker):
    delivery_rows = [_row(Status="Delivered", count=3)]
    pool = AsyncMock()
    pool.fetchval.return_value = 7
    pool.fetch.return_value = delivery_rows
    mocker.patch("app.mcp_server.get_pool", new=AsyncMock(return_value=pool))
    result = json.loads(await get_district_status("d1"))
    assert result["active_drivers"] == 7
    assert result["district_id"] == "d1"
    assert result["deliveries_by_status"][0]["Status"] == "Delivered"


async def test_get_district_status_handles_zero_drivers(mocker):
    pool = AsyncMock()
    pool.fetchval.return_value = None  # no active drivers
    pool.fetch.return_value = []
    mocker.patch("app.mcp_server.get_pool", new=AsyncMock(return_value=pool))
    result = json.loads(await get_district_status("d1"))
    assert result["active_drivers"] == 0


# ── get_stalled_drivers ──────────────────────────────────────────────────────

async def test_get_stalled_drivers_default_minutes(mocker):
    rows = [_row(DriverId="uid3", Name="Samer", DistrictId="d1", LastSeen="2026-07-01")]
    mocker.patch("app.mcp_server.get_pool", new=AsyncMock(return_value=_make_pool(rows)))
    result = json.loads(await get_stalled_drivers())
    assert result[0]["Name"] == "Samer"


async def test_get_stalled_drivers_custom_minutes(mocker):
    rows = []
    mocker.patch("app.mcp_server.get_pool", new=AsyncMock(return_value=_make_pool(rows)))
    result = json.loads(await get_stalled_drivers(minutes=30))
    assert result == []


# ── get_activity_trend (ClickHouse) ─────────────────────────────────────────

async def test_get_activity_trend_returns_hourly_data(mocker):
    from datetime import datetime, timezone
    ch_result = MagicMock()
    ch_result.result_rows = [
        (datetime(2026, 7, 1, 10, tzinfo=timezone.utc), 5, 120),
    ]
    mocker.patch("app.mcp_server.ch_query", new=AsyncMock(return_value=ch_result))
    result = json.loads(await get_activity_trend("d1", hours=24))
    assert len(result) == 1
    assert result[0]["unique_drivers"] == 5
    assert result[0]["total_pings"] == 120


# ── get_peak_hours (ClickHouse) ──────────────────────────────────────────────

async def test_get_peak_hours_returns_hourly_averages(mocker):
    ch_result = MagicMock()
    ch_result.result_rows = [(14, 10, 300), (15, 12, 360)]
    mocker.patch("app.mcp_server.ch_query", new=AsyncMock(return_value=ch_result))
    result = json.loads(await get_peak_hours("d1", days=7))
    assert len(result) == 2
    assert result[0]["hour_of_day"] == 14
    assert result[1]["total_pings"] == 360
