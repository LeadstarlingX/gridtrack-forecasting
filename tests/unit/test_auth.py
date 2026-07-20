import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt

from app.auth import scope_pg_sql, get_allowed_districts

_SECRET = "test-secret-32-chars-minimum-ok!"


def _token(role: str, district_ids: list[str] | None = None) -> str:
    claims: dict = {"sub": "u1", "role": role}
    if district_ids is not None:
        claims["districtId"] = district_ids
    return jwt.encode(claims, _SECRET, algorithm="HS256")


def _creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


# ── scope_pg_sql ──────────────────────────────────────────────────────────────

def test_scope_noop_for_none():
    sql = 'SELECT "DeliveryId" FROM "Deliveries" LIMIT 10'
    assert scope_pg_sql(sql, None) == sql


def test_scope_injects_filter_for_non_join_query():
    """Non-JOIN queries get the district filter injected into WHERE (before LIMIT)."""
    sql = 'SELECT "DeliveryId" FROM "Deliveries" LIMIT 10'
    result = scope_pg_sql(sql, ["mezzeh", "malki"])
    assert '"DistrictId" IN' in result
    assert "'mezzeh'" in result
    assert "'malki'" in result
    assert "LIMIT 10" in result


def test_scope_wraps_join_query_for_observer():
    """JOIN queries are outer-wrapped so DistrictId from the join is filterable."""
    sql = (
        'SELECT d."Name", d."DistrictId", COUNT(del."DeliveryId") '
        'FROM "Drivers" d JOIN "Deliveries" del ON del."AssignedDriverId" = d."DriverId" '
        'GROUP BY d."DriverId", d."Name", d."DistrictId"'
    )
    result = scope_pg_sql(sql, ["mezzeh"])
    assert "__rbac" in result
    assert "'mezzeh'" in result
    assert sql in result


def test_scope_empty_list_returns_no_rows():
    result = scope_pg_sql('SELECT "DeliveryId" FROM "Deliveries"', [])
    assert "FALSE" in result


def test_scope_sanitises_injection_attempt():
    result = scope_pg_sql('SELECT 1', ["legit'; DROP TABLE x--"])
    assert "'; DROP" not in result  
    assert ";" not in result 


def test_scope_injects_filter_before_group_by():
    """Non-JOIN with GROUP BY gets the filter injected before GROUP BY, not after."""
    sql = (
        'SELECT "DistrictId", COUNT(*) FROM "Deliveries" '
        'GROUP BY "DistrictId" ORDER BY COUNT(*) DESC LIMIT 10'
    )
    result = scope_pg_sql(sql, ["kafrsousa"])
    assert '"DistrictId" IN' in result
    assert "'kafrsousa'" in result
    assert "GROUP BY" in result
    assert result.index('"DistrictId" IN') < result.index("GROUP BY")


# ── get_allowed_districts ─────────────────────────────────────────────────────

async def test_dev_mode_no_secret_returns_none():
    """When JWT_SECRET is empty, all requests pass through without filtering."""
    with patch("app.auth.settings") as s:
        s.jwt_secret = ""
        result = get_allowed_districts(creds=None)
    assert result is None


async def test_general_observer_returns_none():
    token = _token("GeneralObserver")
    with patch("app.auth.settings") as s:
        s.jwt_secret = _SECRET
        result = get_allowed_districts(creds=_creds(token))
    assert result is None


async def test_no_role_claim_returns_none():
    """Clerk tokens and legacy tokens with no role claim get no filter."""
    bare = jwt.encode({"sub": "u1"}, _SECRET, algorithm="HS256")
    with patch("app.auth.settings") as s:
        s.jwt_secret = _SECRET
        result = get_allowed_districts(creds=_creds(bare))
    assert result is None


async def test_observer_returns_district_list():
    token = _token("Observer", ["mezzeh", "malki"])
    with patch("app.auth.settings") as s:
        s.jwt_secret = _SECRET
        result = get_allowed_districts(creds=_creds(token))
    assert result == ["mezzeh", "malki"]


async def test_observer_scalar_district_coerced_to_list():
    """Single districtId string (not array) is normalised to a list."""
    claims = {"sub": "u1", "role": "Observer", "districtId": "mezzeh"}
    token = jwt.encode(claims, _SECRET, algorithm="HS256")
    with patch("app.auth.settings") as s:
        s.jwt_secret = _SECRET
        result = get_allowed_districts(creds=_creds(token))
    assert result == ["mezzeh"]


async def test_missing_token_when_secret_configured_raises_401():
    with patch("app.auth.settings") as s:
        s.jwt_secret = _SECRET
        with pytest.raises(HTTPException) as exc:
            get_allowed_districts(creds=None)
    assert exc.value.status_code == 401


async def test_invalid_token_raises_401():
    with patch("app.auth.settings") as s:
        s.jwt_secret = _SECRET
        with pytest.raises(HTTPException) as exc:
            get_allowed_districts(creds=_creds("not.a.valid.token"))
    assert exc.value.status_code == 401


# ── _run_tool SQL scoping ─────────────────────────────────────────────────────

async def test_run_tool_applies_district_filter_for_observer():
    from app.services.chatbot import _run_tool, _allowed_districts_ctx

    tok = _allowed_districts_ctx.set(["mezzeh"])
    try:
        mock_pool = MagicMock()
        mock_pool.fetch = AsyncMock(return_value=[])

        with patch("app.db.get_pool", new=AsyncMock(return_value=mock_pool)):
            await _run_tool(
                "query_postgres",
                {"sql": 'SELECT "DeliveryId" FROM "Deliveries" LIMIT 10'},
            )

        executed_sql = mock_pool.fetch.call_args[0][0]
        assert '"DistrictId" IN' in executed_sql
        assert "'mezzeh'" in executed_sql
    finally:
        _allowed_districts_ctx.reset(tok)


async def test_run_tool_no_filter_for_general_observer():
    from app.services.chatbot import _run_tool, _allowed_districts_ctx

    tok = _allowed_districts_ctx.set(None)
    try:
        mock_pool = MagicMock()
        mock_pool.fetch = AsyncMock(return_value=[])

        with patch("app.db.get_pool", new=AsyncMock(return_value=mock_pool)):
            await _run_tool(
                "query_postgres",
                {"sql": 'SELECT "DeliveryId" FROM "Deliveries" LIMIT 10'},
            )

        executed_sql = mock_pool.fetch.call_args[0][0]
        assert "__rbac" not in executed_sql
    finally:
        _allowed_districts_ctx.reset(tok)