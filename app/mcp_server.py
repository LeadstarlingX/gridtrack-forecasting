"""MCP server exposing read-only GridTrack DB tools for external agents.

Mount point: /mcp (set by main.py).
Auth: Bearer token via MCP_API_KEY env var.
Transport: Streamable HTTP (SSE).
"""

import json
from datetime import timedelta

from mcp.server.fastmcp import FastMCP

from app.db import get_pool
from app.ch import ch_query

mcp = FastMCP("GridTrack Live Data")


class _BearerAuth:
    """ASGI middleware: require Authorization: Bearer <api_key> on every request."""

    def __init__(self, asgi_app, api_key: str) -> None:
        self._app = asgi_app
        self._expected = f"Bearer {api_key}"

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            headers = dict(scope.get("headers", []))
            auth = headers.get(b"authorization", b"").decode()
            if auth != self._expected:
                await send({
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [[b"content-type", b"text/plain"]],
                })
                await send({"type": "http.response.body", "body": b"Unauthorized"})
                return
        await self._app(scope, receive, send)


@mcp.tool()
async def get_active_drivers(district_id: str = "") -> str:
    """Return active drivers from the database, optionally filtered by district."""
    pool = await get_pool()
    if district_id:
        rows = await pool.fetch(
            'SELECT "DriverId", "Name", "DistrictId",'
            ' ST_Y("Location") AS lat, ST_X("Location") AS lng'
            ' FROM public."Drivers"'
            ' WHERE "IsActive" = true AND "DistrictId" = $1 LIMIT 50',
            district_id,
        )
    else:
        rows = await pool.fetch(
            'SELECT "DriverId", "Name", "DistrictId",'
            ' ST_Y("Location") AS lat, ST_X("Location") AS lng'
            ' FROM public."Drivers" WHERE "IsActive" = true LIMIT 50'
        )
    return json.dumps([dict(r) for r in rows], default=str)


@mcp.tool()
async def get_anomalies(district_id: str = "", hours: int = 1) -> str:
    """Return deliveries flagged as anomalies within the last N hours."""
    pool = await get_pool()
    delta = timedelta(hours=hours)
    if district_id:
        rows = await pool.fetch(
            'SELECT "DeliveryId", "Status", "DistrictId", "CreatedAt", "ExpectedEta"'
            ' FROM public."Deliveries"'
            ' WHERE "AnomalyFlag" = true AND "CreatedAt" >= NOW() - $1 AND "DistrictId" = $2'
            ' ORDER BY "CreatedAt" DESC LIMIT 20',
            delta, district_id,
        )
    else:
        rows = await pool.fetch(
            'SELECT "DeliveryId", "Status", "DistrictId", "CreatedAt", "ExpectedEta"'
            ' FROM public."Deliveries"'
            ' WHERE "AnomalyFlag" = true AND "CreatedAt" >= NOW() - $1'
            ' ORDER BY "CreatedAt" DESC LIMIT 20',
            delta,
        )
    return json.dumps([dict(r) for r in rows], default=str)


@mcp.tool()
async def get_deliveries_summary() -> str:
    """Return delivery counts grouped by status and district for the last hour."""
    pool = await get_pool()
    rows = await pool.fetch(
        'SELECT "Status", "DistrictId", COUNT(*) AS count'
        ' FROM public."Deliveries"'
        " WHERE \"CreatedAt\" >= NOW() - '1 hour'::interval"
        ' GROUP BY "Status", "DistrictId"'
        ' ORDER BY "DistrictId", "Status"'
    )
    return json.dumps([dict(r) for r in rows], default=str)


@mcp.tool()
async def get_district_status(district_id: str) -> str:
    """Return combined operational status for a district (drivers + deliveries)."""
    pool = await get_pool()
    driver_count = await pool.fetchval(
        'SELECT COUNT(*) FROM public."Drivers"'
        ' WHERE "IsActive" = true AND "DistrictId" = $1',
        district_id,
    )
    rows = await pool.fetch(
        'SELECT "Status", COUNT(*) AS count FROM public."Deliveries"'
        " WHERE \"DistrictId\" = $1 AND \"CreatedAt\" >= NOW() - '1 hour'::interval"
        ' GROUP BY "Status"',
        district_id,
    )
    return json.dumps({
        "district_id": district_id,
        "active_drivers": driver_count or 0,
        "deliveries_by_status": [dict(r) for r in rows],
    }, default=str)


@mcp.tool()
async def get_stalled_drivers(minutes: int = 15) -> str:
    """Return active drivers who have not sent a GPS ping in the last N minutes."""
    pool = await get_pool()
    rows = await pool.fetch(
        'SELECT "DriverId", "Name", "DistrictId", "LastSeen"'
        ' FROM public."Drivers"'
        ' WHERE "IsActive" = true AND "LastSeen" < NOW() - $1'
        ' ORDER BY "LastSeen" ASC LIMIT 20',
        timedelta(minutes=minutes),
    )
    return json.dumps([dict(r) for r in rows], default=str)


@mcp.tool()
async def get_activity_trend(district_id: str, hours: int = 24) -> str:
    """Return hourly GPS ping counts and unique driver counts for a district (ClickHouse)."""
    result = await ch_query(
        "SELECT toStartOfHour(recorded_at) AS hour,"
        "       countDistinct(driver_id) AS unique_drivers,"
        "       count() AS total_pings"
        " FROM driver_positions"
        " WHERE district_id = {district_id:String}"
        "   AND recorded_at >= now() - INTERVAL {hours:UInt32} HOUR"
        " GROUP BY hour ORDER BY hour ASC",
        {"district_id": district_id, "hours": hours},
    )
    rows = [
        {"hour": str(r[0]), "unique_drivers": r[1], "total_pings": r[2]}
        for r in result.result_rows
    ]
    return json.dumps(rows)


@mcp.tool()
async def get_peak_hours(district_id: str, days: int = 7) -> str:
    """Return average activity by hour-of-day over the last N days for a district (ClickHouse)."""
    result = await ch_query(
        "SELECT toHour(recorded_at) AS hour_of_day,"
        "       countDistinct(driver_id) AS unique_drivers,"
        "       count() AS total_pings"
        " FROM driver_positions"
        " WHERE district_id = {district_id:String}"
        "   AND recorded_at >= now() - INTERVAL {days:UInt32} DAY"
        " GROUP BY hour_of_day ORDER BY hour_of_day ASC",
        {"district_id": district_id, "days": days},
    )
    rows = [
        {"hour_of_day": r[0], "unique_drivers": r[1], "total_pings": r[2]}
        for r in result.result_rows
    ]
    return json.dumps(rows)


def make_mcp_app(api_key: str):
    """Return the FastMCP ASGI app wrapped with Bearer auth."""
    return _BearerAuth(mcp.streamable_http_app(), api_key)
