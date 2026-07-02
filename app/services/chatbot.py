"""LLM chat service.

Primary: Groq llama-3.3-70b-versatile.
Fallback: Gemini Flash (if Groq fails).

Exposes:
  call_llm(prompt)              — non-streaming, returns full string
  stream_llm(prompt)            — async generator of token strings
  call_llm_with_tools(prompt)   — tool-calling loop, returns (answer, tools_used)
  stream_with_tools(prompt)     — streaming with tool events: yields JSON frames
"""

import json
import logging
from collections.abc import AsyncGenerator
from datetime import timedelta
from typing import Any

from groq import AsyncGroq

from app.config import settings

logger = logging.getLogger(__name__)

_groq          = AsyncGroq(api_key=settings.groq_api_key)
_FAST_MODEL    = "llama-3.1-8b-instant"       # urgency notes, short tasks
_QUALITY_MODEL = "llama-3.3-70b-versatile"    # chatbot, tools, incidents, staffing


# ── Context compression ──────────────────────────────────────────────────────

# Groq free tier: 12 000 TPM.  Reserve ~400 tokens for the response and ~150
# for system text / question, leaving ~11 450 tokens of input headroom.
# 8 000 chars ≈ 2 000 tokens — conservative enough to absorb several concurrent
# requests in the same minute without hitting the per-minute ceiling.
_CONTEXT_CHAR_BUDGET = 8_000


def compress_context(ctx: dict, char_budget: int = _CONTEXT_CHAR_BUDGET) -> str:
    """Compress a context dict to fit inside *char_budget* characters.

    Returns the raw JSON serialisation unchanged if it already fits.
    Otherwise applies progressively tighter structural compression:

    - Lists are truncated to a leading sample with a trailing omission note
      (e.g. "…18 more items omitted") so the LLM knows it has a sample.
    - Long string values are tail-truncated with a "…" marker.
    - Dict keys and nesting are always preserved.

    Falls back to a hard string truncation as a last resort so the budget
    is guaranteed regardless of input shape.
    """
    raw = json.dumps(ctx)
    if len(raw) <= char_budget:
        return raw

    def _compress(v: object, max_items: int, max_str: int) -> object:
        if isinstance(v, dict):
            return {k: _compress(val, max_items, max_str) for k, val in v.items()}
        if isinstance(v, list):
            if len(v) <= max_items:
                return [_compress(i, max_items, max_str) for i in v]
            sample = [_compress(i, max_items, max_str) for i in v[:max_items]]
            return sample + [f"…{len(v) - max_items} more items omitted"]
        if isinstance(v, str) and len(v) > max_str:
            return v[:max_str] + "…"
        return v

    # Progressively tighter passes: (max_list_items, max_string_chars)
    for max_items, max_str in [(10, 300), (5, 150), (3, 80), (2, 50), (1, 30)]:
        result = json.dumps(_compress(ctx, max_items, max_str))
        if len(result) <= char_budget:
            return result

    # Guarantee: hard-truncate the tightest pass — never blows the budget
    return json.dumps(_compress(ctx, 1, 30))[:char_budget] + "…"


# ── Tools available to the chatbot ──────────────────────────────────────────

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_district_activity",
            "description": (
                "Returns the current real-time activity for a specific district: "
                "how many position events arrived in the last hour and how many "
                "unique drivers are currently active."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "district_id": {
                        "type": "string",
                        "description": "District identifier, e.g. 'mezzeh' or 'kafrsousa'",
                    }
                },
                "required": ["district_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_districts_summary",
            "description": (
                "Returns a summary of ALL known districts: position events in the "
                "last hour and active driver count for each. Use this when the "
                "question is about comparing districts or asking about the overall fleet."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_active_drivers",
            "description": (
                "Queries the database for drivers currently marked active. "
                "Optionally filter by district_id. Returns driver IDs, names, and districts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "district_id": {
                        "type": "string",
                        "description": "Filter to a specific district (optional).",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_anomalies",
            "description": (
                "Queries the database for deliveries flagged as anomalies within the "
                "last N hours. Optionally filter by district_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "district_id": {
                        "type": "string",
                        "description": "Filter to a specific district (optional).",
                    },
                    "hours": {
                        "type": "integer",
                        "description": "Look-back window in hours (default 1).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_deliveries_summary",
            "description": (
                "Returns delivery counts grouped by status and district for the last hour. "
                "Use this for questions about pending, in-transit, or completed deliveries."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_district_status",
            "description": (
                "Returns combined operational status for a specific district: "
                "active driver count and deliveries broken down by status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "district_id": {
                        "type": "string",
                        "description": "District identifier, e.g. 'mezzeh'.",
                    }
                },
                "required": ["district_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stalled_drivers",
            "description": (
                "Returns active drivers who have not sent a GPS ping in the last N minutes — "
                "likely offline, disconnected, or stalled mid-route. "
                "Use when asked about missing, silent, inactive, or offline drivers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {
                        "type": "integer",
                        "description": "Inactivity threshold in minutes (default 15).",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_activity_trend",
            "description": (
                "Queries ClickHouse for hourly GPS ping counts and unique driver counts "
                "for a district over the last N hours. "
                "Use for trend and momentum questions: 'is activity rising?', "
                "'how busy was Mezzeh this morning?', 'when did things slow down?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "district_id": {
                        "type": "string",
                        "description": "District identifier.",
                    },
                    "hours": {
                        "type": "integer",
                        "description": "Look-back window in hours (default 24).",
                    },
                },
                "required": ["district_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_peak_hours",
            "description": (
                "Queries ClickHouse for average activity broken down by hour-of-day "
                "over the last N days for a district. "
                "Use for staffing and scheduling questions: "
                "'when is Mezzeh busiest?', 'should we add drivers at noon?', "
                "'what are the quiet hours?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "district_id": {
                        "type": "string",
                        "description": "District identifier.",
                    },
                    "days": {
                        "type": "integer",
                        "description": "History window in days (default 7).",
                    },
                },
                "required": ["district_id"],
            },
        },
    },
]


async def _run_tool(name: str, args: dict[str, Any]) -> str:
    from datetime import datetime, timezone
    from app.services.forecast import _windows, _active_drivers  # live in-memory state

    now    = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=1)

    if name == "get_district_activity":
        d       = args.get("district_id", "")
        window  = _windows.get(d)
        count   = sum(1 for t in (window or []) if t >= cutoff)
        drivers = len(_active_drivers.get(d, set()))
        return json.dumps({"district": d, "position_events_last_hour": count, "active_drivers": drivers})

    if name == "get_all_districts_summary":
        districts = set(_windows.keys()) | set(_active_drivers.keys())
        summary   = []
        for d in sorted(districts):
            window  = _windows.get(d)
            count   = sum(1 for t in (window or []) if t >= cutoff)
            drivers = len(_active_drivers.get(d, set()))
            summary.append({"district": d, "events_last_hour": count, "active_drivers": drivers})
        return json.dumps(summary)

    # DB-backed tools — lazy import to avoid startup failure if postgres is unavailable
    from app.db import get_pool
    pool = await get_pool()

    if name == "get_active_drivers":
        district_id = args.get("district_id", "")
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

    if name == "get_anomalies":
        district_id = args.get("district_id", "")
        hours = int(args.get("hours", 1))
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

    if name == "get_deliveries_summary":
        rows = await pool.fetch(
            'SELECT "Status", "DistrictId", COUNT(*) AS count'
            ' FROM public."Deliveries"'
            " WHERE \"CreatedAt\" >= NOW() - '1 hour'::interval"
            ' GROUP BY "Status", "DistrictId"'
            ' ORDER BY "DistrictId", "Status"'
        )
        return json.dumps([dict(r) for r in rows], default=str)

    if name == "get_district_status":
        district_id = args.get("district_id", "")
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

    if name == "get_stalled_drivers":
        minutes = int(args.get("minutes", 15))
        rows = await pool.fetch(
            'SELECT "DriverId", "Name", "DistrictId", "LastSeen"'
            ' FROM public."Drivers"'
            ' WHERE "IsActive" = true AND "LastSeen" < NOW() - $1'
            ' ORDER BY "LastSeen" ASC LIMIT 20',
            timedelta(minutes=minutes),
        )
        return json.dumps([dict(r) for r in rows], default=str)

    # ── ClickHouse tools ─────────────────────────────────────────────────────
    from app.ch import ch_query

    if name == "get_activity_trend":
        district_id = args.get("district_id", "")
        hours = int(args.get("hours", 24))
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

    if name == "get_peak_hours":
        district_id = args.get("district_id", "")
        days = int(args.get("days", 7))
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

    return json.dumps({"error": f"Unknown tool: {name}"})


# ── Public API ───────────────────────────────────────────────────────────────

async def call_llm(prompt: str) -> str:
    """Non-streaming call. Groq primary, Gemini fallback."""
    try:
        return await _call_groq(prompt)
    except Exception as exc:
        logger.warning("Groq failed (%s), trying Gemini fallback", exc)
        return await _call_gemini(prompt)


async def call_llm_fast(prompt: str) -> str:
    """Fast / cheap model for short structured outputs (urgency notes etc.)."""
    try:
        resp = await _groq.chat.completions.create(
            model=_FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("Fast Groq failed (%s), falling back to quality model", exc)
        return await _call_groq(prompt)


async def stream_llm(prompt: str) -> AsyncGenerator[str, None]:
    """Streaming generator — yields token strings. Falls back to single chunk on error."""
    try:
        stream = await _groq.chat.completions.create(
            model=_QUALITY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=400,
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token
    except Exception as exc:
        logger.warning("Groq streaming failed (%s), falling back", exc)
        result = await _call_gemini(prompt)
        yield result


async def stream_with_tools(prompt: str) -> AsyncGenerator[str, None]:
    """Streaming with tool calling.

    Yields JSON-encoded frames (not raw SSE lines):
      {"tool": "get_active_drivers"}   — emitted immediately when a tool is called
      {"token": "Based on data..."}    — final answer tokens
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    msg = None

    for _ in range(5):
        resp = await _groq.chat.completions.create(
            model=_QUALITY_MODEL,
            messages=messages,
            tools=_TOOLS,
            tool_choice="auto",
            max_tokens=500,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            yield json.dumps({"token": (msg.content or "").strip()})
            return

        tool_calls_payload: list[dict[str, Any]] = [
            {
                "id":       tc.id,
                "type":     "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
        messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls_payload})

        for tc in msg.tool_calls:
            yield json.dumps({"tool": tc.function.name})
            args   = json.loads(tc.function.arguments or "{}")
            result = await _run_tool(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Fallback if max iterations hit without settling
    yield json.dumps({"token": (msg.content or "").strip() if msg else ""})


async def call_llm_with_tools(prompt: str) -> tuple[str, list[str]]:
    """Tool-calling loop. Returns (answer, tools_used list)."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    tools_used: list[str] = []
    msg = None

    for _ in range(5):
        resp = await _groq.chat.completions.create(
            model=_QUALITY_MODEL,
            messages=messages,
            tools=_TOOLS,
            tool_choice="auto",
            max_tokens=500,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return (msg.content or "").strip(), tools_used

        tool_calls_payload: list[dict[str, Any]] = [
            {
                "id":       tc.id,
                "type":     "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
        messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls_payload})

        for tc in msg.tool_calls:
            tools_used.append(tc.function.name)
            args   = json.loads(tc.function.arguments or "{}")
            result = await _run_tool(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return (msg.content or "").strip() if msg else "", tools_used


# ── Private helpers ──────────────────────────────────────────────────────────

async def _call_groq(prompt: str) -> str:
    resp = await _groq.chat.completions.create(
        model=_QUALITY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


async def _call_gemini(prompt: str) -> str:
    if not settings.google_api_key:
        raise RuntimeError("Gemini fallback disabled: GOOGLE_API_KEY not set")
    import asyncio
    import google.generativeai as genai
    genai.configure(api_key=settings.google_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp  = await asyncio.to_thread(model.generate_content, prompt)
    return resp.text
