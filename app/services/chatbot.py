"""LLM chat service.

Primary: OpenRouter (Llama 3.3 70B, global, free tier).
Fallback chain: Groq llama-3.3-70b-versatile → Gemini Flash.

Exposes:
  call_llm(prompt)              — non-streaming, returns full string
  stream_llm(prompt)            — async generator of token strings
  call_llm_with_tools(prompt)   — tool-calling loop, returns (answer, tools_used)
  stream_with_tools(prompt)     — streaming with tool events: yields JSON frames
"""

import asyncio
import contextvars
import json
import logging
from collections.abc import AsyncGenerator
from datetime import timedelta
from typing import Any

from anthropic import AsyncAnthropic
from groq import AsyncGroq
from openai import AsyncOpenAI

from app.auth import scope_pg_sql
from app.config import settings

logger = logging.getLogger(__name__)

_groq          = AsyncGroq(api_key=settings.groq_api_key)
_FAST_MODEL    = "llama-3.1-8b-instant"       # urgency notes, short tasks
_QUALITY_MODEL = "llama-3.3-70b-versatile"    # chatbot, tools, incidents, staffing

_CLAUDE_MODEL      = "claude-opus-5"
_CLAUDE_FAST_MODEL = "claude-haiku-4-5"

# Ordered by quality — all confirmed to support tool_choice on OpenRouter free tier.
# First model that doesn't 429/error wins; daily limits are per-model so cycling extends capacity.
_OR_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "poolside/laguna-s-2.1:free",
    "inclusionai/ling-3.0-flash:free",
    "poolside/laguna-xs-2.1:free",
]

_anthropic: AsyncAnthropic | None = None

# Per-account OR pool: list of [AsyncOpenAI client, exhausted_until_epoch].
# Built lazily from the three optional key slots; empty slots are skipped.
_or_pool: list = []


def _get_anthropic() -> AsyncAnthropic:
    global _anthropic
    if _anthropic is None:
        _anthropic = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _anthropic


def _or_pool_init() -> None:
    global _or_pool
    if _or_pool:
        return
    keys = [k for k in [
        settings.openrouter_api_key,
        settings.openrouter_api_key_2,
        settings.openrouter_api_key_3,
    ] if k]
    _or_pool = [
        [AsyncOpenAI(api_key=k, base_url="https://openrouter.ai/api/v1"), 0.0]
        for k in keys
    ]


_allowed_districts_ctx: contextvars.ContextVar[list[str] | None] = \
    contextvars.ContextVar("_allowed_districts", default=None)

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

_PG_SCHEMA = """
PostgreSQL tables (schema: public, double-quote all identifiers):
- "Drivers": "DriverId" UUID PK, "Name" TEXT, "ShortName" TEXT, "IsActive" BOOL, "DistrictId" TEXT, "LastSeen" TIMESTAMPTZ
- "Deliveries": "DeliveryId" UUID PK, "AssignedDriverId" UUID FK->Drivers, "Status" INTEGER, "DistrictId" TEXT, "AnomalyFlag" BOOL, "AnomalyReason" TEXT, "CreatedAt" TIMESTAMPTZ, "PickedUpAt" TIMESTAMPTZ, "DeliveredAt" TIMESTAMPTZ, "ExpectedEta" TIMESTAMPTZ, "UrgencyScore" INTEGER, "RouteDistanceMeters" FLOAT

CRITICAL RULES (violations cause SQL errors):
1. "Status" is INTEGER — NEVER use strings. Values: 0=Created 1=Assigned 2=PickedUp 3=InTransit 4=Delivered 5=Cancelled 6=Anomalous
2. FK in "Deliveries" is "AssignedDriverId" — there is NO "DriverId" column in "Deliveries"
3. No "Districts" table exists — only DistrictId text key
4. Use double-quotes for identifiers, never backticks
5. SELECT only needed columns. LIMIT 20. No SELECT *
6. Time: make_interval(hours => 1), make_interval(days => 7)

EXAMPLE QUERIES (copy the pattern exactly):
- Active drivers: SELECT "DriverId", "Name", "DistrictId" FROM "Drivers" WHERE "IsActive" = true LIMIT 20
- In-transit deliveries: SELECT "DeliveryId", "DistrictId", "AssignedDriverId" FROM "Deliveries" WHERE "Status" = 3 LIMIT 20
- Delivered today: SELECT COUNT("DeliveryId") FROM "Deliveries" WHERE "Status" = 4 AND "DeliveredAt" >= NOW() - make_interval(hours => 24)
- Anomalies: SELECT "DeliveryId", "DistrictId", "AnomalyReason" FROM "Deliveries" WHERE "AnomalyFlag" = true LIMIT 20
- District delivery counts: SELECT "DistrictId", COUNT(*) FROM "Deliveries" WHERE "Status" = 4 GROUP BY "DistrictId" ORDER BY COUNT(*) DESC LIMIT 10
- Driver with their deliveries: SELECT d."DriverId", d."Name", d."DistrictId", COUNT(del."DeliveryId") FROM "Drivers" d LEFT JOIN "Deliveries" del ON del."AssignedDriverId" = d."DriverId" GROUP BY d."DriverId", d."Name", d."DistrictId" LIMIT 20
- Best drivers today (most deliveries completed on time): SELECT d."Name", d."DistrictId", COUNT(del."DeliveryId") AS completed FROM "Drivers" d JOIN "Deliveries" del ON del."AssignedDriverId" = d."DriverId" WHERE del."Status" = 4 AND del."DeliveredAt" >= NOW() - make_interval(hours => 24) GROUP BY d."DriverId", d."Name", d."DistrictId" ORDER BY completed DESC LIMIT 5
- On-time drivers (delivered before ETA): SELECT d."Name", d."DistrictId", COUNT(del."DeliveryId") AS on_time FROM "Drivers" d JOIN "Deliveries" del ON del."AssignedDriverId" = d."DriverId" WHERE del."Status" = 4 AND del."DeliveredAt" IS NOT NULL AND del."ExpectedEta" IS NOT NULL AND del."DeliveredAt" <= del."ExpectedEta" GROUP BY d."DriverId", d."Name", d."DistrictId" ORDER BY on_time DESC LIMIT 5
"""

_CH_SCHEMA = """
ClickHouse table: driver_positions
Columns: driver_id String, lat Float64, lng Float64, district_id String, recorded_at DateTime64(3,'UTC')
WARNING: This table has NO delivery or driver name data. Only use the 5 columns listed above. Never reference PostgreSQL columns here.
Rules: SELECT only. LIMIT 20. SELECT only the columns needed to answer the question — never SELECT *.
IMPORTANT: For time windows use INTERVAL syntax without single quotes: now() - INTERVAL 1 HOUR, now() - INTERVAL 7 DAY.
Use toStartOfHour(), countDistinct(), toHour() for aggregations. Use count() not count(column) for row counts.
"""

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_district_activity",
            "description": (
                "Returns real-time in-memory activity for a specific district: "
                "GPS ping count in the last hour and unique active driver count. "
                "Fastest option for current activity — use before querying the DB."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "district_id": {"type": "string", "description": "District id — a numeric string key, e.g. '13433880'. NOT a name."}
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
                "Returns real-time in-memory summary for ALL districts: ping count and "
                "active driver count per district. Use for fleet-wide comparisons."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_postgres",
            "description": (
                "Execute any read-only SELECT query against the GridTrack PostgreSQL database. "
                "Use for: top drivers by deliveries, anomaly counts, district delivery stats, "
                "stalled drivers, pending orders, anything needing Drivers or Deliveries data.\n"
                + _PG_SCHEMA
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A valid PostgreSQL SELECT statement."}
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_clickhouse",
            "description": (
                "Execute any read-only SELECT query against the GridTrack ClickHouse database. "
                "Use for: hourly/daily activity trends, peak hours, next-hour demand prediction "
                "(compare current pings vs same hour yesterday/last week), driver movement history.\n"
                + _CH_SCHEMA
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A valid ClickHouse SELECT statement."}
                },
                "required": ["sql"],
            },
        },
    },
]


_MAX_RESULT_CHARS = 3_000  # ~750 tokens per tool call

# Delivery status names → integer values (Groq llama often generates string comparisons)
_STATUS_INTS = {"Created": 0, "Assigned": 1, "PickedUp": 2,
                "InTransit": 3, "Delivered": 4, "Cancelled": 5, "Anomalous": 6}

def _fix_pg_sql(sql: str) -> str:
    """Auto-correct common LLM SQL mistakes before sending to PostgreSQL."""
    import re
    # Replace "Status" = 'Name' or "Status" = "Name" with integer value
    def _replace_status(m):
        name = m.group(1)
        return f'"Status" = {_STATUS_INTS.get(name, m.group(0))}'
    sql = re.sub(r'"Status"\s*=\s*[\'"](\w+)[\'"]', _replace_status, sql)
    # Replace backtick identifiers with double-quote
    sql = re.sub(r'`([^`]+)`', r'"\1"', sql)
    return sql

def _trim_result(result: str) -> str:
    if len(result) <= _MAX_RESULT_CHARS:
        return result
    return result[:_MAX_RESULT_CHARS] + "…(truncated)"


_GEMINI_SYSTEM = (
    "You are a delivery operations assistant in Damascus. "
    "Answer concisely, using numbers. Use tools to fetch live data. "
    "When ranking or listing results, always include the name (not just ID) and the metric value used for ranking. "
    "When writing SQL: SELECT only the columns needed, never SELECT *. "
    "IMPORTANT: Call at most 2-3 tools per question. If a tool returns empty or zero results, accept that and answer directly — do not call more tools to verify."
)

# Ordered Gemini models to try — primary first, fallback chain after.
# Verify API IDs against Google AI Studio if names change.
_GEMINI_MODELS = [
    "gemini-2.0-flash",       # Gemini 2.0 Flash  (current workhorse)
    "gemini-2.0-flash-lite",  # Gemini Flash Lite (high RPD, lighter model)
]

# Lazy-cached tool lists (avoids importing google.genai at startup)
_gemini_tools_cache: list | None = None
_claude_tools_cache: list | None = None


def _get_claude_tools() -> list:
    global _claude_tools_cache
    if _claude_tools_cache is None:
        _claude_tools_cache = [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            }
            for t in _TOOLS
        ]
    return _claude_tools_cache
# Per-model circuit-breakers: timestamp when the model was marked dead, or None.
# Quota errors (PerDay) reset after 6 h. Key/permission errors reset after 1 h
# (a restart or key rotation fixes those; the timer just avoids an infinite dead state).
import time as _time
_gemini_dead: dict[str, float] = {}   # model → epoch seconds when tripped
_GEMINI_DEAD_TTL = 6 * 3600            # 6 hours

def _or_daily_limit_hit(exc: Exception) -> bool:
    """True when the error is an account-wide daily cap (not a per-model error)."""
    s = str(exc)
    return "429" in s and "free-models-per-day" in s


def _is_gemini_dead(model: str) -> bool:
    tripped_at = _gemini_dead.get(model)
    if tripped_at is None:
        return False
    if _time.time() - tripped_at > _GEMINI_DEAD_TTL:
        del _gemini_dead[model]
        logger.info("Gemini %s circuit breaker reset after TTL — will retry", model)
        return False
    return True


def _next_gemini_model() -> str | None:
    """Return the first Gemini model that hasn't hit its quota/error TTL, or None."""
    return next((m for m in _GEMINI_MODELS if not _is_gemini_dead(m)), None)


def _get_gemini_tools() -> list:
    global _gemini_tools_cache
    if _gemini_tools_cache is None:
        from google.genai import types
        _gemini_tools_cache = [types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["function"]["name"],
                description=t["function"]["description"],
                parameters=t["function"].get("parameters"),
            )
            for t in _TOOLS
        ])]
    return _gemini_tools_cache


async def _run_tool(name: str, args: dict[str, Any], allowed_districts: list[str] | None = None) -> str:
    from datetime import datetime, timezone
    from app.services.forecast import _windows, _active_drivers  # live in-memory state

    now    = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=1)

    if name == "get_district_activity":
        d       = args.get("district_id", "")
        window  = _windows.get(d)
        count   = sum(1 for t in (window or []) if t >= cutoff)
        drivers = len(_active_drivers.get(d, set()))
        result  = json.dumps({"district": d, "position_events_last_hour": count, "active_drivers": drivers})
        logger.info("tool=%s district=%s result_len=%d", name, d, len(result))
        return result

    if name == "get_all_districts_summary":
        districts = set(_windows.keys()) | set(_active_drivers.keys())
        summary   = []
        for d in sorted(districts):
            window  = _windows.get(d)
            count   = sum(1 for t in (window or []) if t >= cutoff)
            drivers = len(_active_drivers.get(d, set()))
            summary.append({"district": d, "events_last_hour": count, "active_drivers": drivers})
        result = json.dumps(summary)
        logger.info("tool=%s districts=%d result_len=%d", name, len(summary), len(result))
        return result

    if name == "query_postgres":
        sql = args.get("sql", "").strip()
        if not sql.upper().startswith("SELECT"):
            return json.dumps({"error": "Only SELECT statements are allowed"})
        sql = _fix_pg_sql(sql)
        sql = scope_pg_sql(sql, _allowed_districts_ctx.get())
        from app.db import get_pool
        pool = await get_pool()
        try:
            rows = await pool.fetch(sql, timeout=5.0)
            result = _trim_result(json.dumps([dict(r) for r in rows[:20]], default=str))
            logger.info("tool=%s rows=%d result_len=%d sql=%.120s", name, len(rows), len(result), sql)
            return result
        except Exception as exc:
            logger.warning("tool=%s error=%s sql=%.120s", name, exc, sql)
            return json.dumps({"error": str(exc)})

    if name == "query_clickhouse":
        sql = args.get("sql", "").strip()
        if not sql.upper().startswith("SELECT"):
            return json.dumps({"error": "Only SELECT statements are allowed"})
        from app.ch import ch_query
        try:
            ch_result = await asyncio.wait_for(ch_query(sql), timeout=5.0)
            rows = [
                dict(zip(ch_result.column_names, row))
                for row in ch_result.result_rows[:20]
            ]
            result = _trim_result(json.dumps(rows, default=str))
            logger.info("tool=%s rows=%d result_len=%d sql=%.120s", name, len(rows), len(result), sql)
            return result
        except Exception as exc:
            logger.warning("tool=%s error=%s sql=%.120s", name, exc, sql)
            return json.dumps({"error": str(exc)})

    return json.dumps({"error": f"Unknown tool: {name}"})


# ── Public API ───────────────────────────────────────────────────────────────

async def call_llm(prompt: str, *, response_format: dict | None = None) -> str:
    """Non-streaming call. OpenRouter primary, Groq secondary, Gemini multi-model fallback."""
    _or_pool_init()
    for _or_entry in _or_pool:
        if _time.time() < _or_entry[1]:
            continue
        _or_daily_hit = False
        for or_model in _OR_MODELS:
            try:
                resp = await _or_entry[0].chat.completions.create(
                    model=or_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    **({"response_format": response_format} if response_format else {}),
                )
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    return content
                logger.warning("OpenRouter %s empty content, trying next", or_model)
            except Exception as exc:
                if _or_daily_limit_hit(exc):
                    _or_entry[1] = _time.time() + 3600
                    logger.warning("OpenRouter account daily limit hit — trying next account")
                    _or_daily_hit = True
                    break
                logger.warning("OpenRouter %s failed (%s), trying next", or_model, exc)
        if not _or_daily_hit:
            break
    try:
        return await _call_groq(prompt, response_format=response_format)
    except Exception as exc:
        logger.warning("Groq failed (%s), trying Gemini models", exc)
    for model in _GEMINI_MODELS:
        if _is_gemini_dead(model):
            continue
        try:
            return await _call_gemini(prompt, model)
        except Exception as exc:
            logger.warning("Gemini %s failed (%s), trying next model", model, exc)
    dead = [m for m in _GEMINI_MODELS if _is_gemini_dead(m)]
    if dead:
        logger.warning("All Gemini models circuit-broken (%s) — no LLM available", dead)
    raise RuntimeError("All LLM providers failed")


async def call_llm_fast(prompt: str) -> str:
    """Fast / cheap model for short structured outputs (urgency notes etc.)."""
    _or_pool_init()
    for _or_entry in _or_pool:
        if _time.time() < _or_entry[1]:
            continue
        _or_daily_hit = False
        for or_model in _OR_MODELS:
            try:
                resp = await _or_entry[0].chat.completions.create(
                    model=or_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=60,
                )
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    return content
                logger.warning("OpenRouter fast %s empty content, trying next", or_model)
            except Exception as exc:
                if _or_daily_limit_hit(exc):
                    _or_entry[1] = _time.time() + 3600
                    logger.warning("OpenRouter account daily limit hit — trying next account")
                    _or_daily_hit = True
                    break
                logger.warning("OpenRouter fast %s failed (%s), trying next", or_model, exc)
        if not _or_daily_hit:
            break
    try:
        resp = await _groq.chat.completions.create(
            model=_FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("Groq fast failed (%s), falling back to call_llm chain", exc)
    return await call_llm(prompt)


async def stream_llm(prompt: str) -> AsyncGenerator[str, None]:
    """Streaming generator — yields token strings. OpenRouter primary, Groq/Gemini fallback."""
    _or_pool_init()
    for _or_entry in _or_pool:
        if _time.time() < _or_entry[1]:
            continue
        _or_daily_hit = False
        for or_model in _OR_MODELS:
            try:
                stream = await _or_entry[0].chat.completions.create(
                    model=or_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    max_tokens=400,
                )
                async for chunk in stream:
                    token = chunk.choices[0].delta.content
                    if token:
                        yield token
                return
            except Exception as exc:
                if _or_daily_limit_hit(exc):
                    _or_entry[1] = _time.time() + 3600
                    logger.warning("OpenRouter account daily limit hit — trying next account")
                    _or_daily_hit = True
                    break
                logger.warning("OpenRouter %s streaming failed (%s), trying next", or_model, exc)
        if not _or_daily_hit:
            break
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
        logger.warning("Groq streaming failed (%s), falling back to Gemini", exc)
        result = await _call_gemini(prompt)
        yield result


def _extract_question(prompt: str) -> str:
    """Pull just the user question (used only by the bare _call_gemini fallback)."""
    if "\nQuestion:" in prompt:
        return prompt.split("\nQuestion:")[-1].strip()
    return prompt


def _user_content(prompt: str) -> str:
    """Extract the operational context + question from build_prompt() output.

    Strips the static _BASE header so Gemini's system_instruction handles the role,
    while the user message carries the live context snapshot and question.
    """
    for marker in ("Operational context:", "Question:"):
        if marker in prompt:
            return prompt[prompt.index(marker):]
    return prompt


def _recover_tool_calls(exc: Exception) -> list[tuple[str, dict]] | None:
    """Fix Groq 400s where the model escapes single quotes as \\' in SQL inside JSON.

    Returns list of (tool_name, args_dict) if recoverable, else None.
    """
    import re
    body = getattr(exc, "body", None) or {}
    failed = body.get("error", {}).get("failed_generation", "")
    if not failed:
        return None
    fixed = failed.replace("\\'", "'")
    # Handles: name/>, name{}, name={}, name={"sql":...}></function> endings
    # The [>\s]* after the JSON handles the `>` that closes the opening element tag
    # before the </function> closing tag in the format: <function=name,{...}></function>
    matches = re.findall(
        r"<function=(\w+)[\[\]\s,=]*((?:\{.*?\})?)[>\s]*(?:/>|</function>)",
        fixed,
        re.DOTALL,
    )
    if not matches:
        return None
    result = []
    for name, args_json in matches:
        if not args_json:
            result.append((name, {}))
        else:
            try:
                result.append((name, json.loads(args_json) or {}))
            except (json.JSONDecodeError, ValueError):
                return None
    return result or None


async def stream_with_tools(prompt: str, allowed_districts: list[str] | None = None) -> AsyncGenerator[str, None]:
    """Streaming with tool calling. Claude primary, Gemini chain secondary, Groq fallback.

    Yields JSON-encoded frames:
      {"tool": "name"}       — emitted when a tool is called
      {"token": "text..."}   — final answer
    """
    _tok = _allowed_districts_ctx.set(allowed_districts)
    try:
        # OpenRouter primary — cycle through accounts, then models within each
        _or_pool_init()
        for _or_entry in _or_pool:
            if _time.time() < _or_entry[1]:
                continue
            _or_daily_hit = False
            for or_model in _OR_MODELS:
                got_answer = False
                try:
                    async for frame in _stream_groq_with_tools(
                        prompt, client=_or_entry[0], model=or_model
                    ):
                        yield frame
                        if not got_answer:
                            try:
                                got_answer = bool(json.loads(frame).get("token"))
                            except (json.JSONDecodeError, AttributeError):
                                pass
                    if got_answer:
                        return
                    logger.warning("OpenRouter %s empty answer, trying next", or_model)
                except Exception as exc:
                    if _or_daily_limit_hit(exc):
                        _or_entry[1] = _time.time() + 3600
                        logger.warning("OpenRouter account daily limit hit — trying next account")
                        _or_daily_hit = True
                        break
                    logger.warning("OpenRouter %s stream failed (%s), trying next", or_model, exc)
            if not _or_daily_hit:
                break

        # Gemini fallback
        for model in _GEMINI_MODELS:
            if _gemini_dead.get(model):
                continue
            got_answer = False
            try:
                async for frame in _stream_gemini_with_tools(prompt, model):
                    yield frame
                    if not got_answer:
                        try:
                            got_answer = bool(json.loads(frame).get("token"))
                        except (json.JSONDecodeError, AttributeError):
                            pass
                if got_answer:
                    return
                logger.warning("Gemini %s returned empty answer, trying next model", model)
            except Exception as exc:
                logger.warning("Gemini %s failed (%s), trying next model", model, exc)

        try:
            async for frame in _stream_groq_with_tools(prompt):
                yield frame
        except Exception as groq_exc:
            logger.warning("Groq fallback also failed (%s)", groq_exc)
            yield json.dumps({"token": "Both AI providers are temporarily unavailable. Please try again in a moment."})
    finally:
        _allowed_districts_ctx.reset(_tok)


async def call_llm_with_tools(prompt: str, allowed_districts: list[str] | None = None) -> tuple[str, list[str]]:
    """Tool-calling loop. Claude primary, Gemini chain secondary, Groq fallback. Returns (answer, tools_used)."""
    _tok = _allowed_districts_ctx.set(allowed_districts)
    try:
        # OpenRouter primary — cycle through accounts, then models within each
        _or_pool_init()
        for _or_entry in _or_pool:
            if _time.time() < _or_entry[1]:
                continue
            _or_daily_hit = False
            for or_model in _OR_MODELS:
                try:
                    answer, tools_used = await _call_groq_with_tools(
                        prompt, client=_or_entry[0], model=or_model
                    )
                    if answer:
                        return answer, tools_used
                    logger.warning("OpenRouter %s empty answer, trying next", or_model)
                except Exception as exc:
                    if _or_daily_limit_hit(exc):
                        _or_entry[1] = _time.time() + 3600
                        logger.warning("OpenRouter account daily limit hit — trying next account")
                        _or_daily_hit = True
                        break
                    logger.warning("OpenRouter %s failed (%s), trying next", or_model, exc)
            if not _or_daily_hit:
                break

        # Gemini fallback
        for model in _GEMINI_MODELS:
            if _gemini_dead.get(model):
                continue
            try:
                answer, tools_used = await _call_gemini_with_tools(prompt, model)
                if answer:
                    return answer, tools_used
                logger.warning("Gemini %s returned empty answer, trying next model", model)
            except Exception as exc:
                logger.warning("Gemini %s failed (%s), trying next model", model, exc)

        try:
            return await _call_groq_with_tools(prompt)
        except Exception as groq_exc:
            logger.warning("Groq fallback also failed (%s)", groq_exc)
            return "Both AI providers are temporarily unavailable. Please try again in a moment.", []
    finally:
        _allowed_districts_ctx.reset(_tok)


# ── Private helpers ──────────────────────────────────────────────────────────

async def _call_groq(prompt: str, *, response_format: dict | None = None) -> str:
    resp = await _groq.chat.completions.create(
        model=_QUALITY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        **({"response_format": response_format} if response_format else {}),
    )
    return resp.choices[0].message.content.strip()


async def _call_gemini(prompt: str, model: str | None = None) -> str:
    if not settings.google_api_key:
        raise RuntimeError("Gemini fallback disabled: GOOGLE_API_KEY not set")
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=settings.google_api_key)
    clean = prompt.split("You have access to tools")[0].strip() if "You have access to tools" in prompt else prompt
    model = model or _next_gemini_model() or _GEMINI_MODELS[-1]
    config = types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    resp = await _gemini_generate(client, model, clean, config)
    return _gemini_resp_text(resp)


async def _stream_groq_with_tools(prompt: str, allowed_districts: list[str] | None = None, *, client=None, model: str | None = None) -> AsyncGenerator[str, None]:
    """OpenAI-compatible tool-calling streaming loop. Defaults to Groq."""
    client = client or _groq
    model  = model  or _QUALITY_MODEL
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    msg = None
    for _ in range(3):
        try:
            resp = await client.chat.completions.create(
                model=model, messages=messages, tools=_TOOLS,
                tool_choice="auto", max_tokens=800,
            )
        except Exception as call_exc:
            recovered = _recover_tool_calls(call_exc)
            if not recovered:
                raise
            logger.warning("Recovered %d tool call(s) from malformed Groq JSON", len(recovered))
            fake_payload = [
                {"id": f"r{i}", "type": "function",
                 "function": {"name": n, "arguments": json.dumps(a)}}
                for i, (n, a) in enumerate(recovered)
            ]
            messages.append({"role": "assistant", "content": None, "tool_calls": fake_payload})
            for i, (name, args) in enumerate(recovered):
                yield json.dumps({"tool": name})
                result = await _run_tool(name, args)
                messages.append({"role": "tool", "tool_call_id": f"r{i}", "content": result})
            continue

        msg = resp.choices[0].message
        if not msg.tool_calls:
            yield json.dumps({"token": (msg.content or "").strip()})
            return

        tool_calls_payload: list[dict[str, Any]] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
        messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls_payload})
        for tc in msg.tool_calls:
            yield json.dumps({"tool": tc.function.name})
            args   = json.loads(tc.function.arguments or "{}") or {}
            result = await _run_tool(tc.function.name, args, allowed_districts)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Loop exhausted with tool context — force a final answer
    if msg and msg.tool_calls:
        try:
            final = await client.chat.completions.create(
                model=model, messages=messages, tool_choice="none", max_tokens=400,
            )
            yield json.dumps({"token": (final.choices[0].message.content or "").strip()})
            return
        except Exception:
            pass
    yield json.dumps({"token": (msg.content or "").strip() if msg else ""})


async def _call_groq_with_tools(prompt: str, allowed_districts: list[str] | None = None, *, client=None, model: str | None = None) -> tuple[str, list[str]]:
    """OpenAI-compatible tool-calling loop. Defaults to Groq. Returns (answer, tools_used)."""
    client = client or _groq
    model  = model  or _QUALITY_MODEL
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    tools_used: list[str] = []
    msg = None
    for _ in range(3):
        try:
            resp = await client.chat.completions.create(
                model=model, messages=messages, tools=_TOOLS,
                tool_choice="auto", max_tokens=800,
            )
        except Exception as call_exc:
            recovered = _recover_tool_calls(call_exc)
            if not recovered:
                raise
            logger.warning("Recovered %d tool call(s) from malformed JSON", len(recovered))
            fake_payload = [
                {"id": f"r{i}", "type": "function",
                 "function": {"name": n, "arguments": json.dumps(a)}}
                for i, (n, a) in enumerate(recovered)
            ]
            messages.append({"role": "assistant", "content": None, "tool_calls": fake_payload})
            for i, (name, args) in enumerate(recovered):
                tools_used.append(name)
                result = await _run_tool(name, args, allowed_districts)
                messages.append({"role": "tool", "tool_call_id": f"r{i}", "content": result})
            continue

        msg = resp.choices[0].message
        if not msg.tool_calls:
            return (msg.content or "").strip(), tools_used

        tool_calls_payload: list[dict[str, Any]] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
        messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls_payload})
        for tc in msg.tool_calls:
            tools_used.append(tc.function.name)
            args   = json.loads(tc.function.arguments or "{}") or {}
            result = await _run_tool(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Loop exhausted with tool context — force a final answer
    if msg and msg.tool_calls:
        try:
            final = await client.chat.completions.create(
                model=model, messages=messages, tool_choice="none", max_tokens=400,
            )
            return (final.choices[0].message.content or "").strip(), tools_used
        except Exception:
            pass
    return (msg.content or "").strip() if msg else "", tools_used


async def _gemini_generate(client, model: str, contents, config):
    """Call Gemini with auto-retry on 503 and short 429 delays (≤ 30 s).
    Sets _gemini_dead[model] on daily quota exhaustion so the circuit-breaker
    skips that model and tries the next one in the fallback chain.
    """
    global _gemini_dead
    if _gemini_dead.get(model):
        raise RuntimeError(f"Gemini {model} daily quota exhausted — skipping")
    for attempt in range(2):
        try:
            return await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            exc_str = str(exc)
            # Daily quota or invalid/leaked key — mark dead, skip this model permanently.
            if (
                "PerDay" in exc_str
                or ("free_tier_requests" in exc_str and "limit: 0" in exc_str)
                or ("403" in exc_str and "PERMISSION_DENIED" in exc_str)
                or "API_KEY_INVALID" in exc_str
                or "leaked" in exc_str.lower()
            ):
                _gemini_dead[model] = _time.time()
                logger.warning(
                    "Gemini %s circuit breaker tripped — will retry in %.0f h (%s)",
                    model, _GEMINI_DEAD_TTL / 3600, exc_str[:120],
                )
                raise
            if attempt == 0:
                if "503" in exc_str:
                    logger.warning("Gemini %s 503, retrying in 2 s…", model)
                    await asyncio.sleep(2)
                    continue
                if "429" in exc_str:
                    import re as _re
                    m = _re.search(r"retryDelay.*?(\d+)s", exc_str)
                    wait = int(m.group(1)) if m else 0
                    if 0 < wait <= 30:
                        logger.warning("Gemini %s 429, retrying in %d s…", model, wait)
                        await asyncio.sleep(wait)
                        continue
            raise


def _to_dict(result: str) -> dict:
    """Coerce a tool result JSON string to a dict for FunctionResponse.response."""
    try:
        parsed = json.loads(result)
        return parsed if isinstance(parsed, dict) else {"result": parsed}
    except (json.JSONDecodeError, ValueError):
        return {"result": result}


def _gemini_resp_text(resp) -> str:
    try:
        return resp.text or ""
    except Exception:
        parts = resp.candidates[0].content.parts
        return " ".join(p.text for p in parts if hasattr(p, "text") and p.text)


async def _stream_gemini_with_tools(prompt: str, model: str, allowed_districts: list[str] | None = None) -> AsyncGenerator[str, None]:
    """Gemini streaming with full tool access (single model)."""
    if not settings.google_api_key:
        yield json.dumps({"token": "Gemini unavailable: GOOGLE_API_KEY not set."})
        return
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.google_api_key)
    config = types.GenerateContentConfig(
        system_instruction=_GEMINI_SYSTEM,
        tools=_get_gemini_tools(),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents: list = [types.Content(role="user", parts=[types.Part(text=_user_content(prompt))])]

    for _ in range(3):
        resp = await _gemini_generate(client, model, contents, config)
        candidate = resp.candidates[0]
        fn_calls = [p.function_call for p in candidate.content.parts
                    if hasattr(p, "function_call") and p.function_call]

        if not fn_calls:
            yield json.dumps({"token": _gemini_resp_text(resp)})
            return

        contents.append(candidate.content)
        tool_parts = []
        for fc in fn_calls:
            yield json.dumps({"tool": fc.name})
            args = dict(fc.args) if fc.args else {}
            result = await _run_tool(fc.name, args, allowed_districts)
            tool_parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=fc.name,
                    response=_to_dict(result),
                )
            ))
        contents.append(types.Content(role="tool", parts=tool_parts))

    # Loop exhausted — force a final text answer, no more tool calls
    contents.append(types.Content(
        role="user",
        parts=[types.Part(text="Based on the data collected, give a direct answer now. Do not call any more tools.")],
    ))
    try:
        resp = await _gemini_generate(client, model, contents, config)
        yield json.dumps({"token": _gemini_resp_text(resp)})
    except Exception:
        yield json.dumps({"token": ""})


async def _call_claude(prompt: str) -> str:
    if not settings.anthropic_api_key:
        raise RuntimeError("Claude unavailable: ANTHROPIC_API_KEY not set")
    resp = await _get_anthropic().messages.create(
        model=_CLAUDE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
    )
    return "".join(b.text for b in resp.content if b.type == "text")


async def _call_claude_with_tools(prompt: str) -> tuple[str, list[str]]:
    if not settings.anthropic_api_key:
        raise RuntimeError("Claude unavailable: ANTHROPIC_API_KEY not set")
    client = _get_anthropic()
    messages: list[dict[str, Any]] = [{"role": "user", "content": _user_content(prompt)}]
    tools_used: list[str] = []

    for _ in range(3):
        resp = await client.messages.create(
            model=_CLAUDE_MODEL,
            system=_GEMINI_SYSTEM,
            tools=_get_claude_tools(),
            messages=messages,
            max_tokens=1024,
        )
        tool_blocks = [b for b in resp.content if b.type == "tool_use"]
        if not tool_blocks:
            return "".join(b.text for b in resp.content if b.type == "text"), tools_used

        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for b in tool_blocks:
            tools_used.append(b.name)
            result = await _run_tool(b.name, dict(b.input))
            tool_results.append({"type": "tool_result", "tool_use_id": b.id, "content": result})
        messages.append({"role": "user", "content": tool_results})

    # Loop exhausted — force final answer
    messages.append({"role": "user", "content": "Based on the data collected, give a direct answer now. Do not call any more tools."})
    try:
        final = await client.messages.create(
            model=_CLAUDE_MODEL, system=_GEMINI_SYSTEM, messages=messages, max_tokens=400,
        )
        return "".join(b.text for b in final.content if b.type == "text"), tools_used
    except Exception:
        return "", tools_used


async def _stream_claude_with_tools(prompt: str) -> AsyncGenerator[str, None]:
    if not settings.anthropic_api_key:
        yield json.dumps({"token": "Claude unavailable: ANTHROPIC_API_KEY not set."})
        return
    client = _get_anthropic()
    messages: list[dict[str, Any]] = [{"role": "user", "content": _user_content(prompt)}]

    for _ in range(3):
        resp = await client.messages.create(
            model=_CLAUDE_MODEL,
            system=_GEMINI_SYSTEM,
            tools=_get_claude_tools(),
            messages=messages,
            max_tokens=1024,
        )
        tool_blocks = [b for b in resp.content if b.type == "tool_use"]
        if not tool_blocks:
            yield json.dumps({"token": "".join(b.text for b in resp.content if b.type == "text")})
            return

        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for b in tool_blocks:
            yield json.dumps({"tool": b.name})
            result = await _run_tool(b.name, dict(b.input))
            tool_results.append({"type": "tool_result", "tool_use_id": b.id, "content": result})
        messages.append({"role": "user", "content": tool_results})

    # Loop exhausted — force final answer
    messages.append({"role": "user", "content": "Based on the data collected, give a direct answer now. Do not call any more tools."})
    try:
        final = await client.messages.create(
            model=_CLAUDE_MODEL, system=_GEMINI_SYSTEM, messages=messages, max_tokens=400,
        )
        yield json.dumps({"token": "".join(b.text for b in final.content if b.type == "text")})
    except Exception:
        yield json.dumps({"token": ""})


async def _call_gemini_with_tools(prompt: str, model: str, allowed_districts: list[str] | None = None) -> tuple[str, list[str]]:
    """Gemini non-streaming with full tool access (single model)."""
    if not settings.google_api_key:
        raise RuntimeError("Gemini unavailable: GOOGLE_API_KEY not set")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.google_api_key)
    config = types.GenerateContentConfig(
        system_instruction=_GEMINI_SYSTEM,
        tools=_get_gemini_tools(),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents: list = [types.Content(role="user", parts=[types.Part(text=_user_content(prompt))])]
    tools_used: list[str] = []

    for _ in range(3):
        resp = await _gemini_generate(client, model, contents, config)
        candidate = resp.candidates[0]
        fn_calls = [p.function_call for p in candidate.content.parts
                    if hasattr(p, "function_call") and p.function_call]

        if not fn_calls:
            return _gemini_resp_text(resp), tools_used

        contents.append(candidate.content)
        tool_parts = []
        for fc in fn_calls:
            tools_used.append(fc.name)
            args = dict(fc.args) if fc.args else {}
            result = await _run_tool(fc.name, args, allowed_districts)
            tool_parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=fc.name,
                    response=_to_dict(result),
                )
            ))
        contents.append(types.Content(role="tool", parts=tool_parts))

    # Loop exhausted — force a final text answer, no more tool calls
    contents.append(types.Content(
        role="user",
        parts=[types.Part(text="Based on the data collected, give a direct answer now. Do not call any more tools.")],
    ))
    try:
        resp = await _gemini_generate(client, model, contents, config)
        return _gemini_resp_text(resp), tools_used
    except Exception:
        return "", tools_used
