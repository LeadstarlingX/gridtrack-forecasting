"""LLM chat service.

Primary: Groq llama-3.3-70b-versatile.
Fallback: Gemini Flash (if Groq fails).

Exposes:
  call_llm(prompt)              — non-streaming, returns full string
  stream_llm(prompt)            — async generator of token strings
  call_llm_with_tools(prompt)   — tool-calling loop, returns (answer, tools_used)
  stream_with_tools(prompt)     — streaming with tool events: yields JSON frames
"""

import asyncio
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

_PG_SCHEMA = """
PostgreSQL tables (schema: public, double-quote all identifiers):
- "Drivers": "DriverId" UUID PK, "Name" TEXT, "IsActive" BOOL, "DistrictId" TEXT, "LastSeen" TIMESTAMPTZ, "Location" GEOGRAPHY
- "Deliveries": "DeliveryId" UUID PK, "DriverId" UUID FK->Drivers, "Status" TEXT (Pending/InTransit/Delivered/Cancelled), "DistrictId" TEXT, "AnomalyFlag" BOOL, "CreatedAt" TIMESTAMPTZ, "ExpectedEta" TIMESTAMPTZ, "ActualEta" TIMESTAMPTZ
WARNING: There is NO "Districts" table. District names are not stored — only DistrictId (a text key) exists. Never query for district names.
Rules: SELECT only. LIMIT 20. Double-quote identifiers. SELECT only the columns needed to answer the question — never SELECT *.
IMPORTANT: For time arithmetic use make_interval() to avoid JSON escaping issues.
Examples: NOW() - make_interval(hours => 1), NOW() - make_interval(days => 7).
Single quotes in SQL do NOT need backslash-escaping inside JSON.
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

# Lazy-cached Gemini tool list (avoids importing google.genai at startup)
_gemini_tools_cache: list | None = None


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
    # Handles: name/>, name{}, name[]{}, name={}, name={"sql":...}>, </function> endings
    matches = re.findall(
        r"<function=(\w+)[\[\]\s,=]*((?:\{.*?\})?)\s*(?:/>|</function>)",
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


async def stream_with_tools(prompt: str) -> AsyncGenerator[str, None]:
    """Streaming with tool calling. Gemini primary, Groq fallback.

    Yields JSON-encoded frames:
      {"tool": "name"}       — emitted when a tool is called
      {"token": "text..."}   — final answer
    """
    got_answer = False
    try:
        async for frame in _stream_gemini_with_tools(prompt):
            yield frame
            if not got_answer:
                try:
                    got_answer = bool(json.loads(frame).get("token"))
                except (json.JSONDecodeError, AttributeError):
                    pass
        if got_answer:
            return
        logger.warning("Gemini returned empty answer, falling back to Groq")
    except Exception as gemini_exc:
        logger.warning("Gemini primary failed (%s), falling back to Groq", gemini_exc)

    try:
        async for frame in _stream_groq_with_tools(prompt):
            yield frame
    except Exception as groq_exc:
        logger.warning("Groq fallback also failed (%s)", groq_exc)
        yield json.dumps({"token": "Both AI providers are temporarily unavailable. Please try again in a moment."})


async def call_llm_with_tools(prompt: str) -> tuple[str, list[str]]:
    """Tool-calling loop. Gemini primary, Groq fallback. Returns (answer, tools_used)."""
    try:
        answer, tools_used = await _call_gemini_with_tools(prompt)
        if answer:
            return answer, tools_used
        logger.warning("Gemini returned empty answer, falling back to Groq")
    except Exception as gemini_exc:
        logger.warning("Gemini primary failed (%s), falling back to Groq", gemini_exc)

    try:
        return await _call_groq_with_tools(prompt)
    except Exception as groq_exc:
        logger.warning("Groq fallback also failed (%s)", groq_exc)
        return "Both AI providers are temporarily unavailable. Please try again in a moment.", []


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
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=settings.google_api_key)
    clean = prompt.split("You have access to tools")[0].strip() if "You have access to tools" in prompt else prompt
    resp = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-2.0-flash",
        contents=clean,
        config=types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    try:
        return resp.text
    except Exception:
        parts = resp.candidates[0].content.parts
        return " ".join(p.text for p in parts if hasattr(p, "text") and p.text)


async def _stream_groq_with_tools(prompt: str) -> AsyncGenerator[str, None]:
    """Groq tool-calling streaming loop (fallback)."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    msg = None
    for _ in range(3):
        try:
            resp = await _groq.chat.completions.create(
                model=_QUALITY_MODEL, messages=messages, tools=_TOOLS,
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
            result = await _run_tool(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Loop exhausted with tool context — force a final answer
    if msg and msg.tool_calls:
        try:
            final = await _groq.chat.completions.create(
                model=_QUALITY_MODEL, messages=messages, tool_choice="none", max_tokens=400,
            )
            yield json.dumps({"token": (final.choices[0].message.content or "").strip()})
            return
        except Exception:
            pass
    yield json.dumps({"token": (msg.content or "").strip() if msg else ""})


async def _call_groq_with_tools(prompt: str) -> tuple[str, list[str]]:
    """Groq tool-calling loop (fallback). Returns (answer, tools_used)."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    tools_used: list[str] = []
    msg = None
    for _ in range(3):
        try:
            resp = await _groq.chat.completions.create(
                model=_QUALITY_MODEL, messages=messages, tools=_TOOLS,
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
                tools_used.append(name)
                result = await _run_tool(name, args)
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
            final = await _groq.chat.completions.create(
                model=_QUALITY_MODEL, messages=messages, tool_choice="none", max_tokens=400,
            )
            return (final.choices[0].message.content or "").strip(), tools_used
        except Exception:
            pass
    return (msg.content or "").strip() if msg else "", tools_used


async def _gemini_generate(client, contents, config):
    """Call Gemini with auto-retry on 503 and short 429 delays (≤ 30 s)."""
    for attempt in range(2):
        try:
            return await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.0-flash",
                contents=contents,
                config=config,
            )
        except Exception as exc:
            exc_str = str(exc)
            if attempt == 0:
                if "503" in exc_str:
                    logger.warning("Gemini 503, retrying in 2 s…")
                    await asyncio.sleep(2)
                    continue
                if "429" in exc_str:
                    import re as _re
                    m = _re.search(r"retryDelay.*?(\d+)s", exc_str)
                    wait = int(m.group(1)) if m else 0
                    if 0 < wait <= 30:
                        logger.warning("Gemini 429, retrying in %d s…", wait)
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


async def _stream_gemini_with_tools(prompt: str) -> AsyncGenerator[str, None]:
    """Gemini primary streaming with full tool access."""
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
        resp = await _gemini_generate(client, contents, config)
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
            result = await _run_tool(fc.name, args)
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
        resp = await _gemini_generate(client, contents, config)
        yield json.dumps({"token": _gemini_resp_text(resp)})
    except Exception:
        yield json.dumps({"token": ""})


async def _call_gemini_with_tools(prompt: str) -> tuple[str, list[str]]:
    """Gemini primary non-streaming with full tool access."""
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
        resp = await _gemini_generate(client, contents, config)
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
            result = await _run_tool(fc.name, args)
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
        resp = await _gemini_generate(client, contents, config)
        return _gemini_resp_text(resp), tools_used
    except Exception:
        return "", tools_used
