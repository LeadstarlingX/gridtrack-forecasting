"""LLM chat service.

Primary: Groq llama-3.3-70b-versatile.
Fallback: Gemini Flash (if Groq fails).

Exposes:
  call_llm(prompt)              — non-streaming, returns full string
  stream_llm(prompt)            — async generator of token strings
  call_llm_with_tools(prompt)   — tool-calling loop, uses in-memory forecast state
"""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from groq import AsyncGroq

from app.config import settings

logger = logging.getLogger(__name__)

_groq         = AsyncGroq(api_key=settings.groq_api_key)
_FAST_MODEL   = "llama-3.1-8b-instant"       # urgency notes, short tasks
_QUALITY_MODEL = "llama-3.3-70b-versatile"   # chatbot, tools, incidents, staffing


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
            return sample + [f"\u2026{len(v) - max_items} more items omitted"]
        if isinstance(v, str) and len(v) > max_str:
            return v[:max_str] + "\u2026"
        return v
 
    # Progressively tighter passes: (max_list_items, max_string_chars)
    for max_items, max_str in [(10, 300), (5, 150), (3, 80), (2, 50), (1, 30)]:
        result = json.dumps(_compress(ctx, max_items, max_str))
        if len(result) <= char_budget:
            return result
 
    # Guarantee: hard-truncate the tightest pass — never blows the budget
    return json.dumps(_compress(ctx, 1, 30))[:char_budget] + "\u2026"



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
]


def _run_tool(name: str, args: dict[str, Any]) -> str:
    from datetime import datetime, timedelta, timezone
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


async def call_llm_with_tools(prompt: str) -> str:
    """Tool-calling loop. Groq asks for live district data via tools, then synthesises."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

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
            return (msg.content or "").strip()

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
            args   = json.loads(tc.function.arguments or "{}")
            result = _run_tool(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return (msg.content or "").strip()


# ── Private helpers ──────────────────────────────────────────────────────────

async def _call_groq(prompt: str) -> str:
    resp = await _groq.chat.completions.create(
        model=_QUALITY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


async def _call_gemini(prompt: str) -> str:
    import asyncio
    import google.generativeai as genai
    genai.configure(api_key=settings.google_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp  = await asyncio.to_thread(model.generate_content, prompt)
    return resp.text
