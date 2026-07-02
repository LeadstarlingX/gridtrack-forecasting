"""PDF report generator.

Makes 3 focused LLM calls (< 1 500 input tokens each) to stay well within
Groq's 12 000 TPM free-tier limit, then assembles a 1-page PDF with fpdf2.

Call chain:
  1. Situation summary   (~300 in, ~150 out)
  2. Recommendations     (~200 in, ~200 out)   — only receives call-1 output
  3. Priority actions    (~400 in, ~150 out)   — receives call-1 + call-2 output
"""

import asyncio
import logging
from datetime import datetime, timezone

from fpdf import FPDF

from app.services.chatbot import _groq, _QUALITY_MODEL, compress_context

logger = logging.getLogger(__name__)

_MAX_CTX_CHARS = 2_000   # ~500 tokens — keeps each call cheap


def _trim_messages(messages: list[dict[str, str]], max_chars: int = 1_500) -> str:
    """Last ≤8 turns as a compact string, capped at max_chars."""
    lines: list[str] = []
    total = 0
    for m in reversed(messages[-8:]):
        role    = m.get("role", "user").upper()
        content = m.get("content", "")[:300]
        line    = f"{role}: {content}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(reversed(lines))


def _sanitize(text: str) -> str:
    """Replace common non-Latin-1 chars so fpdf2 core fonts don't choke."""
    return (
        text
        .replace("•", "-")    # •
        .replace("–", "-")    # en-dash
        .replace("—", "--")   # em-dash
        .replace("‘", "'").replace("’", "'")
        .replace("“", '"').replace("”", '"')
        .encode("latin-1", errors="replace")
        .decode("latin-1")
    )


async def generate_report(messages: list[dict[str, str]], context: dict) -> bytes:
    """Run 3 LLM calls → return PDF bytes."""
    ctx  = compress_context(context, char_budget=_MAX_CTX_CHARS)
    conv = _trim_messages(messages)

    # ── Call 1: situation summary ────────────────────────────────────────────
    r1 = await _groq.chat.completions.create(
        model=_QUALITY_MODEL,
        messages=[{"role": "user", "content": (
            "You are a logistics operations analyst for a Damascus delivery service.\n"
            f"Operational data snapshot: {ctx}\n"
            f"Recent conversation:\n{conv}\n\n"
            "Write a concise 2-3 sentence situation summary for an operations report. "
            "Include current delivery status and any active issues. Plain text only."
        )}],
        max_tokens=150,
    )
    summary = r1.choices[0].message.content.strip()
    logger.debug("Report call 1 done (%d chars)", len(summary))

    await asyncio.sleep(1)   # pace calls; total ~500 tokens → well under 12 K TPM

    # ── Call 2: recommendations ──────────────────────────────────────────────
    r2 = await _groq.chat.completions.create(
        model=_QUALITY_MODEL,
        messages=[{"role": "user", "content": (
            f"Current situation: {summary}\n\n"
            "Write exactly 4 actionable recommendations to improve delivery performance. "
            "Each on its own line starting with '- '. No headers, no extra text."
        )}],
        max_tokens=200,
    )
    recommendations = r2.choices[0].message.content.strip()

    await asyncio.sleep(1)

    # ── Call 3: priority actions ─────────────────────────────────────────────
    r3 = await _groq.chat.completions.create(
        model=_QUALITY_MODEL,
        messages=[{"role": "user", "content": (
            f"Situation: {summary}\n"
            f"Recommendations: {recommendations}\n\n"
            "List exactly 3 immediate priority actions (specific: who does what, when). "
            "Each on its own line starting with '- '. No headers, no extra text."
        )}],
        max_tokens=150,
    )
    actions = r3.choices[0].message.content.strip()

    return _build_pdf(summary, recommendations, actions)


def _build_pdf(summary: str, recommendations: str, actions: str) -> bytes:
    pdf = FPDF()
    pdf.set_margins(20, 20, 20)
    pdf.add_page()

    # ── Title block ──────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", style="B", size=18)
    pdf.multi_cell(0, 12, "GridTrack Operations Report")

    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 6, f"Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")
    pdf.ln(4)

    pdf.set_draw_color(200, 200, 200)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(6)

    # ── Sections ─────────────────────────────────────────────────────────────
    def section(title: str, body: str) -> None:
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", style="B", size=11)
        pdf.multi_cell(0, 8, title)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 6, _sanitize(body))
        pdf.ln(3)

    section("Situation Summary", summary)
    section("Key Recommendations", recommendations)
    section("Priority Actions", actions)

    # ── Footer ───────────────────────────────────────────────────────────────
    pdf.set_y(-18)
    pdf.set_font("Helvetica", style="I", size=8)
    pdf.set_text_color(160, 160, 160)
    pdf.cell(0, 5, "Generated by GridTrack AI Assistant", align="C")

    return bytes(pdf.output())
