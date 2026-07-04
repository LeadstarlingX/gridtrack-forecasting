"""Tests for app/services/report.py — PDF generation."""
from unittest.mock import AsyncMock, MagicMock

from app.services.report import _trim_messages, _sanitize, _build_pdf, generate_report


# ── _trim_messages ────────────────────────────────────────────────────────────

def test_trim_messages_formats_roles_and_content():
    msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    result = _trim_messages(msgs)
    assert "USER:" in result
    assert "ASSISTANT:" in result
    assert "hello" in result


def test_trim_messages_respects_char_limit():
    msgs = [{"role": "user", "content": "x" * 1000}] * 20
    result = _trim_messages(msgs, max_chars=500)
    assert len(result) <= 500


def test_trim_messages_empty_returns_empty():
    assert _trim_messages([]) == ""


def test_trim_messages_truncates_long_content():
    msgs = [{"role": "user", "content": "a" * 400}]
    result = _trim_messages(msgs)
    # content is capped at 300 chars per message
    assert len(result) < 400


# ── _sanitize ─────────────────────────────────────────────────────────────────

def test_sanitize_replaces_bullet():
    assert _sanitize("• item") == "- item"


def test_sanitize_replaces_em_dash():
    assert _sanitize("a — b") == "a -- b"


def test_sanitize_replaces_en_dash():
    assert _sanitize("a – b") == "a - b"


def test_sanitize_replaces_curly_quotes():
    result = _sanitize("“quoted”")
    assert '"quoted"' in result


def test_sanitize_plain_text_unchanged():
    assert _sanitize("hello world 123") == "hello world 123"


# ── _build_pdf ────────────────────────────────────────────────────────────────

def test_build_pdf_returns_bytes():
    pdf_bytes = _build_pdf("Situation OK.", "- Rec 1\n- Rec 2", "- Action 1\n- Action 2")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 100  # non-trivial PDF


def test_build_pdf_contains_pdf_header():
    pdf_bytes = _build_pdf("summary", "recommendations", "actions")
    assert pdf_bytes[:4] == b"%PDF"


def test_build_pdf_handles_special_chars_in_body():
    # Should not raise even with chars that need sanitizing
    pdf_bytes = _build_pdf("Status — good", "- Rec • one", "- Action")
    assert isinstance(pdf_bytes, bytes)


# ── generate_report ───────────────────────────────────────────────────────────

async def test_generate_report_makes_three_llm_calls(mocker):
    mocker.patch("asyncio.sleep", new=AsyncMock())

    def _make_resp(text):
        r = MagicMock()
        r.choices = [MagicMock()]
        r.choices[0].message.content = text
        return r

    mock_create = AsyncMock(side_effect=[
        _make_resp("Situation: 10 deliveries active."),
        _make_resp("- Monitor district 1\n- Reduce ETA errors"),
        _make_resp("- Call dispatcher\n- Check route"),
    ])
    mocker.patch("app.services.report._groq.chat.completions.create", new=mock_create)

    result = await generate_report(
        [{"role": "user", "content": "How many deliveries?"},
         {"role": "assistant", "content": "There are 10."}],
        {"activeDrivers": 5, "totalDeliveries": 10},
    )

    assert isinstance(result, bytes)
    assert result[:4] == b"%PDF"
    assert mock_create.await_count == 3
