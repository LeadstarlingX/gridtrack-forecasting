import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.chatbot import call_llm, _call_groq, _call_gemini


async def test_returns_groq_response(mocker):
    mocker.patch("app.services.chatbot._call_groq", return_value="groq answer")
    result = await call_llm("what is the status?")
    assert result == "groq answer"


async def test_falls_back_to_gemini_when_groq_fails(mocker):
    mocker.patch("app.services.chatbot._call_groq", side_effect=Exception("timeout"))
    mocker.patch("app.services.chatbot._call_gemini", return_value="gemini answer")
    result = await call_llm("what is the status?")
    assert result == "gemini answer"


async def test_raises_if_both_fail(mocker):
    mocker.patch("app.services.chatbot._call_groq", side_effect=Exception("groq down"))
    mocker.patch("app.services.chatbot._call_gemini", side_effect=Exception("gemini down"))
    with pytest.raises(Exception):
        await call_llm("what is the status?")


# ── _call_groq (direct) ───────────────────────────────────────────────────────

async def test_call_groq_returns_stripped_content(mocker):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "  answer with spaces  "
    mocker.patch(
        "app.services.chatbot._groq.chat.completions.create",
        new=AsyncMock(return_value=mock_resp),
    )
    result = await _call_groq("test prompt")
    assert result == "answer with spaces"


async def test_call_groq_passes_prompt_in_user_message(mocker):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "ok"
    mock_create = AsyncMock(return_value=mock_resp)
    mocker.patch("app.services.chatbot._groq.chat.completions.create", new=mock_create)
    await _call_groq("district load?")
    messages = mock_create.call_args.kwargs["messages"]
    assert any(m["content"] == "district load?" for m in messages)


# ── _call_gemini (direct) ─────────────────────────────────────────────────────

async def test_call_gemini_returns_response_text(mocker):
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text = "gemini response"
    mocker.patch("google.generativeai.configure")
    mocker.patch("google.generativeai.GenerativeModel", return_value=mock_model)
    result = await _call_gemini("test prompt")
    assert result == "gemini response"


async def test_call_gemini_passes_prompt_to_model(mocker):
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text = "ok"
    mocker.patch("google.generativeai.configure")
    mocker.patch("google.generativeai.GenerativeModel", return_value=mock_model)
    await _call_gemini("how many drivers?")
    mock_model.generate_content.assert_called_once_with("how many drivers?")
