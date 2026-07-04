import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import app.services.chatbot as chatbot_module
from app.services.chatbot import (
    call_llm, _call_groq, _call_gemini,
    compress_context, _trim_result, _recover_tool_calls,
    _to_dict, _gemini_resp_text, _user_content, _extract_question,
    call_llm_fast, stream_llm, call_llm_with_tools, stream_with_tools,
    _run_tool, _stream_groq_with_tools, _call_groq_with_tools,
    _gemini_generate, _stream_gemini_with_tools, _call_gemini_with_tools,
    _get_gemini_tools,
)


async def _collect(gen):
    return [x async for x in gen]


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
    mocker.patch.object(chatbot_module, "settings", MagicMock(google_api_key="fake-key"))
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value.text = "gemini response"
    mocker.patch("google.genai.Client", return_value=mock_client)
    result = await _call_gemini("test prompt")
    assert result == "gemini response"


async def test_call_gemini_passes_prompt_to_model(mocker):
    mocker.patch.object(chatbot_module, "settings", MagicMock(google_api_key="fake-key"))
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value.text = "ok"
    mocker.patch("google.genai.Client", return_value=mock_client)
    await _call_gemini("how many drivers?")
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.0-flash"
    assert call_kwargs["contents"] == "how many drivers?"


# ── compress_context ──────────────────────────────────────────────────────────

def test_compress_context_returns_raw_when_fits():
    ctx = {"a": 1}
    result = compress_context(ctx)
    assert result == json.dumps(ctx)


def test_compress_context_truncates_large_lists():
    ctx = {"drivers": list(range(100))}
    result = compress_context(ctx, char_budget=50)
    assert "omitted" in result


def test_compress_context_truncates_long_strings():
    ctx = {"note": "x" * 2000}
    result = compress_context(ctx, char_budget=100)
    assert len(result) <= 100  # budget enforced; truncation marker is in the raw value


def test_compress_context_preserves_dict_keys():
    ctx = {"a": {"b": {"c": "deep"}}}
    result = compress_context(ctx)
    assert '"a"' in result and '"b"' in result


# ── _trim_result ──────────────────────────────────────────────────────────────

def test_trim_result_returns_unchanged_when_short():
    s = "hello"
    assert _trim_result(s) == s


def test_trim_result_truncates_at_limit():
    long = "x" * 5000
    result = _trim_result(long)
    assert result.endswith("…(truncated)")
    assert len(result) <= 3015


# ── _recover_tool_calls ───────────────────────────────────────────────────────

def _make_exc(failed: str):
    exc = Exception("tool_use_failed")
    exc.body = {"error": {"failed_generation": failed}}
    return exc


def test_recover_tool_calls_standard_format():
    # name{} format — no separator between name and args, closing tag without leading >
    exc = _make_exc('<function=query_postgres{"sql":"SELECT 1"}</function>')
    result = _recover_tool_calls(exc)
    assert result is not None
    assert result[0][0] == "query_postgres"
    assert result[0][1] == {"sql": "SELECT 1"}


def test_recover_tool_calls_bracket_format():
    # name[]{} format — brackets between name and args
    exc = _make_exc('<function=query_postgres[]{"sql":"SELECT 1"}</function>')
    result = _recover_tool_calls(exc)
    assert result is not None
    assert result[0][1] == {"sql": "SELECT 1"}


def test_recover_tool_calls_self_closing_no_args():
    exc = _make_exc("<function=get_all_districts_summary/>")
    result = _recover_tool_calls(exc)
    assert result is not None
    assert result[0] == ("get_all_districts_summary", {})


def test_recover_tool_calls_equals_separator():
    # name={} format — equals sign between name and args
    exc = _make_exc('<function=query_postgres={"sql":"SELECT 1"}</function>')
    result = _recover_tool_calls(exc)
    assert result is not None
    assert result[0][1] == {"sql": "SELECT 1"}


def test_recover_tool_calls_fixes_escaped_quotes():
    exc = _make_exc("<function=query_postgres{\"sql\":\"SELECT \\'1\\'\"}</function>")
    result = _recover_tool_calls(exc)
    assert result is not None


def test_recover_tool_calls_no_body_returns_none():
    exc = Exception("plain error")
    assert _recover_tool_calls(exc) is None


def test_recover_tool_calls_no_failed_generation_returns_none():
    exc = Exception("x")
    exc.body = {"error": {}}
    assert _recover_tool_calls(exc) is None


def test_recover_tool_calls_no_matches_returns_none():
    exc = _make_exc("this is not a function call")
    assert _recover_tool_calls(exc) is None


# ── _to_dict ─────────────────────────────────────────────────────────────────

def test_to_dict_wraps_list():
    assert _to_dict("[1,2]") == {"result": [1, 2]}


def test_to_dict_passes_through_dict():
    assert _to_dict('{"key": "val"}') == {"key": "val"}


def test_to_dict_wraps_invalid_json():
    assert _to_dict("not json") == {"result": "not json"}


# ── _gemini_resp_text ─────────────────────────────────────────────────────────

def test_gemini_resp_text_returns_text_attr():
    resp = MagicMock()
    resp.text = "the answer"
    assert _gemini_resp_text(resp) == "the answer"


def test_gemini_resp_text_falls_back_to_parts():
    from unittest.mock import PropertyMock
    resp = MagicMock()
    type(resp).text = PropertyMock(side_effect=Exception("no text"))
    part = MagicMock()
    part.text = "part text"
    resp.candidates = [MagicMock()]
    resp.candidates[0].content.parts = [part]
    assert _gemini_resp_text(resp) == "part text"


# ── _user_content / _extract_question ────────────────────────────────────────

def test_user_content_extracts_from_marker():
    prompt = "System stuff\nOperational context: live data\nQuestion: how many?"
    result = _user_content(prompt)
    assert result.startswith("Operational context:")


def test_user_content_returns_full_when_no_marker():
    prompt = "plain prompt"
    assert _user_content(prompt) == "plain prompt"


def test_extract_question_strips_prefix():
    prompt = "context stuff\nQuestion: how many drivers?"
    assert _extract_question(prompt) == "how many drivers?"


def test_extract_question_returns_full_when_no_marker():
    prompt = "bare prompt"
    assert _extract_question(prompt) == "bare prompt"


# ── call_llm_fast ────────────────────────────────────────────────────────────

async def test_call_llm_fast_returns_groq_response(mocker):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "  urgency: 7  "
    mocker.patch("app.services.chatbot._groq.chat.completions.create",
                 new=AsyncMock(return_value=mock_resp))
    result = await call_llm_fast("score this")
    assert result == "urgency: 7"


async def test_call_llm_fast_falls_back_to_quality_model_on_error(mocker):
    mocker.patch(
        "app.services.chatbot._groq.chat.completions.create",
        new=AsyncMock(side_effect=[Exception("fast model down"),
                                   _make_groq_resp("fallback answer")]),
    )
    result = await call_llm_fast("score this")
    assert result == "fallback answer"


def _make_groq_resp(text: str):
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = text
    return r


# ── stream_llm ───────────────────────────────────────────────────────────────

async def test_stream_llm_yields_tokens(mocker):
    chunk1, chunk2 = MagicMock(), MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "hello "
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = "world"

    async def fake_stream():
        yield chunk1
        yield chunk2

    mocker.patch("app.services.chatbot._groq.chat.completions.create",
                 new=AsyncMock(return_value=fake_stream()))
    tokens = await _collect(stream_llm("hi"))
    assert any("hello" in t for t in tokens)


async def test_stream_llm_falls_back_on_error(mocker):
    mocker.patch("app.services.chatbot._groq.chat.completions.create",
                 new=AsyncMock(side_effect=Exception("stream broken")))
    mocker.patch("app.services.chatbot._call_gemini",
                 new=AsyncMock(return_value="gemini fallback"))
    tokens = await _collect(stream_llm("hi"))
    assert tokens == ["gemini fallback"]


# ── _run_tool ────────────────────────────────────────────────────────────────

async def test_run_tool_district_activity(mocker):
    import app.services.forecast as forecast_module
    now = datetime.now(timezone.utc)
    mocker.patch.object(forecast_module, "_windows", {"d1": [now]})
    mocker.patch.object(forecast_module, "_active_drivers", {"d1": {"drv1", "drv2"}})
    result = json.loads(await _run_tool("get_district_activity", {"district_id": "d1"}))
    assert result["district"] == "d1"
    assert result["active_drivers"] == 2


async def test_run_tool_all_districts_summary(mocker):
    import app.services.forecast as forecast_module
    now = datetime.now(timezone.utc)
    mocker.patch.object(forecast_module, "_windows", {"d1": [now], "d2": []})
    mocker.patch.object(forecast_module, "_active_drivers", {"d1": {"drv1"}})
    result = json.loads(await _run_tool("get_all_districts_summary", {}))
    assert isinstance(result, list)
    districts = [r["district"] for r in result]
    assert "d1" in districts and "d2" in districts


async def test_run_tool_query_postgres_select(mocker):
    import app.db  # ensure module is in sys.modules before patching
    mock_pool = AsyncMock()
    mock_pool.fetch.return_value = [{"count": 5}]
    mocker.patch("app.db.get_pool", new=AsyncMock(return_value=mock_pool))
    result = json.loads(await _run_tool("query_postgres", {"sql": 'SELECT COUNT(*) FROM "Drivers"'}))
    assert result[0]["count"] == 5


async def test_run_tool_query_postgres_blocks_non_select(mocker):
    result = json.loads(await _run_tool("query_postgres", {"sql": "DROP TABLE Drivers"}))
    assert "error" in result


async def test_run_tool_query_clickhouse(mocker):
    import app.ch  # ensure module is in sys.modules before patching
    ch_result = MagicMock()
    ch_result.column_names = ["total"]
    ch_result.result_rows = [[42]]
    mocker.patch("app.ch.ch_query", new=AsyncMock(return_value=ch_result))
    result = json.loads(await _run_tool("query_clickhouse",
                                        {"sql": "SELECT count() AS total FROM driver_positions"}))
    assert result[0]["total"] == 42


async def test_run_tool_query_clickhouse_blocks_non_select():
    result = json.loads(await _run_tool("query_clickhouse", {"sql": "INSERT INTO t VALUES (1)"}))
    assert "error" in result


async def test_run_tool_unknown_returns_error():
    result = json.loads(await _run_tool("nonexistent_tool", {}))
    assert "error" in result


# ── _gemini_generate ─────────────────────────────────────────────────────────

async def test_gemini_generate_returns_on_success(mocker):
    mocker.patch("asyncio.sleep", new=AsyncMock())
    mock_resp = MagicMock()
    mocker.patch("asyncio.to_thread", new=AsyncMock(return_value=mock_resp))
    client, config = MagicMock(), MagicMock()
    result = await _gemini_generate(client, [], config)
    assert result is mock_resp


async def test_gemini_generate_retries_on_503(mocker):
    mocker.patch("asyncio.sleep", new=AsyncMock())
    mock_resp = MagicMock()
    mocker.patch("asyncio.to_thread",
                 new=AsyncMock(side_effect=[Exception("503 Service Unavailable"), mock_resp]))
    result = await _gemini_generate(MagicMock(), [], MagicMock())
    assert result is mock_resp


async def test_gemini_generate_retries_on_short_429(mocker):
    mocker.patch("asyncio.sleep", new=AsyncMock())
    mock_resp = MagicMock()
    mocker.patch("asyncio.to_thread",
                 new=AsyncMock(side_effect=[
                     Exception("429 Too Many Requests retryDelay 5s"),
                     mock_resp,
                 ]))
    result = await _gemini_generate(MagicMock(), [], MagicMock())
    assert result is mock_resp


async def test_gemini_generate_does_not_retry_on_long_429(mocker):
    mocker.patch("asyncio.sleep", new=AsyncMock())
    mocker.patch("asyncio.to_thread",
                 new=AsyncMock(side_effect=Exception("429 Too Many Requests retryDelay 60s")))
    with pytest.raises(Exception, match="429"):
        await _gemini_generate(MagicMock(), [], MagicMock())


# ── _stream_groq_with_tools ───────────────────────────────────────────────────

async def test_stream_groq_direct_answer(mocker):
    """Groq answers immediately with no tool calls."""
    mocker.patch("app.services.chatbot._groq.chat.completions.create",
                 new=AsyncMock(return_value=_make_groq_resp_with_tools(None, "42 drivers")))
    frames = await _collect(_stream_groq_with_tools("how many drivers?"))
    assert any(json.loads(f).get("token") == "42 drivers" for f in frames)


async def test_stream_groq_calls_tool_then_answers(mocker):
    """Groq calls one tool, then gives a final answer."""
    tc = _make_tool_call("get_all_districts_summary", "{}")
    resp1 = _make_groq_resp_with_tools([tc], None)
    resp2 = _make_groq_resp_with_tools(None, "5 active districts")
    mocker.patch("app.services.chatbot._groq.chat.completions.create",
                 new=AsyncMock(side_effect=[resp1, resp2]))
    mocker.patch("app.services.chatbot._run_tool", new=AsyncMock(return_value='{"count": 5}'))
    frames = await _collect(_stream_groq_with_tools("district summary"))
    tool_frames = [f for f in frames if "tool" in json.loads(f)]
    token_frames = [f for f in frames if "token" in json.loads(f)]
    assert tool_frames[0] == json.dumps({"tool": "get_all_districts_summary"})
    assert any(json.loads(f)["token"] == "5 active districts" for f in token_frames)


async def test_stream_groq_force_answer_after_loop_exhaustion(mocker):
    """After 3 rounds of tool calls, forces a text answer."""
    tc = _make_tool_call("get_all_districts_summary", "{}")
    resp_with_tools = _make_groq_resp_with_tools([tc], None)
    final_resp = _make_groq_resp_with_tools(None, "Final forced answer")
    mocker.patch("app.services.chatbot._groq.chat.completions.create",
                 new=AsyncMock(side_effect=[resp_with_tools] * 3 + [final_resp]))
    mocker.patch("app.services.chatbot._run_tool", new=AsyncMock(return_value="{}"))
    frames = await _collect(_stream_groq_with_tools("question"))
    assert any("Final forced answer" in json.loads(f).get("token", "") for f in frames)


# ── _call_groq_with_tools ─────────────────────────────────────────────────────

async def test_call_groq_with_tools_direct_answer(mocker):
    mocker.patch("app.services.chatbot._groq.chat.completions.create",
                 new=AsyncMock(return_value=_make_groq_resp_with_tools(None, "direct")))
    answer, tools = await _call_groq_with_tools("question")
    assert answer == "direct"
    assert tools == []


async def test_call_groq_with_tools_records_tool_names(mocker):
    tc = _make_tool_call("query_postgres", '{"sql":"SELECT 1"}')
    resp1 = _make_groq_resp_with_tools([tc], None)
    resp2 = _make_groq_resp_with_tools(None, "answer")
    mocker.patch("app.services.chatbot._groq.chat.completions.create",
                 new=AsyncMock(side_effect=[resp1, resp2]))
    mocker.patch("app.services.chatbot._run_tool", new=AsyncMock(return_value="[]"))
    answer, tools = await _call_groq_with_tools("q")
    assert "query_postgres" in tools


# ── _stream_gemini_with_tools ─────────────────────────────────────────────────

async def test_stream_gemini_unavailable_when_no_key(mocker):
    mocker.patch.object(chatbot_module, "settings", MagicMock(google_api_key=""))
    frames = await _collect(_stream_gemini_with_tools("q"))
    assert any("unavailable" in json.loads(f).get("token", "") for f in frames)


async def test_stream_gemini_direct_answer(mocker):
    mocker.patch.object(chatbot_module, "settings", MagicMock(google_api_key="key"))
    mocker.patch("google.genai.Client")
    mocker.patch("google.genai.types", MagicMock())
    mocker.patch("app.services.chatbot._get_gemini_tools", return_value=[])
    mock_resp = _make_gemini_resp_no_tools("Direct answer")
    mocker.patch("app.services.chatbot._gemini_generate", new=AsyncMock(return_value=mock_resp))
    frames = await _collect(_stream_gemini_with_tools("q"))
    assert any(json.loads(f).get("token") == "Direct answer" for f in frames)


async def test_stream_gemini_calls_tool_then_answers(mocker):
    mocker.patch.object(chatbot_module, "settings", MagicMock(google_api_key="key"))
    mocker.patch("google.genai.Client")
    mocker.patch("google.genai.types", MagicMock())
    mocker.patch("app.services.chatbot._get_gemini_tools", return_value=[])
    mocker.patch("app.services.chatbot._run_tool", new=AsyncMock(return_value='{"data": 1}'))
    resp_with_tools = _make_gemini_resp_with_tools("get_all_districts_summary", {})
    resp_no_tools = _make_gemini_resp_no_tools("5 districts total")
    mocker.patch("app.services.chatbot._gemini_generate",
                 new=AsyncMock(side_effect=[resp_with_tools, resp_no_tools]))
    frames = await _collect(_stream_gemini_with_tools("district summary"))
    tool_frames = [f for f in frames if "tool" in json.loads(f)]
    assert tool_frames[0] == json.dumps({"tool": "get_all_districts_summary"})
    assert any("5 districts total" in json.loads(f).get("token", "") for f in frames)


# ── _call_gemini_with_tools ───────────────────────────────────────────────────

async def test_call_gemini_raises_when_no_key(mocker):
    mocker.patch.object(chatbot_module, "settings", MagicMock(google_api_key=""))
    with pytest.raises(RuntimeError, match="unavailable"):
        await _call_gemini_with_tools("q")


async def test_call_gemini_with_tools_direct_answer(mocker):
    mocker.patch.object(chatbot_module, "settings", MagicMock(google_api_key="key"))
    mocker.patch("google.genai.Client")
    mocker.patch("google.genai.types", MagicMock())
    mocker.patch("app.services.chatbot._get_gemini_tools", return_value=[])
    mock_resp = _make_gemini_resp_no_tools("42 drivers")
    mocker.patch("app.services.chatbot._gemini_generate", new=AsyncMock(return_value=mock_resp))
    answer, tools = await _call_gemini_with_tools("how many drivers?")
    assert answer == "42 drivers"
    assert tools == []


async def test_call_gemini_with_tools_records_tool_names(mocker):
    mocker.patch.object(chatbot_module, "settings", MagicMock(google_api_key="key"))
    mocker.patch("google.genai.Client")
    mocker.patch("google.genai.types", MagicMock())
    mocker.patch("app.services.chatbot._get_gemini_tools", return_value=[])
    mocker.patch("app.services.chatbot._run_tool", new=AsyncMock(return_value='{"data": 1}'))
    resp1 = _make_gemini_resp_with_tools("query_postgres", {"sql": "SELECT 1"})
    resp2 = _make_gemini_resp_no_tools("answer")
    mocker.patch("app.services.chatbot._gemini_generate",
                 new=AsyncMock(side_effect=[resp1, resp2]))
    answer, tools = await _call_gemini_with_tools("q")
    assert "query_postgres" in tools
    assert answer == "answer"


# ── call_llm_with_tools (facade) ──────────────────────────────────────────────

async def test_call_llm_with_tools_returns_gemini_answer(mocker):
    mocker.patch("app.services.chatbot._call_gemini_with_tools",
                 new=AsyncMock(return_value=("gemini answer", ["tool1"])))
    answer, tools = await call_llm_with_tools("q")
    assert answer == "gemini answer"
    assert tools == ["tool1"]


async def test_call_llm_with_tools_falls_back_to_groq_on_empty(mocker):
    mocker.patch("app.services.chatbot._call_gemini_with_tools",
                 new=AsyncMock(return_value=("", [])))
    mocker.patch("app.services.chatbot._call_groq_with_tools",
                 new=AsyncMock(return_value=("groq answer", [])))
    answer, tools = await call_llm_with_tools("q")
    assert answer == "groq answer"


async def test_call_llm_with_tools_returns_error_when_both_fail(mocker):
    mocker.patch("app.services.chatbot._call_gemini_with_tools",
                 new=AsyncMock(side_effect=Exception("gemini fail")))
    mocker.patch("app.services.chatbot._call_groq_with_tools",
                 new=AsyncMock(side_effect=Exception("groq fail")))
    answer, tools = await call_llm_with_tools("q")
    assert "unavailable" in answer.lower()


# ── stream_with_tools (facade) ────────────────────────────────────────────────

async def test_stream_with_tools_gemini_success(mocker):
    async def fake_gemini(prompt):
        yield json.dumps({"token": "gemini answer"})

    mocker.patch("app.services.chatbot._stream_gemini_with_tools", side_effect=fake_gemini)
    frames = await _collect(stream_with_tools("q"))
    assert any(json.loads(f).get("token") == "gemini answer" for f in frames)


async def test_stream_with_tools_falls_back_to_groq_on_gemini_failure(mocker):
    async def fail_gemini(prompt):
        raise Exception("gemini fail")
        yield  # make it a generator

    async def fake_groq(prompt):
        yield json.dumps({"token": "groq fallback"})

    mocker.patch("app.services.chatbot._stream_gemini_with_tools", side_effect=fail_gemini)
    mocker.patch("app.services.chatbot._stream_groq_with_tools", side_effect=fake_groq)
    frames = await _collect(stream_with_tools("q"))
    assert any(json.loads(f).get("token") == "groq fallback" for f in frames)


async def test_stream_with_tools_returns_error_when_both_fail(mocker):
    async def fail(prompt):
        raise Exception("fail")
        yield

    mocker.patch("app.services.chatbot._stream_gemini_with_tools", side_effect=fail)
    mocker.patch("app.services.chatbot._stream_groq_with_tools", side_effect=fail)
    frames = await _collect(stream_with_tools("q"))
    assert any("unavailable" in json.loads(f).get("token", "") for f in frames)


# ── compress_context — additional branches ────────────────────────────────────

def test_compress_context_small_list_not_truncated():
    """Line 63: list ≤ max_items branch — items preserved, but outer string still compressed."""
    ctx = {"drivers": [1, 2, 3], "report": "x" * 600}
    result = compress_context(ctx, char_budget=100)
    data = json.loads(result)
    assert "drivers" in data
    assert isinstance(data["drivers"], list)


def test_compress_context_hard_truncate_when_all_passes_fail():
    """Line 77: even the tightest compression pass exceeds budget → hard-truncate."""
    ctx = {"a": {"b": {"c": {"d": {"e": "x" * 1000}}}}}
    result = compress_context(ctx, char_budget=5)
    assert result.endswith("…")


# ── _get_gemini_tools — cache-miss path ──────────────────────────────────────

def test_get_gemini_tools_populates_cache_on_miss(mocker):
    """Lines 195-205: when cache is None, builds the tool list from google.genai.types."""
    mocker.patch.object(chatbot_module, "_gemini_tools_cache", None)
    mocker.patch("google.genai.types", MagicMock())
    result = _get_gemini_tools()
    assert isinstance(result, list)
    assert len(result) > 0


# ── _run_tool — exception handlers ───────────────────────────────────────────

async def test_run_tool_postgres_exception_returns_error_json(mocker):
    """Lines 247-249: pool.fetch raises → JSON error response."""
    import app.db
    mock_pool = AsyncMock()
    mock_pool.fetch.side_effect = RuntimeError("connection lost")
    mocker.patch("app.db.get_pool", new=AsyncMock(return_value=mock_pool))
    result = json.loads(await _run_tool("query_postgres", {"sql": 'SELECT 1'}))
    assert "error" in result
    assert "connection lost" in result["error"]


async def test_run_tool_clickhouse_exception_returns_error_json(mocker):
    """Lines 265-267: ch_query raises → JSON error response."""
    import app.ch
    mocker.patch("app.ch.ch_query", new=AsyncMock(side_effect=RuntimeError("ch timeout")))
    result = json.loads(await _run_tool("query_clickhouse",
                                        {"sql": "SELECT count() FROM driver_positions"}))
    assert "error" in result
    assert "ch timeout" in result["error"]


# ── _recover_tool_calls — JSON decode failure ─────────────────────────────────

def test_recover_tool_calls_returns_none_on_invalid_args_json():
    """Lines 361-362: args portion is not valid JSON → returns None."""
    exc = _make_exc('<function=query_postgres{"bad:json}</function>')
    assert _recover_tool_calls(exc) is None


# ── _call_gemini — edge cases ─────────────────────────────────────────────────

async def test_call_gemini_raises_when_no_api_key(mocker):
    """Line 426: empty google_api_key → RuntimeError."""
    mocker.patch.object(chatbot_module, "settings", MagicMock(google_api_key=""))
    with pytest.raises(RuntimeError, match="Gemini fallback disabled"):
        await _call_gemini("test")


async def test_call_gemini_falls_back_to_parts_when_text_raises(mocker):
    """Lines 441-443: resp.text raises → joins candidate parts."""
    from unittest.mock import PropertyMock
    mocker.patch.object(chatbot_module, "settings", MagicMock(google_api_key="key"))
    mock_resp = MagicMock()
    type(mock_resp).text = PropertyMock(side_effect=Exception("no text"))
    part = MagicMock()
    part.text = "part answer"
    mock_resp.candidates = [MagicMock()]
    mock_resp.candidates[0].content.parts = [part]
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_resp
    mocker.patch("google.genai.Client", return_value=mock_client)
    mocker.patch("google.genai.types", MagicMock())
    result = await _call_gemini("test")
    assert result == "part answer"


# ── stream_with_tools — Gemini empty-answer fallback paths ───────────────────

async def test_stream_with_tools_handles_non_json_gemini_frame(mocker):
    """Lines 380-381: frame fails JSON parse → silently ignored, got_answer stays False."""
    async def gemini_non_json(prompt):
        yield "not valid json"

    async def groq_answer(prompt):
        yield json.dumps({"token": "groq stepped in"})

    mocker.patch("app.services.chatbot._stream_gemini_with_tools", side_effect=gemini_non_json)
    mocker.patch("app.services.chatbot._stream_groq_with_tools", side_effect=groq_answer)
    frames = await _collect(stream_with_tools("q"))
    # frames includes the raw non-JSON frame and the Groq token frame
    assert json.dumps({"token": "groq stepped in"}) in frames


async def test_stream_with_tools_falls_back_when_gemini_yields_no_token(mocker):
    """Line 384: Gemini completes with only tool frames → falls back to Groq."""
    async def gemini_tool_only(prompt):
        yield json.dumps({"tool": "get_all_districts_summary"})

    async def groq_answer(prompt):
        yield json.dumps({"token": "groq answer"})

    mocker.patch("app.services.chatbot._stream_gemini_with_tools", side_effect=gemini_tool_only)
    mocker.patch("app.services.chatbot._stream_groq_with_tools", side_effect=groq_answer)
    frames = await _collect(stream_with_tools("q"))
    assert any("groq answer" in json.loads(f).get("token", "") for f in frames)


# ── _stream_groq_with_tools — exception recovery path ────────────────────────

async def test_stream_groq_recovers_tool_calls_from_malformed_exception(mocker):
    """Lines 456-471: Groq raises with recoverable failed_generation → tool yielded, continues."""
    exc = Exception("tool_use_failed")
    exc.body = {"error": {"failed_generation": "<function=get_all_districts_summary/>"}}

    final_resp = _make_groq_resp_with_tools(None, "recovered answer")
    mocker.patch("app.services.chatbot._groq.chat.completions.create",
                 new=AsyncMock(side_effect=[exc, final_resp]))
    mocker.patch("app.services.chatbot._run_tool", new=AsyncMock(return_value='{}'))
    frames = await _collect(_stream_groq_with_tools("q"))
    assert any("tool" in json.loads(f) for f in frames)
    assert any("recovered answer" in json.loads(f).get("token", "") for f in frames)


async def test_stream_groq_survives_force_answer_exception(mocker):
    """Lines 498-500: force-final call raises → falls back to msg.content."""
    tc = _make_tool_call("get_all_districts_summary", "{}")
    resp_with_tools = _make_groq_resp_with_tools([tc], "partial content")
    mocker.patch("app.services.chatbot._groq.chat.completions.create",
                 new=AsyncMock(side_effect=[
                     resp_with_tools, resp_with_tools, resp_with_tools,
                     Exception("API error on final"),
                 ]))
    mocker.patch("app.services.chatbot._run_tool", new=AsyncMock(return_value="{}"))
    frames = await _collect(_stream_groq_with_tools("q"))
    assert any("token" in json.loads(f) for f in frames)


# ── _call_groq_with_tools — exception recovery path ─────────────────────────

async def test_call_groq_with_tools_recovers_from_malformed_exception(mocker):
    """Lines 514-529: Groq raises with recoverable failed_generation → continues."""
    exc = Exception("tool_use_failed")
    exc.body = {"error": {"failed_generation": "<function=get_all_districts_summary/>"}}

    final_resp = _make_groq_resp_with_tools(None, "recovered")
    mocker.patch("app.services.chatbot._groq.chat.completions.create",
                 new=AsyncMock(side_effect=[exc, final_resp]))
    mocker.patch("app.services.chatbot._run_tool", new=AsyncMock(return_value='{}'))
    answer, tools = await _call_groq_with_tools("q")
    assert "get_all_districts_summary" in tools
    assert answer == "recovered"


async def test_call_groq_with_tools_survives_force_answer_exception(mocker):
    """Lines 548-556: force-final call raises → returns msg.content fallback."""
    tc = _make_tool_call("get_all_districts_summary", "{}")
    resp_with_tools = _make_groq_resp_with_tools([tc], "fallback content")
    mocker.patch("app.services.chatbot._groq.chat.completions.create",
                 new=AsyncMock(side_effect=[
                     resp_with_tools, resp_with_tools, resp_with_tools,
                     Exception("API down"),
                 ]))
    mocker.patch("app.services.chatbot._run_tool", new=AsyncMock(return_value="{}"))
    answer, tools = await _call_groq_with_tools("q")
    assert isinstance(answer, str)


# ── _stream_gemini_with_tools — loop-exhausted force-final ───────────────────

async def test_stream_gemini_force_answer_after_loop_exhaustion(mocker):
    """Lines 645-651: 3 tool-call iterations → forces a final text answer."""
    mocker.patch.object(chatbot_module, "settings", MagicMock(google_api_key="key"))
    mocker.patch("google.genai.Client")
    mocker.patch("google.genai.types", MagicMock())
    mocker.patch("app.services.chatbot._get_gemini_tools", return_value=[])
    mocker.patch("app.services.chatbot._run_tool", new=AsyncMock(return_value='{"data": 1}'))
    resp_with_tools = _make_gemini_resp_with_tools("query_postgres", {"sql": "SELECT 1"})
    resp_final = _make_gemini_resp_no_tools("forced answer")
    mocker.patch("app.services.chatbot._gemini_generate",
                 new=AsyncMock(side_effect=[resp_with_tools] * 3 + [resp_final]))
    frames = await _collect(_stream_gemini_with_tools("q"))
    assert any("forced answer" in json.loads(f).get("token", "") for f in frames)


async def test_stream_gemini_force_answer_exception_yields_empty_token(mocker):
    """Lines 652-653: force-final call raises → yields empty token frame."""
    mocker.patch.object(chatbot_module, "settings", MagicMock(google_api_key="key"))
    mocker.patch("google.genai.Client")
    mocker.patch("google.genai.types", MagicMock())
    mocker.patch("app.services.chatbot._get_gemini_tools", return_value=[])
    mocker.patch("app.services.chatbot._run_tool", new=AsyncMock(return_value='{}'))
    resp_with_tools = _make_gemini_resp_with_tools("query_postgres", {"sql": "SELECT 1"})
    mocker.patch("app.services.chatbot._gemini_generate",
                 new=AsyncMock(side_effect=[resp_with_tools] * 3 + [Exception("force failed")]))
    frames = await _collect(_stream_gemini_with_tools("q"))
    assert any("token" in json.loads(f) for f in frames)


# ── _call_gemini_with_tools — loop-exhausted force-final ─────────────────────

async def test_call_gemini_with_tools_force_answer_after_loop_exhaustion(mocker):
    """Lines 696-702: 3 tool-call iterations → forces final text answer."""
    mocker.patch.object(chatbot_module, "settings", MagicMock(google_api_key="key"))
    mocker.patch("google.genai.Client")
    mocker.patch("google.genai.types", MagicMock())
    mocker.patch("app.services.chatbot._get_gemini_tools", return_value=[])
    mocker.patch("app.services.chatbot._run_tool", new=AsyncMock(return_value='{}'))
    resp_with_tools = _make_gemini_resp_with_tools("query_postgres", {"sql": "SELECT 1"})
    resp_final = _make_gemini_resp_no_tools("final answer")
    mocker.patch("app.services.chatbot._gemini_generate",
                 new=AsyncMock(side_effect=[resp_with_tools] * 3 + [resp_final]))
    answer, tools = await _call_gemini_with_tools("q")
    assert answer == "final answer"


async def test_call_gemini_with_tools_force_answer_exception_returns_empty(mocker):
    """Lines 703-704: force-final call raises → returns ("", tools)."""
    mocker.patch.object(chatbot_module, "settings", MagicMock(google_api_key="key"))
    mocker.patch("google.genai.Client")
    mocker.patch("google.genai.types", MagicMock())
    mocker.patch("app.services.chatbot._get_gemini_tools", return_value=[])
    mocker.patch("app.services.chatbot._run_tool", new=AsyncMock(return_value='{}'))
    resp_with_tools = _make_gemini_resp_with_tools("query_postgres", {"sql": "SELECT 1"})
    mocker.patch("app.services.chatbot._gemini_generate",
                 new=AsyncMock(side_effect=[resp_with_tools] * 3 + [Exception("force failed")]))
    answer, tools = await _call_gemini_with_tools("q")
    assert answer == ""
    assert isinstance(tools, list)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_groq_resp_with_tools(tool_calls, content):
    msg = MagicMock()
    msg.tool_calls = tool_calls
    msg.content = content
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


def _make_tool_call(name: str, arguments: str):
    tc = MagicMock()
    tc.id = f"call_{name}"
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _make_gemini_resp_no_tools(text: str):
    resp = MagicMock()
    resp.text = text
    resp.candidates = [MagicMock()]
    resp.candidates[0].content.parts = []
    return resp


def _make_gemini_resp_with_tools(tool_name: str, args: dict):
    fc = MagicMock()
    fc.name = tool_name
    fc.args = args
    part = MagicMock()
    part.function_call = fc
    resp = MagicMock()
    resp.candidates = [MagicMock()]
    resp.candidates[0].content.parts = [part]
    resp.candidates[0].content.role = "model"
    return resp
