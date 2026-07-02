import pytest
from httpx import AsyncClient, ASGITransport


async def _fake_consumer():
    """Drop-in for start_consumer that never touches RabbitMQ."""
    import asyncio
    await asyncio.Future()


@pytest.fixture
async def client(mocker):
    """ASGI test client with the consumer mocked out."""
    mocker.patch("app.main.start_consumer", new=_fake_consumer)
    from app.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def ready_client(mocker):
    """ASGI test client with consumer ready event pre-set."""
    mocker.patch("app.main.start_consumer", new=_fake_consumer)
    from app.messaging import consumer as _consumer
    from app.main import app
    _consumer.ready.set()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    _consumer.ready.clear()


# ── /health ──────────────────────────────────────────────────────────────────

async def test_health_returns_ok(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_health_rejects_post(client):
    resp = await client.post("/health")
    assert resp.status_code == 405


# ── /chat ─────────────────────────────────────────────────────────────────────

async def test_chat_returns_llm_answer(client, mocker):
    mocker.patch("app.main.call_llm_with_tools", return_value=("5 active deliveries in mezzeh.", []))
    resp = await client.post(
        "/chat",
        json={"question": "How many deliveries?", "context": {"district": "mezzeh", "active": 5}},
    )
    assert resp.status_code == 200
    assert resp.json()["answer"] == "5 active deliveries in mezzeh."


async def test_chat_prompt_includes_question_and_context(client, mocker):
    captured: dict = {}

    async def capture_llm(prompt: str) -> tuple[str, list[str]]:
        captured["prompt"] = prompt
        return ("ok", [])

    mocker.patch("app.main.call_llm_with_tools", side_effect=capture_llm)
    await client.post(
        "/chat",
        json={"question": "which district is busiest?", "context": {"district": "babtouma"}},
    )
    assert "which district is busiest?" in captured["prompt"]
    assert "babtouma" in captured["prompt"]


async def test_chat_missing_question_returns_422(client):
    resp = await client.post("/chat", json={"context": {}})
    assert resp.status_code == 422


async def test_chat_missing_context_returns_422(client):
    resp = await client.post("/chat", json={"question": "hello"})
    assert resp.status_code == 422


async def test_chat_empty_body_returns_422(client):
    resp = await client.post("/chat", json={})
    assert resp.status_code == 422


# ── /ready ────────────────────────────────────────────────────────────────────

async def test_ready_returns_503_when_consumer_not_subscribed(client):
    """Consumer event not set → 503 with detail message."""
    from app.messaging import consumer as _consumer
    _consumer.ready.clear()
    resp = await client.get("/ready")
    assert resp.status_code == 503
    assert "not yet connected" in resp.json()["detail"]


async def test_ready_returns_200_when_consumer_subscribed(ready_client):
    """`/ready` returns 200 once the consumer has subscribed to all exchanges."""
    resp = await ready_client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


async def test_ready_rejects_post(client):
    resp = await client.post("/ready")
    assert resp.status_code == 405


# ── /recommend ────────────────────────────────────────────────────────────────

_RECOMMEND_BODY = {
    "delivery_id": "aaaaaaaa-0000-0000-0000-000000000001",
    "district_id": "mezzeh",
    "anomaly_type": "EtaExceeded",
    "anomaly_reason": "Driver stalled for 20 minutes",
    "candidates": [
        {"rank": 1, "name": "Ahmad Hassan", "distance_m": 250, "on_time_rate_pct": 0.85, "score": 0.82},
        {"rank": 2, "name": "Mohamed Ali",  "distance_m": 800, "on_time_rate_pct": 0.60, "score": 0.65},
    ],
}

_VALID_LLM_JSON = '{"recommended_action": "Reassign", "candidate_rank": 1, "reason": "Closest driver with best on-time rate.", "urgency_score": 7}'


async def test_recommend_returns_structured_response(client, mocker):
    mocker.patch("app.services.recommendation.call_llm", return_value=_VALID_LLM_JSON)
    resp = await client.post("/recommend", json=_RECOMMEND_BODY)
    assert resp.status_code == 200
    data = resp.json()
    assert data["recommended_action"] == "Reassign"
    assert data["candidate_rank"] == 1
    assert data["urgency_score"] == 7
    assert "reason" in data


async def test_recommend_accepts_llm_with_markdown_fences(client, mocker):
    mocker.patch(
        "app.services.recommendation.call_llm",
        return_value=f"```json\n{_VALID_LLM_JSON}\n```",
    )
    resp = await client.post("/recommend", json=_RECOMMEND_BODY)
    assert resp.status_code == 200
    assert resp.json()["recommended_action"] == "Reassign"


async def test_recommend_falls_back_to_monitor_on_garbage_llm_output(client, mocker):
    mocker.patch("app.services.recommendation.call_llm", return_value="I cannot decide.")
    resp = await client.post("/recommend", json=_RECOMMEND_BODY)
    assert resp.status_code == 200
    data = resp.json()
    assert data["recommended_action"] == "Monitor"
    assert data["candidate_rank"] is None


async def test_recommend_missing_delivery_id_returns_422(client):
    body = {k: v for k, v in _RECOMMEND_BODY.items() if k != "delivery_id"}
    resp = await client.post("/recommend", json=body)
    assert resp.status_code == 422


async def test_recommend_no_candidates_still_valid(client, mocker):
    mocker.patch(
        "app.services.recommendation.call_llm",
        return_value='{"recommended_action": "Monitor", "candidate_rank": null, "reason": "No drivers.", "urgency_score": 6}',
    )
    body = {**_RECOMMEND_BODY, "candidates": []}
    resp = await client.post("/recommend", json=body)
    assert resp.status_code == 200
    assert resp.json()["recommended_action"] == "Monitor"


# ── /staffing ─────────────────────────────────────────────────────────────────

_STAFFING_BODY = {
    "district": "mezzeh",
    "target_datetime": "2026-06-15T09:00:00Z",
    "day_of_week": 6,
    "target_hour": 9,
    "historical_avg_deliveries": 12.0,
    "recent_surge_detected": False,
}

_VALID_STAFFING_JSON = (
    '{"recommended_drivers": 4, "confidence": "high", '
    '"reasoning": "Historical avg of 12 suggests 4 drivers."}'
)


async def test_staffing_returns_structured_response(client, mocker):
    mocker.patch("app.services.staffing._call_groq", return_value=_VALID_STAFFING_JSON)
    resp = await client.post("/staffing", json=_STAFFING_BODY)
    assert resp.status_code == 200
    data = resp.json()
    assert data["recommended_drivers"] == 4
    assert data["confidence"] == "high"
    assert "reasoning" in data


async def test_staffing_missing_district_returns_422(client):
    body = {k: v for k, v in _STAFFING_BODY.items() if k != "district"}
    resp = await client.post("/staffing", json=body)
    assert resp.status_code == 422


async def test_staffing_missing_target_hour_returns_422(client):
    body = {k: v for k, v in _STAFFING_BODY.items() if k != "target_hour"}
    resp = await client.post("/staffing", json=body)
    assert resp.status_code == 422


async def test_staffing_groq_failure_falls_back_to_safe_default(client, mocker):
    mocker.patch("app.services.staffing._call_groq", side_effect=RuntimeError("Groq down"))
    resp = await client.post("/staffing", json=_STAFFING_BODY)
    assert resp.status_code == 200
    data = resp.json()
    # safe_default: max(1, round(12.0 / 3)) = 4
    assert data["recommended_drivers"] == 4
    assert data["confidence"] == "low"
    assert "AI unavailable" in data["reasoning"]


async def test_staffing_llm_returns_zero_drivers_clamped_to_one(client, mocker):
    mocker.patch(
        "app.services.staffing._call_groq",
        return_value='{"recommended_drivers": 0, "confidence": "medium", "reasoning": "No demand."}',
    )
    body = {**_STAFFING_BODY, "historical_avg_deliveries": 0.5}
    resp = await client.post("/staffing", json=body)
    assert resp.status_code == 200
    assert resp.json()["recommended_drivers"] >= 1


async def test_staffing_surge_flag_accepted(client, mocker):
    mocker.patch("app.services.staffing._call_groq", return_value=_VALID_STAFFING_JSON)
    resp = await client.post("/staffing", json={**_STAFFING_BODY, "recent_surge_detected": True})
    assert resp.status_code == 200
    assert resp.json()["recommended_drivers"] == 4


async def test_staffing_garbage_llm_json_returns_parse_fallback(client, mocker):
    mocker.patch("app.services.staffing._call_groq", return_value="not json at all")
    resp = await client.post("/staffing", json=_STAFFING_BODY)
    assert resp.status_code == 200
    data = resp.json()
    assert data["recommended_drivers"] >= 1
    assert data["confidence"] == "low"


async def test_staffing_markdown_fenced_json_is_accepted(client, mocker):
    mocker.patch(
        "app.services.staffing._call_groq",
        return_value=f"```json\n{_VALID_STAFFING_JSON}\n```",
    )
    resp = await client.post("/staffing", json=_STAFFING_BODY)
    assert resp.status_code == 200
    assert resp.json()["recommended_drivers"] == 4


# ── /chat/stream ──────────────────────────────────────────────────────────────


async def test_chat_stream_post_returns_sse_content_type(client, mocker):
    async def fake_stream(_prompt: str):
        yield '{"token": "hello"}'

    mocker.patch("app.main.stream_with_tools", side_effect=fake_stream)
    resp = await client.post(
        "/chat/stream",
        json={"question": "How many deliveries?", "context": {}},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


async def test_chat_stream_post_emits_token_events_and_done(client, mocker):
    async def fake_stream(_prompt: str):
        for token in ["Hello", " world"]:
            yield f'{{"token": "{token}"}}'

    mocker.patch("app.main.stream_with_tools", side_effect=fake_stream)
    resp = await client.post(
        "/chat/stream",
        json={"question": "Test", "context": {"district": "mezzeh"}},
    )
    assert resp.status_code == 200
    body = resp.text
    assert 'data: {"token": "Hello"}' in body
    assert 'data: {"token": " world"}' in body
    assert "data: [DONE]" in body


async def test_chat_stream_get_returns_sse_via_query_params(client, mocker):
    async def fake_stream(_prompt: str):
        yield '{"token": "ok"}'

    mocker.patch("app.main.stream_with_tools", side_effect=fake_stream)
    resp = await client.get("/chat/stream?question=test&context=%7B%7D")
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "data: [DONE]" in resp.text


async def test_chat_stream_done_terminator_present_after_error(client, mocker):
    async def failing_stream(_prompt: str):
        yield '{"token": "partial"}'
        raise RuntimeError("Network error mid-stream")

    mocker.patch("app.main.stream_with_tools", side_effect=failing_stream)
    resp = await client.post(
        "/chat/stream",
        json={"question": "Fail halfway", "context": {}},
    )
    assert resp.status_code == 200
    assert "data: [DONE]" in resp.text


async def test_chat_stream_missing_question_returns_422(client):
    resp = await client.post("/chat/stream", json={"context": {}})
    assert resp.status_code == 422


# ── /transcribe ───────────────────────────────────────────────────────────────


async def test_transcribe_returns_text(client, mocker):
    mock_groq = mocker.MagicMock()
    mock_groq.audio.transcriptions.create = mocker.AsyncMock(
        return_value=mocker.MagicMock(text="hello world")
    )
    mocker.patch("groq.AsyncGroq", return_value=mock_groq)

    resp = await client.post(
        "/transcribe",
        files={"file": ("audio.webm", b"fake audio bytes", "audio/webm")},
    )
    assert resp.status_code == 200
    assert resp.json()["text"] == "hello world"


async def test_transcribe_passes_filename_to_groq(client, mocker):
    mock_groq = mocker.MagicMock()
    create_mock = mocker.AsyncMock(return_value=mocker.MagicMock(text="ok"))
    mock_groq.audio.transcriptions.create = create_mock
    mocker.patch("groq.AsyncGroq", return_value=mock_groq)

    await client.post(
        "/transcribe",
        files={"file": ("my_recording.wav", b"data", "audio/wav")},
    )
    call_kwargs = create_mock.call_args.kwargs
    assert call_kwargs["file"][0] == "my_recording.wav"


async def test_transcribe_groq_failure_returns_503(client, mocker):
    mock_groq = mocker.MagicMock()
    mock_groq.audio.transcriptions.create = mocker.AsyncMock(
        side_effect=RuntimeError("Whisper unavailable")
    )
    mocker.patch("groq.AsyncGroq", return_value=mock_groq)

    resp = await client.post(
        "/transcribe",
        files={"file": ("audio.webm", b"data", "audio/webm")},
    )
    assert resp.status_code == 503


async def test_transcribe_missing_file_returns_422(client):
    resp = await client.post("/transcribe")
    assert resp.status_code == 422
