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
    mocker.patch("app.main.call_llm", return_value="5 active deliveries in mezzeh.")
    resp = await client.post(
        "/chat",
        json={"question": "How many deliveries?", "context": {"district": "mezzeh", "active": 5}},
    )
    assert resp.status_code == 200
    assert resp.json()["answer"] == "5 active deliveries in mezzeh."


async def test_chat_prompt_includes_question_and_context(client, mocker):
    captured: dict = {}

    async def capture_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "ok"

    mocker.patch("app.main.call_llm", side_effect=capture_llm)
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
