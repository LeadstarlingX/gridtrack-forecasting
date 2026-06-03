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
