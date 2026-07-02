import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from app.messaging import consumer as _consumer
from app.messaging.consumer import start_consumer
from app.models import ChatBody, RecommendationRequest, RecommendationResponse, ReportRequest, StaffingRequest, StaffingResponse
from app.services.chatbot import call_llm, call_llm_with_tools, compress_context, stream_with_tools

from app.services.recommendation import get_recommendation
from app.services.staffing import get_staffing

logger = logging.getLogger(__name__)
_consumer_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _consumer_task
    logger.info("Starting RabbitMQ consumer...")
    _consumer_task = asyncio.create_task(start_consumer())
    yield
    logger.info("Shutting down consumer...")
    if _consumer_task:
        _consumer_task.cancel()
        try:
            await _consumer_task
        except asyncio.CancelledError:
            pass
    from app.db import close_pool
    await close_pool()


app = FastAPI(title="GridTrack Forecasting Service", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    if not _consumer.ready.is_set():
        raise HTTPException(status_code=503, detail="Consumer not yet connected to RabbitMQ")
    return {"status": "ready"}


# ── AI recommendations ───────────────────────────────────────────────────────

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(body: RecommendationRequest) -> RecommendationResponse:
    return await get_recommendation(body)


# ── Analytics chatbot ────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(body: ChatBody):
    """Non-streaming chat. Uses tool-calling so Groq can query live district state."""
    prompt = (
        f"You are a delivery operations assistant in Damascus.\n"
        f"You have access to tools that can fetch real-time district data.\n"
        f"Operational context (analytics snapshot): {compress_context(body.context)}\n"
        f"Question: {body.question}\n"
        f"Answer concisely, using numbers. Prefer tool calls for live data."
    )
    answer, tools_used = await call_llm_with_tools(prompt)
    return {"answer": answer, "tools_used": tools_used}


@app.get("/chat/stream")
async def chat_stream(question: str, context: str = "{}"):
    """Streaming SSE chat endpoint (GET, query-param form — kept for backward compat)."""
    try:
        ctx = json.loads(context)
    except Exception:
        ctx = {}
    return _stream_response(question, ctx)


@app.post("/chat/stream")
async def chat_stream_post(body: ChatBody):
    """Streaming SSE chat endpoint (POST body — avoids URL-length limits for large CSV)."""
    return _stream_response(body.question, body.context)


def _stream_response(question: str, ctx: dict):
    prompt = (
        f"You are a delivery operations assistant in Damascus.\n"
        f"You have access to tools that can fetch real-time district data.\n"
        f"Operational context: {compress_context(ctx)}\n"
        f"Question: {question}\n"
        f"Answer concisely, using numbers. Prefer tool calls for live data."
    )

    async def event_generator():
        try:
            async for frame in stream_with_tools(prompt):
                yield f"data: {frame}\n\n"
        except Exception as exc:
            logger.warning("Streaming error: %s", exc)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── PDF report ───────────────────────────────────────────────────────────────

@app.post("/chat/report")
async def chat_report(body: ReportRequest):
    """Generate a 1-page PDF operations report from the conversation (3 LLM calls)."""
    from app.services.report import generate_report
    from fastapi.responses import Response

    pdf_bytes = await generate_report(body.messages, body.context)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="gridtrack-report.pdf"'},
    )


# ── Staffing assistant ───────────────────────────────────────────────────────

@app.post("/staffing", response_model=StaffingResponse)
async def staffing(body: StaffingRequest) -> StaffingResponse:
    return await get_staffing(body)


# ── Audio transcription ──────────────────────────────────────────────────────

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribe audio via Groq Whisper. Accepts any audio format Whisper supports."""
    from groq import AsyncGroq
    from app.config import settings

    groq   = AsyncGroq(api_key=settings.groq_api_key)
    audio  = await file.read()
    filename = file.filename or "audio.webm"

    try:
        resp = await groq.audio.transcriptions.create(
            file=(filename, audio, file.content_type or "audio/webm"),
            model="whisper-large-v3",
        )
        return {"text": resp.text}
    except Exception as exc:
        logger.warning("Whisper transcription failed: %s", exc)
        raise HTTPException(status_code=503, detail="Transcription service unavailable")


# ── MCP server (external agent access) ──────────────────────────────────────

from app.config import settings as _settings

if _settings.mcp_api_key:
    try:
        from app.mcp_server import make_mcp_app
        app.mount("/mcp", make_mcp_app(_settings.mcp_api_key))
        logger.info("MCP server mounted at /mcp")
    except (ImportError, AttributeError) as exc:
        logger.warning("MCP server disabled (%s). Requires mcp>=1.0.0", exc)
