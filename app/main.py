import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.messaging import consumer as _consumer
from app.messaging.consumer import start_consumer
from app.models import ChatBody
from app.services.chatbot import call_llm

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


app = FastAPI(title="GridTrack Forecasting Service", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    if not _consumer.ready.is_set():
        raise HTTPException(status_code=503, detail="Consumer not yet connected to RabbitMQ")
    return {"status": "ready"}


@app.post("/chat")
async def chat(body: ChatBody):
    prompt = (
        f"You are a delivery operations assistant in Damascus.\n"
        f"Operational context: {json.dumps(body.context)}\n"
        f"Question: {body.question}\n"
        f"Answer concisely, using numbers from the context."
    )
    answer = await call_llm(prompt)
    return {"answer": answer}
