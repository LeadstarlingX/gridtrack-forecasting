"""Shared chatbot system prompt."""

from app.services.chatbot import compress_context

_BASE = (
    "You are a delivery operations assistant in Damascus.\n"
    "You have access to tools that can fetch real-time district data and query the database.\n"
    "Answer concisely, using numbers. Prefer tool calls for live data.\n"
    "When ranking or listing results, always include the name (not just ID) and the metric value used for ranking.\n"
    "When writing SQL: SELECT only the columns needed to answer the question, never SELECT *.\n"
)


def build_prompt(question: str, ctx: dict) -> str:
    return (
        f"{_BASE}"
        f"Operational context: {compress_context(ctx)}\n"
        f"Question: {question}"
    )
