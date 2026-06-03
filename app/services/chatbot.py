import logging

from groq import AsyncGroq

from app.config import settings

logger = logging.getLogger(__name__)

_groq = AsyncGroq(api_key=settings.groq_api_key)


async def call_llm(prompt: str) -> str:
    """Call Groq (primary). Falls back to Gemini Flash on any error."""
    try:
        return await _call_groq(prompt)
    except Exception as exc:
        logger.warning("Groq failed (%s), trying Gemini fallback", exc)
        return await _call_gemini(prompt)


async def _call_groq(prompt: str) -> str:
    resp = await _groq.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


async def _call_gemini(prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=settings.google_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    return resp.text
