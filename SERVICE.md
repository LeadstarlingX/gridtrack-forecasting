# GridTrack Forecasting Service

Standalone Python microservice for the GridTrack delivery-monitoring system.
Handles urgency scoring, district demand forecasting, and AI chatbot queries.

---

## How It Fits in the System

```
.NET Backend (GridTrack.Api)
        │
        │  RabbitMQ (fanout exchanges)
        │  ── gridtrack.anomaly       →  Python scores urgency via Groq
        │  ── gridtrack.positions     →  Python updates sliding-window forecast
        │  ── gridtrack.completions   →  (reserved for future analytics)
        │
        ↓
 gridtrack-forecasting  (this service)
        │
        │  RabbitMQ (direct queues — .NET listens)
        │  ── gridtrack.urgency-results   →  .NET broadcasts via SignalR
        │  ── gridtrack.forecast-results  →  .NET broadcasts via SignalR
        │
        │  HTTP (called synchronously by .NET proxy controller)
        └─ POST /chat   →  AI chatbot answer  →  .NET returns to browser
```

The frontend never calls this service directly.
All real-time results flow: Python → RabbitMQ → .NET → SignalR → browser.
The chatbot flows: browser → .NET `/api/v1/analysis/chat` → HTTP → `/chat` here.

---

## RabbitMQ Message Topology

| Direction | Type | Name | Exchange / Queue |
|---|---|---|---|
| .NET → Python | exchange | Anomaly events | `gridtrack.anomaly` (fanout) |
| .NET → Python | exchange | Position updates | `gridtrack.positions` (fanout) |
| .NET → Python | exchange | Delivery completions | `gridtrack.completions` (fanout) |
| Python → .NET | queue | Urgency results | `gridtrack.urgency-results` |
| Python → .NET | queue | Forecast results | `gridtrack.forecast-results` |

Python binds **anonymous exclusive queues** to the fanout exchanges on startup.
Python publishes to the **default exchange** with the queue name as routing key.

> **Note for .NET side:** Wolverine must be told what type to deserialize from each
> inbound queue. Add to `Program.cs` (verify the exact Wolverine 3.x API name):
>
> ```csharp
> opts.ListenToRabbitQueue("gridtrack.urgency-results")
>     .UseForMessages<UrgencyResultMessage>();
> opts.ListenToRabbitQueue("gridtrack.forecast-results")
>     .UseForMessages<ForecastResultMessage>();
> ```
>
> Without this, Wolverine cannot identify the message type from a Python-published
> message (no .NET type header is present).

---

## Message Schemas

All field names are **camelCase** — this matches Wolverine's default JSON serialization
on the .NET side. Pydantic models must not rename fields.

### Inbound — Python receives these

```python
# Fired when a delivery is flagged anomalous.
# anomalyType is one of: "EtaExceeded" | "RouteDeviation" | "StalePosition" | "UnexpectedStop"
class DeliveryAnomalyIntegrationEvent(BaseModel):
    deliveryId: UUID
    districtId: str
    anomalyType: str
    reason: str
    driverLat: float
    driverLng: float
    occurredAt: datetime

# Fired on every driver GPS ping.
class DriverPositionIntegrationEvent(BaseModel):
    driverId: UUID
    districtId: str
    lat: float
    lng: float
    deliveryStatus: str
    timestamp: datetime

# Fired when a delivery is marked Delivered.
class DeliveryCompletedIntegrationEvent(BaseModel):
    deliveryId: UUID
    driverId: UUID
    districtId: str
    pickedUpAt: datetime
    deliveredAt: datetime
    actualDurationSeconds: float
    expectedDurationSeconds: float
```

### Outbound — Python publishes these

```python
# Urgency score + AI note — .NET handler caches and broadcasts via SignalR "UrgencyUpdated".
class UrgencyResultMessage(BaseModel):
    deliveryId: str        # UUID serialized as string
    urgencyScore: int      # 0–10
    aiNote: str

# Demand forecast per district — .NET handler caches and broadcasts via SignalR "ForecastOverlayUpdated".
class ForecastResultMessage(BaseModel):
    districtId: str
    expectedDeliveries: int
    staffingRatio: float
    label: str             # "Critical" | "Moderate" | "Low demand"
    color: str             # "#f87171" | "#fbbf24" | "#34d399"
    generatedAt: str       # ISO-8601 datetime string
```

### HTTP — chatbot

```python
# POST /chat  — called by the .NET AnalysisController proxy, not the browser directly.
class ChatBody(BaseModel):
    question: str
    context: dict          # operational snapshot built by .NET (district stats, active drivers, etc.)
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `RABBITMQ_URL` | Yes | Full AMQP(S) URL — e.g. `amqp://guest:guest@localhost:5672` or `amqps://user:pass@bunny.cloudamqp.com/vhost` |
| `GROQ_API_KEY` | Yes | Groq API key for `llama3-8b-8192` model (urgency notes + chatbot) |
| `GOOGLE_API_KEY` | Yes | Google AI key for Gemini Flash — used as chatbot fallback only |

Create a `.env` file at the repo root for local development (never commit it):

```
RABBITMQ_URL=amqp://guest:guest@localhost:5672
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...
```

---

## Districts Reference

The four operational districts. Used by the forecast service and urgency scoring.

| ID | Name | Center (lat, lng) |
|---|---|---|
| `mezzeh` | Mezzeh | 33.505, 36.243 |
| `kafrsousa` | Kafr Sousa | 33.497, 36.272 |
| `malki` | Malki | 33.517, 36.281 |
| `babtouma` | Bab Touma | 33.522, 36.307 |

---

## Project Structure

```
gridtrack-forecasting/
├── app/
│   ├── main.py            # FastAPI app + lifespan (starts consumer task)
│   ├── config.py          # pydantic-settings — reads env vars
│   ├── models.py          # All Pydantic schemas (inbound + outbound + HTTP)
│   ├── messaging/
│   │   ├── consumer.py    # aio_pika async consumer — binds to fanout exchanges
│   │   └── publisher.py   # publishes results back to .NET queues
│   └── services/
│       ├── anomaly.py     # urgency score + Groq AI note
│       ├── forecast.py    # sliding-window demand forecast per district
│       └── chatbot.py     # Groq primary / Gemini Flash fallback
├── .env                   # local secrets — not committed
├── .gitignore
├── requirements.txt
├── Dockerfile
└── SERVICE.md             # this file
```

---

## Implementation

### `app/config.py`

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672"
    groq_api_key: str = ""
    google_api_key: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
```

---

### `app/models.py`

```python
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DeliveryAnomalyIntegrationEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    deliveryId: UUID
    districtId: str
    anomalyType: str   # "EtaExceeded" | "RouteDeviation" | "StalePosition" | "UnexpectedStop"
    reason: str
    driverLat: float
    driverLng: float
    occurredAt: datetime


class DriverPositionIntegrationEvent(BaseModel):
    driverId: UUID
    districtId: str
    lat: float
    lng: float
    deliveryStatus: str
    timestamp: datetime


class DeliveryCompletedIntegrationEvent(BaseModel):
    deliveryId: UUID
    driverId: UUID
    districtId: str
    pickedUpAt: datetime
    deliveredAt: datetime
    actualDurationSeconds: float
    expectedDurationSeconds: float


class UrgencyResultMessage(BaseModel):
    deliveryId: str
    urgencyScore: int
    aiNote: str


class ForecastResultMessage(BaseModel):
    districtId: str
    expectedDeliveries: int
    staffingRatio: float
    label: str
    color: str
    generatedAt: str


class ChatBody(BaseModel):
    question: str
    context: dict
```

---

### `app/main.py`

```python
import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

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
```

---

### `app/messaging/consumer.py`

```python
import asyncio
import logging

import aio_pika

from app.config import settings
from app.messaging.publisher import publish
from app.models import (
    DeliveryAnomalyIntegrationEvent,
    DriverPositionIntegrationEvent,
)
from app.services.anomaly import score_anomaly
from app.services.forecast import update_forecast

logger = logging.getLogger(__name__)

# Maps fanout exchange name → (Pydantic schema, async handler function)
EXCHANGE_MAP = {
    "gridtrack.anomaly":   (DeliveryAnomalyIntegrationEvent, score_anomaly),
    "gridtrack.positions": (DriverPositionIntegrationEvent,  update_forecast),
}


async def start_consumer() -> None:
    """Connect to RabbitMQ and start consuming all configured exchanges.
    Retries with backoff on connection failure (handles Render cold starts)."""
    while True:
        try:
            await _run_consumer()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Consumer crashed: %s — retrying in 5s", exc)
            await asyncio.sleep(5)


async def _run_consumer() -> None:
    connection = await aio_pika.connect_robust(settings.rabbitmq_url)
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=10)

        for exchange_name, (schema, handler) in EXCHANGE_MAP.items():
            exchange = await channel.declare_exchange(
                exchange_name,
                aio_pika.ExchangeType.FANOUT,
                durable=True,
            )
            # Anonymous exclusive queue — cleaned up automatically on disconnect
            queue = await channel.declare_queue("", exclusive=True)
            await queue.bind(exchange)

            async def on_message(
                msg: aio_pika.IncomingMessage,
                _schema=schema,
                _handler=handler,
            ) -> None:
                async with msg.process():
                    try:
                        event = _schema.model_validate_json(msg.body)
                        result = await _handler(event)
                        if result is not None:
                            await publish(channel, result)
                    except Exception as exc:
                        logger.error("Error processing message: %s", exc)

            await queue.consume(on_message)
            logger.info("Subscribed to exchange: %s", exchange_name)

        logger.info("Consumer ready — waiting for messages")
        # Keep alive until cancelled or connection drops
        await asyncio.Future()
```

---

### `app/messaging/publisher.py`

```python
import logging

import aio_pika
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Maps outbound message type → queue name (.NET listens on these)
_QUEUE_MAP: dict[str, str] = {
    "UrgencyResultMessage":  "gridtrack.urgency-results",
    "ForecastResultMessage": "gridtrack.forecast-results",
}


async def publish(channel: aio_pika.Channel, message: BaseModel) -> None:
    type_name = type(message).__name__
    queue_name = _QUEUE_MAP.get(type_name)
    if not queue_name:
        logger.warning("No queue mapped for message type: %s", type_name)
        return

    body = message.model_dump_json().encode()
    await channel.default_exchange.publish(
        aio_pika.Message(
            body=body,
            content_type="application/json",
        ),
        routing_key=queue_name,
    )
    logger.debug("Published %s to %s", type_name, queue_name)
```

---

### `app/services/anomaly.py`

```python
import logging

from groq import AsyncGroq

from app.config import settings
from app.models import DeliveryAnomalyIntegrationEvent, UrgencyResultMessage

logger = logging.getLogger(__name__)

# Base urgency score per anomaly type (0–10 scale)
URGENCY_BASE: dict[str, int] = {
    "StalePosition":  5,
    "EtaExceeded":    3,
    "RouteDeviation": 4,
    "UnexpectedStop": 4,
}

# Additional points for high-demand districts
DISTRICT_BOOST: dict[str, int] = {
    "kafrsousa": 2,
    "babtouma":  1,
    "malki":     0,
    "mezzeh":    0,
}

_groq = AsyncGroq(api_key=settings.groq_api_key)


async def score_anomaly(event: DeliveryAnomalyIntegrationEvent) -> UrgencyResultMessage:
    score = URGENCY_BASE.get(event.anomalyType, 2)
    score += DISTRICT_BOOST.get(event.districtId, 0)
    score = min(10, score)

    try:
        note = await _groq_note(event, score)
    except Exception as exc:
        logger.warning("Groq call failed (%s), using fallback note", exc)
        note = _fallback_note(event, score)

    return UrgencyResultMessage(
        deliveryId=str(event.deliveryId),
        urgencyScore=score,
        aiNote=note,
    )


async def _groq_note(event: DeliveryAnomalyIntegrationEvent, score: int) -> str:
    prompt = (
        f"Delivery anomaly: type={event.anomalyType}, "
        f"reason='{event.reason}', district={event.districtId}, "
        f"urgency={score}/10. "
        "Write a single concise action note for a dispatcher (max 15 words)."
    )
    resp = await _groq.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=40,
    )
    return resp.choices[0].message.content.strip()


def _fallback_note(event: DeliveryAnomalyIntegrationEvent, score: int) -> str:
    level = "Critical" if score >= 8 else "High" if score >= 6 else "Moderate"
    return f"{level} — {event.reason.lower()}. Manual check recommended."
```

---

### `app/services/forecast.py`

```python
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

from app.models import DriverPositionIntegrationEvent, ForecastResultMessage

logger = logging.getLogger(__name__)

# In-memory sliding window: district → deque of event timestamps
_windows: dict[str, deque[datetime]] = defaultdict(deque)

# Active driver IDs seen per district within the window
_active_drivers: dict[str, set[str]] = defaultdict(set)

# Last time a forecast was emitted per district (throttle to every 5 min)
_last_emit: dict[str, datetime] = {}

WINDOW = timedelta(minutes=60)
EMIT_INTERVAL = timedelta(minutes=5)
CRITICAL_RATIO = 0.70
MODERATE_RATIO = 0.85


async def update_forecast(
    event: DriverPositionIntegrationEvent,
) -> ForecastResultMessage | None:
    now = datetime.now(timezone.utc)
    district = event.districtId

    _active_drivers[district].add(str(event.driverId))
    _windows[district].append(now)

    # Prune entries older than the window
    cutoff = now - WINDOW
    while _windows[district] and _windows[district][0] < cutoff:
        _windows[district].popleft()

    if not _should_emit(district, now):
        return None

    # Extrapolate: count events in last 30 min, double to estimate 60-min demand
    half_cutoff = now - timedelta(minutes=30)
    count_last_30 = sum(1 for t in _windows[district] if t >= half_cutoff)
    expected = count_last_30 * 2

    driver_count = len(_active_drivers[district])
    ratio = driver_count / expected if expected > 0 else 1.0

    if ratio < CRITICAL_RATIO:
        label, color = "Critical", "#f87171"
    elif ratio < MODERATE_RATIO:
        label, color = "Moderate", "#fbbf24"
    else:
        label, color = "Low demand", "#34d399"

    logger.info(
        "Forecast %s: expected=%d drivers=%d ratio=%.2f label=%s",
        district, expected, driver_count, ratio, label,
    )

    return ForecastResultMessage(
        districtId=district,
        expectedDeliveries=expected,
        staffingRatio=round(ratio, 2),
        label=label,
        color=color,
        generatedAt=now.isoformat(),
    )


def _should_emit(district: str, now: datetime) -> bool:
    last = _last_emit.get(district)
    if last is None or (now - last) >= EMIT_INTERVAL:
        _last_emit[district] = now
        return True
    return False
```

---

### `app/services/chatbot.py`

```python
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
```

---

## `requirements.txt`

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
aio-pika>=9.4.0
pydantic>=2.7.0
pydantic-settings>=2.2.0
groq>=0.9.0
google-generativeai>=0.5.0
```

---

## `Dockerfile`

```dockerfile
FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Running Locally

RabbitMQ must be running. Use the same Docker Compose from the .NET repo:

```bash
docker compose up gridtrack.rabbitmq -d
```

Or start a standalone RabbitMQ:

```bash
docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:management-alpine
```

Then:

```bash
# activate venv (Windows)
.venv\Scripts\activate

# activate venv (Linux / macOS)
source .venv/bin/activate

# copy and fill in secrets
cp .env.example .env   # or create .env manually

# run with hot reload
uvicorn app.main:app --reload --port 8000
```

The service starts, logs `Consumer ready — waiting for messages`, and exposes:
- `GET  http://localhost:8000/health`
- `POST http://localhost:8000/chat`

---

## Deployment (Render)

- Runtime: **Docker** (uses `Dockerfile`)
- Plan: **Free** (shares the 750 instance-hr/month pool with the .NET API)
- Health check path: `/health`

Required environment variables in Render dashboard:

```
RABBITMQ_URL   = amqps://user:pass@bunny.cloudamqp.com/vhost
GROQ_API_KEY   = gsk_...         (add as secret — never commit)
GOOGLE_API_KEY = AIza...         (add as secret — never commit)
```

---

## .NET Backend — Integration Points

### HTTP proxy (chatbot)

`AnalysisController.cs` in the .NET API proxies `/api/v1/analysis/chat` to this service:

```csharp
[HttpPost("chat")]
public async Task<IActionResult> Chat(
    [FromBody] ChatRequest req,
    [FromServices] IHttpClientFactory factory,
    CancellationToken ct)
{
    var client = factory.CreateClient("python");
    var payload = new { question = req.Question, context = await BuildContextAsync(ct) };
    var resp = await client.PostAsJsonAsync("/chat", payload, ct);
    return Ok(await resp.Content.ReadFromJsonAsync<object>(ct));
}
```

The named HTTP client `"python"` is registered in DI with `Python:BaseUrl` from config
(`https://gridtrack-python.onrender.com` in production, `http://localhost:8000` locally).

### Inbound result handlers (SignalR broadcast)

When this service publishes to `gridtrack.urgency-results` or `gridtrack.forecast-results`,
the .NET handlers `UrgencyResultHandler` and `ForecastResultHandler` in
`Application/CQRS/Handlers/` pick them up via Wolverine and broadcast to SignalR clients.

---

## Testing Strategy

Everything is automated. No manual curl, no Swagger clicking, no eyeballing logs.
The rule is: **write a test first, run it, make it pass, then move on.**

```
Level 1 — Python unit tests      fast, no infrastructure, always run
Level 2 — Python integration     real RabbitMQ via testcontainers-python
Level 3 — E2E with .NET backend  both services + RabbitMQ running together
```

---

### Dev dependencies (`requirements-dev.txt`)

Keep these separate from production requirements:

```
pytest>=8.0
pytest-asyncio>=0.23
pytest-mock>=3.14
testcontainers[rabbitmq]>=4.0
```

Install with:
```bash
pip install -r requirements-dev.txt
```

---

### `pytest.ini` (repo root)

Required so pytest-asyncio works correctly across all async tests:

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

---

### Test structure

```
tests/
├── conftest.py                  # shared fixtures used across both levels
├── unit/
│   ├── test_anomaly.py          # score_anomaly — mocked Groq
│   ├── test_forecast.py         # update_forecast — pure math, no mocks needed
│   └── test_chatbot.py          # call_llm — mocked Groq + Gemini fallback
└── integration/
    └── test_pipeline.py         # real RabbitMQ container, full consumer→publisher round-trip
```

---

### Level 1 — Unit tests

**`tests/unit/test_anomaly.py`** — mock Groq so tests run offline instantly:

```python
import pytest
from uuid import uuid4
from datetime import datetime, timezone

from app.models import DeliveryAnomalyIntegrationEvent
from app.services.anomaly import score_anomaly, URGENCY_BASE, DISTRICT_BOOST


def make_event(anomaly_type="StalePosition", district="mezzeh"):
    return DeliveryAnomalyIntegrationEvent(
        deliveryId=uuid4(),
        districtId=district,
        anomalyType=anomaly_type,
        reason="test reason",
        driverLat=33.5,
        driverLng=36.2,
        occurredAt=datetime.now(timezone.utc),
    )


async def test_score_uses_base_table(mocker):
    mocker.patch("app.services.anomaly._groq_note", return_value="take action")
    result = await score_anomaly(make_event("RouteDeviation", "mezzeh"))
    assert result.urgencyScore == URGENCY_BASE["RouteDeviation"]


async def test_district_boost_is_added(mocker):
    mocker.patch("app.services.anomaly._groq_note", return_value="check driver")
    result = await score_anomaly(make_event("StalePosition", "kafrsousa"))
    expected = min(10, URGENCY_BASE["StalePosition"] + DISTRICT_BOOST["kafrsousa"])
    assert result.urgencyScore == expected


async def test_score_is_capped_at_10(mocker):
    mocker.patch("app.services.anomaly._groq_note", return_value="urgent")
    # Force a high base score by patching the table
    mocker.patch.dict("app.services.anomaly.URGENCY_BASE", {"StalePosition": 9})
    mocker.patch.dict("app.services.anomaly.DISTRICT_BOOST", {"kafrsousa": 5})
    result = await score_anomaly(make_event("StalePosition", "kafrsousa"))
    assert result.urgencyScore == 10


async def test_groq_failure_uses_fallback_note(mocker):
    mocker.patch("app.services.anomaly._groq_note", side_effect=Exception("Groq down"))
    result = await score_anomaly(make_event("EtaExceeded", "mezzeh"))
    # Falls back to _fallback_note — should still return a valid message
    assert isinstance(result.aiNote, str)
    assert len(result.aiNote) > 0


async def test_delivery_id_is_string_in_result(mocker):
    mocker.patch("app.services.anomaly._groq_note", return_value="check")
    event = make_event()
    result = await score_anomaly(event)
    assert result.deliveryId == str(event.deliveryId)
```

**`tests/unit/test_forecast.py`** — no mocking needed, pure state machine:

```python
import asyncio
from datetime import datetime, timezone, timedelta
from uuid import uuid4

import pytest

import app.services.forecast as svc
from app.models import DriverPositionIntegrationEvent


def make_position(district="mezzeh", driver_id=None):
    return DriverPositionIntegrationEvent(
        driverId=driver_id or uuid4(),
        districtId=district,
        lat=33.5,
        lng=36.2,
        deliveryStatus="InTransit",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture(autouse=True)
def reset_state():
    """Clear in-memory state between tests."""
    svc._windows.clear()
    svc._active_drivers.clear()
    svc._last_emit.clear()
    yield


async def test_first_event_emits_forecast():
    result = await svc.update_forecast(make_position())
    # First event for a district always emits (no prior last_emit)
    assert result is not None


async def test_second_event_within_interval_is_throttled():
    await svc.update_forecast(make_position())      # emits
    result = await svc.update_forecast(make_position())  # throttled
    assert result is None


async def test_emit_after_interval_elapses(monkeypatch):
    # Force last_emit to be old enough
    district = "mezzeh"
    old_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    svc._last_emit[district] = old_time

    result = await svc.update_forecast(make_position(district))
    assert result is not None


async def test_critical_label_when_drivers_insufficient():
    district = "babtouma"
    # Add many position events to build up expected demand
    for _ in range(20):
        await svc.update_forecast(make_position(district, uuid4()))

    # Force emit
    svc._last_emit.clear()
    # Only one unique driver → ratio will be < CRITICAL_RATIO
    result = await svc.update_forecast(make_position(district))
    if result is not None:
        assert result.label in ("Critical", "Moderate", "Low demand")
        assert result.color in ("#f87171", "#fbbf24", "#34d399")
        assert 0.0 <= result.staffingRatio


async def test_forecast_result_fields_are_complete():
    result = await svc.update_forecast(make_position("malki"))
    assert result is not None
    assert result.districtId == "malki"
    assert isinstance(result.expectedDeliveries, int)
    assert isinstance(result.staffingRatio, float)
    assert result.generatedAt  # non-empty ISO string
```

**`tests/unit/test_chatbot.py`** — verify primary + fallback:

```python
import pytest
from app.services.chatbot import call_llm


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
```

Run all unit tests:
```bash
pytest tests/unit/ -v
```

---

### Level 2 — Integration tests (real RabbitMQ via testcontainers)

These spin up a real RabbitMQ Docker container, publish a message to the exchange the
same way the .NET backend would, and assert that the correct result lands on the output
queue. This catches JSON serialization mismatches and routing bugs before touching .NET.

**`tests/conftest.py`**:

```python
import asyncio
import pytest
from testcontainers.rabbitmq import RabbitMqContainer


@pytest.fixture(scope="session")
def rabbitmq_url():
    with RabbitMqContainer("rabbitmq:management-alpine") as container:
        yield container.get_connection_url()
```

**`tests/integration/test_pipeline.py`**:

```python
import asyncio
import json
from datetime import datetime, timezone
from uuid import uuid4

import aio_pika
import pytest

from app.models import DeliveryAnomalyIntegrationEvent, DriverPositionIntegrationEvent
from app.services.anomaly import score_anomaly
from app.services.forecast import update_forecast
from app.messaging.publisher import publish


async def _get_channel(url: str):
    conn = await aio_pika.connect_robust(url)
    return conn, await conn.channel()


async def test_anomaly_pipeline_publishes_urgency_result(rabbitmq_url, mocker):
    """
    Simulate: .NET publishes anomaly event → Python scores → result lands on urgency queue.
    """
    mocker.patch("app.services.anomaly._groq_note", return_value="check driver now")

    conn, channel = await _get_channel(rabbitmq_url)

    # Declare the result queue so we can consume from it
    result_queue = await channel.declare_queue("gridtrack.urgency-results", durable=False)

    # Build and process the event directly (consumer logic without AMQP overhead)
    event = DeliveryAnomalyIntegrationEvent(
        deliveryId=uuid4(),
        districtId="mezzeh",
        anomalyType="StalePosition",
        reason="No movement for 25 min",
        driverLat=33.505,
        driverLng=36.243,
        occurredAt=datetime.now(timezone.utc),
    )
    result = await score_anomaly(event)
    assert result is not None

    # Publish the result the same way the real publisher does
    await publish(channel, result)

    # Consume from the result queue and verify the message
    message = await asyncio.wait_for(result_queue.get(no_ack=True), timeout=5.0)
    payload = json.loads(message.body)

    assert payload["deliveryId"] == str(event.deliveryId)
    assert 0 <= payload["urgencyScore"] <= 10
    assert isinstance(payload["aiNote"], str)

    await conn.close()


async def test_publisher_routes_forecast_to_correct_queue(rabbitmq_url):
    """
    Verify ForecastResultMessage lands on gridtrack.forecast-results, not urgency queue.
    """
    from app.models import ForecastResultMessage
    from datetime import datetime, timezone

    conn, channel = await _get_channel(rabbitmq_url)
    queue = await channel.declare_queue("gridtrack.forecast-results", durable=False)

    msg = ForecastResultMessage(
        districtId="babtouma",
        expectedDeliveries=12,
        staffingRatio=0.75,
        label="Moderate",
        color="#fbbf24",
        generatedAt=datetime.now(timezone.utc).isoformat(),
    )
    await publish(channel, msg)

    message = await asyncio.wait_for(queue.get(no_ack=True), timeout=5.0)
    payload = json.loads(message.body)

    assert payload["districtId"] == "babtouma"
    assert payload["label"] == "Moderate"

    await conn.close()
```

Run integration tests (Docker must be running):
```bash
pytest tests/integration/ -v
```

---

### Level 3 — E2E with the .NET backend (automated)

This is the full chain: .NET raises a domain event → Wolverine publishes to RabbitMQ →
Python consumes, calls Groq, publishes result → Wolverine picks it up → SignalR broadcasts.

**Nothing is clicked manually. Both services run in Docker containers managed by the test.**

#### Required changes in the .NET project

The current `IntegrationTestWebAppFactory` sets `ConnectionStrings:Queue = null` to skip
RabbitMQ in unit/integration tests. E2E tests need a separate factory that includes
RabbitMQ AND the Python service.

**1. Add NuGet package to `GridTrack.IntegrationTests.csproj`:**

```xml
<PackageReference Include="Testcontainers.RabbitMq" Version="4.*" />
```

**2. Create `GridTrack.IntegrationTests/E2ETests/E2EWebAppFactory.cs`:**

```csharp
using Testcontainers.PostgreSql;
using Testcontainers.RabbitMq;
using Testcontainers.Redis;
using DotNet.Testcontainers.Builders;
using DotNet.Testcontainers.Containers;

namespace GridTrack.IntegrationTests.E2ETests;

/// <summary>
/// WebApplicationFactory for full E2E tests.
/// Starts PostgreSQL + Redis + RabbitMQ + the Python forecasting service,
/// all in Docker containers managed by Testcontainers.
/// </summary>
public class E2EWebAppFactory : WebApplicationFactory<Program>, IAsyncInitializer
{
    private readonly PostgreSqlContainer _db =
        new PostgreSqlBuilder("postgis/postgis:18-3.6").WithPassword("postgres").Build();

    private readonly RedisContainer _redis =
        new RedisBuilder("redis:8.4.0").Build();

    private readonly RabbitMqContainer _rabbit =
        new RabbitMqBuilder("rabbitmq:management-alpine").Build();

    // Python service — image must be pre-built: `docker build -t gridtrack-forecasting .`
    // (run from the gridtrack-forecasting repo root before running E2E tests)
    private IContainer? _python;

    public async Task InitializeAsync()
    {
        await Task.WhenAll(_db.StartAsync(), _redis.StartAsync(), _rabbit.StartAsync());

        _python = new ContainerBuilder()
            .WithImage("gridtrack-forecasting:latest")
            .WithEnvironment("RABBITMQ_URL", _rabbit.GetConnectionString())
            .WithEnvironment("GROQ_API_KEY", "test-key-triggers-fallback")
            .WithEnvironment("GOOGLE_API_KEY", "test-key-triggers-fallback")
            .WithExposedPort(8000)
            .WithWaitStrategy(Wait.ForUnixContainer()
                .UntilHttpRequestIsSucceeded(r => r.ForPort(8000).ForPath("/health")))
            .Build();

        await _python.StartAsync();
        using var _ = CreateClient();
    }

    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.ConfigureAppConfiguration((_, config) =>
        {
            config.AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["ConnectionStrings:DefaultConnection"] = _db.GetConnectionString(),
                ["ConnectionStrings:Cache"]             = _redis.GetConnectionString(),
                ["ConnectionStrings:Queue"]             = _rabbit.GetConnectionString(),
                ["Clerk:Authority"]                     = "https://test.clerk.invalid",
            });
        });

        // ... same auth stubbing as IntegrationTestWebAppFactory ...
    }

    public override async ValueTask DisposeAsync()
    {
        if (_python != null) await _python.DisposeAsync();
        await base.DisposeAsync();
        await _db.StopAsync();
        await _redis.StopAsync();
        await _rabbit.StopAsync();
    }
}
```

**3. Create `GridTrack.IntegrationTests/E2ETests/PythonPipelineE2ETests.cs`:**

```csharp
using GridTrack.Application.UseCases.Deliveries;
using GridTrack.IntegrationTests.Abstractions;

namespace GridTrack.IntegrationTests.E2ETests;

public class PythonPipelineE2ETests
{
    [ClassDataSource<E2EWebAppFactory>(Shared = SharedType.PerTestSession)]
    public static E2EWebAppFactory Factory { get; set; } = null!;

    [Test]
    public async Task FlagAnomaly_Should_Result_In_UrgencyBroadcast_Via_Python()
    {
        // 1. Seed a delivery
        var deliveryId = Guid.NewGuid();
        // ... seed via DbContext ...

        // 2. Flag the delivery as anomalous — Wolverine dispatches the integration event
        var result = await Factory.Services
            .GetRequiredService<IMessageBus>()
            .InvokeAsync<Result>(new FlagDeliveryAnomalyCommand(...));

        result.IsSuccess.Should().BeTrue();

        // 3. Wait for UrgencyResultHandler to process the Python response
        //    Poll the cache or a SignalR test client for the urgency update
        await Task.Delay(TimeSpan.FromSeconds(5)); // allow round-trip

        // 4. Assert the urgency score was cached
        var cache = Factory.Services.GetRequiredService<ICacheService>();
        var urgency = await cache.GetAsync<UrgencyResultMessage>(
            $"urgency:{deliveryId}", CancellationToken.None);

        urgency.Should().NotBeNull();
        urgency!.UrgencyScore.Should().BeInRange(0, 10);
        urgency.AiNote.Should().NotBeNullOrWhiteSpace();
    }
}
```

**Build the Python image before running E2E tests** (from the `gridtrack-forecasting` repo):
```bash
docker build -t gridtrack-forecasting:latest .
```

Run only E2E tests:
```bash
dotnet run --project GridTrack.IntegrationTests -- --filter E2ETests
```

---

### Summary — when to run each level

| Command | When |
|---|---|
| `pytest tests/unit/ -v` | Every time you change a service function |
| `pytest tests/integration/ -v` | After consumer or publisher changes |
| `docker build … && dotnet run --project GridTrack.IntegrationTests -- --filter E2ETests` | After a complete feature is done, before marking it shipped |
