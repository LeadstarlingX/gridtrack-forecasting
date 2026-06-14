# GridTrack Forecasting Service

Standalone Python microservice for the GridTrack delivery-monitoring system. Handles urgency scoring, district demand forecasting, AI dispatch recommendations, and AI chatbot queries.

## Architecture

```
.NET Backend (GridTrack.Api)
        │
        │  RabbitMQ (fanout exchanges)
        │  ── gridtrack.anomaly       →  Python scores urgency via Groq
        │  ── gridtrack.positions     →  Python updates sliding-window forecast
        │
        ↓
 gridtrack-forecasting  (this service)
        │
        │  RabbitMQ (direct queues — .NET listens)
        │  ── gridtrack.urgency-results   →  .NET broadcasts via SignalR
        │  ── gridtrack.forecast-results  →  .NET broadcasts via SignalR
        │
        │  HTTP (called synchronously by .NET proxy)
        ├─ POST /chat      →  AI chatbot answer  →  .NET returns to browser
        └─ POST /recommend →  structured dispatch recommendation  →  .NET delivery drawer
```

## Tech Stack

- **Runtime**: Python 3.12
- **Framework**: FastAPI + Uvicorn
- **Messaging**: aio-pika (RabbitMQ)
- **AI**: Groq (`llama3-8b-8192`) with Google Gemini Flash fallback
- **Validation**: Pydantic v2

## Prerequisites

- Python 3.12+
- Docker (for RabbitMQ and integration tests)

## Local Development

### 1. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements-dev.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Fill in your secrets
```

`.env` file format:

```
RABBITMQ_URL=amqp://guest:guest@localhost:5672
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...
```

### 4. Start RabbitMQ

```bash
docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:management-alpine
```

### 5. Run the service

```bash
uvicorn app.main:app --reload --port 8000
```

The service starts and logs `Consumer ready — waiting for messages`. Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness probe → `{"status":"ok"}` |
| `GET`  | `/ready` | Readiness probe — 503 until RabbitMQ consumer is connected |
| `POST` | `/chat` | AI chatbot: `{question, context}` → `{answer}` |
| `POST` | `/recommend` | Dispatch recommendation: delivery context + top-3 driver candidates → `{recommended_action, candidate_rank, reason, urgency_score}` |

## Testing

```bash
# Unit tests with coverage (no infrastructure needed)
pytest tests/unit/

# Integration tests (requires Docker for RabbitMQ)
pytest tests/integration/ --no-cov -v
```

Coverage runs automatically on every `pytest tests/unit/` invocation (configured in `pytest.ini`). To generate an HTML report:

```bash
pytest tests/unit/ --cov-report=html
```

## Coverage

Unit-test coverage as of 2026-06-14 (`pytest tests/unit/`):

| Module | Stmts | Miss | Cover | Notes |
|--------|------:|-----:|------:|-------|
| `app/__init__.py` | 0 | 0 | **100%** | |
| `app/config.py` | 7 | 0 | **100%** | |
| `app/main.py` | 41 | 10 | 76% | `lifespan` context manager — tested at integration level |
| `app/messaging/consumer.py` | 45 | 32 | 29% | aio-pika event loop — covered by integration tests |
| `app/messaging/publisher.py` | 19 | 12 | 37% | aio-pika publish path — covered by integration tests |
| `app/models.py` | 58 | 0 | **100%** | |
| `app/services/anomaly.py` | 25 | 3 | 88% | `_groq_note` live API call |
| `app/services/chatbot.py` | 20 | 7 | 65% | `_call_groq` / `_call_gemini` live API calls |
| `app/services/forecast.py` | 40 | 3 | 92% | publish path requires live RabbitMQ |
| `app/services/recommendation.py` | 54 | 6 | 89% | exception handler paths |
| **TOTAL** | **309** | **73** | **76%** | messaging infrastructure excluded from unit scope |

The messaging layer (consumer + publisher) reaches ~90%+ coverage when integration tests run against a real RabbitMQ container. The remaining gaps in `chatbot.py` and `anomaly.py` are the live Groq/Gemini API call paths that are intentionally not called in unit tests.

## Deployment (Render)

- Runtime: **Docker** (uses `Dockerfile`)
- Health check path: `/health`

Required environment variables in Render dashboard:

```
RABBITMQ_URL   = amqps://user:pass@bunny.cloudamqp.com/vhost
GROQ_API_KEY   = gsk_...
GOOGLE_API_KEY = AIza...
```

Build Docker image locally:

```bash
docker build -t gridtrack-forecasting:latest .
docker run -p 8000:8000 \
  -e RABBITMQ_URL=amqp://guest:guest@host.docker.internal:5672 \
  -e GROQ_API_KEY=gsk_... \
  -e GOOGLE_API_KEY=AIza... \
  gridtrack-forecasting:latest
```

## Districts

| ID | Name | Center |
|---|---|---|
| `mezzeh` | Mezzeh | 33.505, 36.243 |
| `kafrsousa` | Kafr Sousa | 33.497, 36.272 |
| `malki` | Malki | 33.517, 36.281 |
| `babtouma` | Bab Touma | 33.522, 36.307 |

## License

MIT — see [LICENSE.md](LICENSE.md)
