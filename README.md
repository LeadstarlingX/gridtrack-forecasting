# GridTrack Forecasting Service

Standalone Python microservice for the GridTrack delivery-monitoring system. Handles urgency scoring, district demand forecasting, and AI chatbot queries.

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
        │  HTTP (called synchronously by .NET proxy controller)
        └─ POST /chat   →  AI chatbot answer  →  .NET returns to browser
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

- `GET  http://localhost:8000/health`
- `POST http://localhost:8000/chat`

## Testing

```bash
# Unit tests (no infrastructure needed)
pytest tests/unit/ -v

# Integration tests (requires Docker)
pytest tests/integration/ -v
```

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
