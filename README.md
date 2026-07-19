In response to trends in the gaming industry, as of 1st of September 2026, GridTrack will cease 
production of the CD pipeline into docker containers and shift to production of floppy disks.

Developers can still order a pigeon carrier to receive the latest updates within a stainless-steel
container right to their doorstep.


# gridtrack-forecasting

Python AI/ML pipeline for the GridTrack AI Agent-integrable delivery-monitoring system. Consumes GPS telemetry and
anomaly events from the .NET backend over RabbitMQ; produces urgency scores, demand forecasts,
surge alerts, and incident clusters — fed back to the .NET backend for live SignalR broadcast.
Also serves a streaming AI chatbot, dispatch recommendations, a staffing forecast, audio
transcription, and an MCP server for external agent access.

## Related repositories

- [GridTrack](https://github.com/LeadstarlingX/GridTrack) — .NET 9 dispatch API (this service's upstream and RabbitMQ host).
- **gridtrack-forecasting** (this repo) — Python AI/ML pipeline.
- [GridTrack.Web](https://github.com/LeadstarlingX/GridTrack.Web) — React real-time operator dashboard (SignalR live map).

## What it does

- **Anomaly urgency scoring** — RabbitMQ consumer on `gridtrack.anomaly` scores each flagged delivery 1–10 via Groq, then publishes the result to `gridtrack.urgency-results` for .NET to broadcast via SignalR.
- **District demand forecasting** — RabbitMQ consumer on `gridtrack.positions` maintains in-memory sliding windows of GPS pings per district; publishes predicted demand to `gridtrack.forecast-results` each flush cycle.
- **Demand surge detection** — rolling z-score on per-district ping counts; surges above threshold are published via RabbitMQ and broadcast live to the dashboard.
- **Incident clustering** — groups co-located anomalies into incidents and pushes cluster summaries live.
- **AI dispatch recommendations** — `POST /recommend` returns a structured ranking of candidate drivers with a recommended action, urgency score, and plain-English reason (Gemini primary, Groq fallback).
- **Staffing forecast** — `POST /staffing` returns per-district recommended driver counts based on forecasted demand.
- **Analytics chatbot** — `POST /chat/stream` (SSE) and `POST /chat` drive the operator chatbot; Gemini 2.0 Flash primary with Groq fallback; tool-calling loop queries live PostgreSQL and ClickHouse to answer natural-language questions about fleet state.
- **PDF report** — `POST /chat/report` generates a 1-page operations PDF from a conversation history.
- **Audio transcription** — `POST /transcribe` accepts any audio format and returns a text transcript via Groq Whisper.
- **MCP server** — `/mcp/sse` (SSE) + `/mcp/messages` (JSON-RPC) expose 7 read-only tools for external AI agents: `get_active_drivers`, `get_anomalies`, `get_deliveries_summary`, `get_district_status`, `get_stalled_drivers`, `get_activity_trend`, `get_peak_hours`. Bearer-token auth.

## Architecture & development methodology

```
.NET Backend (GridTrack.Api)
        │
        │  RabbitMQ fanout exchanges (inbound)
        │  ── gridtrack.anomaly     →  urgency scoring  →  publishes gridtrack.urgency-results
        │  ── gridtrack.positions   →  sliding-window forecast  →  publishes gridtrack.forecast-results
        │                                                        →  surge / incident detection
        ↓
 gridtrack-forecasting  (this service)
        │
        │  HTTP — called synchronously by the .NET proxy
        ├─ POST /recommend            →  dispatch recommendation
        ├─ POST /staffing             →  per-district staffing advice
        ├─ POST /chat                 →  chatbot (non-streaming)
        ├─ POST /chat/stream (SSE)    →  streaming chatbot with tool events
        ├─ POST /chat/report          →  PDF operations report
        ├─ POST /transcribe           →  Whisper audio transcription
        └─ GET  /mcp/sse              →  MCP SSE transport for external agents
           POST /mcp/messages         →  MCP JSON-RPC tool calls
```

Key design points:

- **In-memory sliding windows** — per-district GPS ping counts are accumulated in RAM for sub-millisecond read latency; ClickHouse is used for historical trend queries only.
- **Gemini primary / Groq fallback** — chatbot and recommendation calls try Gemini 2.0 Flash first (1500 RPD free tier); 429s with a short `retryDelay` are absorbed with a wait; longer waits fall through to Groq. Both providers failing yields a graceful error message.
- **Tool-call discipline** — the chatbot LLM loop is hard-capped at 3 tool-call rounds; if the model has not produced a text answer by then, a force-answer turn is injected before falling back. Prevents free-tier RPM exhaustion on a single question.
- **MCP SSE transport** — FastMCP `sse_app()` mounted at `/mcp`; sessions established over `GET /mcp/sse`, JSON-RPC messages sent via `POST /mcp/messages/?session_id=...`.

## Running locally

**Full stack (recommended) — from the [GridTrack](https://github.com/LeadstarlingX/GridTrack) repo:**
```bash
docker compose up -d    # starts all services including this one on :8000
```
Secrets (`GROQ_API_KEY`, `GOOGLE_API_KEY`, `MCP_API_KEY`) come from `.env` at the GridTrack repo root.

**Standalone:**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements-dev.txt
cp .env.example .env    # fill in RABBITMQ_URL, GROQ_API_KEY, GOOGLE_API_KEY

# Infrastructure only (from the GridTrack repo):
docker compose up -d gridtrack.rabbitmq gridtrack.db gridtrack.clickhouse

uvicorn app.main:app --reload --port 8000
```

The service logs `Consumer ready — waiting for messages` once the RabbitMQ consumer is connected.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness probe → `{"status":"ok"}` |
| `GET`  | `/ready` | Readiness — 503 until RabbitMQ consumer connected |
| `POST` | `/chat` | Chatbot (non-streaming): `{question, context}` → `{answer, tools_used}` |
| `POST` | `/chat/stream` | Chatbot (SSE): yields `{"tool":"name"}` frames then `{"token":"..."}` |
| `POST` | `/chat/report` | PDF report generated from a conversation history |
| `POST` | `/transcribe` | Audio → text via Groq Whisper |
| `POST` | `/recommend` | Dispatch recommendation for a delivery and candidate drivers |
| `POST` | `/staffing` | Per-district staffing forecast |
| `GET`  | `/mcp/sse` | MCP SSE connection (Bearer auth required) |
| `POST` | `/mcp/messages` | MCP JSON-RPC tool calls (Bearer auth required) |

## MCP server — AI agent integration

gridtrack-forecasting exposes a **Model Context Protocol (MCP) server** so that external AI agents
(Claude Code, Cursor, custom LLM agents, etc.) can query live fleet data as native tools — no
REST wrappers needed.

### Transport

| Endpoint | Purpose |
|----------|---------|
| `GET  http://localhost:8000/mcp/sse` | Open an SSE session — use this as the MCP URL |
| `POST http://localhost:8000/mcp/messages/?session_id=...` | Send JSON-RPC tool calls |

Auth: `Authorization: Bearer <MCP_API_KEY>` on every request.  
`MCP_API_KEY` is defined in `.env` at the GridTrack repo root and injected into the container via `compose.yaml`.

### Available tools

| Tool | Parameters | What it returns |
|------|-----------|-----------------|
| `get_active_drivers` | `district_id?` | Active drivers (id, name, district, lat/lng) |
| `get_anomalies` | `district_id?`, `hours=1` | Recent anomalies with type, reason, driver |
| `get_deliveries_summary` | — | Fleet-wide counts by status |
| `get_district_status` | `district_id` | Per-district: active drivers, pings, deliveries |
| `get_stalled_drivers` | `minutes=15` | Drivers with no GPS ping in the last N minutes |
| `get_activity_trend` | `district_id`, `hours=24` | Hourly delivery counts for trend charts |
| `get_peak_hours` | `district_id`, `days=7` | Busiest hours over the last N days |

### Connecting from Claude Code

Add to `.claude/settings.json` in your project root:

```json
{
  "mcpServers": {
    "gridtrack": {
      "type": "sse",
      "url": "http://localhost:8000/mcp",
      "headers": {
        "Authorization": "Bearer gridtrack-mcp-2026"
      }
    }
  }
}
```

Then set `MCP_API_KEY` in your shell and restart Claude Code:

```powershell
# Windows PowerShell
$env:MCP_API_KEY = "gridtrack-mcp-2026"
claude
```

```bash
# macOS / Linux
export MCP_API_KEY="gridtrack-mcp-2026"
claude
```

Claude will list the 7 tools above and can call them to read live data directly.

### Connecting from any MCP-compatible agent

Any client that supports the MCP SSE transport works. Example with the MCP Python SDK:

```python
from mcp import ClientSession
from mcp.client.sse import sse_client

async with sse_client(
    "http://localhost:8000/mcp",
    headers={"Authorization": "Bearer gridtrack-mcp-2026"},
) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("get_active_drivers", {})
        print(result.content)
```

## Testing

```bash
# Unit tests with coverage (no infrastructure needed)
pytest tests/unit/ --cov=app --cov-report=term-missing

# Integration tests (requires RabbitMQ via Docker)
pytest tests/integration/ --no-cov -v

# HTML coverage report
pytest tests/unit/ --cov=app --cov-report=html
```

## Coverage

<!-- COVERAGE_START -->
*Auto-updated by CI on every push.*
<!-- COVERAGE_END -->

## License

MIT — see [LICENSE.md](LICENSE.md)
