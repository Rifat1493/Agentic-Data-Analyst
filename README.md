# Agentic-Data-Analyst

A supervisor-orchestrated, multi-agent stock-analysis system built on **LangGraph**
and served over an **async FastAPI** API. A central supervisor LLM dynamically
routes work between three worker agents:

| Agent | Job |
|-------|-----|
| `data_collector` | Downloads historical OHLCV data from Yahoo Finance. |
| `code_generator` | Writes & safely executes Plotly code to produce a chart PNG. |
| `report_generator` | Writes a Markdown financial-analysis report. |

It also ships the production concerns around that core: **Auth0 JWT auth + per-agent
RBAC**, **rate limiting**, **code guardrails**, **OpenTelemetry tracing**, short- and
long-term **memory** (Postgres-backed), **human-in-the-loop approval**, **Docker
packaging**, and **CI/CD to Docker Hub**.

---

## Quickstart

```bash
# 1. Install deps into a project venv (creates .venv, reads uv.lock)
uv sync

# 2. Configure secrets
cp .env.dev.example .env.dev   # then fill in DASHSCOPE_API_KEY etc.

# 3. Run the API (Swagger UI at http://localhost:8000/docs)
uv run uvicorn app:app --reload
```

### API endpoints

The interactive **Swagger UI** is auto-generated at **`/docs`** (ReDoc at `/redoc`,
raw schema at `/openapi.json`).

| Method & path | Purpose |
|---------------|---------|
| `GET /health` | Liveness/readiness probe. |
| `POST /analyze` | Run the workflow for a prompt. Returns artifacts, or pauses for approval. |
| `POST /analyze/{thread_id}/resume` | Resume a paused (human-in-the-loop) run. |

```bash
# Fire-and-forget analysis
curl -X POST localhost:8000/analyze \
  -H 'content-type: application/json' \
  -d '{"prompt": "Analyze AAPL over the last month and write a short report."}'
```

---

## Human-in-the-loop (HITL)

Writing a report is the one externally visible action, so it is gated behind an
optional human approval step. Start a run with `"require_approval": true`:

```bash
# 1. Start — pauses before the report is written
curl -X POST localhost:8000/analyze \
  -d '{"prompt": "Report on AAPL.", "require_approval": true}'
# -> {"status": "pending_approval", "thread_id": "abc...", "interrupt": {...}}

# 2. Approve (or reject) — resumes the graph from where it paused
curl -X POST localhost:8000/analyze/abc.../resume \
  -d '{"approved": true}'
```

Under the hood the `report_generator` node calls LangGraph's `interrupt()`, which
suspends the graph (the **checkpointer** persists the exact state by `thread_id`).
The reviewer's verdict is fed back via `Command(resume=...)`, and the run continues.
See [analyst_workflow.py](src/workflows/analyst_workflow.py).

---

## Why async programming is necessary here

**The work is I/O-bound, not CPU-bound.** A single `/analyze` request spends almost
all of its wall-clock time *waiting on the network*:

- multiple **LLM calls** to DashScope (the supervisor + each worker agent),
- **market-data downloads** from Yahoo Finance,
- **Postgres / Redis** round-trips for memory and caching.

The CPU sits idle during every one of those waits.

**Synchronous (`python app.py` style, one thread blocking per request).**
With blocking calls, a worker thread issues an LLM request and then *sleeps* until
the bytes come back — doing nothing useful for hundreds of milliseconds to several
seconds. To serve N concurrent users you need N threads, each mostly parked. That
caps throughput, wastes memory on stacks, and adds context-switch overhead. One
slow LLM call stalls a whole thread.

**Asynchronous (this project).** With `async def` handlers and `await`, when one
request hits a network wait the event loop *parks that coroutine and runs another
request's ready work on the same thread*. A single process can therefore keep
dozens of in-flight analyses progressing concurrently, because at any instant most
of them are just waiting on I/O. Concretely:

- FastAPI + uvicorn are **ASGI** (async) — handlers are `async def` and the server
  drives them on an event loop.
- LangGraph/LangChain expose a **native async path**, so we drive the whole
  pipeline with `await workflow.ainvoke(...)` instead of the blocking `.invoke(...)`.
- Sync leaf calls (e.g. yfinance) are run in a threadpool by the framework so they
  don't block the loop.

So async isn't decoration — for a service whose latency is dominated by external
I/O, it's what lets one process handle real concurrency instead of one-request-at-
a-time. (Note: async helps with **I/O concurrency**, not CPU-bound parallelism;
heavy CPU work would still need processes/threads.)

> Aside — `uv run app.py` vs `python app.py`: `uv run` syncs the project's
> environment (creates the venv, installs the locked deps, picks the right Python)
> *before* executing, so the script always runs against a reproducible environment.
> `python app.py` just runs against whatever interpreter/venv happens to be active.

---

## Testing & coverage

Offline unit tests (no LLM/network) run by default; the agent/workflow suites that
need a live `DASHSCOPE_API_KEY` are excluded via `pyproject.toml`.

```bash
uv run pytest                      # offline suite
uv run pytest --cov --cov-report=term-missing   # with coverage
```

Coverage is configured in `pyproject.toml` (`[tool.coverage.*]`, measuring `src`
and `app`).

---

## Docker

A multi-stage, uv-based image runs the API as a non-root user with a `/health`
healthcheck.

```bash
docker build -t jamiur1493/agentic-data-analyst .
docker run --rm -p 8000:8000 --env-file .env.dev jamiur1493/agentic-data-analyst
# Swagger UI -> http://localhost:8000/docs
```

---

## CI/CD

[.github/workflows/ci.yml](.github/workflows/ci.yml) runs on every push/PR:

1. **test** — `uv sync` + `uv run pytest --cov`.
2. **docker** — on pushes to `main`, builds and pushes the image to Docker Hub at
   **`jamiur1493/agentic-data-analyst`** (`latest` + a short-SHA tag).

Required repository secrets: `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`
(a Docker Hub access token).

---

## Roadmap / notes

Planned next steps (not yet implemented):

- reduce latency (semantic cache tuning, parallel tool calls)
- MCP — how to manage tool servers
- microservice split
- Bedrock / Foundry model backends
- .\.venv\Scripts\Activate.ps1
- 10.0.0.156

