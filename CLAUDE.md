# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`Ultimate Macroeconomics Dashboard` is a 10-container Docker stack: a Streamlit multi-page dashboard backed by Postgres + Qdrant, with FastAPI micro-services for the AI analyst, forecasting, clustering, on-demand data ingestion, and a sandboxed Python executor. Read `README.md` for the full description; the sections below cover only what isn't obvious from the code.

## Running the stack

There is no test suite or linter. Everything runs inside containers via Docker Compose. Every service is Python 3.12 and uses [uv](https://docs.astral.sh/uv/) for dependency management — each service has its own `pyproject.toml` + `uv.lock`, and the Dockerfile runs `uv sync --frozen` into `/opt/venv`.

```bash
# Full stack (build + run, foreground)
docker compose up --build

# Single service rebuild
docker compose build agent
docker compose up -d agent

# Logs
docker compose logs -f app
docker compose logs -f agent
```

For local iteration without rebuilding the image, work in any service directory:

```bash
cd app           # or agent, forecaster, etc.
uv sync          # creates .venv from pyproject.toml + uv.lock
uv run python -m streamlit run app.py   # or uvicorn main:app for FastAPI services
uv add <package>      # add a dependency (updates pyproject.toml + uv.lock)
uv lock --upgrade     # refresh the lockfile
```

First boot requires `_container_data/.env` (copy from `_container_data/.env.example`) and a populated LLM section in `_container_data/config.yaml`. The `db_init` container runs once to provision Postgres roles; `downloader_general` then runs once (~1–2h) to ingest World Bank + Yahoo Finance + Webz.io news. The dashboard is at `http://localhost:8501`.

If the host has no NVIDIA GPU, remove the `deploy:` block from the `forecaster` service in `docker-compose.yaml` (the `chronos` model will be skipped; `pmdarima` and `prophet` still work).

## Architecture

### Service map and ports

| Service              | Port | Role                                                                                  |
| -------------------- | ---- | ------------------------------------------------------------------------------------- |
| `db`                 | 5432 | Postgres 18 — World Bank + Yahoo Finance tabular data                                 |
| `vector_db`          | 6333 | Qdrant — news article embeddings                                                      |
| `db_init`            | —    | One-shot: creates Postgres roles (superuser + read-only `POSTGRES_LLM_*`) and schema  |
| `downloader_general` | —    | One-shot: clones `Webhose/free-news-datasets`, fetches WB + Yahoo, populates both DBs |
| `app`                | 8501 | Streamlit dashboard (entry point: `app/app.py`)                                       |
| `agent`              | 8000 | FastAPI — LangGraph multi-agent AI analyst                                            |
| `forecaster`         | 8001 | FastAPI — `pmdarima` / `prophet` / `chronos` time-series forecasting                  |
| `clustering`         | 8002 | FastAPI — KMeans / DBSCAN                                                             |
| `downloader_extra`   | 8003 | FastAPI — on-demand World Bank indicator ingestion (called by the agent)              |
| `python_sandbox`     | 8004 | FastAPI — isolated executor for LLM-generated Plotly/Polars code                      |

Inside the Compose network, services address each other by container name and the port from `config.yaml` (e.g. `http://agent:8000`, `http://forecaster:8001`). The `app` resolves these via `app/core/api_client.py`, which also honours `*_BASE_URL` env vars as overrides.

### Configuration

`_container_data/config.yaml` is the **single source of truth** for ports, hostnames, LLM/embedding settings, forecaster toggles, etc. It is bind-mounted read-only into every service.

Important: `docker-compose.yaml` duplicates the ports and bind-mount paths declared in `config.yaml`. Changing a port or path in one file requires changing it in the other. The two files are not auto-synced.

Other config files in `_container_data/`:
- `.env` — secrets (Postgres creds, Qdrant API key, `OPENAI_API_KEY`). Never commit; gitignored.
- `database_schema.yaml` — column-level documentation of Postgres tables; mounted into `agent` so the SQL worker can ground its queries.
- `_configs/world_bank_download_config.json` — list of WB indicators grouped by dashboard page. Append here to add indicators on next clean boot; or add at runtime via the AI analyst (it calls `downloader_extra`).
- `_configs/news_download_config.json` — news topics for the RAG corpus.
- `_configs/yahoo_download_config.json` — Yahoo Finance tickers.
- `themes.yaml` — colour palettes. `active:` key selects one; bundled themes are `dark`, `dark-blue`, `light-green`. Drives both the registered Plotly template (`"app"`) and Streamlit chrome. **Deploy-time only** — the runtime theme picker was removed in v0.6.
- `app/.streamlit/config.toml` — Streamlit's own theme/server config. Mirror of `themes.yaml` for the chrome side; edit `server.address = "0.0.0.0"` to expose the local dev build on the LAN.

### `app` (Streamlit)

Entry point `app/app.py` registers the Plotly template, sets up `st.session_state` (chat history, per-service health flags), declares the multi-page navigation, and shows a one-time data disclaimer dialog. Pages live under `app/pages/` and are numbered `01_…` through `16_…` for ordering; the numbers also encode the v0.3 navigation renormalisation. Shared infrastructure is in `app/core/`:

- `api_client.py` — typed wrappers around every backend HTTP endpoint (forecaster, agent SSE stream, clustering, plot interpretation, downloader_extra). Always use these wrappers rather than `requests.post` directly — they handle the base-URL resolution and request logging.
- `postgres_client.py` / `qdrant_client.py` — connection helpers with retries (hardened in v0.5).
- `plotting.py` — Plotly helpers; pages call `get_color` / `get_colorway` rather than hard-coding hex values, so palette swaps work via `themes.yaml`.
- `theming.py` — registers the `"app"` Plotly template from the active theme.
- `token_usage.py` — in-memory aggregator displayed on the Settings page; cleared on session end.
- `app_logging.py` — centralised page-render and HTTP-request logging.

### `agent` (LangGraph supervisor)

`agent/agent/graph.py` (~1k lines) is the heart of the AI analyst. The flow is:

1. **`GuardrailAgent`** — screens the latest user message; out-of-scope or abusive requests get a polite markdown refusal and skip the rest of the graph.
2. **`MacroSupervisorAgent`** — plans, picks the next worker, and decides when to finish.
3. **Workers** (one of `WORKER_NAMES` in `graph.py`):
   - `sql_agent` — 3-step World Bank exploration + Yahoo Finance lookup, grounded in `database_schema.yaml`.
   - `plotly_agent` — generates Plotly code, runs it in `python_sandbox`, returns the figure as an artifact.
   - `table_agent` — Polars transformations on prior worker output.
   - `rag_agent` — Qdrant semantic search over the news corpus.
   - `web_search` — DuckDuckGo fallback.
   - `downloader_agent` — calls `downloader_extra` to ingest WB indicators on demand.
   - `chat_agent` — conversational synthesis / final answer.

Streaming protocol is SSE on `POST /chat/stream` with `step` / `token` / `final` / `error` events; the `final` event carries the answer plus an `artifacts` dict and a `usage` block. `POST /plots/interpret` is a separate vision endpoint that reads a base64 Plotly screenshot with two modes (`no_hallucinations` strict description vs. analyst interpretation).

Per-LLM-call token accounting is attached via `UsageTracker` (`agent/agent/usage.py`) on every LangChain LLM in the graph (guardrail, supervisor, each worker, final synthesis stream). Worker output schemas live in `agent/agent/schemas.py`.

### Data ingestion

`downloader_general/src/` is split into `core/` (orchestration, schema validation in `utils/schema.py`), `extractors/` (one module per source: `world_bank`, `yahoo`, `github` for the news repo), and `utils/`. It's a one-shot job — its container exits after success. Re-running it from scratch requires removing the `_container_data/downloader_general/.download_completed` marker (gitignored) and the persistent volumes (`postgres_data`, `qdrant_data`).

For incremental WB indicator additions during a live stack, the agent's `downloader_agent` worker calls `downloader_extra` (port 8003), which writes directly into the running Postgres without touching the marker.

## Conventions worth knowing

- The codebase uses **Polars**, not Pandas. Don't introduce `pandas` in new code.
- Charts are always **Plotly** going through the `"app"` registered template; pull colours from `core/theming` helpers rather than hard-coding.
- Agent worker outputs are **Pydantic models** (`agent/agent/schemas.py`); structured-output LLM calls use `with_structured_output(...)`. Adding a worker means: schema in `schemas.py`, tool wrappers in `tools.py`, node + supervisor routing in `graph.py`, and the worker name in `WORKER_NAMES`.
- Every backend HTTP endpoint should have a typed wrapper in `app/core/api_client.py` — don't bypass it from pages.
- The agent's Postgres role is intentionally read-only (`POSTGRES_LLM_USERNAME`); don't grant it write permissions. The superuser role is only for `db_init` and operator tasks.
