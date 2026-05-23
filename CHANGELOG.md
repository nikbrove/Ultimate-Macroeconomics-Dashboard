# Changelog

All notable changes to **Ultimate Macroeconomics Dashboard** are documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [v0.9]

AI-analyst quality and latency pass plus a critical fix to the read-only LLM Postgres role's permissions, on top of a broader maintenance pass: new `tests/` suites under every service (`agent`, `app`, `clustering`, `downloader_extra`, `downloader_general`, `forecaster`, `python_sandbox`), two new app pages (`17_token_usage`, `18_monitoring`) replacing the legacy `16_settings`, retirement of the standalone `_container_data/db_init.sh` (its job is now done by `downloader_general`'s bootstrap), and assorted refactors across every service. Same architecture, same overall UI; the multi-agent graph is shorter (one fewer LLM call per turn) and the supervisor / SQL / chat / RAG / web-search workers are now grounded in chat history, a centralised macro context block, and a deterministic worker-status channel.

The notes below detail the AI-analyst + DB-permission work — the broader file-by-file diffs across the rest of the stack are visible in `git log` for the v0.9 commit.

### Fixed
- **`downloader_general`: read-only LLM role had no table privileges.** `ensure_llm_role` only created/altered the role but never granted any `SELECT`, so the AI analyst's `sql_agent` failed with `permission denied for table databases / database_indicators / indicators / …` on every query. The bootstrap now also issues `GRANT USAGE ON SCHEMA public`, `GRANT SELECT ON ALL TABLES IN SCHEMA public`, and `ALTER DEFAULT PRIVILEGES … GRANT SELECT ON TABLES` (run as the superuser so future tables created by `downloader_general` / `downloader_extra` are readable automatically). The new test `test_ensure_llm_role_grants_select_on_existing_and_future_tables` pins the contract.
- **`downloader_general` entrypoint never re-ran the bootstrap on upgrade.** The shell entrypoint exited early on the `.download_completed` marker, so any change to `ensure_llm_role` (such as the grant fix above) would never take effect on an existing volume. The marker check now lives inside `main.py`: bootstrap runs every container start, downloads still run only once.

### Added — agent quality
- **Pass chat history to workers.** `SQLAgent`, `RAGAgent`, `WebSearchAgent`, and `ChatAgent` now receive the last 3 user/assistant turns from `AgentState.messages` in their prompts so follow-ups like "now Germany too" don't require the supervisor to re-state context. Plot / table workers don't get it — they operate on the already-fetched artifact.
- **Worked SQL examples (few-shot).** The `SQLAgent` preamble carries 3 canonical examples (single-country WDI lookup, ticker-known Yahoo fetch, multi-country WDI with `IN (...)`) so first-shot SQL quality improves on smaller models.
- **`last_worker_status` channel.** New `AgentState` field carrying a `Literal["SUCCESS","EMPTY","ERROR","NEEDS_DOWNLOAD","BLOCKED","UNKNOWN"]` tag returned by every worker. The supervisor's branching is now driven by this enum instead of regex-matching `"SQL_AGENT INDICATOR_NOT_DOWNLOADED:"` prose, and the prompt explicitly references the tag (e.g. "route to `downloader_agent` when `last_worker_status` is `NEEDS_DOWNLOAD`").
- **Centralised macro context.** The supervisor preamble now states the always-on assumptions (WDI is `db_id = 2`; `indicators.economy` holds ISO-3 codes; Yahoo is the only stock/index source; today's date) instead of expecting the planner to re-derive them.
- **Short-circuit WDI lookup in `sql_agent`.** The SQL prompt now defaults `database_id = 2` (WDI) and skips the `databases`-table lookup step unless the user explicitly names another World Bank database — drops one LLM call from most macro queries.

### Changed — agent speed
- **Dropped the FINAL_SYNTHESIS LLM call.** The supervisor already writes the final markdown answer to `isolated_worker_task` when it picks `FINISH`; `MacroAgentGraph._stream_final_synthesis` re-fed it through an extra streaming LLM call. Replaced with a chunked character-streamer (`_stream_supervisor_draft`) that emits the draft in ~24-char bursts directly — no model call, no risk of the synthesis pass altering numbers. A small leak filter (`_sanitize_draft`) strips any line containing worker names / sandbox / traceback tokens as a last-mile guard.
- **Heuristic-first guardrail.** The previous unconditional LLM screening step is gone; `GuardrailAgent` now uses three regexes (auto-allow for short greetings + in-scope keywords, auto-block for obvious red flags) and only escalates ambiguous messages to the structured-output LLM. Same safety profile, no LLM call for the typical user message.
- **Shared `httpx.AsyncClient`.** `execute_code_in_sandbox` and `download_indicator` used to spin up `async with httpx.AsyncClient()` per call (full TLS handshake every time). One pool is now built lazily in `agent.tools._get_httpx_client()` and closed via a FastAPI `shutdown` hook.
- **Cached `get_database_schema_text()`.** The YAML-to-text rendering used to run on every SQL step; it's now wrapped in `functools.lru_cache(maxsize=1)` (and invalidated by `configure_runtime` for tests).
- **Prefix-cacheable prompt layout.** Supervisor / SQL / Plotly / Polars / RAG / web-search / chat prompts have been restructured so the static role + scope + rules + (where applicable) database schema and few-shot examples form the prompt prefix, and only the per-call dynamic data lives in the trailing suffix. Providers that auto-cache identical prefixes (OpenAI, Anthropic) can now reuse them across turns.
- **Trimmed supervisor `worker_results`.** `worker_results` uses `operator.add` and grew unboundedly across turns; the supervisor prompt now keeps the last two verbatim and summarises older entries to a single line each.
- **Healthcheck noise reduced.** Compose healthchecks for the HTTP services (`agent` / `app` / `forecaster` / `clustering` / `downloader_extra` / `python_sandbox` / `vector_db`) bumped from `interval: 10s` to `30s` so `/health` polling no longer drowns real errors in `docker compose logs`. `db` stays at 5s (it gates `depends_on: condition: service_healthy` on stack startup).

### Removed
- **`MacroAgentGraph._stream_final_synthesis`** and its `FINAL_SYNTHESIS_SYSTEM_PROMPT` — replaced by direct chunked emission of the supervisor's draft.

### Operator notes
- **Existing deployments** need a one-time rebuild of `downloader_general` so the new bootstrap runs against their already-populated DB: `docker compose build downloader_general && docker compose up -d downloader_general`. The container will re-apply role + grants in seconds, find the download marker, and exit without re-running the multi-hour ingestion.
- **No `.env` / `config.yaml` changes** required.

## [v0.8]

Codebase-wide refactoring pass across all ten containers. No new user-facing features — the goals were dedup, dead-code removal, async-safety, error-handling hardening, and modular structure. Same architecture, same UI, cleaner internals.

### Added
- **`app/core/page_helpers.py`** — shared `prepare_indicator_slice` + `fetch_indicator_slice` helpers that replace 10 verbatim copies of `_prepare_indicator_slice` previously inlined in `app/pages/01..10_*.py`. Dashboard pages now import these instead of re-implementing the same World Bank normalization recipe each time.
- **`agent/agent/prompts.py`** — central location for the LangGraph system prompts (147-line supervisor prompt and the guardrail prompt), pulled out of `graph.py` for readability.
- **`python_sandbox`: `RLIMIT_AS` (2 GB) and `RLIMIT_CPU`** applied to the subprocess running LLM-generated code (via `preexec_fn`). Previously only a wall-clock timeout existed; misbehaving code now gets killed by the kernel instead of starving the container.
- **`agent /chat/stream`: 5-minute SSE stream timeout.** An `asyncio.timeout(...)` guard wraps `astream_events`; on timeout the client receives a final `error` event instead of an open-ended stream.
- **`forecaster`: `FORECASTER_CONFIG_PATH` env var** to override the default `config.yaml` location.

### Changed
- **`forecaster /predict` and `clustering /cluster` are now `async def`** with the CPU-bound work dispatched via `fastapi.concurrency.run_in_threadpool(...)`. Previously the sync handlers blocked the FastAPI event loop for the entire duration of `Prophet.fit()` / `auto_arima()` / `TSNE.fit_transform()`.
- **`forecaster` model cache is now thread-safe.** Moved from a module-level dict to `app.state.model_cache` + `asyncio.Lock` with double-checked locking; concurrent first-time requests for the same model no longer race on heavy ML imports.
- **`agent /plots/interpret` no longer blocks the event loop.** The blocking `openai.OpenAI.chat.completions.create(...)` call is now wrapped with `asyncio.to_thread(...)`.
- **`app/core/api_client.py` error messages.** All `RuntimeError`s wrapping HTTP failures now include the actual status code and a body excerpt instead of the misleading "No available base URL candidates" string.
- **`app/core/plotting.py: GraphBox.render_streamlit_ui`** decomposed: the settings popover and the dropped-log-points caption were extracted into helper methods, reducing the 435-line monolith.
- **`agent/agent/graph.py`** worker `except Exception` blocks now call `logger.exception(...)` before returning the fallback result, so failures show up in container logs instead of vanishing.
- **`python_sandbox/main.py`** logs subprocess start, finish, and temp-file cleanup outcomes.
- **`downloader_general`**: post-download `sleep(10)` is now a configurable class attribute (`between_download_sleep_seconds`).
- **`forecaster /models`, `agent /models`, and assorted page settings** narrowed bare `except Exception:` to specific exceptions with logging.

### Removed
- **10 duplicated copies** of `_prepare_indicator_slice` across `app/pages/01..10_*.py` (~250 lines of dead duplication).
- **Unused `pandas==3.0.1`** from `downloader_extra/requirements.txt` (~50 MB image bloat).
- **`successfull_connections` field** from `downloader_general/src/extractors/world_bank_download.py` (set but never read; also fixed the typo by deletion).
- **Duplicated `all_records = []` / `offset = None` initialization** in `app/core/qdrant_client.py` (copy-paste artifact).

### Deferred (called out, not done in this pass)
- Splitting `agent/agent/graph.py` (~1.5k lines) into per-worker modules under `agent/agent/workers/`.
- Rewriting `plotly.express` chart calls in dashboard pages to consume polars natively (removing the remaining `.to_pandas()` conversions). The remaining sites rely on pandas-specific boolean masking / `.empty` / `.fillna` patterns and would need a careful go.Figure rewrite.
- Shared `wb_api.py` between `downloader_general` and `downloader_extra` (requires multi-service file sharing).
- Replacing pandas with polars internally in the `forecaster` models.
- Adding the missing PostgreSQL indexes via a `db_init` migration.

## [v0.7]

Improvements of descriptions for the dashboard

### Added
- `.drawio` and `.png` diagrams with description of dashboard architecture and agentic system architecture
- Renamed `TODO` into `TODO.md`
- updated `.gitignore`
- presentation of the dashboard
- text of the dashboard introduction in $\LaTeX$ and its compiled version in `.pptx`

## [v0.6]

The "hosting-ready" release. Adds an alternate deployment topology for public VPS hosting (on the `hosting` branch), introduces in-session LLM token-usage accounting, and removes the runtime theme picker.

### Added
- **Session token usage panel.** The Settings page now shows a per-model breakdown of prompt / completion / total tokens consumed by the AI analyst during the current Streamlit session, with a "Reset token counter" button. Tracking is in-memory only and clears on session end.
- **Token usage in the agent API.** The `agent` service attaches a per-request `UsageTracker` callback to every LangChain LLM call (guardrail, supervisor, every worker, plus the final synthesis stream) and surfaces aggregated token counts in the `final` SSE event of `/chat/stream`. `/plots/interpret` now also returns a `usage` block.
- **`CHANGELOG.md`** (this file).
- **Hardened production deployment topology** on the new `hosting` branch (added in a follow-up commit on that branch): `nginx` reverse proxy as the sole public entry point, internal-only Docker network for backend services, per-session user-supplied LLM credentials forwarded to the agent via request headers, embedding/RAG credentials kept server-side. See `README.md` § "Deployment (hosting)".

### Changed
- **README.md** — full rewrite: clearer prose, fixed spelling, dedicated `.env` variable table, separated "local development" and "Deployment (hosting)" flows, refreshed configuration section.
- `themes.yaml` is now the only place to switch themes; this is a deploy-time concern, not a runtime UI option.

### Removed
- **Runtime theme picker** from the Settings page. The dropdown, "Apply theme" button and the code that rewrote `app/.streamlit/config.toml` are gone. `core/theming.set_active_theme`, `list_theme_names`, `get_active_theme_name`, and the `_sync_streamlit_config` helpers were removed (they had no remaining callers).

## [v0.5]

### Added
- `_container_data/.env.example` to make the required environment variables explicit.

### Changed
- Sturdier external-service clients: `app/core/postgres_client.py` and `app/core/qdrant_client.py` got better error handling and retries.
- `python_sandbox/main.py` — significant rework of sandbox execution and timeouts.
- `clustering` service: new dependencies, more robust input handling.
- `downloader_general`, `downloader_extra` and `forecaster` received reliability fixes around file handling and configuration loading.
- Minor adjustments in `agent/main.py`, `app/core/plotting.py` and `app/pages/16_settings.py`.

## [v0.4]

### Added
- **Theming system** under `_container_data/themes.yaml` plus `app/core/theming.py`:
  - Three bundled themes (`dark`, `dark-blue`, `light-green`).
  - A registered Plotly template (`"app"`) derived from the active theme.
  - A Streamlit-config sync that mirrored the active theme into `app/.streamlit/config.toml`.
- A theme picker on the Settings page (later removed in v0.6).

### Changed
- All dashboard pages migrated off hard-coded chart colours and onto the new theme tokens (`get_color`, `get_colorway`).
- `app/.streamlit/config.toml` now derives from the active theme rather than holding fixed colours.

## [v0.3]

### Added
- **Multi-agent architecture** for the AI analyst built on LangGraph. `agent/agent/graph.py` was rewritten end-to-end (~1k lines) to introduce:
  - A `GuardrailAgent` that screens incoming user messages.
  - A `MacroSupervisorAgent` that plans, delegates and finishes.
  - Specialised workers: `sql_agent` (3-step World Bank exploration + Yahoo Finance lookup), `plotly_agent` (Plotly code generation in a sandboxed runtime), `table_agent` (Polars transformations), `rag_agent` (Qdrant news search), `web_search` (DuckDuckGo), `downloader_agent` (calls `downloader_extra` to ingest new World Bank indicators), `chat_agent` (conversational synthesis).
  - Streaming SSE protocol from `/chat/stream` (`step` / `token` / `final` / `error` events).
- New `agent/agent/schemas.py` Pydantic models for every structured worker output.

### Changed
- AI analyst page rebuilt around the new streaming protocol with execution log, plot artifact rendering and a dedicated data-table expander.
- Page numbering renormalised under the new navigation structure (pages `06`–`16`).

### Removed
- The legacy single-prompt agent path in `app/core/api_client.py` and the older `10_ai_agent_chat.py` page.

## [v0.2]

### Added
- Six new dashboard pages: Governance & Institutions, Technology & Innovation, Health & Wellbeing, Education & Human Capital, Environment, **AI Analyst**, **Custom Plot Constructor**, **Clustering Sandbox**, **Yahoo Finance**, **News Explorer**, and the first **Settings** page.
- World Bank download configuration extended with additional indicators across the new domains.
- Centralised page-render and HTTP-request logging in `app/core/app_logging.py`.
- `db_init` container that creates the application's PostgreSQL roles and schema (added shortly before this tag, alongside `_container_data/db_init.sh` and `init-user.sh`).
- `downloader_general` got a structured `src/utils/schema.py` validator and improved retry handling for World Bank fetches.

### Changed
- Database schema (`_container_data/database_schema.yaml`) refactored — column names and types tightened.
- `downloader_extra/client_wb.py`, `downloader_general/main.py` and the `world_bank` / `yahoo` / `github` extractors hardened against API timeouts and partial pages.
- README polished and a `TODO` file added.

## [Initial commit]

### Added
- Initial scaffolding of the project: ten-container `docker-compose.yaml`, the Streamlit `app` skeleton with the first World Bank indicator pages, the `agent` FastAPI service, `forecaster`, `clustering`, `downloader_general`, `downloader_extra`, `python_sandbox`, plus `db` (PostgreSQL) and `vector_db` (Qdrant).
- World Bank, Yahoo Finance and Webz.io news ingestion pipelines.
- Core dashboard infrastructure: postgres / qdrant clients, plotting helpers, asset templates, base config files (`config.yaml`, `database_schema.yaml`, download configs).
- README with installation instructions; MIT license.
