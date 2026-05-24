# Ultimate Macroeconomics Dashboard

![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![uv](https://img.shields.io/badge/uv-%23DE5FE9.svg?style=for-the-badge&logo=uv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![LangGraph](https://img.shields.io/badge/langgraph-%231C3C3C.svg?style=for-the-badge&logo=langgraph&logoColor=white)
![Postgres](https://img.shields.io/badge/postgres-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)
![Qdrant](https://img.shields.io/badge/qdrant-%23dc2626.svg?style=for-the-badge&logo=qdrant&logoColor=white)

[`Ultimate Macroeconomics Dashboard`](https://github.com/alexveider1/Ultimate-Macroeconomics-Dashboard) is an AI-powered macroeconomic analytics suite: a Streamlit dashboard backed by Postgres + Qdrant, plus FastAPI services for an LLM analyst, forecasting, clustering, on-demand ingestion, and a Python sandbox. **70+** World Bank indicators, **30 000+** news articles, **50+** Yahoo Finance tickers, **80+** prebuilt charts.

Full stack: nine Docker containers — `db`, `vector_db`, `downloader_general`, `app`, `agent`, `forecaster`, `clustering`, `downloader_extra`, `python_sandbox`. Architecture details and conventions live in [`CLAUDE.md`](CLAUDE.md); deep configuration lives in [`_container_data/config.yaml`](_container_data/config.yaml).

## Quick start

Prerequisites: Docker with the Compose plugin. NVIDIA GPU is optional (improves the forecaster).

```bash
# 1. Clone
git clone https://github.com/alexveider1/Ultimate-Macroeconomics-Dashboard
cd Ultimate-Macroeconomics-Dashboard/

# 2. Create the env file (fill in your secrets)
cp _container_data/.env.example _container_data/.env
$EDITOR _container_data/.env

# 3. Point the agent at an OpenAI-compatible LLM
$EDITOR _container_data/config.yaml      # set shared.openai_* keys

# 4. (No GPU) remove the `deploy:` block under `forecaster:` in docker-compose.yaml

# 5. Build and run
docker compose up --build
```

Open <http://localhost:8501>. First boot does a one-shot ingestion (~1–2h) that fills both databases; the dashboard stays available while it runs.

### Required `.env` variables

| Variable                   | Used by                       | Purpose                                                                |
| -------------------------- | ----------------------------- | ---------------------------------------------------------------------- |
| `POSTGRES_USER`            | `db`, `downloader_general`, `downloader_extra`, `app` | Postgres superuser created natively by the `postgres:18` image on first boot. |
| `POSTGRES_PASSWORD`        | same                          | Password for the superuser.                                            |
| `POSTGRES_DB`              | `db`                          | Default database created on first boot (typically `postgres`).         |
| `POSTGRES_LLM_USER`        | `downloader_general`, `agent`, `app` | Read-only role used by the AI analyst and the dashboard's bulk reads to query the database. |
| `POSTGRES_LLM_PASSWORD`    | same                          | Password for the read-only role (rotatable; takes effect on next boot).|
| `QDRANT__SERVICE__API_KEY` | `vector_db`, `agent`, `app`   | Bearer token protecting the Qdrant HTTP API.                           |
| `OPENAI_API_KEY`           | `agent`                       | API key for the LLM/embedding provider in `config.yaml`.               |

> Never commit `_container_data/.env`. It is gitignored.

### Required `config.yaml` keys

Set these under `shared:` to point at your LLM provider (any OpenAI-compatible API works):

```yaml
shared:
  openai_base_url: https://api.openai.com/v1
  openai_llm_model: gpt-5.4
  openai_embedding_model: openai/text-embedding-3-small
```

Everything else has working defaults. See [`_container_data/config.yaml`](_container_data/config.yaml) for the full schema.

## LLM requirements

The agent needs a model with reasoning, tool/function calling, vision (to read rendered charts), and ≥256k context. Any recent flagship from OpenAI, Google, Anthropic, Qwen, or DeepSeek works. Local models served via [vLLM](https://github.com/vllm-project/vllm) on a strong GPU also work.

## Local development

Per service (each has its own `pyproject.toml` + `uv.lock`):

```bash
cd app           # or agent, forecaster, clustering, downloader_*, python_sandbox
uv sync --group dev
uv run pytest tests
uv run streamlit run app.py        # for the app
uv run uvicorn main:app --reload   # for FastAPI services
```

Linting / type-checking: `uv run ruff check .` and `uv run ty .`. Tests under `<service>/tests/`; testcontainers-backed integration tests need Docker running.

## Illustrations

|                               |                                  |
| ----------------------------- | -------------------------------- |
| ![](app/assets/structure.png) | ![](app/assets/ai_structure.png) |
| ![](app/assets/1.png)         | ![](app/assets/2.png)            |
| ![](app/assets/3.png)         | ![](app/assets/4.png)            |

## Disclaimer

All data is sourced from third-party providers and presented as-is. The author makes no representations about its accuracy or completeness.

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
