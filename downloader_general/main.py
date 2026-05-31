"""Entry point for the ingestion container.

On every container start: load env + config, then **always** bootstrap the
read-only LLM Postgres role (idempotent CREATE/ALTER + SELECT grants — cheap
to re-run, and required for password rotation, upgrades, and grants on tables
that didn't exist at the last bootstrap). After that, run the three
downloaders (World Bank → news → Yahoo) **once only**, gated by a marker
file (``.download_completed``) written after a successful run; subsequent
boots see the marker and skip downloads but still re-apply the bootstrap.
"""

import logging
import os
import sys
from pathlib import Path

import tqdm
import yaml
from dotenv import load_dotenv

from src.extractors import NewsDownloader, WorldBankDownloader, YahooDownloader
from src.utils.db_bootstrap import ensure_llm_role
from src.utils.downloads import _get_sql_config
from src.utils.schema import load_database_schema


class _TqdmHandler(logging.StreamHandler):
    """Logging handler that writes through ``tqdm.tqdm.write``.

    Without this, log records would clobber any active tqdm progress bar; the
    handler routes the formatted record through ``tqdm.write`` so the bar
    redraws cleanly above the log line.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Route ``record`` to stdout via ``tqdm.tqdm.write``.

        Args:
            record: The log record to emit.
        """
        try:
            tqdm.tqdm.write(self.format(record), file=sys.stdout)
        except Exception:
            self.handleError(record)


CONFIG_PATH = Path("config.yaml")
DEFAULT_DOWNLOAD_MARKER = Path("_container_data/.download_completed")


def _require(mapping: dict, *path: str) -> object:
    """Look up a nested config key, failing loudly on any missing segment.

    Args:
        mapping: Root config dict.
        *path: Sequence of keys to walk into ``mapping``.

    Returns:
        The value at ``mapping[path[0]][path[1]]...``.

    Raises:
        KeyError: When any segment is absent.
    """
    current: object = mapping
    for segment in path:
        if not isinstance(current, dict) or segment not in current:
            raise KeyError(f"Missing required config key '{'.'.join(path)}' in {CONFIG_PATH}")
        current = current[segment]
    return current


def main() -> None:
    """Run the bootstrap (LLM role + grants) and, if not yet done, the downloads.

    The bootstrap runs on every container start — it's idempotent and cheap,
    and it's the path that grants ``SELECT`` on tables added since the marker
    was first written (e.g. tables created on a fresh deploy after a schema
    upgrade). Downloads themselves stay one-shot, gated by
    ``DEFAULT_DOWNLOAD_MARKER`` (overrideable via ``DOWNLOADER_ONCE_MARKER``).
    Each downloader's ``_initialize_connections`` is checked before ``run()``
    so a failed health check skips that source rather than aborting the whole
    job. A failure in the LLM-role bootstrap is logged and ignored — the rest
    of the ingestion can still proceed.
    """
    container_data_dir = Path("_container_data")
    news_output_dir = container_data_dir / "news"
    marker_path = Path(os.getenv("DOWNLOADER_ONCE_MARKER", str(DEFAULT_DOWNLOAD_MARKER)))

    container_data_dir.mkdir(parents=True, exist_ok=True)
    news_output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(
                container_data_dir / "app.log",
                mode="w",
                encoding="utf-8",
            ),
            _TqdmHandler(),
        ],
    )

    args = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    env_file = _require(args, "shared", "env_file")
    postgres_host = _require(args, "postgres", "host")
    postgres_port = _require(args, "postgres", "port")
    # config.yaml holds the fallback DB name; POSTGRES_DB in .env wins because
    # that's the value the postgres image uses on first volume init.
    load_dotenv(env_file)
    postgres_db = os.getenv("POSTGRES_DB") or _require(args, "postgres", "database")
    qdrant_host = _require(args, "qdrant", "host")
    qdrant_port = _require(args, "qdrant", "port")
    database_schema_path = _require(args, "shared", "database_schema")
    database_schema = load_database_schema(database_schema_path)
    world_bank_download_config = _require(args, "shared", "world_bank_download_config")
    news_download_config = _require(args, "shared", "news_download_config")
    yahoo_download_config = _require(args, "shared", "yahoo_download_config")
    repo_url = _require(args, "downloader_general", "repo_url")
    openai_base_url = _require(args, "shared", "openai_base_url")
    openai_embedding_model = _require(args, "shared", "openai_embedding_model")
    openai_embedding_model_max_tokens = _require(
        args, "shared", "openai_embedding_model_max_tokens"
    )
    openai_model_dimensions = _require(args, "shared", "openai_embedding_model_dimensions")

    superuser_uri = _get_sql_config(
        username=os.getenv("POSTGRES_USER", ""),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        host=str(postgres_host),
        port=int(postgres_port),
        db=str(postgres_db),
    )
    try:
        ensure_llm_role(
            sql_uri=superuser_uri,
            llm_username=os.getenv("POSTGRES_LLM_USER", ""),
            llm_password=os.getenv("POSTGRES_LLM_PASSWORD", ""),
        )
    except Exception:
        logging.exception("LLM role bootstrap failed; continuing with downloads")

    if marker_path.exists():
        logging.info(
            "Download marker present (%s); bootstrap re-applied, skipping downloads.",
            marker_path,
        )
        return

    world_bank_downloader = WorldBankDownloader(
        env_path=env_file,
        download_config_path=world_bank_download_config,
        database_schema=database_schema,
    )
    if world_bank_downloader._initialize_connections(
        host=postgres_host,
        port=postgres_port,
        db=postgres_db,
    ):
        world_bank_downloader.run()

    news_downloader = NewsDownloader(
        env_file=env_file,
        repo_url=repo_url,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        config_path=news_download_config,
        save_path=news_output_dir,
        openai_base_url=openai_base_url,
        openai_embedding_model=openai_embedding_model,
        openai_token_limit=openai_embedding_model_max_tokens,
        openai_model_dimensions=openai_model_dimensions,
    )
    if news_downloader._initialize_connections():
        news_downloader.run()

    yahoo_downloader = YahooDownloader(
        env_path=env_file,
        download_config_path=yahoo_download_config,
        database_schema=database_schema,
    )
    if yahoo_downloader._initialize_connections(
        host=postgres_host,
        port=postgres_port,
        db=postgres_db,
    ):
        yahoo_downloader.run()

    # Mark the one-shot download as completed so future container starts
    # only re-apply the bootstrap and skip the multi-hour ingestion.
    try:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.touch()
        logging.info("Download marker written: %s", marker_path)
    except OSError:
        logging.exception("Could not write download marker at %s", marker_path)


if __name__ == "__main__":
    main()
