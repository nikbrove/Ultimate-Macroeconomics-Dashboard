"""Ensure stub config.yaml + .env exist before `main` is imported."""

from __future__ import annotations

from pathlib import Path


def _ensure_config_files() -> None:
    cwd = Path.cwd()
    config_path = cwd / "config.yaml"
    if not config_path.is_file():
        config_path.write_text("python_sandbox:\n  port: 8004\n", encoding="utf-8")
    env_path = cwd / ".env"
    if not env_path.is_file():
        env_path.write_text("", encoding="utf-8")


_ensure_config_files()
