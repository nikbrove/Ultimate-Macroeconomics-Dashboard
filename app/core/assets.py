import json
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any


_ASSETS_DIR = Path("assets")
_PLOT_MARKUP_TEMPLATES_PATH = _ASSETS_DIR / "plot_markup_templates.json"


@lru_cache(maxsize=1)
def _load_plot_markup_templates() -> dict[str, str]:
    payload = json.loads(_PLOT_MARKUP_TEMPLATES_PATH.read_text(encoding="utf-8"))
    return {
        str(key): str(value)
        for key, value in payload.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def get_markup_template(name: str) -> str:
    templates = _load_plot_markup_templates()
    if name not in templates:
        raise KeyError(f"Unknown markup template: {name}")
    return templates[name]


def render_markup_template(name: str, **substitutions: Any) -> str:
    return Template(get_markup_template(name)).substitute(substitutions)
