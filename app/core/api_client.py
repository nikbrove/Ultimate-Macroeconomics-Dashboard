"""Typed HTTP wrappers around every backend FastAPI service.

Every Streamlit page must use these helpers instead of calling ``httpx``
directly: they centralise base-URL resolution (explicit arg → env var →
Compose default), request logging via :mod:`core.app_logging`, timeouts,
and error normalisation. Each wrapper raises ``RuntimeError`` with a
user-readable message when the underlying call fails so the pages can
surface the failure with a plain ``st.error``.
"""

import json
import os
from typing import Any

import httpx

from core.app_logging import log_http_request


def _http_error_message(endpoint: str, exc: httpx.HTTPError) -> str:
    """Render a one-line error string from an ``httpx`` exception.

    Includes the response status code and up to 300 chars of the response
    body when present, otherwise falls back to the exception message.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return f"{endpoint} request failed: {exc}"
    body_excerpt = (response.text or "")[:300].strip()
    return f"{endpoint} returned HTTP {response.status_code}" + (
        f": {body_excerpt}" if body_excerpt else ""
    )


def _resolve_base_url(
    explicit_base_url: str | None,
    env_var_name: str,
    default_base_url: str,
) -> str:
    """Pick the first usable base URL: explicit > env var > Compose default.

    Args:
        explicit_base_url: Direct override supplied by the caller.
        env_var_name: Name of the env var that can override the default.
        default_base_url: Compose-network default (e.g. ``http://agent:8000``).

    Returns:
        Base URL stripped of any trailing ``/``.
    """
    candidates = [
        explicit_base_url,
        os.getenv(env_var_name),
        default_base_url,
    ]

    for candidate in candidates:
        if candidate and str(candidate).strip():
            return str(candidate).strip().rstrip("/")

    return default_base_url.rstrip("/")


def resolve_forecaster_base_url(base_url: str | None = None) -> str:
    """Return the forecaster URL (uses ``FORECASTER_BASE_URL`` env var if set)."""
    return _resolve_base_url(base_url, "FORECASTER_BASE_URL", "http://forecaster:8001")


def resolve_agent_base_url(base_url: str | None = None) -> str:
    """Return the agent URL (uses ``AGENT_BASE_URL`` env var if set)."""
    return _resolve_base_url(base_url, "AGENT_BASE_URL", "http://agent:8000")


def resolve_clustering_base_url(base_url: str | None = None) -> str:
    """Return the clustering URL (uses ``CLUSTERING_BASE_URL`` env var if set)."""
    return _resolve_base_url(base_url, "CLUSTERING_BASE_URL", "http://clustering:8002")


def forecast_timeseries(
    base_url: str,
    dates: list[str],
    values: list[float],
    n_prev: int,
    n_predict: int,
    alpha: float = 0.05,
    model_type: str = "prophet",
) -> dict[str, Any]:
    """Call ``POST /predict`` on the forecaster service.

    Args:
        base_url: Forecaster service URL (or empty/``None`` to use the default).
        dates: ISO date strings of the historical series.
        values: Numeric values aligned with ``dates``.
        n_prev: Maximum number of past points to feed the model.
        n_predict: Number of forecast steps to emit.
        alpha: Confidence level alpha (e.g. ``0.05`` → 95% CI).
        model_type: ``"prophet"`` / ``"arima"`` / ``"chronos"``.

    Returns:
        Raw JSON dict with at least a ``forecast`` key (list of points).

    Raises:
        RuntimeError: When the service returns a non-2xx response.
    """
    resolved_base_url = resolve_forecaster_base_url(base_url)
    payload = {
        "model_type": model_type,
        "dates": dates,
        "values": values,
        "n_prev": n_prev,
        "n_predict": n_predict,
        "alpha": alpha,
    }
    try:
        log_http_request(
            resolved_base_url,
            "/predict",
            "POST",
            summary=(
                f"model_type={model_type} history_points={len(values)} "
                f"n_prev={n_prev} n_predict={n_predict}"
            ),
        )
        with httpx.Client(timeout=60.0) as client:
            response = client.post(f"{resolved_base_url}/predict", json=payload)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        raise RuntimeError(_http_error_message("/predict", exc)) from exc


def agent_chat_stream(
    user_message: str,
    chat_history: list[dict[str, str]] | None = None,
    base_url: str | None = None,
):
    """Open an SSE stream to ``POST /chat/stream`` on the agent and yield events.

    Each yielded value is a decoded ``dict`` from one ``data:`` line of the
    SSE response (``step``, ``token``, ``final``, or ``error`` events).

    Args:
        user_message: Latest user message.
        chat_history: Prior chat turns in ``{"role", "content"}`` form.
        base_url: Agent service URL (or ``None`` to use the default).

    Yields:
        One decoded event dict per SSE frame.

    Raises:
        RuntimeError: When the stream errors out or an event is invalid JSON.
    """
    resolved_base_url = resolve_agent_base_url(base_url)
    payload = {
        "user_message": user_message,
        "chat_history": chat_history or [],
    }
    try:
        log_http_request(
            resolved_base_url,
            "/chat/stream",
            "POST",
            summary=(
                f"message_length={len(user_message)} "
                f"history_items={len(chat_history or [])} stream=true"
            ),
        )
        timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
        with httpx.Client(timeout=timeout) as client:
            with client.stream(
                "POST",
                f"{resolved_base_url}/chat/stream",
                json=payload,
            ) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    line = str(raw_line or "").strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line.removeprefix("data:").strip()
                    if not line or line.startswith(":"):
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"Invalid streaming payload from agent service: {exc}"
                        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(_http_error_message("/chat/stream", exc)) from exc


def interpret_plot_image(
    image_base64: str,
    mode: str,
    chart_context: str = "",
    base_url: str | None = None,
) -> dict[str, Any]:
    """Call ``POST /plots/interpret`` on the agent to describe a rendered chart.

    Args:
        image_base64: PNG payload encoded as base64 (no data: prefix).
        mode: ``"no_hallucinations"`` for a strict factual description or
            ``"creative"`` for an interpretive narrative.
        chart_context: Optional human-readable context appended to the prompt
            (indicator name, year, chart kind).
        base_url: Agent service URL (or ``None`` for the default).

    Returns:
        Dict with at least ``description``, ``mode``, and a ``usage`` block.

    Raises:
        RuntimeError: When the agent returns a non-2xx response.
    """
    resolved_base_url = resolve_agent_base_url(base_url)
    payload = {
        "image_base64": image_base64,
        "mode": mode,
        "chart_context": chart_context,
    }
    try:
        log_http_request(
            resolved_base_url,
            "/plots/interpret",
            "POST",
            summary=(
                f"mode={mode} chart_context_length={len(chart_context)} "
                f"image_base64_length={len(image_base64)}"
            ),
        )
        with httpx.Client(timeout=90.0) as client:
            response = client.post(
                f"{resolved_base_url}/plots/interpret",
                json=payload,
            )
        response.raise_for_status()
        result = response.json()
        if isinstance(result, dict):
            return result
        return {"description": str(result), "mode": mode}
    except httpx.HTTPError as exc:
        raise RuntimeError(_http_error_message("/plots/interpret", exc)) from exc


def cluster_dataframe(
    dataframe: list[dict[str, Any]],
    method: str,
    feature_columns: list[str],
    k: int = 3,
    n_init: int = 10,
    random_state: int = 42,
    eps: float = 0.5,
    min_samples: int = 5,
    reduction_method: str = "tsne",
    base_url: str | None = None,
) -> dict[str, Any]:
    """Call ``POST /cluster`` on the clustering service.

    Args:
        dataframe: Row-oriented payload (one dict per row).
        method: ``"kmeans"`` or ``"dbscan"``.
        feature_columns: Numeric columns to feed the algorithm.
        k: Number of clusters for KMeans.
        n_init: Number of KMeans initialisations.
        random_state: Seed for reproducibility.
        eps: DBSCAN neighbourhood radius.
        min_samples: DBSCAN minimum cluster size.
        reduction_method: ``"tsne"`` or ``"pca"`` for the 2-D projection.
        base_url: Clustering service URL (or ``None`` for the default).

    Returns:
        Raw JSON dict with cluster labels and 2-D projection coordinates.

    Raises:
        RuntimeError: When the service returns a non-2xx response.
    """
    resolved_base_url = resolve_clustering_base_url(base_url)
    payload = {
        "method": method,
        "dataframe": dataframe,
        "feature_columns": feature_columns,
        "k": k,
        "n_init": n_init,
        "random_state": random_state,
        "eps": eps,
        "min_samples": min_samples,
        "reduction_method": reduction_method,
    }
    try:
        log_http_request(
            resolved_base_url,
            "/cluster",
            "POST",
            summary=(
                f"method={method} rows={len(dataframe)} feature_columns={len(feature_columns)}"
            ),
        )
        with httpx.Client(timeout=60.0) as client:
            response = client.post(f"{resolved_base_url}/cluster", json=payload)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        raise RuntimeError(_http_error_message("/cluster", exc)) from exc


def list_agent_models(base_url: str | None = None) -> list[str]:
    """Return the list of LLM model ids the agent currently knows about.

    Args:
        base_url: Agent service URL (or ``None`` for the default).

    Returns:
        List of model ids; empty list on transport error / empty payload.

    Raises:
        RuntimeError: When the agent returns a non-2xx response.
    """
    resolved_base_url = resolve_agent_base_url(base_url)
    try:
        log_http_request(resolved_base_url, "/models", "GET")
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{resolved_base_url}/models")
        response.raise_for_status()
        payload = response.json()

        if isinstance(payload, dict):
            models = payload.get("models", [])
            if isinstance(models, list):
                return [str(model) for model in models if str(model).strip()]

        if isinstance(payload, list):
            return [str(model) for model in payload if str(model).strip()]
    except httpx.HTTPError as exc:
        raise RuntimeError(_http_error_message("/models", exc)) from exc

    return []
