"""FastAPI service exposing pmdarima / Prophet / Chronos time-series forecasting.

Heavy ML imports happen lazily inside :func:`_get_forecaster` so the container
boots fast even when only a subset of models is enabled. Each model is
instantiated at most once per process; subsequent requests reuse the cached
instance behind an ``asyncio.Lock`` that protects the first-call race.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import polars as pl
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from forecasters.core.base import BaseForecaster
from schemas import ForecastPoint, ForecastRequest, ForecastResponse

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(os.environ.get("FORECASTER_CONFIG_PATH", "config.yaml"))

CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
FORECASTER_CONFIG = CONFIG.get("forecaster", {})

ARIMA_AVAILABLE = bool(FORECASTER_CONFIG.get("ARIMA_AVAILABLE"))
PROPHET_AVAILABLE = bool(FORECASTER_CONFIG.get("PROPHET_AVAILABLE"))
CHRONOS_AVAILABLE = bool(FORECASTER_CONFIG.get("CHRONOS_AVAILABLE"))
CHRONOS_MODEL_NAME = FORECASTER_CONFIG.get("CHRONOS_MODEL")
CHRONOS_DEFAULT_MODEL_NAME = "amazon/chronos-t5-small"


async def _get_forecaster(app: FastAPI, model_type: str) -> BaseForecaster:
    """Return a cached forecaster, lazily importing + constructing under a lock.

    Without the lock, two concurrent first-time requests for the same model
    would each import the heavy ML library (prophet, chronos…) and race on
    the dict assignment. The lock makes initialization atomic across the
    asyncio event loop. Heavy fit/predict still runs off-loop via
    ``run_in_threadpool``.

    Args:
        app: FastAPI instance whose ``state.model_cache`` holds the singletons.
        model_type: ``prophet`` / ``chronos`` / ``arima``.

    Returns:
        A ready-to-use :class:`BaseForecaster` subclass.

    Raises:
        HTTPException: 400 when the requested model is disabled or unknown;
            500 when the underlying library fails to import or instantiate.
    """
    cache: dict[str, BaseForecaster] = app.state.model_cache
    lock: asyncio.Lock = app.state.model_cache_lock

    if model_type == "prophet":
        if not PROPHET_AVAILABLE:
            raise HTTPException(status_code=400, detail="Model 'prophet' is disabled.")
        if "prophet" in cache:
            return cache["prophet"]
        async with lock:
            if "prophet" in cache:
                return cache["prophet"]
            try:
                from forecasters.prophet_model import ProphetForecaster
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to initialize Prophet forecaster: {str(e)}",
                )
            cache["prophet"] = ProphetForecaster()
            return cache["prophet"]

    if model_type == "chronos":
        if not CHRONOS_AVAILABLE:
            raise HTTPException(status_code=400, detail="Model 'chronos' is disabled.")
        if "chronos" in cache:
            return cache["chronos"]
        async with lock:
            if "chronos" in cache:
                return cache["chronos"]
            try:
                from forecasters.chronos_model import ChronosForecaster
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to initialize Chronos forecaster: {str(e)}",
                )
            cache["chronos"] = (
                ChronosForecaster(CHRONOS_MODEL_NAME) if CHRONOS_MODEL_NAME else ChronosForecaster()
            )
            return cache["chronos"]

    if model_type == "arima":
        if not ARIMA_AVAILABLE:
            raise HTTPException(status_code=400, detail="Model 'arima' is disabled.")
        if "arima" in cache:
            return cache["arima"]
        async with lock:
            if "arima" in cache:
                return cache["arima"]
            try:
                from forecasters.arima_model import ArimaForecaster
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to initialize ARIMA forecaster: {str(e)}",
                )
            cache["arima"] = ArimaForecaster()
            return cache["arima"]

    raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the per-process model cache and lock on startup.

    Chronos is eagerly loaded onto RAM/GPU at boot so the first inference
    request doesn't pay the multi-second checkpoint-download + weight-load
    cost. ARIMA/Prophet stay lazy since they are cheap to construct.
    """
    app.state.model_cache = {}
    app.state.model_cache_lock = asyncio.Lock()

    if CHRONOS_AVAILABLE:
        try:
            from forecasters.chronos_model import ChronosForecaster

            chronos_label = CHRONOS_MODEL_NAME or CHRONOS_DEFAULT_MODEL_NAME
            logger.info("Preloading Chronos pipeline: %s", chronos_label)
            app.state.model_cache["chronos"] = await run_in_threadpool(
                ChronosForecaster, CHRONOS_MODEL_NAME
            ) if CHRONOS_MODEL_NAME else await run_in_threadpool(ChronosForecaster)
            logger.info("Chronos pipeline ready (%s)", chronos_label)
        except Exception as exc:
            logger.warning("Failed to preload Chronos on startup: %s", exc, exc_info=True)

    yield


app = FastAPI(
    title="Time Series Forecasting API",
    description="A unified API for Prophet, Chronos, and ARIMA forecasting.",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    """Return a static welcome banner — used as a liveness signal."""
    return {"message": "Welcome to the Time Series Forecasting API"}


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return ``{"status": "ok"}`` for the Compose healthcheck."""
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict[str, list[str]]:
    """Return the labels of every enabled model.

    Chronos additionally embeds the underlying checkpoint name in parentheses
    so the dashboard can show which weights are loaded.
    """
    available_models: list[str] = []
    if ARIMA_AVAILABLE:
        available_models.append("arima")
    if PROPHET_AVAILABLE:
        available_models.append("prophet")
    if CHRONOS_AVAILABLE:
        chronos_label = CHRONOS_MODEL_NAME or CHRONOS_DEFAULT_MODEL_NAME
        available_models.append(f"chronos ({chronos_label})")

    return {"available_models": available_models}


@app.post("/predict", response_model=ForecastResponse)
async def generate_prediction(request: ForecastRequest) -> ForecastResponse:
    """Build a forecast for the supplied history using the requested model.

    Args:
        request: Validated :class:`ForecastRequest` body.

    Returns:
        ForecastResponse with the model label and predicted points.

    Raises:
        HTTPException: 400 for unparseable dates or invalid inputs;
            500 if the underlying model raises during fit/predict.
    """
    df = pl.DataFrame({"ds": request.dates, "y": request.values}).with_columns(
        pl.col("ds").str.to_datetime(strict=False)
    )

    if df["ds"].null_count() > 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format found in 'dates'. Use ISO datetime-compatible strings.",
        )

    df = df.group_by("ds", maintain_order=True).agg(pl.col("y").last()).sort("ds")

    if request.n_prev is not None and request.n_prev < len(df):
        df_context = df.tail(request.n_prev)
    else:
        df_context = df

    forecaster = await _get_forecaster(app, request.model_type)

    try:
        forecast_df = await run_in_threadpool(
            forecaster.predict,
            df=df_context,
            n_predict=request.n_predict,
            alpha=request.alpha,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid forecasting input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting failed: {str(e)}")

    forecast_df = forecast_df.with_columns(pl.col("ds").dt.strftime("%Y-%m-%d %H:%M:%S"))

    points = [ForecastPoint(**row) for row in forecast_df.to_dicts()]

    return ForecastResponse(model_used=request.model_type, forecast=points)
