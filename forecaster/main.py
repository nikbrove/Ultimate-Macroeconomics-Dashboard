import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import polars as pl
import yaml

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from schemas import ForecastPoint, ForecastRequest, ForecastResponse
from forecasters.core.base import BaseForecaster

CONFIG_PATH = Path(os.environ.get("FORECASTER_CONFIG_PATH", "config.yaml"))

CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
FORECASTER_CONFIG = CONFIG.get("forecaster", {})

ARIMA_AVAILABLE = bool(FORECASTER_CONFIG.get("ARIMA_AVAILABLE"))
PROPHET_AVAILABLE = bool(FORECASTER_CONFIG.get("PROPHET_AVAILABLE"))
CHRONOS_AVAILABLE = bool(FORECASTER_CONFIG.get("CHRONOS_AVAILABLE"))
CHRONOS_MODEL_NAME = FORECASTER_CONFIG.get("CHRONOS_MODEL")
CHRONOS_DEFAULT_MODEL_NAME = "amazon/chronos-t5-small"


async def _get_forecaster(app: FastAPI, model_type: str) -> BaseForecaster:
    """Return a cached forecaster, lazily importing+constructing under a lock.

    Without the lock, two concurrent first-time requests for the same model
    would each import the heavy ML library (prophet, chronos…) and race on
    the dict assignment. The lock makes initialization atomic across the
    asyncio event loop. Heavy fit/predict still runs off-loop via
    ``run_in_threadpool``.
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
                ChronosForecaster(CHRONOS_MODEL_NAME)
                if CHRONOS_MODEL_NAME
                else ChronosForecaster()
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
    app.state.model_cache = {}
    app.state.model_cache_lock = asyncio.Lock()
    yield


app = FastAPI(
    title="Time Series Forecasting API",
    description="A unified API for Prophet, Chronos, and ARIMA forecasting.",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Welcome to the Time Series Forecasting API"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict[str, list[str]]:
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
        raise HTTPException(
            status_code=400, detail=f"Invalid forecasting input: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting failed: {str(e)}")

    forecast_df = forecast_df.with_columns(
        pl.col("ds").dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    points = [ForecastPoint(**row) for row in forecast_df.to_dicts()]

    return ForecastResponse(model_used=request.model_type, forecast=points)
