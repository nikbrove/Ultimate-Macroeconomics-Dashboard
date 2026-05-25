"""Pydantic request/response schemas for the forecasting FastAPI service."""

import math
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator

ModelType = Literal[
    "prophet",
    "chronos",
    "auto_arima",
    "arima",
    "sarima",
    "moving_average",
    "xgboost",
]


class ForecastRequest(BaseModel):
    """Body accepted by ``POST /predict``.

    Args:
        model_type: Which underlying model to use. Models can be disabled
            via ``config.yaml`` toggles; requesting a disabled model yields
            a 400.
        dates: ISO timestamps for the historical points (aligned to ``values``).
        values: Historical observations; must be the same length as ``dates``
            and contain only finite numbers.
        n_prev: How many trailing points to keep as context for fitting
            (truncates ``dates`` / ``values`` if smaller than their length).
        n_predict: How many future points to produce.
        alpha: Significance level for confidence intervals (e.g. ``0.05`` -> 95%).
        model_params: Optional model-specific hyperparameters forwarded to
            the wrapper's ``predict`` call. Unknown keys are ignored by the
            individual models, so callers can pass a flat dict.
    """

    model_type: ModelType = Field(default="prophet", description="Choose the forecasting model.")

    dates: List[str] = Field(..., description="Timestamps for the historical data (ISO format).")
    values: List[float] = Field(..., description="Historical time series values.")

    n_prev: int = Field(..., gt=0, description="Number of previous points to consider for fitting.")
    n_predict: int = Field(..., gt=0, description="Number of future points to predict.")
    alpha: float = Field(
        0.05,
        ge=0.01,
        le=0.2,
        description="Significance level for CI",
    )
    model_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model-specific hyperparameters (e.g. p/d/q for arima, window for MA).",
    )

    @field_validator("values")
    def check_lengths_match(cls, v, info):
        """Reject mismatched lengths, empty inputs, and non-finite numbers."""
        if "dates" in info.data and len(v) != len(info.data["dates"]):
            raise ValueError("The number of dates and values must be strictly equal.")
        if len(v) == 0:
            raise ValueError("At least one historical point is required.")
        if any(not math.isfinite(val) for val in v):
            raise ValueError("All values must be finite numbers.")
        return v


class ForecastPoint(BaseModel):
    """One predicted point with its confidence interval.

    Args:
        ds: Timestamp formatted as ``%Y-%m-%d %H:%M:%S``.
        yhat: Point forecast.
        yhat_lower: Lower bound of the confidence interval.
        yhat_upper: Upper bound of the confidence interval.
    """

    ds: str
    yhat: float
    yhat_lower: float
    yhat_upper: float


class ForecastResponse(BaseModel):
    """Response returned by ``POST /predict``.

    Args:
        model_used: Echo of the model that produced the forecast.
        forecast: Future points sorted ascending by timestamp.
    """

    model_used: str
    forecast: List[ForecastPoint]
