"""XGBoost forecaster on a feature-engineered version of the series.

Features per training row are the last ``lags`` lagged values, the mean
and standard deviation of those lagged values, and the integer time
position. The model is trained once on the historical window and then
called recursively to roll the forecast forward ``n_predict`` steps.
"""

import math

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm
from xgboost import XGBRegressor

from .core.base import BaseForecaster, resolve_forecast_frequency


def _build_training_matrix(values: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` aligned for supervised training over lag features."""
    n = len(values)
    rows: list[list[float]] = []
    targets: list[float] = []
    for t in range(lags, n):
        window = values[t - lags : t]
        feat: list[float] = list(window)
        feat.append(float(np.mean(window)))
        feat.append(float(np.std(window, ddof=0)))
        feat.append(float(t))
        rows.append(feat)
        targets.append(float(values[t]))
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def _features_for_step(history: np.ndarray, lags: int, t_index: int) -> np.ndarray:
    """Return the 1×F feature row used to predict the next step."""
    window = history[-lags:]
    feat: list[float] = list(window)
    feat.append(float(np.mean(window)))
    feat.append(float(np.std(window, ddof=0)))
    feat.append(float(t_index))
    return np.asarray(feat, dtype=float).reshape(1, -1)


class XgboostForecaster(BaseForecaster):
    """Recursive XGBoost forecaster on lag + rolling features.

    Confidence intervals come from the in-sample residual standard
    deviation, widened by ``sqrt(horizon)`` (random-walk-style growth) —
    the same shape used by :class:`MovingAverageForecaster`. This is a
    pragmatic baseline, not a calibrated prediction interval.
    """

    def __init__(self):
        """Forward to the base no-op constructor; nothing to set up."""
        super().__init__()

    def predict(
        self,
        df: pl.DataFrame,
        n_predict: int,
        alpha: float,
        *,
        lags: int = 5,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        **kwargs,
    ) -> pl.DataFrame:
        """Train on lag features and roll the forecast forward ``n_predict`` steps.

        Args:
            df: Two-column ``(ds, y)`` Polars frame sorted ascending by ``ds``.
            n_predict: Forecast horizon in points.
            alpha: Significance level for the confidence interval.
            lags: Number of lagged values used as features.
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Step shrinkage applied to each tree.

        Returns:
            Polars frame with ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.

        Raises:
            ValueError: When the history is too short to build at least
                two training rows for the requested ``lags`` value.
        """
        y = df["y"].to_numpy().astype(float)
        lags = max(1, int(lags))
        if len(y) <= lags + 1:
            raise ValueError(
                f"Need at least {lags + 2} historical points to train XGBoost with lags={lags}."
            )

        X, target = _build_training_matrix(y, lags)

        model = XGBRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            objective="reg:squarederror",
            verbosity=0,
            random_state=42,
            tree_method="hist",
        )
        model.fit(X, target)

        in_sample_pred = np.asarray(model.predict(X), dtype=float)
        residuals = target - in_sample_pred
        sigma = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
        if not math.isfinite(sigma):
            sigma = 0.0

        z = float(norm.ppf(1.0 - alpha / 2.0))

        history = y.copy()
        forecasts: list[float] = []
        for _ in range(n_predict):
            t_index = len(history)
            feat = _features_for_step(history, lags, t_index)
            next_pred = float(model.predict(feat)[0])
            forecasts.append(next_pred)
            history = np.append(history, next_pred)

        yhat = np.asarray(forecasts, dtype=float)
        horizons = np.arange(1, n_predict + 1, dtype=float)
        margin = z * sigma * np.sqrt(horizons)

        last_date = df["ds"].max()
        freq = resolve_forecast_frequency(pd.DatetimeIndex(df["ds"].to_list()))
        future_dates = pd.date_range(start=last_date, periods=n_predict + 1, freq=freq)[1:]

        return pl.DataFrame(
            {
                "ds": future_dates,
                "yhat": yhat,
                "yhat_lower": yhat - margin,
                "yhat_upper": yhat + margin,
            }
        )
