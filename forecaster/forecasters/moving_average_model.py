"""Naive moving-average baseline forecaster.

Predicts every future point as the mean of the last ``window`` historical
values; the confidence interval comes from the in-sample residual standard
deviation, widened by ``sqrt(horizon)`` to mimic random-walk uncertainty
growth.
"""

import math

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm

from .core.base import BaseForecaster, resolve_forecast_frequency


class MovingAverageForecaster(BaseForecaster):
    """Flat-mean baseline; useful as a sanity check against richer models."""

    def __init__(self):
        """Forward to the base no-op constructor; nothing to set up."""
        super().__init__()

    def predict(
        self,
        df: pl.DataFrame,
        n_predict: int,
        alpha: float,
        *,
        window: int = 5,
        **kwargs,
    ) -> pl.DataFrame:
        """Forecast ``n_predict`` constant points using the trailing ``window`` mean.

        Args:
            df: Two-column ``(ds, y)`` Polars frame sorted ascending by ``ds``.
            n_predict: Forecast horizon in points.
            alpha: Significance level for the confidence interval.
            window: Number of trailing historical points to average over.
                Clamped to ``[1, len(y)]``; an out-of-range value is replaced
                with a quarter of the history.

        Returns:
            Polars frame with ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.
        """
        y = df["y"].to_numpy().astype(float)
        n = len(y)
        window = int(window)
        if window <= 0 or window > n:
            window = max(1, min(n, n // 4 if n >= 4 else n))

        forecast_value = float(np.mean(y[-window:]))
        yhat = np.full(n_predict, forecast_value, dtype=float)

        if n > window:
            in_sample_means = np.array(
                [float(np.mean(y[i - window : i])) for i in range(window, n)],
                dtype=float,
            )
            residuals = y[window:] - in_sample_means
            sigma = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
        else:
            sigma = float(np.std(y, ddof=1)) if n > 1 else 0.0

        if not math.isfinite(sigma):
            sigma = 0.0

        z = float(norm.ppf(1.0 - alpha / 2.0))
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
