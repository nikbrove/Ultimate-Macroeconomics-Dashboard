"""Facebook Prophet forecaster wrapper."""

import pandas as pd
import polars as pl
from prophet import Prophet

from .core.base import BaseForecaster, resolve_forecast_frequency


class ProphetForecaster(BaseForecaster):
    """Stateless wrapper around ``prophet.Prophet`` (re-fits each call)."""

    def __init__(self):
        """Forward to base constructor; Prophet's own state lives per-call."""
        super().__init__()

    def predict(self, df: pl.DataFrame, n_predict: int, alpha: float) -> pl.DataFrame:
        """Fit a Prophet model on ``df`` and return ``n_predict`` future points.

        Args:
            df: Two-column ``(ds, y)`` Polars frame sorted ascending by ``ds``.
            n_predict: Number of future points to emit.
            alpha: Significance level; passed to Prophet as ``interval_width=1-alpha``.

        Returns:
            Polars frame with ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.
        """
        interval_width = 1.0 - alpha

        pdf = pd.DataFrame(
            {
                "ds": pd.to_datetime(df["ds"].to_list()),
                "y": df["y"].to_list(),
            }
        )

        model = Prophet(interval_width=interval_width)
        model.fit(pdf)

        freq = resolve_forecast_frequency(pd.DatetimeIndex(pdf["ds"]))
        future = model.make_future_dataframe(periods=n_predict, freq=freq)
        forecast = model.predict(future)

        future_forecast = forecast.tail(n_predict)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

        return pl.from_pandas(future_forecast)
