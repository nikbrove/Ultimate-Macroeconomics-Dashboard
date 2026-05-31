"""Auto-tuned ARIMA forecaster powered by ``pmdarima.auto_arima`` (non-seasonal)."""

import pandas as pd
import pmdarima as pm
import polars as pl

from .core.base import BaseForecaster, resolve_forecast_frequency


class AutoArimaForecaster(BaseForecaster):
    """Stateless wrapper around ``pmdarima.auto_arima``.

    Re-fits on every call, so the wrapper is thread-safe at the cost of
    re-tuning per request. Use the manual :class:`ArimaForecaster` when
    you already know the ``(p, d, q)`` orders.
    """

    def __init__(self):
        """Forward to the base no-op constructor; nothing to set up."""
        super().__init__()

    def predict(
        self,
        df: pl.DataFrame,
        n_predict: int,
        alpha: float,
        **kwargs,
    ) -> pl.DataFrame:
        """Fit ``auto_arima`` on ``df.y`` and return ``n_predict`` future points."""
        y = df["y"].to_numpy()

        model = pm.auto_arima(y, seasonal=False, suppress_warnings=True)

        forecasts, conf_int = model.predict(n_periods=n_predict, return_conf_int=True, alpha=alpha)

        last_date = df["ds"].max()
        freq = resolve_forecast_frequency(pd.DatetimeIndex(df["ds"].to_list()))
        future_dates = pd.date_range(start=last_date, periods=n_predict + 1, freq=freq)[1:]

        return pl.DataFrame(
            {
                "ds": future_dates,
                "yhat": forecasts,
                "yhat_lower": conf_int[:, 0],
                "yhat_upper": conf_int[:, 1],
            }
        )
