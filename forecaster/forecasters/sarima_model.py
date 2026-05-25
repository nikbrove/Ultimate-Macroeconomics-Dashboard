"""SARIMA forecaster using ``statsmodels.tsa.statespace.sarimax`` with manual orders."""

import numpy as np
import pandas as pd
import polars as pl
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .core.base import BaseForecaster, resolve_forecast_frequency


class SarimaForecaster(BaseForecaster):
    """SARIMA wrapper that takes both the non-seasonal and seasonal orders.

    The fit relaxes the stationarity/invertibility checks so user-supplied
    orders that lie outside the strict region still produce a forecast
    (state-space filtering remains valid).
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
        p: int = 1,
        d: int = 1,
        q: int = 1,
        P: int = 0,
        D: int = 0,
        Q: int = 0,
        s: int = 12,
        **kwargs,
    ) -> pl.DataFrame:
        """Fit ``SARIMA((p, d, q), (P, D, Q, s))`` and return ``n_predict`` future points.

        Args:
            df: Two-column ``(ds, y)`` Polars frame sorted ascending by ``ds``.
            n_predict: Forecast horizon in points.
            alpha: Significance level for the confidence interval.
            p: Non-seasonal AR order.
            d: Non-seasonal integration order.
            q: Non-seasonal MA order.
            P: Seasonal AR order.
            D: Seasonal integration order.
            Q: Seasonal MA order.
            s: Seasonal period.

        Returns:
            Polars frame with ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.
        """
        y = df["y"].to_numpy().astype(float)
        order = (int(p), int(d), int(q))
        seasonal_order = (int(P), int(D), int(Q), int(s))

        model = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        forecast = model.get_forecast(steps=n_predict)
        yhat = np.asarray(forecast.predicted_mean, dtype=float)
        conf = np.asarray(forecast.conf_int(alpha=alpha), dtype=float)

        last_date = df["ds"].max()
        freq = resolve_forecast_frequency(pd.DatetimeIndex(df["ds"].to_list()))
        future_dates = pd.date_range(start=last_date, periods=n_predict + 1, freq=freq)[1:]

        return pl.DataFrame(
            {
                "ds": future_dates,
                "yhat": yhat,
                "yhat_lower": conf[:, 0],
                "yhat_upper": conf[:, 1],
            }
        )
