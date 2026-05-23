"""Abstract base class + shared helpers for every forecaster implementation."""

from abc import ABC, abstractmethod

import pandas as pd
import polars as pl


class BaseForecaster(ABC):
    """Abstract base for every concrete forecaster (ARIMA, Prophet, Chronos)."""

    def __init__(self):
        """No-op constructor — subclasses override with model-specific setup."""

    @abstractmethod
    def predict(
        self,
        df: pl.DataFrame,
        n_predict: int,
        alpha: float,
    ) -> pl.DataFrame:
        """Produce ``n_predict`` future points with a ``(1-alpha)`` confidence band.

        Args:
            df: Two-column Polars frame with ``ds`` (datetime) and ``y`` (float),
                already sorted ascending by ``ds``.
            n_predict: Number of future points to emit.
            alpha: Significance level for the confidence interval.

        Returns:
            Polars frame with columns ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.
        """


def resolve_forecast_frequency(
    datetimes: pd.DatetimeIndex | list[pd.Timestamp],
    default: str = "D",
) -> str:
    """Infer a stable pandas frequency string for ``datetimes``.

    First asks pandas to infer the frequency; if that fails (e.g. the series
    is irregular), falls back to the modal positive gap between successive
    timestamps. Returns ``default`` for series with fewer than two points or
    when no positive gap is available.

    Args:
        datetimes: Historical timestamps; need not be sorted.
        default: Frequency to return when nothing better can be inferred.

    Returns:
        A pandas frequency string suitable for ``pd.date_range(freq=...)``.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(datetimes)).sort_values()
    if len(idx) < 2:
        return default

    inferred = pd.infer_freq(idx)
    if inferred:
        return inferred

    deltas = idx.to_series().diff().dropna()
    positive_deltas = deltas[deltas > pd.Timedelta(0)]
    if positive_deltas.empty:
        return default

    most_common_delta = positive_deltas.mode().iloc[0]
    try:
        return pd.tseries.frequencies.to_offset(most_common_delta).freqstr
    except Exception:
        return default
