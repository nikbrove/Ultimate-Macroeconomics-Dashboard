"""Smoke tests covering every concrete forecaster on a small synthetic series.

These don't validate accuracy — only that each wrapper returns the expected
shape, dtypes, and a confidence band that brackets the point forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from forecasters.arima_model import ArimaForecaster
from forecasters.auto_arima_model import AutoArimaForecaster
from forecasters.moving_average_model import MovingAverageForecaster
from forecasters.sarima_model import SarimaForecaster


def _synthetic_series(n: int = 60) -> pl.DataFrame:
    """Return a noisy upward-trending monthly series — same fixture for every smoke test."""
    rng = np.random.default_rng(seed=0)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    values = np.linspace(0, 10, n) + rng.normal(scale=0.5, size=n)
    return pl.DataFrame({"ds": dates, "y": values})


def _assert_forecast_shape(result: pl.DataFrame, expected_rows: int) -> None:
    """Common shape + CI assertions reused across every smoke test."""
    assert set(result.columns) == {"ds", "yhat", "yhat_lower", "yhat_upper"}
    assert result.height == expected_rows
    lower = result["yhat_lower"].to_numpy()
    upper = result["yhat_upper"].to_numpy()
    point = result["yhat"].to_numpy()
    assert np.all(lower <= point)
    assert np.all(point <= upper)


def test_auto_arima_returns_expected_shape_and_columns() -> None:
    result = AutoArimaForecaster().predict(df=_synthetic_series(), n_predict=6, alpha=0.05)
    _assert_forecast_shape(result, expected_rows=6)


def test_manual_arima_uses_supplied_orders() -> None:
    result = ArimaForecaster().predict(
        df=_synthetic_series(), n_predict=6, alpha=0.05, p=2, d=1, q=1
    )
    _assert_forecast_shape(result, expected_rows=6)


def test_sarima_with_seasonal_orders() -> None:
    result = SarimaForecaster().predict(
        df=_synthetic_series(),
        n_predict=6,
        alpha=0.05,
        p=1,
        d=0,
        q=1,
        P=1,
        D=0,
        Q=0,
        s=12,
    )
    _assert_forecast_shape(result, expected_rows=6)


def test_moving_average_returns_flat_forecast() -> None:
    result = MovingAverageForecaster().predict(
        df=_synthetic_series(), n_predict=6, alpha=0.05, window=4
    )
    _assert_forecast_shape(result, expected_rows=6)
    yhat = result["yhat"].to_numpy()
    # Naive MA forecast is constant across the horizon.
    assert np.allclose(yhat, yhat[0])


def test_xgboost_recursive_forecast_shape() -> None:
    pytest.importorskip("xgboost")
    from forecasters.xgboost_model import XgboostForecaster

    result = XgboostForecaster().predict(
        df=_synthetic_series(),
        n_predict=6,
        alpha=0.05,
        lags=4,
        n_estimators=50,
        max_depth=2,
        learning_rate=0.1,
    )
    _assert_forecast_shape(result, expected_rows=6)
