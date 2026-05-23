"""Smoke test: ARIMA forecaster returns the right shape on a small synthetic series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from forecasters.arima_model import ArimaForecaster


def test_arima_returns_expected_shape_and_columns() -> None:
    rng = np.random.default_rng(seed=0)
    n = 60
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    values = np.linspace(0, 10, n) + rng.normal(scale=0.5, size=n)
    df = pl.DataFrame({"ds": dates, "y": values})

    forecaster = ArimaForecaster()
    result = forecaster.predict(df=df, n_predict=6, alpha=0.05)

    assert set(result.columns) == {"ds", "yhat", "yhat_lower", "yhat_upper"}
    assert result.height == 6
    # Confidence band should bracket the point forecast.
    lower = result["yhat_lower"].to_numpy()
    upper = result["yhat_upper"].to_numpy()
    point = result["yhat"].to_numpy()
    assert np.all(lower <= point)
    assert np.all(point <= upper)
