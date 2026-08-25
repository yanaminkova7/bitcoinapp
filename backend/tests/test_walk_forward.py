import numpy as np
import pandas as pd
import pytest

from app.evaluation.walk_forward import walk_forward_validate
from app.forecasting.baselines import NaiveForecaster


def make_ohlcv(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 2, n)
    low = close - rng.uniform(0, 2, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(1000, 5000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates
    )


def test_walk_forward_uses_only_past_data_for_each_fold() -> None:
    df = make_ohlcv(300)
    result = walk_forward_validate(
        df, NaiveForecaster, "naive", horizon=1, n_test_folds=20, min_train_size=100
    )
    for fold in result.folds:
        # The training window end must always be strictly before the date being predicted.
        assert fold.train_end < fold.test_date


def test_walk_forward_train_window_expands_monotonically() -> None:
    df = make_ohlcv(300)
    result = walk_forward_validate(
        df, NaiveForecaster, "naive", horizon=1, n_test_folds=20, min_train_size=100
    )
    train_ends = [f.train_end for f in result.folds]
    assert train_ends == sorted(train_ends)
    assert len(set(train_ends)) == len(train_ends)  # strictly increasing, no repeats


def test_walk_forward_naive_prediction_equals_last_train_close() -> None:
    df = make_ohlcv(300)
    result = walk_forward_validate(
        df, NaiveForecaster, "naive", horizon=1, n_test_folds=10, min_train_size=100
    )
    for fold in result.folds:
        expected = df.loc[df.index <= fold.train_end, "close"].iloc[-1]
        assert fold.predicted == pytest.approx(expected)


def test_walk_forward_metrics_are_finite() -> None:
    df = make_ohlcv(300)
    result = walk_forward_validate(
        df, NaiveForecaster, "naive", horizon=1, n_test_folds=30, min_train_size=100
    )
    metrics = result.metrics()
    assert metrics["n_folds"] == 30
    assert np.isfinite(metrics["mae"])
    assert np.isfinite(metrics["rmse"])
    assert 0.0 <= metrics["directional_accuracy"] <= 1.0


def test_walk_forward_raises_on_insufficient_data() -> None:
    df = make_ohlcv(50)
    with pytest.raises(ValueError):
        walk_forward_validate(df, NaiveForecaster, "naive", n_test_folds=20, min_train_size=100)
