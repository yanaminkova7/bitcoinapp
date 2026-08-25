import numpy as np
import pandas as pd
import pytest

from app.forecasting.features import (
    add_calendar_features,
    add_momentum_features,
    add_price_features,
    add_target,
    add_volatility_features,
    add_volume_features,
    build_features,
)


def make_ohlcv(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 2, n)
    low = close - rng.uniform(0, 2, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(1000, 5000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates
    )


FEATURE_STEPS = [add_price_features, add_momentum_features, add_volatility_features, add_volume_features]


@pytest.mark.parametrize("step", FEATURE_STEPS)
def test_no_leakage_from_future_prices(step) -> None:
    """Changing prices strictly after cutoff_idx must not change any feature value
    computed for rows at or before cutoff_idx."""
    df = make_ohlcv(100)
    cutoff_idx = 60

    df_modified = df.copy()
    future_slice = df_modified.index[cutoff_idx + 1 :]
    df_modified.loc[future_slice, ["open", "high", "low", "close", "volume"]] *= 5.0

    original_features = step(df)
    modified_features = step(df_modified)

    pd.testing.assert_frame_equal(
        original_features.iloc[: cutoff_idx + 1],
        modified_features.iloc[: cutoff_idx + 1],
    )


def test_no_leakage_full_pipeline() -> None:
    df = make_ohlcv(150)
    cutoff_idx = 100

    df_modified = df.copy()
    future_slice = df_modified.index[cutoff_idx + 1 :]
    df_modified.loc[future_slice, ["open", "high", "low", "close", "volume"]] *= 3.0

    original = build_features(df, horizon=1, target_type="price")
    modified = build_features(df_modified, horizon=1, target_type="price")

    feature_cols = [c for c in original.columns if c != "target"]
    common_dates = original.index.intersection(modified.index)
    common_dates = common_dates[common_dates <= df.index[cutoff_idx]]

    pd.testing.assert_frame_equal(
        original.loc[common_dates, feature_cols],
        modified.loc[common_dates, feature_cols],
    )


def test_target_price_is_shifted_close() -> None:
    df = make_ohlcv(20)
    result = add_target(df, horizon=1, target_type="price")
    assert result["target"].iloc[0] == pytest.approx(df["close"].iloc[1])
    assert pd.isna(result["target"].iloc[-1])


def test_target_return_matches_manual_calculation() -> None:
    df = make_ohlcv(20)
    result = add_target(df, horizon=3, target_type="return")
    expected = df["close"].iloc[5] / df["close"].iloc[2] - 1
    assert result["target"].iloc[2] == pytest.approx(expected)


def test_target_invalid_type_raises() -> None:
    df = make_ohlcv(10)
    with pytest.raises(ValueError):
        add_target(df, target_type="not_a_real_type")


def test_calendar_features_are_correct() -> None:
    df = make_ohlcv(10)
    result = add_calendar_features(df)
    assert result["day_of_week"].iloc[0] == df.index[0].dayofweek
    assert result["month"].iloc[0] == df.index[0].month


def test_build_features_raises_on_missing_columns() -> None:
    df = make_ohlcv(10).drop(columns=["volume"])
    with pytest.raises(ValueError):
        build_features(df)


def test_build_features_drops_nan_rows() -> None:
    df = make_ohlcv(100)
    result = build_features(df, horizon=1, target_type="price")
    assert result.isna().sum().sum() == 0
    assert len(result) < len(df)
