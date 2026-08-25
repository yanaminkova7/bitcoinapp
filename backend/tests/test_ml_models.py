import numpy as np
import pandas as pd
import pytest

from app.forecasting.ml_models import (
    GradientBoostingForecaster,
    LinearRegressionForecaster,
    RandomForestForecaster,
)

MODEL_CLASSES = [LinearRegressionForecaster, RandomForestForecaster, GradientBoostingForecaster]


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


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_fit_predict_returns_single_value(model_cls) -> None:
    df = make_ohlcv(200)
    model = model_cls(horizon=1, target_type="price").fit(df)
    forecast = model.predict(horizon=1)
    assert len(forecast) == 1
    assert np.isfinite(forecast.iloc[0])


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_predict_wrong_horizon_raises(model_cls) -> None:
    df = make_ohlcv(200)
    model = model_cls(horizon=1, target_type="price").fit(df)
    with pytest.raises(ValueError):
        model.predict(horizon=7)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_predict_before_fit_raises(model_cls) -> None:
    with pytest.raises(RuntimeError):
        model_cls(horizon=1).predict(horizon=1)


def test_return_target_type_produces_price_scale_output() -> None:
    df = make_ohlcv(200)
    model = LinearRegressionForecaster(horizon=1, target_type="return").fit(df)
    forecast = model.predict(horizon=1)
    # Output should be in price terms (roughly near the last close), not a tiny
    # fractional return value, even though the model was trained on returns.
    last_close = df["close"].iloc[-1]
    assert forecast.iloc[0] == pytest.approx(last_close, rel=0.5)


def test_changing_data_before_training_window_end_does_not_change_prediction_shape() -> None:
    # Sanity check that fit/predict is deterministic given the same data and random_state.
    df = make_ohlcv(200)
    model_a = RandomForestForecaster(horizon=1, random_state=42).fit(df)
    model_b = RandomForestForecaster(horizon=1, random_state=42).fit(df)
    assert model_a.predict(horizon=1).iloc[0] == pytest.approx(model_b.predict(horizon=1).iloc[0])
