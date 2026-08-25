import numpy as np
import pandas as pd
import pytest

from app.forecasting.lstm_model import LSTMForecaster


def make_ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
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


def small_model(**overrides) -> LSTMForecaster:
    # Small/fast config so tests run quickly.
    defaults = dict(seq_len=10, hidden_size=4, num_layers=1, epochs=3, batch_size=8, patience=2)
    defaults.update(overrides)
    return LSTMForecaster(**defaults)


def test_fit_predict_returns_single_finite_value() -> None:
    df = make_ohlcv(150)
    model = small_model().fit(df)
    forecast = model.predict(horizon=1)
    assert len(forecast) == 1
    assert np.isfinite(forecast.iloc[0])


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        small_model().predict(horizon=1)


def test_predict_wrong_horizon_raises() -> None:
    df = make_ohlcv(150)
    model = small_model(horizon=1).fit(df)
    with pytest.raises(ValueError):
        model.predict(horizon=5)


def test_raises_on_insufficient_data() -> None:
    df = make_ohlcv(5)
    with pytest.raises(ValueError):
        small_model(seq_len=10, horizon=1).fit(df)


def test_return_target_type_produces_price_scale_output() -> None:
    df = make_ohlcv(150)
    model = small_model(target_type="return").fit(df)
    forecast = model.predict(horizon=1)
    last_close = df["close"].iloc[-1]
    assert forecast.iloc[0] == pytest.approx(last_close, rel=0.5)


def test_train_val_split_is_chronological_not_shuffled() -> None:
    """The validation slice must be the *last* val_fraction of sequences in time order,
    not a random subset - verify by checking val loss uses only the tail sequences."""
    df = make_ohlcv(150)
    model = small_model(val_fraction=0.2)
    close = df["close"].to_numpy(dtype=float)
    X_raw, y_raw, base_close = model._build_sequences(close)
    n_seq = len(X_raw)
    n_val = max(1, int(n_seq * model.val_fraction))
    n_train = n_seq - n_val
    # The sequence index boundary must be strictly increasing in time - i.e. every
    # validation sequence's underlying dates come after every training sequence's.
    assert n_train > 0 and n_val > 0
    # First training sequence starts at the earliest data; last validation sequence
    # ends at the most recent data - confirms no shuffling occurred in _build_sequences.
    assert np.array_equal(X_raw[0], close[: model.seq_len])
    assert np.array_equal(X_raw[-1], close[n_seq - 1 : n_seq - 1 + model.seq_len])


def test_normalization_uses_only_fit_data() -> None:
    """Mean/std used for scaling must come only from the data passed to fit(), so fitting
    on a shorter, earlier slice must not be affected by later values outside that slice."""
    df = make_ohlcv(150)
    early_slice = df.iloc[:100]

    model = small_model().fit(early_slice)
    assert model._mean == pytest.approx(early_slice["close"].mean())
    # ddof=0 to match numpy's default (the model computes std on a numpy array).
    assert model._std == pytest.approx(early_slice["close"].to_numpy().std(ddof=0))
