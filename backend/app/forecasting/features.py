"""Reusable, leakage-safe feature engineering for Bitcoin price forecasting.

Every function here computes each row using only that row's own value and values from
*earlier* rows (via `.shift()` / backward-looking `.rolling()`). None of them look ahead.
`add_target()` is the one intentional exception - a supervised-learning label is, by
definition, a future value - but it is kept clearly separate from the feature columns so
it is never accidentally fed back in as a feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def add_price_features(df: pd.DataFrame, lags: tuple[int, ...] = (1, 2, 3, 7, 14)) -> pd.DataFrame:
    """All values are computed from `close` shifted by at least 1, i.e. every feature at
    row t uses only data through t-1 - today's own bar is never used, only past ones."""
    df = df.copy()
    prior_close = df["close"].shift(1)

    for lag in lags:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        # N-day return ending the day before row t (never touches row t's own close).
        df[f"return_lag_{lag}"] = prior_close.pct_change(lag)

    for window in (7, 14, 30):
        df[f"rolling_mean_{window}"] = prior_close.rolling(window).mean()
        df[f"rolling_std_{window}"] = prior_close.rolling(window).std()
        df[f"rolling_min_{window}"] = prior_close.rolling(window).min()
        df[f"rolling_max_{window}"] = prior_close.rolling(window).max()

    return df


def add_momentum_features(df: pd.DataFrame, rsi_window: int = 14) -> pd.DataFrame:
    df = df.copy()
    prior_close = df["close"].shift(1)

    delta = prior_close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_window).mean()
    avg_loss = loss.rolling(rsi_window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema_12 = prior_close.ewm(span=12, adjust=False).mean()
    ema_26 = prior_close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_histogram"] = macd_line - signal_line

    df["roc_10"] = prior_close.pct_change(10)

    return df


def add_volatility_features(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    prior_close = df["close"].shift(1)
    prior_high = df["high"].shift(1)
    prior_low = df["low"].shift(1)

    df["rolling_volatility_14"] = prior_close.pct_change().rolling(window).std()

    prior_prior_close = prior_close.shift(1)
    true_range = pd.concat(
        [
            prior_high - prior_low,
            (prior_high - prior_prior_close).abs(),
            (prior_low - prior_prior_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = true_range.rolling(window).mean()

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    prior_volume = df["volume"].shift(1)

    df["volume_change"] = prior_volume.pct_change()
    df["rolling_volume_7"] = prior_volume.rolling(7).mean()
    df["volume_ma_30"] = prior_volume.rolling(30).mean()

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    return df


def add_target(df: pd.DataFrame, horizon: int = 1, target_type: str = "price") -> pd.DataFrame:
    """Add the supervised-learning label. This is the one place future information is
    intentionally used - `target` for row t is the close price (or return) at t+horizon.
    Never use a `target` column as a model input feature."""
    df = df.copy()
    if target_type == "price":
        df["target"] = df["close"].shift(-horizon)
    elif target_type == "return":
        df["target"] = df["close"].shift(-horizon) / df["close"] - 1
    else:
        raise ValueError(f"Unknown target_type: {target_type!r}")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the feature pipeline only - no target column, no dropping of rows. The most
    recent row(s) will have NaN features during the warm-up window, and that is expected:
    callers that need a clean training matrix should use `build_features` instead; callers
    that need the latest row for live inference (which never has a target yet) use this."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    result = df.copy()
    result = add_price_features(result)
    result = add_momentum_features(result)
    result = add_volatility_features(result)
    result = add_volume_features(result)
    result = add_calendar_features(result)
    return result


def build_features(df: pd.DataFrame, horizon: int = 1, target_type: str = "price") -> pd.DataFrame:
    """Full training matrix: features plus target, with any row containing a NaN feature
    (the initial warm-up window) or NaN target (the final `horizon` rows) dropped, since
    those rows can't be used for training or evaluation."""
    result = compute_features(df)
    result = add_target(result, horizon=horizon, target_type=target_type)
    return result.dropna()


def feature_columns(df_with_features: pd.DataFrame) -> list[str]:
    """Given a features (+ optionally target) DataFrame, return just the model input
    columns - excludes raw OHLCV, the target/label, and `daily_return`/`log_return`
    (preprocessing artifacts computed from *today's own* close, which breaks the
    shift-by-1 convention every feature in this module otherwise follows - use
    `return_lag_1` instead, which is the equivalent value already correctly lagged)."""
    exclude = set(REQUIRED_COLUMNS) | {"target", "daily_return", "log_return"}
    return [c for c in df_with_features.columns if c not in exclude]
