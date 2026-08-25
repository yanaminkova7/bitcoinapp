from __future__ import annotations

import pandas as pd

from app.forecasting.base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """Tomorrow's price = today's price, repeated for every step in the horizon."""

    name = "naive"

    def __init__(self) -> None:
        self._last_value: float | None = None

    def fit(self, df: pd.DataFrame) -> "NaiveForecaster":
        self._last_value = float(df["close"].iloc[-1])
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._last_value is None:
            raise RuntimeError("Call fit() before predict()")
        return pd.Series([self._last_value] * horizon)


class MovingAverageForecaster(BaseForecaster):
    """Tomorrow's price = the trailing moving average of the last `window` observations."""

    name = "moving_average"

    def __init__(self, window: int = 7) -> None:
        self.window = window
        self._average: float | None = None

    def fit(self, df: pd.DataFrame) -> "MovingAverageForecaster":
        close = df["close"]
        if len(close) < self.window:
            raise ValueError(f"Series has {len(close)} points, fewer than window={self.window}")
        self._average = float(close.iloc[-self.window :].mean())
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._average is None:
            raise RuntimeError("Call fit() before predict()")
        return pd.Series([self._average] * horizon)
