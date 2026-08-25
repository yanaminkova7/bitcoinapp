from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import pandas as pd


class BaseForecaster(ABC):
    """Common interface every forecasting model implements, so models can be swapped
    without changing the code that calls them."""

    name: str = "base"

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseForecaster":
        """Fit on a chronologically ordered OHLCV DataFrame (columns: open, high, low,
        close, volume). Simple models may only look at `close`; feature-based models use
        the full frame to compute their own inputs."""

    @abstractmethod
    def predict(self, horizon: int) -> pd.Series:
        """Return `horizon` forecasted values for the periods immediately after
        the data the model was fit on."""

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "BaseForecaster":
        return joblib.load(path)
