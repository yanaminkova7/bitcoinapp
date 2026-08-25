"""Empirical, residual-based prediction intervals.

Approach: take a model's historical walk-forward residuals (predicted - actual) from a
held-out calibration period, and use their empirical quantiles to build an interval
around a new point forecast. This is inspired by conformal prediction but does NOT carry
its usual distribution-free coverage guarantee, because that guarantee relies on
exchangeability, which time-series data with trends and volatility clustering (see
notebooks/exploratory_analysis.ipynb) does not strictly satisfy. Treat the resulting
interval as an honest, data-driven range reflecting this model's actual historical
error - not a calibrated probability guarantee.

Quantiles are taken on the *signed* residual distribution (not absolute value), so the
interval can be asymmetric - which matters here: our own backtests show models tend to
underpredict sharply during rallies more than they overpredict during calm periods (fat
right tail in price moves), and a symmetric interval would understate that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PredictionInterval:
    point_forecast: float
    lower: float
    upper: float
    confidence: float

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError(f"lower ({self.lower}) must not exceed upper ({self.upper})")


def prediction_interval(point_forecast: float, residuals: pd.Series, confidence: float = 0.9) -> PredictionInterval:
    """Build an interval around `point_forecast` from historical residuals.

    `residuals` must be `predicted - actual` values from past forecasts of the same
    model at the same horizon (e.g. from a WalkForwardResult), not from the point being
    forecast now. Requires at least 20 residuals - too few and empirical quantiles are
    unreliable.
    """
    if len(residuals) < 20:
        raise ValueError(f"Need at least 20 historical residuals to calibrate an interval, got {len(residuals)}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

    alpha = 1 - confidence
    # actual = predicted - residual, so applying the historical residual distribution to
    # a new point forecast: subtracting a large residual gives the low end of the range,
    # subtracting a small/negative residual gives the high end.
    high_residual_quantile = float(np.quantile(residuals, 1 - alpha / 2))
    low_residual_quantile = float(np.quantile(residuals, alpha / 2))

    lower = point_forecast - high_residual_quantile
    upper = point_forecast - low_residual_quantile

    return PredictionInterval(point_forecast=point_forecast, lower=lower, upper=upper, confidence=confidence)
