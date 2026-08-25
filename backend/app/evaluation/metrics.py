from __future__ import annotations

import numpy as np
import pandas as pd


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.mean(np.abs(actual.to_numpy() - predicted.to_numpy())))


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.sqrt(np.mean((actual.to_numpy() - predicted.to_numpy()) ** 2)))


def mape(actual: pd.Series, predicted: pd.Series) -> float:
    actual_arr = actual.to_numpy()
    predicted_arr = predicted.to_numpy()
    nonzero = actual_arr != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((actual_arr[nonzero] - predicted_arr[nonzero]) / actual_arr[nonzero])) * 100)


def directional_accuracy(actual: pd.Series, predicted: pd.Series, previous_actual: pd.Series) -> float:
    """Fraction of steps where the predicted direction of change (vs. the previous
    actual value) matches the actual direction of change."""
    actual_direction = np.sign(actual.to_numpy() - previous_actual.to_numpy())
    predicted_direction = np.sign(predicted.to_numpy() - previous_actual.to_numpy())
    return float(np.mean(actual_direction == predicted_direction))
