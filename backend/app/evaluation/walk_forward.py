"""Walk-forward (expanding-window) backtesting.

This is the ONLY sanctioned way to evaluate a forecasting model in this project. Never
use `sklearn.model_selection.train_test_split` (it shuffles by default, and even with
shuffle=False it produces just one split) - time-series evaluation requires repeatedly
training on an expanding window of the past and testing on the period immediately after
it, then moving forward. See Section 28, rule 1 of the project plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from app.evaluation.metrics import directional_accuracy, mae, mape, rmse
from app.forecasting.features import build_features, feature_columns


@dataclass
class FoldResult:
    train_end: pd.Timestamp
    test_date: pd.Timestamp
    actual: float
    predicted: float
    previous_actual: float


@dataclass
class WalkForwardResult:
    model_name: str
    horizon: int
    target_type: str
    folds: list[FoldResult] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "train_end": [f.train_end for f in self.folds],
                "test_date": [f.test_date for f in self.folds],
                "actual": [f.actual for f in self.folds],
                "predicted": [f.predicted for f in self.folds],
            }
        )

    def residuals(self) -> pd.Series:
        """predicted - actual for every fold, in the form `prediction_interval()` expects."""
        frame = self.to_frame()
        return frame["predicted"] - frame["actual"]

    def metrics(self) -> dict[str, float]:
        frame = self.to_frame()
        actual = frame["actual"]
        predicted = frame["predicted"]
        previous = pd.Series([f.previous_actual for f in self.folds])
        return {
            "mae": mae(actual, predicted),
            "rmse": rmse(actual, predicted),
            "mape": mape(actual, predicted),
            "directional_accuracy": directional_accuracy(actual, predicted, previous),
            "n_folds": len(self.folds),
        }


def walk_forward_validate(
    df: pd.DataFrame,
    model_factory: Callable[[], object],
    model_name: str,
    horizon: int = 1,
    target_type: str = "price",
    n_test_folds: int = 60,
    min_train_size: int = 200,
    step: int = 1,
) -> WalkForwardResult:
    """Expanding-window walk-forward validation.

    For each fold: fit a fresh model instance on all data strictly before the test
    point, predict `horizon` steps ahead, and compare against the actual value once it's
    known. The training window only ever grows forward in time and is never shuffled.

    `model_factory` objects must implement `.fit(df: pd.DataFrame)` and
    `.predict(horizon: int) -> pd.Series` (see BaseForecaster). Feature-based models
    read `feature_columns(features)` / `features["target"]` from the built feature
    matrix themselves inside their own `.fit()`.
    """
    features = build_features(df, horizon=horizon, target_type=target_type)
    if len(features) < min_train_size + n_test_folds:
        raise ValueError(
            f"Not enough data: need at least {min_train_size + n_test_folds} feature rows, "
            f"got {len(features)}"
        )

    result = WalkForwardResult(model_name=model_name, horizon=horizon, target_type=target_type)

    test_start = len(features) - n_test_folds
    for i in range(test_start, len(features), step):
        train_features = features.iloc[:i]
        if len(train_features) < min_train_size:
            continue

        test_row = features.iloc[i]
        train_end_date = train_features.index[-1]
        raw_train_df = df.loc[df.index <= train_end_date]

        model = model_factory()
        model.fit(raw_train_df)
        prediction = model.predict(horizon=horizon)
        predicted_value = float(prediction.iloc[-1])

        if target_type == "price":
            actual_value = float(test_row["target"])
        else:
            # target is a return; convert both actual and predicted back to price terms
            # so metrics stay comparable across models regardless of target_type.
            base_close = float(df.loc[train_end_date, "close"])
            actual_value = base_close * (1 + float(test_row["target"]))
            predicted_value = base_close * (1 + predicted_value)

        result.folds.append(
            FoldResult(
                train_end=train_end_date,
                test_date=test_row.name,
                actual=actual_value,
                predicted=predicted_value,
                previous_actual=float(df.loc[train_end_date, "close"]),
            )
        )

    return result
