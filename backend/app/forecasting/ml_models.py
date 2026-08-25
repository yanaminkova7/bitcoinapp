"""Traditional ML forecasters (Linear Regression, Random Forest, Gradient Boosting).

Each model is trained to predict a fixed number of steps ahead (`horizon`, set at
construction time) directly from engineered features - this is "direct" multi-step
forecasting, not iterative/autoregressive. `predict()` therefore only supports the
horizon the model was actually trained for; it does not fabricate a day-by-day path.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from app.forecasting.base import BaseForecaster
from app.forecasting.features import build_features, compute_features, feature_columns


class MLForecaster(BaseForecaster):
    name = "ml_base"

    def __init__(self, estimator: RegressorMixin, horizon: int = 1, target_type: str = "price") -> None:
        self.estimator = estimator
        self.horizon = horizon
        self.target_type = target_type
        self._feature_cols: list[str] | None = None
        self._last_close: float | None = None
        self._last_feature_row: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> "MLForecaster":
        train_data = build_features(df, horizon=self.horizon, target_type=self.target_type)
        self._feature_cols = feature_columns(train_data)
        X = train_data[self._feature_cols]
        y = train_data["target"]
        self.estimator.fit(X, y)

        # Features for the most recent available date, used at predict() time - these
        # rows have no target yet (that's what we're forecasting), so build_features'
        # dropna would discard them; compute_features keeps them.
        all_features = compute_features(df)
        self._last_feature_row = all_features[self._feature_cols].iloc[[-1]]
        self._last_close = float(df["close"].iloc[-1])
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._feature_cols is None or self._last_feature_row is None:
            raise RuntimeError("Call fit() before predict()")
        if horizon != self.horizon:
            raise ValueError(
                f"This model was trained to forecast {self.horizon} step(s) ahead (direct "
                f"multi-step, not iterative), so it cannot predict horizon={horizon}. "
                f"Construct a new instance with horizon={horizon} and fit it instead."
            )

        raw_prediction = float(self.estimator.predict(self._last_feature_row)[0])
        if self.target_type == "return":
            price_prediction = self._last_close * (1 + raw_prediction)
        else:
            price_prediction = raw_prediction

        return pd.Series([price_prediction])


class LinearRegressionForecaster(MLForecaster):
    name = "linear_regression"

    def __init__(self, horizon: int = 1, target_type: str = "price") -> None:
        super().__init__(LinearRegression(), horizon=horizon, target_type=target_type)


class RandomForestForecaster(MLForecaster):
    name = "random_forest"

    def __init__(
        self,
        horizon: int = 1,
        target_type: str = "price",
        n_estimators: int = 200,
        max_depth: int | None = 6,
        random_state: int = 42,
    ) -> None:
        estimator = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        super().__init__(estimator, horizon=horizon, target_type=target_type)


class GradientBoostingForecaster(MLForecaster):
    name = "gradient_boosting"

    def __init__(
        self,
        horizon: int = 1,
        target_type: str = "price",
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        estimator = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        super().__init__(estimator, horizon=horizon, target_type=target_type)
