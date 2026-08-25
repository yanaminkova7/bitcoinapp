"""Run walk-forward validation for every forecasting model and record the results.

This is the project's model experiment system (Stage 7): every model - baseline or ML -
is evaluated the same way (expanding-window walk-forward, never shuffled) so results are
directly comparable, and every run is appended to models/trained/experiment_results.csv
for a durable record of what was tried and how it performed.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --n-test-folds 60 --min-train-size 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.evaluation.walk_forward import walk_forward_validate  # noqa: E402
from app.forecasting.baselines import MovingAverageForecaster, NaiveForecaster  # noqa: E402
from app.forecasting.ml_models import (  # noqa: E402
    GradientBoostingForecaster,
    LinearRegressionForecaster,
    RandomForestForecaster,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "BTC-USD_1d.csv"
RESULTS_PATH = PROJECT_ROOT / "models" / "trained" / "experiment_results.csv"

MODEL_FACTORIES = {
    "naive": lambda: NaiveForecaster(),
    "moving_average_7": lambda: MovingAverageForecaster(window=7),
    "linear_regression": lambda: LinearRegressionForecaster(horizon=1),
    "random_forest": lambda: RandomForestForecaster(horizon=1),
    "gradient_boosting": lambda: GradientBoostingForecaster(horizon=1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--n-test-folds", type=int, default=60)
    parser.add_argument("--min-train-size", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data, index_col="timestamp", parse_dates=True)

    run_timestamp = datetime.now(timezone.utc).isoformat()
    rows = []

    for name, factory in MODEL_FACTORIES.items():
        logger.info("Running walk-forward validation: %s", name)
        result = walk_forward_validate(
            df,
            factory,
            name,
            horizon=args.horizon,
            n_test_folds=args.n_test_folds,
            min_train_size=args.min_train_size,
        )
        metrics = result.metrics()
        folds_frame = result.to_frame()
        rows.append(
            {
                "timestamp": run_timestamp,
                "model": name,
                "horizon": args.horizon,
                "train_period_start": str(df.index[0].date()),
                "validation_period_start": str(folds_frame["test_date"].min().date()),
                "validation_period_end": str(folds_frame["test_date"].max().date()),
                "mae": round(metrics["mae"], 4),
                "rmse": round(metrics["rmse"], 4),
                "mape": round(metrics["mape"], 4),
                "directional_accuracy": round(metrics["directional_accuracy"], 4),
                "n_folds": metrics["n_folds"],
            }
        )
        logger.info(
            "%s: MAE=%.2f RMSE=%.2f MAPE=%.2f%% dir_acc=%.1f%%",
            name,
            metrics["mae"],
            metrics["rmse"],
            metrics["mape"],
            metrics["directional_accuracy"] * 100,
        )

    results_df = pd.DataFrame(rows).sort_values("mae")

    print("\n=== Model comparison (sorted by MAE) ===")
    print(results_df[["model", "mae", "rmse", "mape", "directional_accuracy", "n_folds"]].to_string(index=False))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = not RESULTS_PATH.exists()
    results_df.to_csv(RESULTS_PATH, mode="a", header=header, index=False)
    logger.info("Appended results to %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
