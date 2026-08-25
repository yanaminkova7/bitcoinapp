"""Run a walk-forward backtest for one model, storing every prediction and plotting
actual vs. predicted price and prediction error over time.

Per the project's ML rules, this is chronological expanding-window validation only -
never a random train/test split.

Usage:
    python scripts/backtest.py --model random_forest --horizon 1
    python scripts/backtest.py --model naive --horizon 7 --n-test-folds 90
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.evaluation.walk_forward import walk_forward_validate  # noqa: E402
from app.forecasting.baselines import MovingAverageForecaster, NaiveForecaster  # noqa: E402
from app.forecasting.lstm_model import LSTMForecaster  # noqa: E402
from app.forecasting.ml_models import (  # noqa: E402
    GradientBoostingForecaster,
    LinearRegressionForecaster,
    RandomForestForecaster,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "BTC-USD_1d.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "trained" / "backtests"

MODEL_FACTORIES = {
    "naive": lambda h: NaiveForecaster(),
    "moving_average_7": lambda h: MovingAverageForecaster(window=7),
    "linear_regression": lambda h: LinearRegressionForecaster(horizon=h),
    "random_forest": lambda h: RandomForestForecaster(horizon=h),
    "gradient_boosting": lambda h: GradientBoostingForecaster(horizon=h),
    # Retrains at every fold, so this is much slower than the other models here -
    # expect minutes rather than seconds for the default n_test_folds.
    "lstm": lambda h: LSTMForecaster(horizon=h, epochs=30, patience=6),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODEL_FACTORIES.keys()), required=True)
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--target-type", choices=["price", "return"], default="price")
    parser.add_argument("--n-test-folds", type=int, default=90)
    parser.add_argument("--min-train-size", type=int, default=200)
    return parser.parse_args()


def plot_results(frame: pd.DataFrame, model_name: str, horizon: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    error = frame["predicted"] - frame["actual"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(frame["test_date"], frame["actual"], label="actual", linewidth=1.5)
    axes[0].plot(frame["test_date"], frame["predicted"], label="predicted", linewidth=1.2, linestyle="--")
    axes[0].set_title(f"{model_name} - Actual vs Predicted (horizon={horizon})")
    axes[0].legend()

    axes[1].bar(frame["test_date"], error, color="crimson", alpha=0.6)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Prediction Error (predicted - actual)")

    plt.tight_layout()
    plot_path = output_dir / f"{model_name}_h{horizon}.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    logger.info("Saved plot to %s", plot_path)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data, index_col="timestamp", parse_dates=True)
    factory = MODEL_FACTORIES[args.model]

    logger.info(
        "Backtesting %s (horizon=%d, target_type=%s, n_test_folds=%d)",
        args.model,
        args.horizon,
        args.target_type,
        args.n_test_folds,
    )
    result = walk_forward_validate(
        df,
        lambda: factory(args.horizon),
        args.model,
        horizon=args.horizon,
        target_type=args.target_type,
        n_test_folds=args.n_test_folds,
        min_train_size=args.min_train_size,
    )

    frame = result.to_frame()
    metrics = result.metrics()
    logger.info(
        "MAE=%.2f RMSE=%.2f MAPE=%.2f%% directional_accuracy=%.1f%% (%d folds)",
        metrics["mae"],
        metrics["rmse"],
        metrics["mape"],
        metrics["directional_accuracy"] * 100,
        metrics["n_folds"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = OUTPUT_DIR / f"{args.model}_h{args.horizon}_predictions.csv"
    frame.to_csv(predictions_path, index=False)
    logger.info("Saved %d predictions to %s", len(frame), predictions_path)

    plot_results(frame, args.model, args.horizon, OUTPUT_DIR)


if __name__ == "__main__":
    main()
