"""Train an LSTM forecaster and inspect its training/validation loss curve.

Per Stage 8 of the project plan: the validation split is chronological (the most recent
`val_fraction` of sequences), never shuffled, and training only happens after baselines
and traditional ML models are already established as points of comparison (see
scripts/train_model.py / scripts/backtest.py for those).

Usage:
    python scripts/train_lstm.py
    python scripts/train_lstm.py --seq-len 30 --hidden-size 64 --epochs 100
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

from app.evaluation.metrics import mae, rmse  # noqa: E402
from app.forecasting.lstm_model import LSTMForecaster  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "BTC-USD_1d.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "trained"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--target-type", choices=["price", "return"], default="price")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--holdout-days", type=int, default=30, help="Final N days held out entirely for a fair test-set evaluation, never used in fit().")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data, index_col="timestamp", parse_dates=True)

    train_df = df.iloc[: -args.holdout_days]
    test_df = df.iloc[-args.holdout_days - args.seq_len :]

    logger.info(
        "Training LSTM: seq_len=%d hidden_size=%d num_layers=%d epochs=%d train_rows=%d",
        args.seq_len,
        args.hidden_size,
        args.num_layers,
        args.epochs,
        len(train_df),
    )

    model = LSTMForecaster(
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        val_fraction=args.val_fraction,
        horizon=args.horizon,
        target_type=args.target_type,
    )
    model.fit(train_df)
    logger.info(
        "Trained %d epochs (best val loss=%.6f)",
        len(model.train_history),
        min(model.val_history),
    )

    # Held-out test evaluation: predict day-by-day over the last `holdout_days`, each
    # time using only data strictly before that day (never re-touching train_df's fit).
    # Reuse the single trained model's weights for rolling-window inference across the
    # held-out period (no retraining per day) - each step only needs the last seq_len
    # closes before that day, which is still exclusively past data, so no leakage.
    predictions, actuals = [], []
    for i in range(args.holdout_days):
        window_df = df.iloc[: len(train_df) + i]
        recent_close = window_df["close"].to_numpy(dtype=float)[-args.seq_len :]
        model._last_window = recent_close
        model._last_close = float(recent_close[-1])
        pred = model.predict(horizon=args.horizon).iloc[0]
        actual = df["close"].iloc[len(train_df) + i]
        predictions.append(pred)
        actuals.append(actual)

    pred_series = pd.Series(predictions)
    actual_series = pd.Series(actuals)
    logger.info("Held-out test: MAE=%.2f RMSE=%.2f", mae(actual_series, pred_series), rmse(actual_series, pred_series))

    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    axes[0].plot(model.train_history, label="train loss")
    axes[0].plot(model.val_history, label="val loss")
    axes[0].set_title("LSTM Training Curve (scaled MSE)")
    axes[0].legend()

    test_dates = df.index[len(train_df) : len(train_df) + args.holdout_days]
    axes[1].plot(test_dates, actuals, label="actual")
    axes[1].plot(test_dates, predictions, label="predicted", linestyle="--")
    axes[1].set_title(f"Held-out Test ({args.holdout_days} days, never seen during training)")
    axes[1].legend()

    plt.tight_layout()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = MODEL_DIR / "lstm_training_curve.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    logger.info("Saved training curve + held-out test plot to %s", plot_path)

    model_path = MODEL_DIR / "lstm_model.joblib"
    model.save(model_path)
    logger.info("Saved trained model to %s", model_path)


if __name__ == "__main__":
    main()
