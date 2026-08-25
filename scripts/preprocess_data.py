"""Validate a raw OHLCV CSV and write a cleaned, feature-ready copy to data/processed/.

This is a defensive re-validation pass (raw files may be re-downloaded or hand-edited
later) plus a small set of derived columns that downstream EDA and feature engineering
both need. It does not add lookahead features — every derived column here only uses
information available at or before its own row.

Usage:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --input data/raw/BTC-USD_1d.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(RAW_DIR / "BTC-USD_1d.csv"),
        help="Path to a raw OHLCV CSV (default: data/raw/BTC-USD_1d.csv)",
    )
    parser.add_argument("--output", default=None, help="Output CSV path (default: mirrored in data/processed/)")
    return parser.parse_args()


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}. Run scripts/download_data.py first.")
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    missing = set(OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw file is missing required columns: {sorted(missing)}")
    return df


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    before = len(df)
    df = df[~df.index.duplicated(keep="last")]
    if len(df) != before:
        logger.warning("Dropped %d duplicate timestamp(s)", before - len(df))

    df = df.sort_index()

    for col in OHLCV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_missing = int(df[OHLCV_COLUMNS].isna().sum().sum())
    if n_missing:
        logger.warning("Found %d missing/invalid value(s); forward-filling", n_missing)
        df[OHLCV_COLUMNS] = df[OHLCV_COLUMNS].ffill()
        df = df.dropna(subset=OHLCV_COLUMNS)

    invalid_mask = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | (df["volume"] < 0)
    if invalid_mask.any():
        logger.warning("Dropping %d row(s) with invalid prices/volume", int(invalid_mask.sum()))
        df = df[~invalid_mask]

    ohlc_consistent = (df["high"] >= df[["open", "close", "low"]].max(axis=1)) & (
        df["low"] <= df[["open", "close", "high"]].min(axis=1)
    )
    if not ohlc_consistent.all():
        logger.warning("Dropping %d row(s) with inconsistent OHLC relationships", int((~ohlc_consistent).sum()))
        df = df[ohlc_consistent]

    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Uses only the current and prior row's close - no lookahead.
    df["daily_return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def save(df: pd.DataFrame, input_path: Path, output: str | None) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(output) if output else PROCESSED_DIR / input_path.name
    df.to_csv(out_path)
    logger.info("Saved %d rows to %s", len(df), out_path)
    return out_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    try:
        raw_df = load_raw(input_path)
        clean_df = validate_and_clean(raw_df)
        final_df = add_derived_columns(clean_df)
        save(final_df, input_path, args.output)
    except Exception:
        logger.exception("Preprocessing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
