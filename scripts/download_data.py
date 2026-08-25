"""Download historical Bitcoin OHLCV data and store it as raw CSV.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --period 5y --interval 1d
    python scripts/download_data.py --start 2019-01-01 --end 2024-01-01
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTC-USD", help="Ticker symbol (default: BTC-USD)")
    parser.add_argument(
        "--period",
        default="5y",
        help="Lookback period, e.g. 1y, 5y, max (ignored if --start is given)",
    )
    parser.add_argument("--start", default=None, help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date, YYYY-MM-DD (default: today)")
    parser.add_argument("--interval", default="1d", help="Bar interval (default: 1d)")
    parser.add_argument("--output", default=None, help="Output CSV path (default: auto-named in data/raw/)")
    return parser.parse_args()


def download(symbol: str, period: str, start: str | None, end: str | None, interval: str) -> pd.DataFrame:
    logger.info("Downloading %s (%s) ...", symbol, start or period)
    try:
        if start:
            df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
        else:
            df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    except Exception:
        logger.exception("Download request failed for %s", symbol)
        raise

    if df is None or df.empty:
        logger.error("No data returned for %s", symbol)
        raise ValueError(f"No data returned for symbol '{symbol}'")

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df.index.name = "timestamp"

    keep_cols = [c for c in OHLCV_COLUMNS if c in df.columns]
    missing_cols = set(OHLCV_COLUMNS) - set(keep_cols)
    if missing_cols:
        logger.warning("Missing expected columns: %s", sorted(missing_cols))
    df = df[keep_cols]

    before = len(df)
    df = df[~df.index.duplicated(keep="last")]
    if len(df) != before:
        logger.warning("Dropped %d duplicate timestamp(s)", before - len(df))

    df = df.sort_index()

    for col in keep_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_missing = int(df[keep_cols].isna().sum().sum())
    if n_missing:
        logger.warning("Found %d missing/invalid value(s); forward-filling", n_missing)
        df[keep_cols] = df[keep_cols].ffill()
        remaining = int(df[keep_cols].isna().sum().sum())
        if remaining:
            logger.warning("Dropping %d row(s) still missing values after forward-fill", remaining)
            df = df.dropna(subset=keep_cols)

    invalid_price_mask = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if invalid_price_mask.any():
        logger.warning("Dropping %d row(s) with non-positive prices", int(invalid_price_mask.sum()))
        df = df[~invalid_price_mask]

    return df


def save(df: pd.DataFrame, symbol: str, interval: str, output: str | None) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if output:
        out_path = Path(output)
    else:
        safe_symbol = symbol.replace("/", "-")
        out_path = RAW_DATA_DIR / f"{safe_symbol}_{interval}.csv"
    df.to_csv(out_path)
    logger.info("Saved %d rows to %s", len(df), out_path)
    return out_path


def main() -> None:
    args = parse_args()
    try:
        raw_df = download(args.symbol, args.period, args.start, args.end, args.interval)
        clean_df = clean(raw_df)
        save(clean_df, args.symbol, args.interval, args.output)
    except Exception:
        logger.exception("Data download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
