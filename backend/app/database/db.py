"""SQLite persistence for market data, predictions, and model run metrics.

Predictions are keyed on (model, horizon, target_date): re-generating a forecast for a
date/model/horizon already stored overwrites it with the latest estimate rather than
accumulating duplicates from every rerun. This trades away a full history of "how did
the forecast for this date change as we got closer" in exchange for a bounded table size
appropriate for an interactive app; `scripts/backtest.py`'s CSV output is the source of
truth for full walk-forward prediction history instead.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from app.database.schema import ALL_TABLES

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "bitcoin_forecasting.db"


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        for statement in ALL_TABLES:
            conn.execute(statement)
        conn.commit()


@contextmanager
def get_connection(db_path: Path = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def save_market_data(conn: sqlite3.Connection, df: pd.DataFrame, symbol: str) -> int:
    """Upsert OHLCV rows. `df` must be indexed by timestamp with open/high/low/close/volume
    columns. Returns the number of rows written."""
    rows = [
        (idx.isoformat(), symbol, float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"]))
        for idx, r in df.iterrows()
    ]
    conn.executemany(
        """
        INSERT INTO market_data (timestamp, symbol, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(timestamp, symbol) DO UPDATE SET
            open=excluded.open, high=excluded.high, low=excluded.low,
            close=excluded.close, volume=excluded.volume
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def save_prediction(
    conn: sqlite3.Connection,
    model: str,
    horizon: int,
    target_date: str,
    predicted_value: float,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    confidence: float | None = None,
    generated_at: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO predictions (generated_at, model, horizon, target_date, predicted_value, lower_bound, upper_bound, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model, horizon, target_date) DO UPDATE SET
            generated_at=excluded.generated_at, predicted_value=excluded.predicted_value,
            lower_bound=excluded.lower_bound, upper_bound=excluded.upper_bound,
            confidence=excluded.confidence
        """,
        (
            generated_at or datetime.now(timezone.utc).isoformat(),
            model,
            horizon,
            target_date,
            predicted_value,
            lower_bound,
            upper_bound,
            confidence,
        ),
    )
    conn.commit()


def backfill_actuals(conn: sqlite3.Connection, symbol_close: pd.Series) -> int:
    """Fill in `actual_value` for any past predictions whose target_date now has a known
    close price, so forecast error can be tracked over time (Stage 17 monitoring)."""
    rows = conn.execute("SELECT model, horizon, target_date FROM predictions WHERE actual_value IS NULL").fetchall()
    updated = 0
    for model, horizon, target_date in rows:
        ts = pd.Timestamp(target_date)
        if ts in symbol_close.index:
            conn.execute(
                "UPDATE predictions SET actual_value = ? WHERE model = ? AND horizon = ? AND target_date = ?",
                (float(symbol_close.loc[ts]), model, horizon, target_date),
            )
            updated += 1
    conn.commit()
    return updated


def save_model_run(
    conn: sqlite3.Connection,
    model: str,
    horizon: int,
    train_period_start: str,
    validation_period_start: str,
    validation_period_end: str,
    mae: float,
    rmse: float,
    mape: float | None,
    directional_accuracy: float,
    n_folds: int,
    run_at: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO model_runs (run_at, model, horizon, train_period_start, validation_period_start,
                                 validation_period_end, mae, rmse, mape, directional_accuracy, n_folds)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_at or datetime.now(timezone.utc).isoformat(),
            model,
            horizon,
            train_period_start,
            validation_period_start,
            validation_period_end,
            mae,
            rmse,
            mape,
            directional_accuracy,
            n_folds,
        ),
    )
    conn.commit()


def get_predictions(conn: sqlite3.Connection, model: str | None = None) -> pd.DataFrame:
    query = "SELECT * FROM predictions"
    params: tuple = ()
    if model:
        query += " WHERE model = ?"
        params = (model,)
    query += " ORDER BY target_date"
    return pd.read_sql_query(query, conn, params=params)


def get_model_runs(conn: sqlite3.Connection, model: str | None = None, limit: int = 100) -> pd.DataFrame:
    query = "SELECT * FROM model_runs"
    params: tuple = ()
    if model:
        query += " WHERE model = ?"
        params = (model,)
    query += " ORDER BY run_at DESC LIMIT ?"
    params = params + (limit,)
    return pd.read_sql_query(query, conn, params=params)
