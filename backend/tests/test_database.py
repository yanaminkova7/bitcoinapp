from pathlib import Path

import pandas as pd
import pytest

from app.database.db import (
    backfill_actuals,
    get_connection,
    get_model_runs,
    get_predictions,
    save_market_data,
    save_model_run,
    save_prediction,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


def make_ohlcv(n: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1000.0 + i for i in range(n)],
        },
        index=dates,
    )


def test_save_and_query_market_data(db_path: Path) -> None:
    df = make_ohlcv(5)
    with get_connection(db_path) as conn:
        n_written = save_market_data(conn, df, symbol="BTC-USD")
        assert n_written == 5
        rows = conn.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
        assert rows == 5


def test_save_market_data_upserts_on_conflict(db_path: Path) -> None:
    df = make_ohlcv(3)
    with get_connection(db_path) as conn:
        save_market_data(conn, df, symbol="BTC-USD")
        df_updated = df.copy()
        df_updated["close"] = 999.0
        save_market_data(conn, df_updated, symbol="BTC-USD")
        rows = conn.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
        closes = conn.execute("SELECT close FROM market_data").fetchall()
        assert rows == 3  # no duplicates
        assert all(c[0] == 999.0 for c in closes)  # values updated, not duplicated


def test_save_prediction_and_query(db_path: Path) -> None:
    with get_connection(db_path) as conn:
        save_prediction(conn, "naive", 1, "2024-01-10", predicted_value=50000.0, lower_bound=48000.0, upper_bound=52000.0, confidence=0.9)
        df = get_predictions(conn, model="naive")
        assert len(df) == 1
        assert df.iloc[0]["predicted_value"] == 50000.0
        assert df.iloc[0]["actual_value"] is None or pd.isna(df.iloc[0]["actual_value"])


def test_save_prediction_overwrites_same_key(db_path: Path) -> None:
    with get_connection(db_path) as conn:
        save_prediction(conn, "naive", 1, "2024-01-10", predicted_value=50000.0)
        save_prediction(conn, "naive", 1, "2024-01-10", predicted_value=51000.0)
        df = get_predictions(conn, model="naive")
        assert len(df) == 1
        assert df.iloc[0]["predicted_value"] == 51000.0


def test_backfill_actuals_fills_known_close_prices(db_path: Path) -> None:
    with get_connection(db_path) as conn:
        save_prediction(conn, "naive", 1, "2024-01-02", predicted_value=100.0)
        save_prediction(conn, "naive", 1, "2099-01-01", predicted_value=200.0)  # no actual available

        close = pd.Series([100.5, 101.5], index=pd.date_range("2024-01-01", periods=2, freq="D"))
        updated = backfill_actuals(conn, close)

        assert updated == 1
        df = get_predictions(conn, model="naive")
        row = df[df["target_date"] == "2024-01-02"].iloc[0]
        assert row["actual_value"] == pytest.approx(101.5)
        row2 = df[df["target_date"] == "2099-01-01"].iloc[0]
        assert pd.isna(row2["actual_value"])


def test_save_and_query_model_runs(db_path: Path) -> None:
    with get_connection(db_path) as conn:
        save_model_run(
            conn,
            model="random_forest",
            horizon=1,
            train_period_start="2020-01-01",
            validation_period_start="2024-01-01",
            validation_period_end="2024-02-01",
            mae=1000.0,
            rmse=1500.0,
            mape=2.5,
            directional_accuracy=0.55,
            n_folds=30,
        )
        df = get_model_runs(conn, model="random_forest")
        assert len(df) == 1
        assert df.iloc[0]["mae"] == 1000.0
        assert df.iloc[0]["n_folds"] == 30


def test_model_runs_accumulate_history_not_overwrite(db_path: Path) -> None:
    with get_connection(db_path) as conn:
        for mae in [1000.0, 900.0, 800.0]:
            save_model_run(
                conn,
                model="random_forest",
                horizon=1,
                train_period_start="2020-01-01",
                validation_period_start="2024-01-01",
                validation_period_end="2024-02-01",
                mae=mae,
                rmse=1500.0,
                mape=2.5,
                directional_accuracy=0.55,
                n_folds=30,
            )
        df = get_model_runs(conn, model="random_forest")
        assert len(df) == 3
