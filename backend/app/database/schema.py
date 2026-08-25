"""SQLite schema. Plain SQL (no ORM) - kept simple enough to port to PostgreSQL later
by swapping the connection layer, per the project's stated migration path."""

CREATE_MARKET_DATA = """
CREATE TABLE IF NOT EXISTS market_data (
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (timestamp, symbol)
);
"""

CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    generated_at TEXT NOT NULL,
    model TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    target_date TEXT NOT NULL,
    predicted_value REAL NOT NULL,
    lower_bound REAL,
    upper_bound REAL,
    confidence REAL,
    actual_value REAL,
    PRIMARY KEY (model, horizon, target_date)
);
"""

CREATE_MODEL_RUNS = """
CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    model TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    train_period_start TEXT NOT NULL,
    validation_period_start TEXT NOT NULL,
    validation_period_end TEXT NOT NULL,
    mae REAL NOT NULL,
    rmse REAL NOT NULL,
    mape REAL,
    directional_accuracy REAL NOT NULL,
    n_folds INTEGER NOT NULL
);
"""

ALL_TABLES = [CREATE_MARKET_DATA, CREATE_PREDICTIONS, CREATE_MODEL_RUNS]
