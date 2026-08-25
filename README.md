# Bitcoin Forecasting App

A Streamlit application that collects historical Bitcoin market data, trains forecasting models,
evaluates them with time-series-aware backtesting, and presents predictions - with uncertainty
intervals - through an interactive dashboard.

Forecasts are presented as model estimates with uncertainty, never as guaranteed outcomes.

**Status:** data collection, preprocessing, exploratory analysis, feature engineering, baselines,
traditional ML models (Linear Regression, Random Forest, Gradient Boosting), an LSTM, walk-forward
backtesting, empirical prediction intervals, and SQLite-backed persistence/monitoring are all
implemented. Not yet done: automated (unattended) scheduled retraining, and deployment.

## Technology Stack

- **App:** Python, Streamlit
- **ML:** pandas, NumPy, scikit-learn, PyTorch (LSTM), joblib for model persistence
- **Data source:** yfinance
- **Database:** SQLite

## Project Structure

```
app.py                    Streamlit dashboard (entry point)
requirements.txt          Deployment dependencies (Streamlit Cloud reads this)

backend/app/
    config.py              Settings (.env-backed)
    forecasting/
        base.py             BaseForecaster interface (fit/predict/save/load)
        baselines.py        Naive, Moving Average
        ml_models.py        Linear Regression, Random Forest, Gradient Boosting
        lstm_model.py       LSTM (60-day lookback, direct multi-step)
        features.py         Leakage-safe feature engineering
    evaluation/
        metrics.py          MAE, RMSE, MAPE, directional accuracy
        walk_forward.py     Expanding-window walk-forward backtesting
        intervals.py        Empirical residual-quantile prediction intervals
    database/
        schema.py, db.py    SQLite: market data, predictions, model run history
    models/, schemas/, services/, data/   Scaffolding for future phases
backend/requirements.txt   Full local dev dependencies (includes jupyter, pytest, ...)
backend/tests/             pytest suite (59 tests)

data/       raw/ and processed/ market data, plus bitcoin_forecasting.db (all gitignored)
models/     trained model artifacts + backtest outputs (gitignored)
notebooks/  exploratory analysis
scripts/
    download_data.py        Fetch raw OHLCV data from yfinance
    preprocess_data.py      Clean + derive daily_return/log_return
    train_model.py          Walk-forward comparison across all models, logged to CSV
    backtest.py             Walk-forward backtest for one model, with plots
    train_lstm.py           LSTM training with loss curves + held-out test evaluation
```

## Prerequisites

- Python 3.12+ (developed and tested on 3.14)

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

## Running the App

```bash
./venv/bin/streamlit run app.py
```

Open the URL Streamlit prints (default http://localhost:8501). The sidebar lets you pick the
historical period, forecasting model (including LSTM), horizon, backtest window, and prediction
interval confidence level. A "Refresh data now" button forces an immediate data pull; otherwise
data auto-refreshes every 5 minutes.

Every forecast the app generates is logged to SQLite (`data/bitcoin_forecasting.db`); once a
forecast's target date passes, its actual price is backfilled automatically and shown in the
**Live Forecast Monitoring** section, which also flags a model if its live error significantly
exceeds its own backtested MAE.

## Downloading / Preprocessing Data

```bash
./venv/bin/python scripts/download_data.py --period 5y --interval 1d
./venv/bin/python scripts/preprocess_data.py
```

Raw data is written to `data/raw/`, cleaned/derived data to `data/processed/`. Both scripts are
idempotent and safe to re-run to refresh the dataset.

## Training Models / Running Backtests

```bash
# Compare every model via walk-forward validation, log results to models/trained/experiment_results.csv
./venv/bin/python scripts/train_model.py

# Full walk-forward backtest for one model, with actual-vs-predicted + error plots
./venv/bin/python scripts/backtest.py --model random_forest --horizon 1

# Train the LSTM, with a loss-curve plot and held-out test evaluation
./venv/bin/python scripts/train_lstm.py
```

## Exploratory Analysis

`notebooks/exploratory_analysis.ipynb` covers price trends, return distribution, rolling
volatility, drawdowns, and autocorrelation. Run with:

```bash
./venv/bin/jupyter notebook notebooks/exploratory_analysis.ipynb
```

## Running Tests

```bash
cd backend && ../venv/bin/python -m pytest
```

## Environment Variables

Copy `.env.example` to `.env` and adjust as needed. No secrets are currently required (yfinance
needs no API key). If a future data source needs one, add it to `.env` — never commit real values.
`.streamlit/secrets.toml` is also gitignored for Streamlit-native secret handling.

## Model Development Rules

- Time-series data is never randomly shuffled for evaluation — always chronological
  train/holdout splits or walk-forward validation.
- Every new model is benchmarked against the naive and moving-average baselines
  (`backend/app/forecasting/baselines.py`) before being trusted.
- Features must never use information from after the prediction timestamp
  (`backend/app/forecasting/features.py` is tested explicitly for this).
- Prediction intervals are empirical, calibrated from each model's own historical
  out-of-sample residuals — never assumed to be normally distributed, since Bitcoin's
  return distribution is measurably fat-tailed (see the EDA notebook).

## How to Interpret the Results

- **Point forecast**: a single model estimate for the target date, not a guarantee.
- **Prediction interval**: the range the model's own past errors suggest is plausible at
  the chosen confidence level. It reflects historical error, not a statistically
  guaranteed probability - time-series data doesn't strictly satisfy the assumptions
  that guarantee would require.
- **Model Comparison table**: MAE/RMSE/MAPE/directional accuracy from walk-forward
  backtesting - always compare a model's numbers against the naive baseline in the same
  table before trusting it. A lower MAE than naive is a real result; a higher one means
  the added complexity isn't paying off yet.
- **Live Forecast Monitoring**: real-world accuracy as it accumulates, separate from
  backtests. If a model's live error notably exceeds its backtested MAE, the app flags it.

## Limitations

This project is under active, incremental development. Bitcoin price forecasts are model
estimates with inherent uncertainty and should not be treated as financial advice or guaranteed
outcomes. Backtested performance does not guarantee future performance. Prediction intervals are
empirical estimates, not statistically guaranteed probabilities. The app is not yet deployed;
running it requires a local Python environment as described above.
