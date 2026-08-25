# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`app.py` is currently reset to Streamlit's stock "Blank app" starter template (a slider-controlled
Altair spiral chart) and is **disconnected from the Bitcoin forecasting backend** — it does not
import from `backend/`. The Bitcoin forecasting/backtesting system described below still exists in
full under `backend/`, `scripts/`, `data/`, `models/`, and is exercised by the pytest suite and the
offline scripts, but nothing currently wires it back into `app.py`. Root `requirements.txt` was
trimmed to match the blank template's actual imports (`streamlit`, `altair`, `numpy`, `pandas`);
`backend/requirements.txt` (the full ML/dev dependency set) is untouched. When re-connecting
`app.py` to the backend, restore `sys.path` insertion of `backend/` and the imports from the `app`
package there (see git history for the previous full version), and restore the dependencies
`app.py` needs (`yfinance`, `scikit-learn`, `joblib`, `torch`, `pydantic`, `pydantic-settings`) to
root `requirements.txt` since that's what Streamlit Cloud reads for deployment.

## Commands

Install (local dev, includes jupyter/pytest/matplotlib on top of the deploy set in the root
`requirements.txt`):
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

Run the app:
```bash
./venv/bin/streamlit run app.py
```

Run all tests:
```bash
cd backend && ../venv/bin/python -m pytest
```

Run a single test file / test:
```bash
cd backend && ../venv/bin/python -m pytest tests/test_walk_forward.py
cd backend && ../venv/bin/python -m pytest tests/test_walk_forward.py::test_name -v
```

Data pipeline (idempotent, safe to re-run):
```bash
./venv/bin/python scripts/download_data.py --period 5y --interval 1d   # -> data/raw/
./venv/bin/python scripts/preprocess_data.py                            # -> data/processed/
```

Model training / backtesting (offline, outside the Streamlit UI):
```bash
./venv/bin/python scripts/train_model.py                                    # all models, walk-forward, logs to models/trained/experiment_results.csv
./venv/bin/python scripts/backtest.py --model random_forest --horizon 1     # one model, full walk-forward + plots
./venv/bin/python scripts/train_lstm.py                                     # LSTM training + loss curve + held-out eval
```

There is no lint/format command configured in this repo.

## Architecture

The sections below describe the backend as built — it's just not currently invoked from `app.py`
(see "What this is" above).

**`backend/app/forecasting/`** — all models implement `BaseForecaster` (`base.py`):
`fit(df) -> self`, `predict(horizon) -> pd.Series`, plus joblib `save`/`load`. This lets `app.py`
and the scripts treat every model (baselines, ML, LSTM) identically. Two shapes of `predict`
output exist and callers must handle both: baselines return one repeated value per day (a path);
ML/LSTM models return a single direct estimate for `horizon` days out. `features.py` builds
lag/rolling features and is explicitly tested to never leak information from after the prediction
timestamp — any new feature must preserve that.

**`backend/app/evaluation/`** — `walk_forward.py` runs expanding-window, strictly chronological
backtesting (never shuffled) and returns a result object exposing `.metrics()` (MAE/RMSE/MAPE/
directional accuracy) and `.residuals()`. Those residuals feed `intervals.py`, which builds
prediction intervals from empirical quantiles of a model's own past out-of-sample errors —
deliberately not a normal-distribution assumption, since BTC returns are fat-tailed.

**`backend/app/database/`** — SQLite (`db.py` + `schema.py`) persists three things: raw market
data, every prediction the app ever logs (for live monitoring), and model run/backtest history.
the previous `app.py`'s "Live Forecast Monitoring" section compared live realized error against
backtested MAE per model and flagged models whose live error had drifted significantly above
backtest — separate from and complementary to the walk-forward backtests, which replay history
rather than wait for it.

**Performance-driven split (as previously wired into `app.py`)**: the walk-forward comparison
table refits every model at every fold (up to ~90 folds) on each Streamlit rerun, so LSTM was
deliberately excluded from it (`LSTM_ONLY_MODEL`, kept separate from `MODELS`) — refitting an LSTM
that many times would freeze the UI. LSTM was still selectable for a single live forecast, with its
own held-out (not walk-forward) calibration for its prediction interval via
`lstm_calibration_residuals`. A full walk-forward LSTM backtest is only available offline via
`scripts/backtest.py`. Both `compare_all_models` and `lstm_calibration_residuals` are
`st.cache_data`-cached so repeated Streamlit reruns (widget interaction) wouldn't repeatedly
retrain.

`backend/app/config.py` loads settings from `.env` via pydantic-settings (`Settings`); secrets use
`SecretStr` so they never appear in logs/tracebacks. No API key is currently required (yfinance is
key-less).

`backend/app/models/`, `schemas/`, `services/` are scaffolding for future phases and are currently
empty aside from `__init__.py`.

## Model development rules

- Time-series data is never randomly shuffled for evaluation — always chronological
  train/holdout splits or walk-forward validation.
- Every new model is benchmarked against the naive and moving-average baselines
  (`backend/app/forecasting/baselines.py`) before being trusted.
- Features must never use information from after the prediction timestamp — extend
  `backend/tests/test_features.py` coverage for any new feature.
- Prediction intervals are empirical (residual-quantile based), never assumed normal.
