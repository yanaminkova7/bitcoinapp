"""Bitcoin Forecasting Dashboard (Streamlit).

Fetches recent Bitcoin market data, fits a selected forecasting model on it, and
displays the forecast alongside historical prices and honest performance metrics from
walk-forward (chronological, never shuffled) backtesting - both for the selected model
and as a comparison across every model implemented so far.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from app.database.db import (  # noqa: E402
    backfill_actuals,
    get_connection,
    get_predictions,
    save_market_data,
    save_model_run,
    save_prediction,
)
from app.evaluation.intervals import prediction_interval  # noqa: E402
from app.evaluation.walk_forward import walk_forward_validate  # noqa: E402
from app.forecasting.baselines import MovingAverageForecaster, NaiveForecaster  # noqa: E402
from app.forecasting.lstm_model import LSTMForecaster  # noqa: E402
from app.forecasting.ml_models import (  # noqa: E402
    GradientBoostingForecaster,
    LinearRegressionForecaster,
    RandomForestForecaster,
)

SYMBOL = "BTC-USD"
# Each factory takes the forecast horizon and returns a fresh, unfit model instance.
# These are used for BOTH the single-model forecast and the walk-forward comparison
# table - all cheap enough to refit repeatedly.
MODELS = {
    "Naive (tomorrow = today)": lambda h: NaiveForecaster(),
    "Moving Average (7-day)": lambda h: MovingAverageForecaster(window=7),
    "Linear Regression": lambda h: LinearRegressionForecaster(horizon=h),
    "Random Forest": lambda h: RandomForestForecaster(horizon=h),
    "Gradient Boosting": lambda h: GradientBoostingForecaster(horizon=h),
}
# LSTM is offered as a single-model forecast only, never in the walk-forward comparison
# table: that table refits every model at every one of up to 90 folds, and retraining an
# LSTM that many times would take minutes and freeze the interactive UI. A full
# walk-forward LSTM backtest is still available offline via scripts/backtest.py.
LSTM_ONLY_MODEL = {
    "LSTM": lambda h: LSTMForecaster(horizon=h, epochs=30, patience=6),
}
# ML models need enough history to build lag/rolling features and still have data left
# for walk-forward folds; below this, only the plain baselines are offered.
ML_MIN_HISTORY_ROWS = 250

st.set_page_config(page_title="Bitcoin Forecasting Dashboard", layout="wide")


@st.cache_data(ttl=300)
def load_price_history(period: str) -> pd.DataFrame:
    df = yf.download(SYMBOL, period=period, interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df = df.sort_index()
    return df


@st.cache_data(ttl=300)
def sync_market_data_to_db(history: pd.DataFrame, symbol: str) -> int:
    """Persist the latest OHLCV data. Cached at the same TTL as load_price_history so
    this only actually writes when fresh data is fetched, not on every rerun."""
    with get_connection() as conn:
        return save_market_data(conn, history, symbol)


@st.cache_data(ttl=600, show_spinner="Running walk-forward backtests for all models...")
def compare_all_models(
    history: pd.DataFrame, model_labels: tuple[str, ...], horizon: int, n_test_folds: int, min_train_size: int
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    rows = []
    residuals_by_model: dict[str, pd.Series] = {}
    for label in model_labels:
        factory = MODELS[label]
        try:
            result = walk_forward_validate(
                history,
                lambda f=factory: f(horizon),
                label,
                horizon=horizon,
                n_test_folds=n_test_folds,
                min_train_size=min_train_size,
            )
        except ValueError:
            continue
        metrics = result.metrics()
        rows.append(
            {
                "Model": label,
                "MAE": round(metrics["mae"], 2),
                "RMSE": round(metrics["rmse"], 2),
                "MAPE (%)": round(metrics["mape"], 2),
                "Directional Accuracy (%)": round(metrics["directional_accuracy"] * 100, 1),
            }
        )
        residuals_by_model[label] = result.residuals()

        folds_frame = result.to_frame()
        with get_connection() as conn:
            save_model_run(
                conn,
                model=label,
                horizon=horizon,
                train_period_start=str(history.index[0].date()),
                validation_period_start=str(folds_frame["test_date"].min().date()),
                validation_period_end=str(folds_frame["test_date"].max().date()),
                mae=metrics["mae"],
                rmse=metrics["rmse"],
                mape=metrics["mape"],
                directional_accuracy=metrics["directional_accuracy"],
                n_folds=metrics["n_folds"],
            )
    if not rows:
        empty = pd.DataFrame(columns=["Model", "MAE", "RMSE", "MAPE (%)", "Directional Accuracy (%)"])
        return empty, residuals_by_model
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True), residuals_by_model


@st.cache_data(ttl=600, show_spinner="Calibrating LSTM prediction interval...")
def lstm_calibration_residuals(history: pd.DataFrame, horizon: int, n_calib_days: int, seq_len: int) -> pd.Series:
    """Residuals from a model trained on data *before* the calibration window, evaluated
    via rolling inference (no retraining per day) over that held-out window - so these
    residuals are genuinely out-of-sample, not in-sample, the same as the walk-forward
    residuals used for every other model."""
    if len(history) < seq_len + n_calib_days + horizon + 20:
        return pd.Series(dtype=float)

    train_df = history.iloc[:-n_calib_days]
    model = LSTMForecaster(seq_len=seq_len, horizon=horizon, epochs=30, patience=6).fit(train_df)

    residuals = []
    for i in range(n_calib_days):
        target_idx = len(train_df) + i + horizon - 1
        if target_idx >= len(history):
            break
        recent_close = history["close"].to_numpy(dtype=float)[len(train_df) + i - seq_len : len(train_df) + i]
        if len(recent_close) < seq_len:
            continue
        model._last_window = recent_close
        model._last_close = float(recent_close[-1])
        predicted = model.predict(horizon=horizon).iloc[0]
        actual = history["close"].iloc[target_idx]
        residuals.append(predicted - actual)

    return pd.Series(residuals)


st.title("Bitcoin Forecasting Dashboard")
st.caption("Model forecasts are estimates, not guarantees. Uncertainty is inherent to any forecast.")

with st.sidebar:
    st.header("Settings")
    period = st.selectbox("Historical period", ["6mo", "1y", "2y", "5y"], index=1)
    if st.button("Refresh data now"):
        load_price_history.clear()
        st.rerun()
    st.caption("Data also auto-refreshes every 5 minutes.")

history = load_price_history(period)

if history.empty:
    st.error("No data returned from the market data source. Try again shortly.")
    st.stop()

sync_market_data_to_db(history, SYMBOL)
with get_connection() as _conn:
    backfill_actuals(_conn, history["close"])

available_models = list(MODELS.keys())
if len(history) < ML_MIN_HISTORY_ROWS:
    available_models = [m for m in available_models if m in ("Naive (tomorrow = today)", "Moving Average (7-day)")]
    st.sidebar.caption(
        f"Only {len(history)} rows available for '{period}' - ML models need at least "
        f"{ML_MIN_HISTORY_ROWS} to build features and backtest reliably, so only baselines "
        "are offered for this period."
    )
# LSTM is selectable for a single forecast (with the same history requirement as the
# other ML models) but deliberately excluded from `available_models`, which feeds the
# walk-forward comparison table - see LSTM_ONLY_MODEL comment above for why.
forecast_model_options = available_models + (list(LSTM_ONLY_MODEL.keys()) if len(history) >= ML_MIN_HISTORY_ROWS else [])
ALL_MODELS = {**MODELS, **LSTM_ONLY_MODEL}

with st.sidebar:
    model_label = st.selectbox("Forecasting model", forecast_model_options)
    horizon = st.selectbox("Forecast horizon (days)", [1, 3, 7, 14], index=0)
    holdout_days = st.slider("Backtest holdout window (days)", min_value=10, max_value=90, value=30)
    confidence = st.select_slider("Prediction interval confidence", options=[0.8, 0.9, 0.95], value=0.9)

close = history["close"]
current_price = float(close.iloc[-1])
previous_price = float(close.iloc[-2]) if len(close) > 1 else current_price
change_pct = (current_price / previous_price - 1) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Current BTC Price", f"${current_price:,.2f}", f"{change_pct:+.2f}%")
col2.metric("24h Volume", f"{history['volume'].iloc[-1]:,.0f}")
col3.metric("Last Update", str(history.index[-1].date()))

st.subheader("Historical Price")
st.line_chart(close.rename("close"))

# Computed once, up front, so both the Forecast section (for the interval) and the
# Model Comparison section (for the table) can use the same walk-forward run instead of
# duplicating it.
min_train_size = max(60, len(history) - holdout_days - 40 - horizon)
comparison_df, residuals_by_model = compare_all_models(
    history, tuple(available_models), horizon, holdout_days, min_train_size
)

st.subheader("Forecast")
model_factory = ALL_MODELS[model_label]
spinner_msg = "Training LSTM..." if model_label == "LSTM" else f"Fitting {model_label}..."
try:
    with st.spinner(spinner_msg):
        fitted_model = model_factory(horizon).fit(history)
        forecast_values = fitted_model.predict(horizon=horizon)
except ValueError as e:
    st.error(f"Could not fit '{model_label}' for a {horizon}-day horizon: {e}")
    st.stop()

forecast_dates = pd.bdate_range(start=close.index[-1], periods=horizon + 1, freq="D")[1:]
# Baselines return one value per day (a repeated path); ML models return a single direct
# estimate for exactly `horizon` days out - not an intermediate day-by-day path. Handle
# both without fabricating a path the model didn't actually produce.
is_direct_estimate = len(forecast_values) == 1 and horizon > 1
forecast_index = forecast_dates[-1:] if is_direct_estimate else forecast_dates
forecast_series = pd.Series(forecast_values.to_numpy(), index=forecast_index, name="forecast")
point_forecast = float(forecast_values.iloc[-1])

# Prediction interval: empirical quantiles of this model's own past walk-forward
# residuals (or, for LSTM, held-out rolling-inference residuals) applied to today's
# point forecast. See backend/app/evaluation/intervals.py for the full explanation of
# why this is not a calibrated probability guarantee, just an honest historical range.
if model_label == "LSTM":
    calib_residuals = lstm_calibration_residuals(history, horizon, holdout_days, fitted_model.seq_len)
else:
    calib_residuals = residuals_by_model.get(model_label, pd.Series(dtype=float))

interval = None
if len(calib_residuals) >= 20:
    interval = prediction_interval(point_forecast, calib_residuals, confidence=confidence)

with get_connection() as _conn:
    save_prediction(
        _conn,
        model=model_label,
        horizon=horizon,
        target_date=str(forecast_dates[-1].date()),
        predicted_value=point_forecast,
        lower_bound=interval.lower if interval else None,
        upper_bound=interval.upper if interval else None,
        confidence=confidence if interval else None,
    )

combined = pd.concat([close.tail(60).rename("historical"), forecast_series], axis=1)
st.line_chart(combined)

if interval is not None:
    st.write(
        f"**Model forecast for {forecast_dates[-1].date()}:** ${interval.point_forecast:,.2f} "
        f"&nbsp;&nbsp; **{int(confidence * 100)}% interval:** "
        f"${interval.lower:,.2f} – ${interval.upper:,.2f}"
    )
    st.caption(
        f"The interval comes from this model's own historical forecast errors over the last "
        f"{len(calib_residuals)} out-of-sample predictions - it is an empirical estimate of "
        "uncertainty, not a guaranteed probability. Model forecasts are estimates; Bitcoin's "
        "actual future price could fall outside this range."
    )
else:
    st.write(f"**Model forecast for {forecast_dates[-1].date()}:** ${point_forecast:,.2f}")
    st.caption(
        "Not enough historical out-of-sample predictions available to calibrate a prediction "
        "interval for this model/horizon/period combination. This is a single-point model "
        "estimate, not a guaranteed future price."
    )
if is_direct_estimate:
    st.caption(
        f"This model predicts {horizon} days ahead directly (not day-by-day), so only the "
        f"single {forecast_dates[-1].date()} estimate is shown - not an intermediate path."
    )

st.subheader("Model Comparison (walk-forward backtest, no shuffling)")
if comparison_df.empty:
    st.warning("Not enough data for the selected period/horizon to run a backtest.")
else:
    st.dataframe(comparison_df, width="stretch", hide_index=True)
    st.caption(
        f"Evaluated on the last {holdout_days} days for a {horizon}-day horizon; every model "
        "refits at each step using only data available before that point. Naive's directional "
        "accuracy is 0% by construction - it always predicts 'no change', so it never has a "
        "predicted direction to be right or wrong about. LSTM isn't included here: refitting "
        "it at every backtest fold would take minutes and freeze this page. See "
        "scripts/backtest.py for an offline LSTM walk-forward backtest."
    )

st.subheader("Live Forecast Monitoring")
st.caption(
    "Every forecast this dashboard generates is logged. Once a target date arrives, its actual "
    "close price is backfilled here, so real-world forecast error accumulates over time - "
    "separate from the backtests above, which replay history rather than wait for it."
)
with get_connection() as _conn:
    logged_predictions = get_predictions(_conn)

realized = logged_predictions.dropna(subset=["actual_value"]) if not logged_predictions.empty else logged_predictions
if realized.empty:
    st.info(
        "No realized predictions yet - forecasts logged today will show up here once their "
        "target date's actual price becomes available (check back after that date passes)."
    )
else:
    realized = realized.copy()
    realized["abs_error"] = (realized["predicted_value"] - realized["actual_value"]).abs()
    live_mae_by_model = realized.groupby("model")["abs_error"].mean().rename("Live MAE")

    with get_connection() as _conn:
        backtest_runs = get_model_runs(_conn, limit=1000)
    backtest_mae_by_model = (
        backtest_runs.groupby("model")["mae"].mean().rename("Backtest MAE") if not backtest_runs.empty else pd.Series(dtype=float)
    )

    monitor_df = pd.concat([live_mae_by_model, backtest_mae_by_model], axis=1).dropna(subset=["Live MAE"])
    monitor_df["Live MAE"] = monitor_df["Live MAE"].round(2)
    monitor_df["Backtest MAE"] = monitor_df["Backtest MAE"].round(2)
    st.dataframe(monitor_df.reset_index().rename(columns={"index": "Model"}), width="stretch", hide_index=True)

    for model_name, row in monitor_df.iterrows():
        if pd.notna(row["Backtest MAE"]) and row["Backtest MAE"] > 0:
            degradation = row["Live MAE"] / row["Backtest MAE"]
            if degradation > 1.5:
                st.warning(
                    f"**{model_name}**: live forecast error (${row['Live MAE']:,.2f}) is "
                    f"{degradation:.1f}x its backtested MAE (${row['Backtest MAE']:,.2f}) - "
                    "performance may be degrading. Treat this model's current forecasts with "
                    "extra caution rather than assuming past backtest results still apply."
                )

    st.dataframe(
        realized[["model", "horizon", "target_date", "predicted_value", "actual_value", "abs_error"]].sort_values(
            "target_date", ascending=False
        ),
        width="stretch",
        hide_index=True,
    )

