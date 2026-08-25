import pandas as pd
import pytest

from app.forecasting.baselines import MovingAverageForecaster, NaiveForecaster


def make_close_df(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": values})


def test_naive_forecaster_repeats_last_value() -> None:
    df = make_close_df([10.0, 20.0, 30.0])
    model = NaiveForecaster().fit(df)
    forecast = model.predict(horizon=3)
    assert forecast.tolist() == [30.0, 30.0, 30.0]


def test_naive_forecaster_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        NaiveForecaster().predict(horizon=1)


def test_moving_average_forecaster_uses_trailing_window() -> None:
    df = make_close_df([10.0, 20.0, 30.0, 40.0])
    model = MovingAverageForecaster(window=2).fit(df)
    forecast = model.predict(horizon=2)
    assert forecast.tolist() == [35.0, 35.0]


def test_moving_average_forecaster_ignores_data_before_window() -> None:
    # Changing an old value outside the window must not change the forecast.
    df_a = make_close_df([999.0, 20.0, 30.0, 40.0])
    df_b = make_close_df([-999.0, 20.0, 30.0, 40.0])
    forecast_a = MovingAverageForecaster(window=2).fit(df_a).predict(horizon=1)
    forecast_b = MovingAverageForecaster(window=2).fit(df_b).predict(horizon=1)
    assert forecast_a.tolist() == forecast_b.tolist()


def test_moving_average_forecaster_raises_on_short_series() -> None:
    with pytest.raises(ValueError):
        MovingAverageForecaster(window=10).fit(make_close_df([1.0, 2.0]))
