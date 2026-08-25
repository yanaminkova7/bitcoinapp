import numpy as np
import pandas as pd
import pytest

from app.evaluation.intervals import prediction_interval


def test_symmetric_residuals_give_symmetric_interval() -> None:
    residuals = pd.Series(np.linspace(-100, 100, 201))  # symmetric around 0
    result = prediction_interval(point_forecast=1000.0, residuals=residuals, confidence=0.9)
    assert result.lower < 1000.0 < result.upper
    assert (1000.0 - result.lower) == pytest.approx(result.upper - 1000.0, rel=1e-6)


def test_model_that_overpredicts_shifts_interval_down() -> None:
    # predicted - actual is consistently positive -> model overpredicts -> the true
    # value is expected to be *below* the point forecast, so bounds should shift down.
    residuals = pd.Series(np.random.default_rng(0).normal(500, 50, 200))
    result = prediction_interval(point_forecast=1000.0, residuals=residuals, confidence=0.9)
    assert result.upper < 1000.0


def test_model_that_underpredicts_shifts_interval_up() -> None:
    residuals = pd.Series(np.random.default_rng(0).normal(-500, 50, 200))
    result = prediction_interval(point_forecast=1000.0, residuals=residuals, confidence=0.9)
    assert result.lower > 1000.0


def test_asymmetric_residuals_give_asymmetric_interval() -> None:
    # A fat right tail in residuals (occasional large overprediction) should pull the
    # lower bound down further than the upper bound is pulled up.
    rng = np.random.default_rng(0)
    residuals = pd.Series(np.concatenate([rng.normal(0, 20, 190), rng.uniform(500, 1000, 10)]))
    result = prediction_interval(point_forecast=1000.0, residuals=residuals, confidence=0.9)
    lower_width = result.point_forecast - result.lower
    upper_width = result.upper - result.point_forecast
    assert lower_width > upper_width


def test_wider_confidence_gives_wider_interval() -> None:
    residuals = pd.Series(np.random.default_rng(0).normal(0, 100, 200))
    narrow = prediction_interval(point_forecast=1000.0, residuals=residuals, confidence=0.5)
    wide = prediction_interval(point_forecast=1000.0, residuals=residuals, confidence=0.95)
    assert (wide.upper - wide.lower) > (narrow.upper - narrow.lower)


def test_raises_on_too_few_residuals() -> None:
    with pytest.raises(ValueError):
        prediction_interval(point_forecast=1000.0, residuals=pd.Series([1.0, 2.0, 3.0]))


def test_raises_on_invalid_confidence() -> None:
    residuals = pd.Series(np.random.default_rng(0).normal(0, 10, 50))
    with pytest.raises(ValueError):
        prediction_interval(point_forecast=1000.0, residuals=residuals, confidence=1.5)


def test_changing_point_forecast_does_not_change_interval_width() -> None:
    residuals = pd.Series(np.random.default_rng(0).normal(0, 100, 200))
    a = prediction_interval(point_forecast=1000.0, residuals=residuals, confidence=0.9)
    b = prediction_interval(point_forecast=5000.0, residuals=residuals, confidence=0.9)
    assert (a.upper - a.lower) == pytest.approx(b.upper - b.lower)
