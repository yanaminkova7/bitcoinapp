import pandas as pd
import pytest

from app.evaluation.metrics import directional_accuracy, mae, mape, rmse


def test_mae_zero_for_perfect_predictions() -> None:
    actual = pd.Series([1.0, 2.0, 3.0])
    assert mae(actual, actual) == 0.0


def test_mae_basic() -> None:
    actual = pd.Series([10.0, 20.0])
    predicted = pd.Series([12.0, 18.0])
    assert mae(actual, predicted) == 2.0


def test_rmse_basic() -> None:
    actual = pd.Series([0.0, 0.0])
    predicted = pd.Series([3.0, 4.0])
    assert rmse(actual, predicted) == pytest.approx(3.5355339059327378)


def test_mape_basic() -> None:
    actual = pd.Series([100.0, 200.0])
    predicted = pd.Series([110.0, 180.0])
    assert round(mape(actual, predicted), 4) == 10.0


def test_directional_accuracy() -> None:
    previous = pd.Series([100.0, 100.0, 100.0])
    actual = pd.Series([110.0, 90.0, 100.0])
    predicted = pd.Series([105.0, 95.0, 100.0])
    # Step 1: up predicted, up actual -> correct
    # Step 2: down predicted, down actual -> correct
    # Step 3: no change predicted, no change actual -> correct
    assert directional_accuracy(actual, predicted, previous) == 1.0
