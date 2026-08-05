from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.model.histogram_gradient_boosting import (
    SudokuHistogramGradientBoosting,
)


@pytest.fixture
def training_data():
    """Create a small reproducible data split."""
    return create_train_test_split(
        num_solutions=20,
        test_size=0.2,
        random_seed=42,
    )


def test_model_can_be_trained(training_data) -> None:
    model = SudokuHistogramGradientBoosting(
        max_iter=20,
        random_seed=42,
    )

    model.fit(training_data)

    assert isinstance(
        model.model,
        HistGradientBoostingClassifier,
    )
    assert np.array_equal(
        model.classes,
        np.arange(1, 10),
    )


def test_model_returns_predictions_and_probabilities(
    training_data,
) -> None:
    model = SudokuHistogramGradientBoosting(
        max_iter=20,
        random_seed=42,
    )
    model.fit(training_data)

    predictions = model.predict(training_data.X_test)
    probabilities = model.predict_probabilities(
        training_data.X_test
    )

    assert predictions.shape == training_data.y_test.shape
    assert probabilities.shape == (
        len(training_data.X_test),
        9,
    )
    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    )


def test_model_can_be_evaluated(training_data) -> None:
    model = SudokuHistogramGradientBoosting(
        max_iter=20,
        random_seed=42,
    )
    model.fit(training_data)

    accuracy = model.evaluate(training_data)

    assert 0.0 <= accuracy <= 1.0


def test_saved_and_loaded_model_produces_same_predictions(
    training_data,
    tmp_path: Path,
) -> None:
    model = SudokuHistogramGradientBoosting(
        max_iter=20,
        random_seed=42,
    )
    model.fit(training_data)

    predictions_before = model.predict(training_data.X_test)
    model_path = tmp_path / "histogram_gradient_boosting.joblib"
    model.save(model_path)

    loaded_model = SudokuHistogramGradientBoosting.load(model_path)
    predictions_after = loaded_model.predict(training_data.X_test)

    assert model_path.exists()
    assert np.array_equal(
        predictions_before,
        predictions_after,
    )
    assert np.array_equal(
        model.classes,
        loaded_model.classes,
    )


def test_load_rejects_unexpected_object(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid_model.joblib"
    joblib.dump(
        {"not": "histogram gradient boosting"},
        invalid_path,
    )

    with pytest.raises(
        TypeError,
        match="HistGradientBoostingClassifier",
    ):
        SudokuHistogramGradientBoosting.load(invalid_path)
