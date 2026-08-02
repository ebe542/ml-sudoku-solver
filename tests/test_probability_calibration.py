import numpy as np
import pytest

from sudoku_ml.analysis.probability_calibration import (
    SUPPORTED_CALIBRATION_METHODS,
    calibrate_probability_model,
    evaluate_probability_calibration,
)
from sudoku_ml.model.random_forest import SudokuRandomForest


def create_fitted_model() -> SudokuRandomForest:
    """Create a small fitted model for calibration tests."""
    rng = np.random.default_rng(42)

    X_train = rng.random((27, 118))
    y_train = np.tile(
        np.arange(1, 10),
        3,
    )

    model = SudokuRandomForest(
        n_estimators=5,
        random_seed=42,
    )
    model.fit_arrays(
        X_train,
        y_train,
    )

    return model


def test_supported_calibration_methods() -> None:
    assert SUPPORTED_CALIBRATION_METHODS == (
        "sigmoid",
        "isotonic",
    )


@pytest.mark.parametrize(
    "method",
    SUPPORTED_CALIBRATION_METHODS,
)
def test_calibrated_model_returns_probabilities(method: str) -> None:
    rng = np.random.default_rng(123)
    model = create_fitted_model()

    X_calibration = rng.random((27, 118))
    y_calibration = np.tile(
        np.arange(1, 10),
        3,
    )

    calibrated_model = calibrate_probability_model(
        model,
        X_calibration,
        y_calibration,
        method=method,
    )

    probabilities = calibrated_model.predict_probabilities(
        X_calibration[:3]
    )

    assert probabilities.shape == (3, 9)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert np.allclose(
        np.sum(probabilities, axis=1),
        1.0,
    )
    assert np.array_equal(
        calibrated_model.classes,
        np.arange(1, 10),
    )


def test_probability_calibration_rejects_unknown_method() -> None:
    model = create_fitted_model()

    with pytest.raises(
        ValueError,
        match="Unsupported calibration method",
    ):
        calibrate_probability_model(
            model,
            np.zeros((9, 118)),
            np.arange(1, 10),
            method="unknown",
        )


def test_probability_calibration_rejects_empty_data() -> None:
    model = create_fitted_model()

    with pytest.raises(
        ValueError,
        match="At least one calibration sample",
    ):
        calibrate_probability_model(
            model,
            np.empty((0, 118)),
            np.array([]),
            method="sigmoid",
        )


def test_probability_calibration_rejects_mismatched_lengths() -> None:
    model = create_fitted_model()

    with pytest.raises(
        ValueError,
        match="matching lengths"):
        calibrate_probability_model(
            model,
            np.zeros((2, 118)),
            np.array([1]),
            method="sigmoid",
        )

def test_probability_calibration_evaluates_all_methods() -> None:
    rng = np.random.default_rng(202)
    model = create_fitted_model()

    X_calibration = rng.random((27, 118))
    y_calibration = np.tile(
        np.arange(1, 10),
        3,
    )

    X_evaluation = rng.random((9, 118))
    X_evaluation[:, 82:91] = 1.0
    y_evaluation = np.arange(1, 10)

    results = evaluate_probability_calibration(
        model,
        X_calibration,
        y_calibration,
        X_evaluation,
        y_evaluation,
    )

    assert [
        result.name
        for result in results
    ] == [
        "Raw",
        "Sigmoid",
        "Isotonic",
    ]


def test_probability_calibration_reports_both_modes() -> None:
    rng = np.random.default_rng(202)
    model = create_fitted_model()

    X_calibration = rng.random((27, 118))
    y_calibration = np.tile(
        np.arange(1, 10),
        3,
    )

    X_evaluation = rng.random((9, 118))
    X_evaluation[:, 82:91] = 1.0
    y_evaluation = np.arange(1, 10)

    results = evaluate_probability_calibration(
        model,
        X_calibration,
        y_calibration,
        X_evaluation,
        y_evaluation,
    )

    for result in results:
        assert result.raw.sample_count == 9
        assert result.candidate_constrained.sample_count == 9
        assert 0.0 <= result.raw.top_1_accuracy <= 1.0
        assert (
            0.0
            <= result.candidate_constrained.top_1_accuracy
            <= 1.0
        )
        assert result.raw.log_loss >= 0.0
        assert result.candidate_constrained.log_loss >= 0.0
