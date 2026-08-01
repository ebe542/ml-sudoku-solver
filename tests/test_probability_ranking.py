import numpy as np
import pytest

from sudoku_ml.analysis.probability_ranking import (
    analyze_probability_ranking,
    apply_candidate_constraints,
    calculate_expected_calibration_error,
)


class FakeProbabilityModel:
    """Provide controlled probabilities for metric tests."""

    def __init__(self) -> None:
        self.classes = np.array([1, 2, 3])
        self.probabilities = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.6, 0.3, 0.1],
                [0.5, 0.3, 0.2],
            ]
        )

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        return self.probabilities[: len(X)]


def test_probability_ranking_calculates_metrics() -> None:
    model = FakeProbabilityModel()
    X = np.zeros((3, 1))
    y = np.array([1, 2, 3])

    result = analyze_probability_ranking(
        model,
        X,
        y,
        n_bins=2,
    )

    expected_log_loss = -(
        np.log(0.7)
        + np.log(0.3)
        + np.log(0.2)
    ) / 3

    assert result.sample_count == 3
    assert result.top_1_accuracy == pytest.approx(1 / 3)
    assert result.top_2_accuracy == pytest.approx(2 / 3)
    assert result.top_3_accuracy == pytest.approx(1.0)
    assert result.mean_reciprocal_rank == pytest.approx(
        (1.0 + 0.5 + 1 / 3) / 3
    )
    assert result.mean_confidence == pytest.approx(0.6)
    assert result.expected_calibration_error == pytest.approx(
        0.2666667
    )
    assert result.log_loss == pytest.approx(
        expected_log_loss
    )


def test_calibration_error_is_zero_for_perfect_calibration() -> None:
    confidences = np.array([1.0, 1.0])
    correct_predictions = np.array([True, True])

    result = calculate_expected_calibration_error(
        confidences,
        correct_predictions,
    )

    assert result == pytest.approx(0.0)


def test_probability_ranking_rejects_empty_samples() -> None:
    model = FakeProbabilityModel()

    with pytest.raises(
        ValueError,
        match="At least one evaluation sample",
    ):
        analyze_probability_ranking(
            model,
            np.empty((0, 1)),
            np.array([]),
        )


def test_probability_ranking_rejects_mismatched_lengths() -> None:
    model = FakeProbabilityModel()

    with pytest.raises(
        ValueError,
        match="matching lengths",
    ):
        analyze_probability_ranking(
            model,
            np.zeros((2, 1)),
            np.array([1]),
        )


def test_probability_ranking_rejects_unknown_target() -> None:
    model = FakeProbabilityModel()

    with pytest.raises(
        ValueError,
        match="represented in model classes",
    ):
        analyze_probability_ranking(
            model,
            np.zeros((1, 1)),
            np.array([9]),
        )


def test_calibration_error_rejects_invalid_bin_count() -> None:
    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        calculate_expected_calibration_error(
            np.array([0.5]),
            np.array([True]),
            n_bins=0,
        )

def test_candidate_constraints_mask_and_renormalize() -> None:
    probabilities = np.array([[0.4, 0.35, 0.25]])
    classes = np.array([1, 2, 3])

    X = np.zeros((1, 91))
    X[0, 83] = 1.0
    X[0, 84] = 1.0

    result = apply_candidate_constraints(probabilities, classes, X)

    assert result[0, 0] == pytest.approx(0.0)
    assert result[0, 1] == pytest.approx(0.35 / 0.60)
    assert result[0, 2] == pytest.approx(0.25 / 0.60)
    assert np.sum(result[0]) == pytest.approx(1.0)

def test_candidate_constraints_can_correct_raw_top_prediction() -> None:
    model = FakeProbabilityModel()

    X = np.zeros((1, 91))
    X[0, 83] = 1.0
    X[0, 84] = 1.0
    y = np.array([2])

    raw_result = analyze_probability_ranking(model, X, y)
    constrained_result = analyze_probability_ranking(
        model,
        X,
        y,
        candidate_constrained=True,
    )

    assert raw_result.top_1_accuracy == pytest.approx(0.0)
    assert constrained_result.top_1_accuracy == pytest.approx(1.0)
    assert constrained_result.mean_confidence == pytest.approx(0.2 / 0.3)
