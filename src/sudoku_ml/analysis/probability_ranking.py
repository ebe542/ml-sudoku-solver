from dataclasses import dataclass

import numpy as np
from sklearn.metrics import log_loss

from sudoku_ml.model.random_forest import SudokuRandomForest


@dataclass(frozen=True)
class ProbabilityRankingResult:
    """Store model probability-ranking metrics."""

    sample_count: int
    top_1_accuracy: float
    top_2_accuracy: float
    top_3_accuracy: float
    mean_reciprocal_rank: float
    mean_confidence: float
    expected_calibration_error: float
    log_loss: float


def calculate_expected_calibration_error(confidences: np.ndarray, correct_predictions: np.ndarray,
                                         n_bins: int = 10) -> float:
    """Calculate expected calibration error."""
    if n_bins <= 0:
        raise ValueError(
            "Number of calibration bins must be positive."
        )

    bin_boundaries = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    bin_indices = np.digitize(
        confidences,
        bin_boundaries[1:-1],
        right=True,
    )

    calibration_error = 0.0

    for bin_index in range(n_bins):
        in_bin = bin_indices == bin_index

        if not np.any(in_bin):
            continue

        bin_confidence = float(
            np.mean(confidences[in_bin])
        )
        bin_accuracy = float(
            np.mean(correct_predictions[in_bin])
        )
        bin_weight = float(np.mean(in_bin))

        calibration_error += (
            bin_weight
            * abs(bin_accuracy - bin_confidence)
        )

    return calibration_error


CANDIDATE_FEATURE_START = 82
CANDIDATE_FEATURE_END = 91


def apply_candidate_constraints(probabilities: np.ndarray, classes: np.ndarray,
                                X: np.ndarray) -> np.ndarray:
    """Mask invalid Sudoku candidates and renormalize."""
    if X.shape[1] < CANDIDATE_FEATURE_END:
        raise ValueError(
            "Candidate-constrained analysis requires "
            "candidate indicator features."
        )

    candidate_features = X[
        :,
        CANDIDATE_FEATURE_START:CANDIDATE_FEATURE_END,
    ]

    class_digits = classes.astype(int)

    if np.any(
        (class_digits < 1)
        | (class_digits > 9)
    ):
        raise ValueError(
            "Model classes must be Sudoku digits from 1 to 9."
        )

    candidate_mask = candidate_features[
        :,
        class_digits - 1,
    ] > 0.5

    if np.any(
        np.sum(candidate_mask, axis=1) == 0
    ):
        raise ValueError(
            "Every sample must contain at least one candidate."
        )

    constrained_probabilities = np.where(
        candidate_mask,
        probabilities,
        0.0,
    )

    probability_sums = np.sum(
        constrained_probabilities,
        axis=1,
        keepdims=True,
    )

    return constrained_probabilities / probability_sums


def analyze_probability_ranking(model: SudokuRandomForest, X: np.ndarray, y: np.ndarray,
                                n_bins: int = 10, candidate_constrained: bool = False) -> ProbabilityRankingResult:
    """Analyze ranking and calibration of model probabilities."""
    if len(X) == 0:
        raise ValueError(
            "At least one evaluation sample is required."
        )

    if len(X) != len(y):
        raise ValueError(
            "Features and targets must have matching lengths."
        )

    probabilities = model.predict_probabilities(X)
    classes = model.classes

    if candidate_constrained:
        probabilities = apply_candidate_constraints(
            probabilities,
            classes,
            X,
        )
    class_indices = {
        int(digit): index
        for index, digit in enumerate(classes)
    }

    try:
        target_indices = np.asarray(
            [
                class_indices[int(target)]
                for target in y
            ],
            dtype=int,
        )
    except KeyError as error:
        raise ValueError(
            "Every target must be represented in model classes."
        ) from error

    ranked_indices = np.argsort(
        -probabilities,
        axis=1,
    )

    target_ranks = (
        np.argmax(
            ranked_indices
            == target_indices[:, np.newaxis],
            axis=1,
        )
        + 1
    )

    predicted_indices = ranked_indices[:, 0]
    predicted_digits = classes[predicted_indices]
    confidences = probabilities[
        np.arange(len(probabilities)),
        predicted_indices,
    ]
    correct_predictions = predicted_digits == y

    return ProbabilityRankingResult(
        sample_count=len(y),
        top_1_accuracy=float(
            np.mean(target_ranks <= 1)
        ),
        top_2_accuracy=float(
            np.mean(target_ranks <= 2)
        ),
        top_3_accuracy=float(
            np.mean(target_ranks <= 3)
        ),
        mean_reciprocal_rank=float(
            np.mean(1.0 / target_ranks)
        ),
        mean_confidence=float(
            np.mean(confidences)
        ),
        expected_calibration_error=(
            calculate_expected_calibration_error(
                confidences,
                correct_predictions,
                n_bins=n_bins,
            )
        ),
        log_loss=float(
            log_loss(
                y,
                probabilities,
                labels=classes,
            )
        ),
    )
