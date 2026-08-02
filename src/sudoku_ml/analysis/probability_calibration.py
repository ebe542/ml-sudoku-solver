from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from sudoku_ml.analysis.probability_ranking import (
    ProbabilityRankingResult,
    analyze_probability_ranking,
)
from sudoku_ml.model.random_forest import SudokuRandomForest


SUPPORTED_CALIBRATION_METHODS = (
    "sigmoid",
    "isotonic",
)


@dataclass(frozen=True)
class ProbabilityCalibrationResult:
    """Store raw and constrained metrics for one method."""

    name: str
    raw: ProbabilityRankingResult
    candidate_constrained: ProbabilityRankingResult


class CalibratedProbabilityModel:
    """Adapt a calibrated classifier to the ranking interface."""

    def __init__(self, model: CalibratedClassifierCV) -> None:
        self.model = model

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated class probabilities."""
        return self.model.predict_proba(X)

    @property
    def classes(self) -> np.ndarray:
        """Return digit classes learned by the classifier."""
        return self.model.classes_


def calibrate_probability_model(model: SudokuRandomForest, X_calibration: np.ndarray,
                                y_calibration: np.ndarray, method: str = "sigmoid") -> CalibratedProbabilityModel:
    """Calibrate an already fitted Random Forest."""
    if method not in SUPPORTED_CALIBRATION_METHODS:
        raise ValueError(
            f"Unsupported calibration method: {method}."
        )

    if len(X_calibration) == 0:
        raise ValueError(
            "At least one calibration sample is required."
        )

    if len(X_calibration) != len(y_calibration):
        raise ValueError(
            "Calibration features and targets must have "
            "matching lengths."
        )

    calibration_indices = np.arange(len(X_calibration))

    calibrated_classifier = CalibratedClassifierCV(
        estimator=FrozenEstimator(model.model),
        method=method,
        cv=[(calibration_indices, calibration_indices)],
    )
    calibrated_classifier.fit(
        X_calibration,
        y_calibration,
    )

    return CalibratedProbabilityModel(calibrated_classifier)

def evaluate_probability_calibration(model: SudokuRandomForest, X_calibration: np.ndarray,
                                     y_calibration: np.ndarray, X_evaluation: np.ndarray,
                                     y_evaluation: np.ndarray) -> tuple[ProbabilityCalibrationResult, ...]:
    """Compare raw and calibrated model probabilities."""
    evaluated_models = [
        (
            "Raw",
            model,
        )
    ]

    for method in SUPPORTED_CALIBRATION_METHODS:
        calibrated_model = calibrate_probability_model(
            model,
            X_calibration,
            y_calibration,
            method=method,
        )

        evaluated_models.append(
            (
                method.title(),
                calibrated_model,
            )
        )

    results: list[
        ProbabilityCalibrationResult
    ] = []

    for name, evaluated_model in evaluated_models:
        raw_ranking = analyze_probability_ranking(
            evaluated_model,
            X_evaluation,
            y_evaluation,
        )

        constrained_ranking = analyze_probability_ranking(
            evaluated_model,
            X_evaluation,
            y_evaluation,
            candidate_constrained=True,
        )

        results.append(
            ProbabilityCalibrationResult(
                name=name,
                raw=raw_ranking,
                candidate_constrained=(
                    constrained_ranking
                ),
            )
        )

    return tuple(results)
