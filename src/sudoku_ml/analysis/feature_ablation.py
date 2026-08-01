from dataclasses import dataclass

import numpy as np

from sudoku_ml.analysis.probability_ranking import (
    ProbabilityRankingResult,
    analyze_probability_ranking,
)
from sudoku_ml.model.random_forest import SudokuRandomForest


FEATURE_CONFIGURATIONS = (
    ("Grid + position", 82),
    ("Candidate indicators", 91),
    ("Candidate interactions", 118),
)

SUPPORTED_FEATURE_COUNTS = {
    feature_count
    for _, feature_count in FEATURE_CONFIGURATIONS
}


@dataclass(frozen=True)
class FeatureAblationResult:
    """Store the result for one feature configuration."""

    name: str
    feature_count: int
    ranking: ProbabilityRankingResult


def select_features(X: np.ndarray, feature_count: int) -> np.ndarray:
    """Select the requested prefix of the feature vector."""
    if feature_count not in SUPPORTED_FEATURE_COUNTS:
        raise ValueError(
            f"Unsupported feature count: {feature_count}."
        )

    if X.ndim != 2:
        raise ValueError(
            "Feature data must be a two-dimensional array."
        )

    if X.shape[1] < feature_count:
        raise ValueError(
            f"Feature data does not contain {feature_count} features."
        )

    return X[:, :feature_count]


def evaluate_feature_ablation(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                              y_test: np.ndarray, n_estimators: int = 100, random_seed: int | None = None) -> tuple[FeatureAblationResult, ...]:
    """Train and evaluate one model per feature configuration."""
    results: list[FeatureAblationResult] = []

    for name, feature_count in FEATURE_CONFIGURATIONS:
        selected_X_train = select_features(
            X_train,
            feature_count,
        )
        selected_X_test = select_features(
            X_test,
            feature_count,
        )

        model = SudokuRandomForest(
            n_estimators=n_estimators,
            random_seed=random_seed,
        )
        model.fit_arrays(
            selected_X_train,
            y_train,
        )

        ranking = analyze_probability_ranking(
            model,
            selected_X_test,
            y_test,
        )

        results.append(
            FeatureAblationResult(
                name=name,
                feature_count=feature_count,
                ranking=ranking,
            )
        )

    return tuple(results)
