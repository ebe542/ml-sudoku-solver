from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sudoku_ml.analysis.probability_ranking import (
    ProbabilityRankingResult,
    analyze_probability_ranking,
)


COMPARISON_MODEL_NAMES = (
    "Logistic Regression",
    "Random Forest",
    "Extra Trees",
    "Histogram Gradient Boosting",
)


@dataclass(frozen=True)
class ModelComparisonResult:
    """Store ranking metrics for one classifier."""

    name: str
    raw: ProbabilityRankingResult
    candidate_constrained: ProbabilityRankingResult


@dataclass(frozen=True)
class TimedModelComparisonResult:
    """Store metrics and runtimes for one classifier."""

    name: str
    training_seconds: float
    inference_seconds: float
    raw: ProbabilityRankingResult
    candidate_constrained: ProbabilityRankingResult


class ProbabilityModelAdapter:
    """Provide a shared probability-model interface."""

    def __init__(self, name: str, estimator: Any) -> None:
        self.name = name
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the wrapped classifier."""
        self.estimator.fit(X, y)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        return self.estimator.predict_proba(X)

    @property
    def classes(self) -> np.ndarray:
        """Return learned digit classes."""
        return self.estimator.classes_


def create_comparison_models(n_estimators: int = 100, random_seed: int | None = None) -> tuple[ProbabilityModelAdapter, ...]:
    """Create classifiers used in the model comparison."""
    if n_estimators <= 0:
        raise ValueError(
            "Number of estimators must be positive."
        )

    logistic_regression = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1_000,
            random_state=random_seed,
        ),
    )

    random_forest = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_seed,
        n_jobs=-1,
    )

    extra_trees = ExtraTreesClassifier(
        n_estimators=n_estimators,
        random_state=random_seed,
        n_jobs=-1,
    )

    histogram_gradient_boosting = (
        HistGradientBoostingClassifier(
            max_iter=n_estimators,
            random_state=random_seed,
        )
    )

    estimators = (
        logistic_regression,
        random_forest,
        extra_trees,
        histogram_gradient_boosting,
    )

    return tuple(
        ProbabilityModelAdapter(
            name=name,
            estimator=estimator,
        )
        for name, estimator in zip(
            COMPARISON_MODEL_NAMES,
            estimators,
        )
    )

def evaluate_models(models: tuple[Any, ...], X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray) -> tuple[ModelComparisonResult, ...]:
    """Train and evaluate the provided classifiers."""
    results: list[ModelComparisonResult] = []

    for model in models:
        model.fit(
            X_train,
            y_train,
        )

        raw_ranking = analyze_probability_ranking(
            model,
            X_test,
            y_test,
        )

        constrained_ranking = analyze_probability_ranking(
            model,
            X_test,
            y_test,
            candidate_constrained=True,
        )

        results.append(
            ModelComparisonResult(
                name=model.name,
                raw=raw_ranking,
                candidate_constrained=(
                    constrained_ranking
                ),
            )
        )

    return tuple(results)

def evaluate_models_with_timing(models: tuple[Any, ...], X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray) -> tuple[TimedModelComparisonResult, ...]:
    """Train and evaluate classifiers with runtime measurement."""
    results: list[TimedModelComparisonResult] = []

    for model in models:
        training_start = perf_counter()

        model.fit(X_train, y_train)

        training_seconds = (
            perf_counter() - training_start
        )

        inference_start = perf_counter()

        model.predict_probabilities(
            X_test
        )

        inference_seconds = (
            perf_counter() - inference_start
        )

        raw_ranking = analyze_probability_ranking(
            model,
            X_test,
            y_test,
        )

        constrained_ranking = analyze_probability_ranking(
            model,
            X_test,
            y_test,
            candidate_constrained=True,
        )

        results.append(
            TimedModelComparisonResult(
                name=model.name,
                training_seconds=training_seconds,
                inference_seconds=inference_seconds,
                raw=raw_ranking,
                candidate_constrained=(
                    constrained_ranking
                ),
            )
        )

    return tuple(results)
