from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

from sudoku_ml.dataset.generator import create_diverse_dataset
from sudoku_ml.model.random_forest import SudokuRandomForest
from sudoku_ml.preprocessing.features import (
    create_grouped_features_and_targets,
)


@dataclass(frozen=True)
class CrossValidationResult:
    """Store grouped cross-validation accuracy results."""

    fold_accuracies: np.ndarray

    @property
    def mean_accuracy(self) -> float:
        """Return the mean accuracy across folds."""
        return float(np.mean(self.fold_accuracies))

    @property
    def standard_deviation(self) -> float:
        """Return the accuracy standard deviation across folds."""
        return float(np.std(self.fold_accuracies))

    @property
    def minimum_accuracy(self) -> float:
        """Return the minimum fold accuracy."""
        return float(np.min(self.fold_accuracies))

    @property
    def maximum_accuracy(self) -> float:
        """Return the maximum fold accuracy."""
        return float(np.max(self.fold_accuracies))


def evaluate_group_cross_validation(num_solutions: int = 100, n_splits: int = 5, removal_rate: float = 0.5,
                                    n_estimators: int = 100, random_seed: int | None = 42) -> CrossValidationResult:
    """Evaluate the Random Forest using solution-level GroupKFold."""
    if n_splits < 2:
        raise ValueError("At least two folds are required.")

    if num_solutions < n_splits:
        raise ValueError(
            "Number of Sudoku solutions must be at least the number of folds."
        )

    dataset = create_diverse_dataset(
        num_samples=num_solutions,
        removal_rate=removal_rate,
        random_seed=random_seed,
    )

    features, targets, groups = create_grouped_features_and_targets(
        dataset
    )

    splitter = GroupKFold(n_splits=n_splits)
    fold_accuracies: list[float] = []

    for train_indices, test_indices in splitter.split(
        features,
        targets,
        groups,
    ):
        model = SudokuRandomForest(
            n_estimators=n_estimators,
            random_seed=random_seed,
        )

        training_data = type(
            "TrainingData",
            (),
            {
                "X_train": features[train_indices],
                "y_train": targets[train_indices],
            },
        )()

        model.fit(training_data)

        predictions = model.predict(features[test_indices])

        fold_accuracies.append(
            float(
                accuracy_score(
                    targets[test_indices],
                    predictions,
                )
            )
        )

    return CrossValidationResult(
        fold_accuracies=np.asarray(fold_accuracies),
    )
