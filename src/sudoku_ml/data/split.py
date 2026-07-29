from dataclasses import dataclass

import numpy as np

from sudoku_ml.dataset.generator import create_diverse_dataset
from sudoku_ml.preprocessing.features import create_features_and_targets


@dataclass
class MLDataSplit:
    """Store training and test data for machine learning."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def create_train_test_split(num_solutions: int = 100, test_size: float = 0.2,
                            removal_rate: float = 0.5, random_seed: int | None = None) -> MLDataSplit:
    """Create train and test data using a solution-level split."""
    if num_solutions <= 1:
        raise ValueError("At least two Sudoku solutions are required.")

    if not 0.0 < test_size < 1.0:
        raise ValueError("Test size must be between 0 and 1.")

    dataset = create_diverse_dataset(
        num_samples=num_solutions,
        removal_rate=removal_rate,
        random_seed=random_seed,
    )

    rng = np.random.default_rng(random_seed)

    indices = np.arange(num_solutions)
    rng.shuffle(indices)

    test_count = max(1, int(num_solutions * test_size))

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    train_dataset = type(dataset)(
        puzzles=dataset.puzzles[train_indices],
        solutions=dataset.solutions[train_indices],
    )

    test_dataset = type(dataset)(
        puzzles=dataset.puzzles[test_indices],
        solutions=dataset.solutions[test_indices],
    )

    X_train, y_train = create_features_and_targets(train_dataset)
    X_test, y_test = create_features_and_targets(test_dataset)

    return MLDataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )