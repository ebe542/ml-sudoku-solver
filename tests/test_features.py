import numpy as np

from sudoku_ml.dataset.generator import create_dataset
from sudoku_ml.grid import SudokuGrid
from sudoku_ml.preprocessing.features import create_features_and_targets


def test_features_match_empty_cells() -> None:
    values = np.array(
        [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9],
        ]
    )

    solution = SudokuGrid(values)

    dataset = create_dataset(
        solution,
        num_samples=1,
        removal_rate=0.5,
        random_seed=42,
    )

    features, targets = create_features_and_targets(dataset)

    assert features.shape == (40, 91)
    assert targets.shape == (40,)


def test_feature_contains_cell_position() -> None:
    values = np.array(
        [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9],
        ]
    )

    solution = SudokuGrid(values)

    dataset = create_dataset(
        solution,
        num_samples=1,
        removal_rate=0.5,
        random_seed=42,
    )

    features, _ = create_features_and_targets(dataset)

    cell_positions = features[:, -1]

    assert np.all((cell_positions >= 0) & (cell_positions < 81))


def test_targets_are_valid_digits() -> None:
    values = np.array(
        [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9],
        ]
    )

    dataset = create_dataset(
        SudokuGrid(values),
        num_samples=2,
        removal_rate=0.5,
        random_seed=42,
    )

    _, targets = create_features_and_targets(dataset)

    assert np.all((targets >= 1) & (targets <= 9))


def test_feature_contains_candidate_information() -> None:
    values = np.array(
        [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9],
        ]
    )

    solution = SudokuGrid(values)

    dataset = create_dataset(
        solution,
        num_samples=1,
        removal_rate=0.5,
        random_seed=42,
    )

    features, _ = create_features_and_targets(dataset)

    candidate_features = features[:, 82:]

    assert candidate_features.shape[1] == 9
    assert np.all((candidate_features == 0) | (candidate_features == 1))