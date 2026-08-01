import numpy as np

from sudoku_ml.dataset.generator import SudokuDataset
from sudoku_ml.preprocessing.constraints import (
    get_candidate_interactions,
    get_candidates,
)


def create_feature_vector(puzzle: np.ndarray, row: int, column: int) -> np.ndarray:
    """Create the feature vector for one empty Sudoku cell."""
    grid_features = puzzle.flatten().astype(np.float32)
    cell_index = row * 9 + column

    candidates = get_candidates(puzzle, row, column)

    candidate_features = np.array(
        [
            1.0 if digit in candidates else 0.0
            for digit in range(1, 10)
        ],
        dtype=np.float32,
    )

    interaction_features = get_candidate_interactions(
        puzzle,
        row,
        column,
    )

    return np.concatenate(
        [
            grid_features,
            np.array([cell_index], dtype=np.float32),
            candidate_features,
            interaction_features,
        ]
    )


def create_features_and_targets(dataset: SudokuDataset) -> tuple[np.ndarray, np.ndarray]:
    """Convert Sudoku puzzles into cell-level ML features and targets."""
    features: list[np.ndarray] = []
    targets: list[int] = []

    for puzzle, solution in zip(dataset.puzzles, dataset.solutions):
        empty_cells = np.argwhere(puzzle == 0)

        for row, column in empty_cells:
            features.append(
                create_feature_vector(
                    puzzle,
                    int(row),
                    int(column),
                )
            )
            targets.append(int(solution[row, column]))

    return np.asarray(features), np.asarray(targets)


def create_grouped_features_and_targets(dataset: SudokuDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create features, targets, and source-Sudoku group identifiers."""
    features: list[np.ndarray] = []
    targets: list[int] = []
    groups: list[int] = []

    for sudoku_id, (puzzle, solution) in enumerate(
        zip(dataset.puzzles, dataset.solutions)
    ):
        empty_cells = np.argwhere(puzzle == 0)

        for row, column in empty_cells:
            features.append(
                create_feature_vector(
                    puzzle,
                    int(row),
                    int(column),
                )
            )
            targets.append(int(solution[row, column]))
            groups.append(sudoku_id)

    return (
        np.asarray(features),
        np.asarray(targets),
        np.asarray(groups),
    )
