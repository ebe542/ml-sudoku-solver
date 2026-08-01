import numpy as np

from sudoku_ml.dataset.generator import SudokuDataset
from sudoku_ml.preprocessing.constraints import get_candidates


def create_features_and_targets(dataset: SudokuDataset) -> tuple[np.ndarray, np.ndarray]:
    """Convert Sudoku puzzles into cell-level ML features and targets."""
    features: list[np.ndarray] = []
    targets: list[int] = []

    for puzzle, solution in zip(dataset.puzzles, dataset.solutions):
        empty_cells = np.argwhere(puzzle == 0)

        for row, column in empty_cells:
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

            feature_vector = np.concatenate(
                [
                    grid_features,
                    np.array([cell_index], dtype=np.float32),
                    candidate_features,
                ]
            )

            features.append(feature_vector)
            targets.append(int(solution[row, column]))

    return np.asarray(features), np.asarray(targets)
