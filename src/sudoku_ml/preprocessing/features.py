import numpy as np

from sudoku_ml.dataset.generator import SudokuDataset


def create_features_and_targets(dataset: SudokuDataset) -> tuple[np.ndarray, np.ndarray]:
    """Convert Sudoku puzzles into cell-level ML features and targets."""
    features: list[np.ndarray] = []
    targets: list[int] = []

    for puzzle, solution in zip(dataset.puzzles, dataset.solutions):
        empty_cells = np.argwhere(puzzle == 0)

        for row, column in empty_cells:
            grid_features = puzzle.flatten().astype(np.float32)

            cell_index = row * 9 + column

            feature_vector = np.append(grid_features, cell_index)

            features.append(feature_vector)
            targets.append(int(solution[row, column]))

    return np.asarray(features), np.asarray(targets)
