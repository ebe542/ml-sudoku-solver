from dataclasses import dataclass

import numpy as np

from sudoku_ml.grid import SudokuGrid


@dataclass
class SudokuDataset:
    """Store Sudoku inputs and their corresponding solutions."""

    puzzles: np.ndarray
    solutions: np.ndarray


def create_dataset(solution: SudokuGrid, num_samples: int = 100, removal_rate: float = 0.5,
                   random_seed: int | None = None) -> SudokuDataset:
    """Create incomplete Sudoku puzzles from a solved Sudoku grid."""
    if not solution.is_complete():
        raise ValueError("The source grid must be complete.")

    if not solution.is_valid():
        raise ValueError("The source grid must be valid.")

    if not 0.0 <= removal_rate <= 1.0:
        raise ValueError("Removal rate must be between 0 and 1.")

    rng = np.random.default_rng(random_seed)

    puzzles = np.repeat(
        solution.values[np.newaxis, :, :],
        num_samples,
        axis=0,
    )

    solutions = puzzles.copy()

    cells_to_remove = int(81 * removal_rate)

    for puzzle in puzzles:
        indices = rng.choice(81, size=cells_to_remove, replace=False)
        puzzle.flat[indices] = 0

    return SudokuDataset(puzzles=puzzles, solutions=solutions)
