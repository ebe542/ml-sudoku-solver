from dataclasses import dataclass

import numpy as np

from sudoku_ml.grid import SudokuGrid
from sudoku_ml.sudoku_generator import generate_solved_grid


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

    # Determine how many cells should be hidden in each puzzle.
    cells_to_remove = int(81 * removal_rate)

    for puzzle in puzzles:
        # Randomly select cells and replace their values with 0 (empty).
        indices = rng.choice(81, size=cells_to_remove, replace=False)
        puzzle.flat[indices] = 0

    return SudokuDataset(puzzles=puzzles, solutions=solutions)


def create_diverse_dataset(num_samples: int = 100, removal_rate: float = 0.5, random_seed: int | None = None) -> SudokuDataset:
    """Generate a dataset from multiple independent Sudoku solutions."""
    if num_samples <= 0:
        raise ValueError("Number of samples must be positive.")

    rng = np.random.default_rng(random_seed)

    puzzles = []
    solutions = []

    for _ in range(num_samples):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))

        solution = generate_solved_grid(random_seed=seed)

        dataset = create_dataset(
            solution,
            num_samples=1,
            removal_rate=removal_rate,
            random_seed=seed,
        )

        puzzles.append(dataset.puzzles[0])
        solutions.append(dataset.solutions[0])

    return SudokuDataset(
        puzzles=np.asarray(puzzles),
        solutions=np.asarray(solutions),
    )
