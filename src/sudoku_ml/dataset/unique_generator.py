import numpy as np

from sudoku_ml.dataset.generator import SudokuDataset
from sudoku_ml.grid import SudokuGrid
from sudoku_ml.solution_counter import has_unique_solution
from sudoku_ml.sudoku_generator import generate_solved_grid


def create_unique_puzzle(solution: SudokuGrid, removal_rate: float = 0.5,
                         random_seed: int | None = None) -> SudokuGrid:
    """Remove cells while preserving exactly one Sudoku solution."""
    if not solution.is_complete() or not solution.is_valid():
        raise ValueError("The source grid must be complete and valid.")

    if not 0.0 <= removal_rate < 1.0:
        raise ValueError("Removal rate must be between 0 and 1.")

    rng = np.random.default_rng(random_seed)
    puzzle_values = solution.values.copy()
    target_removals = int(81 * removal_rate)
    removed_cells = 0

    if target_removals == 0:
        return SudokuGrid(puzzle_values)

    for cell_index in rng.permutation(81):
        previous_value = int(puzzle_values.flat[cell_index])
        puzzle_values.flat[cell_index] = 0

        if has_unique_solution(SudokuGrid(puzzle_values)):
            removed_cells += 1
        else:
            puzzle_values.flat[cell_index] = previous_value

        if removed_cells == target_removals:
            return SudokuGrid(puzzle_values)

    raise RuntimeError(
        "Could not reach the requested removal rate while preserving uniqueness."
    )


def create_unique_dataset(num_samples: int = 100, removal_rate: float = 0.5,
                          random_seed: int | None = None) -> SudokuDataset:
    """Generate diverse Sudoku puzzles with exactly one solution."""
    if num_samples <= 0:
        raise ValueError("Number of samples must be positive.")

    rng = np.random.default_rng(random_seed)
    puzzles: list[np.ndarray] = []
    solutions: list[np.ndarray] = []

    for _ in range(num_samples):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        solution = generate_solved_grid(random_seed=seed)
        puzzle = create_unique_puzzle(
            solution,
            removal_rate=removal_rate,
            random_seed=seed,
        )

        puzzles.append(puzzle.values)
        solutions.append(solution.values)

    return SudokuDataset(
        puzzles=np.asarray(puzzles),
        solutions=np.asarray(solutions),
    )
