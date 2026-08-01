import numpy as np

from sudoku_ml.grid import SudokuGrid
from sudoku_ml.preprocessing.constraints import get_candidates


def count_solutions(puzzle: SudokuGrid, limit: int = 2) -> int:
    """Count Sudoku solutions up to a configurable limit."""
    if limit <= 0:
        raise ValueError("Solution limit must be positive.")

    if not puzzle.is_valid():
        return 0

    values = puzzle.values.copy()
    return _count_solutions(values, limit)


def has_unique_solution(puzzle: SudokuGrid) -> bool:
    """Return whether a Sudoku puzzle has exactly one solution."""
    return count_solutions(puzzle, limit=2) == 1


def _count_solutions(grid: np.ndarray, limit: int) -> int:
    choice = _select_cell(grid)

    if choice is None:
        return 1

    row, column, candidates = choice

    if not candidates:
        return 0

    solution_count = 0

    for digit in sorted(candidates):
        grid[row, column] = digit
        solution_count += _count_solutions(
            grid,
            limit - solution_count,
        )
        grid[row, column] = 0

        if solution_count >= limit:
            return limit

    return solution_count


def _select_cell(grid: np.ndarray) -> tuple[int, int, set[int]] | None:
    """Choose the empty cell with the fewest candidates."""
    best: tuple[int, int, set[int]] | None = None

    for row, column in np.argwhere(grid == 0):
        row_index = int(row)
        column_index = int(column)
        candidates = get_candidates(
            grid,
            row_index,
            column_index,
        )

        if not candidates:
            return row_index, column_index, candidates

        if best is None or len(candidates) < len(best[2]):
            best = row_index, column_index, candidates

            if len(candidates) == 1:
                break

    return best
