import numpy as np

from sudoku_ml.grid import SudokuGrid


def generate_solved_grid(random_seed: int | None = None) -> SudokuGrid:
    """Generate a random valid and complete Sudoku grid."""
    rng = np.random.default_rng(random_seed)

    grid = np.zeros((9, 9), dtype=int)

    if not _fill_grid(grid, rng):
        raise RuntimeError("Failed to generate a solved Sudoku grid.")

    return SudokuGrid(grid)


def _fill_grid(grid: np.ndarray, rng: np.random.Generator) -> bool:
    """Fill a Sudoku grid using randomized backtracking."""
    empty_cells = np.argwhere(grid == 0)

    if len(empty_cells) == 0:
        return True

    row, column = empty_cells[0]

    candidates = np.arange(1, 10)
    rng.shuffle(candidates)

    for value in candidates:
        if _is_valid_move(grid, row, column, value):
            grid[row, column] = value

            if _fill_grid(grid, rng):
                return True

            grid[row, column] = 0

    return False


def _is_valid_move(grid: np.ndarray, row: int, column: int, value: int) -> bool:
    """Check whether a value can be placed in a Sudoku cell."""
    if value in grid[row, :]:
        return False

    if value in grid[:, column]:
        return False

    block_row = (row // 3) * 3
    block_column = (column // 3) * 3

    block = grid[
        block_row : block_row + 3,
        block_column : block_column + 3,
    ]

    return value not in block