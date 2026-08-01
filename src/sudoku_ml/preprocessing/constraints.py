import numpy as np


def get_candidates(grid: np.ndarray, row: int, column: int) -> set[int]:
    """Return valid candidate digits for an empty Sudoku cell."""
    if grid[row, column] != 0:
        raise ValueError("Candidates can only be calculated for empty cells.")

    candidates = set(range(1, 10))

    # Remove digits already used in the same row.
    candidates -= set(grid[row, :])

    # Remove digits already used in the same column.
    candidates -= set(grid[:, column])

    # Remove digits already used in the same 3x3 block.
    block_row = (row // 3) * 3
    block_column = (column // 3) * 3

    block = grid[
        block_row : block_row + 3,
        block_column : block_column + 3,
    ]

    candidates -= set(block.flat)

    # Zero represents an empty cell and is not a valid candidate.
    candidates.discard(0)

    return candidates
