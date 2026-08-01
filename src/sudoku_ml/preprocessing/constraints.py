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

def get_candidate_interactions(grid: np.ndarray, row: int, column: int) -> np.ndarray:
    """Count candidate occurrences in peer cells of a target cell."""
    if grid[row, column] != 0:
        raise ValueError(
            "Candidate interactions can only be calculated for empty cells."
        )

    row_counts = np.zeros(9, dtype=np.float32)
    column_counts = np.zeros(9, dtype=np.float32)
    block_counts = np.zeros(9, dtype=np.float32)

    # Count candidate occurrences in other empty cells of the same row.
    for peer_column in range(9):
        if peer_column == column or grid[row, peer_column] != 0:
            continue

        for digit in get_candidates(grid, row, peer_column):
            row_counts[digit - 1] += 1.0

    # Count candidate occurrences in other empty cells of the same column.
    for peer_row in range(9):
        if peer_row == row or grid[peer_row, column] != 0:
            continue

        for digit in get_candidates(grid, peer_row, column):
            column_counts[digit - 1] += 1.0

    # Count candidate occurrences in other empty cells of the same 3x3 block.
    block_row = (row // 3) * 3
    block_column = (column // 3) * 3

    for peer_row in range(block_row, block_row + 3):
        for peer_column in range(block_column, block_column + 3):
            if peer_row == row and peer_column == column:
                continue

            if grid[peer_row, peer_column] != 0:
                continue

            for digit in get_candidates(grid, peer_row, peer_column):
                block_counts[digit - 1] += 1.0

    return np.concatenate(
        [
            row_counts,
            column_counts,
            block_counts,
        ]
    )
