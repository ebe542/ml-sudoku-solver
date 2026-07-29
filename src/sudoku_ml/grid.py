from dataclasses import dataclass

import numpy as np


@dataclass
class SudokuGrid:
    """Represent a 9x9 Sudoku grid."""

    values: np.ndarray

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=int)

        if self.values.shape != (9, 9):
            raise ValueError("Sudoku grid must have shape (9, 9).")

        if np.any((self.values < 0) | (self.values > 9)):
            raise ValueError("Sudoku values must be integers between 0 and 9.")

    @property
    def empty_cells(self) -> list[tuple[int, int]]:
        """Return the coordinates of all empty cells."""
        rows, cols = np.where(self.values == 0)
        return list(zip(rows.tolist(), cols.tolist()))

    def is_complete(self) -> bool:
        """Return True if the grid contains no empty cells."""
        return not np.any(self.values == 0)
    