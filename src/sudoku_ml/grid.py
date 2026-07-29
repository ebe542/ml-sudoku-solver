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

    def is_valid(self) -> bool:
        """Return True if the grid has no conflicting values."""
        return (
            self._rows_are_valid()
            and self._columns_are_valid()
            and self._blocks_are_valid()
        )

    def _rows_are_valid(self) -> bool:
        """Check all rows for duplicate non-zero values."""
        return all(self._unit_is_valid(row) for row in self.values)

    def _columns_are_valid(self) -> bool:
        """Check all columns for duplicate non-zero values."""
        return all(self._unit_is_valid(column) for column in self.values.T)

    def _blocks_are_valid(self) -> bool:
        """Check all 3x3 blocks for duplicate non-zero values."""
        for row in range(0, 9, 3):
            for column in range(0, 9, 3):
                block = self.values[row : row + 3, column : column + 3]
                if not self._unit_is_valid(block.flatten()):
                    return False

        return True

    @staticmethod
    def _unit_is_valid(unit: np.ndarray) -> bool:
        """Return True if a row, column, or block has no duplicates."""
        values = unit[unit != 0]
        return len(values) == len(set(values))