import numpy as np
import pytest

from sudoku_ml.grid import SudokuGrid


def test_valid_grid_is_created() -> None:
    values = np.zeros((9, 9), dtype=int)

    grid = SudokuGrid(values)

    assert grid.values.shape == (9, 9)


def test_empty_cells_are_detected() -> None:
    values = np.zeros((9, 9), dtype=int)
    values[0, 0] = 5
    values[4, 4] = 7

    grid = SudokuGrid(values)

    assert (0, 0) not in grid.empty_cells
    assert (4, 4) not in grid.empty_cells
    assert len(grid.empty_cells) == 79


def test_complete_grid_is_detected() -> None:
    values = np.ones((9, 9), dtype=int)

    grid = SudokuGrid(values)

    assert grid.is_complete()


def test_incomplete_grid_is_detected() -> None:
    values = np.ones((9, 9), dtype=int)
    values[0, 0] = 0

    grid = SudokuGrid(values)

    assert not grid.is_complete()


def test_invalid_shape_raises_error() -> None:
    values = np.zeros((8, 9), dtype=int)

    with pytest.raises(ValueError):
        SudokuGrid(values)


def test_invalid_value_raises_error() -> None:
    values = np.zeros((9, 9), dtype=int)
    values[0, 0] = 10

    with pytest.raises(ValueError):
        SudokuGrid(values)


def test_valid_sudoku_grid() -> None:
    values = np.array(
        [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ]
    )
    grid = SudokuGrid(values)

    assert grid.is_valid()


def test_duplicate_in_row_is_invalid() -> None:
    values = np.zeros((9, 9), dtype=int)
    values[0, 0] = 5
    values[0, 1] = 5

    grid = SudokuGrid(values)

    assert not grid.is_valid()


def test_duplicate_in_column_is_invalid() -> None:
    values = np.zeros((9, 9), dtype=int)
    values[0, 0] = 5
    values[1, 0] = 5

    grid = SudokuGrid(values)

    assert not grid.is_valid()


def test_duplicate_in_block_is_invalid() -> None:
    values = np.zeros((9, 9), dtype=int)
    values[0, 0] = 5
    values[1, 1] = 5

    grid = SudokuGrid(values)

    assert not grid.is_valid()
