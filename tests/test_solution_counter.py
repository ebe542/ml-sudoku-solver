import numpy as np
import pytest

from sudoku_ml.grid import SudokuGrid
from sudoku_ml.solution_counter import count_solutions, has_unique_solution


UNIQUE_PUZZLE = SudokuGrid(
    np.array(
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
)


def test_count_solutions_finds_unique_solution() -> None:
    assert count_solutions(UNIQUE_PUZZLE) == 1
    assert has_unique_solution(UNIQUE_PUZZLE)


def test_count_solutions_stops_at_limit() -> None:
    empty_puzzle = SudokuGrid(np.zeros((9, 9), dtype=int))

    assert count_solutions(empty_puzzle, limit=2) == 2
    assert not has_unique_solution(empty_puzzle)


def test_count_solutions_returns_zero_for_invalid_grid() -> None:
    values = np.zeros((9, 9), dtype=int)
    values[0, :2] = 1

    assert count_solutions(SudokuGrid(values)) == 0


def test_count_solutions_does_not_modify_input() -> None:
    original = UNIQUE_PUZZLE.values.copy()

    count_solutions(UNIQUE_PUZZLE)

    assert np.array_equal(UNIQUE_PUZZLE.values, original)


def test_count_solutions_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        count_solutions(UNIQUE_PUZZLE, limit=0)
