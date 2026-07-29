import numpy as np
import pytest

from sudoku_ml.dataset.generator import (
    create_dataset,
    create_diverse_dataset,
)
from sudoku_ml.grid import SudokuGrid


@pytest.fixture
def solved_grid() -> SudokuGrid:
    values = np.array(
        [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9],
        ]
    )

    return SudokuGrid(values)


def test_dataset_shapes(solved_grid: SudokuGrid) -> None:
    dataset = create_dataset(
        solved_grid,
        num_samples=10,
        removal_rate=0.5,
        random_seed=42,
    )

    assert dataset.puzzles.shape == (10, 9, 9)
    assert dataset.solutions.shape == (10, 9, 9)


def test_solutions_match_source_grid(solved_grid: SudokuGrid) -> None:
    dataset = create_dataset(
        solved_grid,
        num_samples=5,
        random_seed=42,
    )

    for solution in dataset.solutions:
        np.testing.assert_array_equal(solution, solved_grid.values)


def test_removed_cells_are_zero(solved_grid: SudokuGrid) -> None:
    dataset = create_dataset(
        solved_grid,
        num_samples=1,
        removal_rate=0.5,
        random_seed=42,
    )

    puzzle = dataset.puzzles[0]

    assert np.count_nonzero(puzzle == 0) == 40


def test_incomplete_puzzles_are_valid(solved_grid: SudokuGrid) -> None:
    dataset = create_dataset(
        solved_grid,
        num_samples=5,
        removal_rate=0.5,
        random_seed=42,
    )

    for puzzle in dataset.puzzles:
        assert SudokuGrid(puzzle).is_valid()


def test_incomplete_source_grid_is_rejected() -> None:
    values = np.zeros((9, 9), dtype=int)
    grid = SudokuGrid(values)

    with pytest.raises(ValueError):
        create_dataset(grid)

def test_diverse_dataset_has_expected_shape() -> None:
    dataset = create_diverse_dataset(
        num_samples=10,
        removal_rate=0.5,
        random_seed=42,
    )

    assert dataset.puzzles.shape == (10, 9, 9)
    assert dataset.solutions.shape == (10, 9, 9)


def test_diverse_dataset_contains_valid_solutions() -> None:
    dataset = create_diverse_dataset(
        num_samples=10,
        removal_rate=0.5,
        random_seed=42,
    )

    for solution in dataset.solutions:
        grid = SudokuGrid(solution)

        assert grid.is_complete()
        assert grid.is_valid()


def test_diverse_dataset_contains_different_solutions() -> None:
    dataset = create_diverse_dataset(
        num_samples=10,
        removal_rate=0.5,
        random_seed=42,
    )

    unique_solutions = {
        solution.tobytes()
        for solution in dataset.solutions
    }

    assert len(unique_solutions) == 10


def test_diverse_dataset_is_reproducible() -> None:
    first = create_diverse_dataset(
        num_samples=10,
        removal_rate=0.5,
        random_seed=42,
    )

    second = create_diverse_dataset(
        num_samples=10,
        removal_rate=0.5,
        random_seed=42,
    )

    np.testing.assert_array_equal(
        first.puzzles,
        second.puzzles,
    )

    np.testing.assert_array_equal(
        first.solutions,
        second.solutions,
    )
