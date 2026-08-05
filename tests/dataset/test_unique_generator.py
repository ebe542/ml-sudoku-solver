import numpy as np
import pytest

from sudoku_ml.dataset.unique_generator import (
    create_unique_dataset,
    create_unique_puzzle,
)
from sudoku_ml.grid import SudokuGrid
from sudoku_ml.solution_counter import has_unique_solution
from sudoku_ml.sudoku_generator import generate_solved_grid


def test_unique_puzzle_has_requested_removals_and_one_solution() -> None:
    solution = generate_solved_grid(random_seed=42)

    puzzle = create_unique_puzzle(
        solution,
        removal_rate=0.5,
        random_seed=42,
    )

    assert len(puzzle.empty_cells) == int(81 * 0.5)
    assert puzzle.is_valid()
    assert has_unique_solution(puzzle)
    assert np.all(
        (puzzle.values == 0)
        | (puzzle.values == solution.values)
    )


def test_unique_puzzle_generation_is_reproducible() -> None:
    solution = generate_solved_grid(random_seed=42)

    first = create_unique_puzzle(solution, random_seed=123)
    second = create_unique_puzzle(solution, random_seed=123)

    assert np.array_equal(first.values, second.values)


def test_zero_removal_rate_returns_complete_copy() -> None:
    solution = generate_solved_grid(random_seed=42)

    puzzle = create_unique_puzzle(
        solution,
        removal_rate=0.0,
        random_seed=123,
    )

    assert puzzle is not solution
    assert np.array_equal(puzzle.values, solution.values)


def test_unique_puzzle_does_not_modify_solution() -> None:
    solution = generate_solved_grid(random_seed=42)
    original = solution.values.copy()

    create_unique_puzzle(solution, random_seed=123)

    assert np.array_equal(solution.values, original)


def test_unique_puzzle_rejects_incomplete_source() -> None:
    with pytest.raises(ValueError, match="complete and valid"):
        create_unique_puzzle(SudokuGrid(np.zeros((9, 9), dtype=int)))


@pytest.mark.parametrize("removal_rate", [-0.1, 1.0, 1.1])
def test_unique_puzzle_rejects_invalid_removal_rate(removal_rate: float) -> None:
    solution = generate_solved_grid(random_seed=42)

    with pytest.raises(ValueError, match="between 0 and 1"):
        create_unique_puzzle(
            solution,
            removal_rate=removal_rate,
        )


def test_unique_dataset_contains_unique_puzzles() -> None:
    dataset = create_unique_dataset(
        num_samples=3,
        removal_rate=0.3,
        random_seed=42,
    )

    assert dataset.puzzles.shape == (3, 9, 9)
    assert dataset.solutions.shape == (3, 9, 9)

    for puzzle_values, solution_values in zip(
        dataset.puzzles,
        dataset.solutions,
    ):
        puzzle = SudokuGrid(puzzle_values)
        solution = SudokuGrid(solution_values)

        assert has_unique_solution(puzzle)
        assert solution.is_complete()
        assert solution.is_valid()


def test_unique_dataset_is_reproducible() -> None:
    first = create_unique_dataset(
        num_samples=2,
        removal_rate=0.3,
        random_seed=42,
    )
    second = create_unique_dataset(
        num_samples=2,
        removal_rate=0.3,
        random_seed=42,
    )

    assert np.array_equal(first.puzzles, second.puzzles)
    assert np.array_equal(first.solutions, second.solutions)


def test_unique_dataset_rejects_invalid_sample_count() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        create_unique_dataset(num_samples=0)
