import numpy as np

from sudoku_ml.sudoku_generator import generate_solved_grid


def test_generated_grid_is_complete() -> None:
    grid = generate_solved_grid(random_seed=42)

    assert grid.is_complete()


def test_generated_grid_is_valid() -> None:
    grid = generate_solved_grid(random_seed=42)

    assert grid.is_valid()


def test_generated_grid_contains_digits_one_to_nine() -> None:
    grid = generate_solved_grid(random_seed=42)

    unique_values = np.unique(grid.values)

    np.testing.assert_array_equal(unique_values, np.arange(1, 10))


def test_same_seed_produces_same_grid() -> None:
    first = generate_solved_grid(random_seed=42)
    second = generate_solved_grid(random_seed=42)

    np.testing.assert_array_equal(first.values, second.values)


def test_different_seeds_produce_different_grids() -> None:
    first = generate_solved_grid(random_seed=42)
    second = generate_solved_grid(random_seed=43)

    assert not np.array_equal(first.values, second.values)
