import numpy as np
import pytest

from sudoku_ml.preprocessing.constraints import (
    get_candidate_interactions,
    get_candidates
)


def test_candidates_respect_row_constraints() -> None:
    grid = np.array(
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

    candidates = get_candidates(grid, 0, 2)

    assert 5 not in candidates
    assert 3 not in candidates
    assert 7 not in candidates


def test_candidates_respect_column_constraints() -> None:
    grid = np.array(
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

    candidates = get_candidates(grid, 0, 2)

    assert 8 not in candidates
    assert 2 in candidates


def test_candidates_respect_block_constraints() -> None:
    grid = np.array(
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

    candidates = get_candidates(grid, 0, 2)

    assert 9 not in candidates
    assert 6 not in candidates


def test_candidates_are_between_one_and_nine() -> None:
    grid = np.zeros((9, 9), dtype=int)

    candidates = get_candidates(grid, 0, 0)

    assert candidates == set(range(1, 10))


def test_candidates_require_empty_cell() -> None:
    grid = np.zeros((9, 9), dtype=int)
    grid[0, 0] = 5

    with pytest.raises(ValueError):
        get_candidates(grid, 0, 0)

def test_candidate_interactions_have_expected_shape() -> None:
    grid = np.zeros((9, 9), dtype=int)

    interactions = get_candidate_interactions(grid, 0, 0)

    assert interactions.shape == (27,)


def test_candidate_interactions_are_non_negative() -> None:
    grid = np.zeros((9, 9), dtype=int)

    interactions = get_candidate_interactions(grid, 0, 0)

    assert np.all(interactions >= 0)


def test_candidate_interactions_require_empty_cell() -> None:
    grid = np.zeros((9, 9), dtype=int)
    grid[0, 0] = 5

    with pytest.raises(ValueError):
        get_candidate_interactions(grid, 0, 0)


def test_candidate_interactions_count_row_candidates() -> None:
    grid = np.zeros((9, 9), dtype=int)

    interactions = get_candidate_interactions(grid, 0, 0)

    row_counts = interactions[:9]

    # Eight other empty cells exist in the row, and every digit is
    # initially a valid candidate for each of them.
    np.testing.assert_array_equal(
        row_counts,
        np.full(9, 8.0, dtype=np.float32),
    )


def test_candidate_interactions_count_column_candidates() -> None:
    grid = np.zeros((9, 9), dtype=int)

    interactions = get_candidate_interactions(grid, 0, 0)

    column_counts = interactions[9:18]

    np.testing.assert_array_equal(
        column_counts,
        np.full(9, 8.0, dtype=np.float32),
    )


def test_candidate_interactions_count_block_candidates() -> None:
    grid = np.zeros((9, 9), dtype=int)

    interactions = get_candidate_interactions(grid, 0, 0)

    block_counts = interactions[18:]

    # A 3x3 block contains eight peer cells besides the target cell.
    np.testing.assert_array_equal(
        block_counts,
        np.full(9, 8.0, dtype=np.float32),
    )