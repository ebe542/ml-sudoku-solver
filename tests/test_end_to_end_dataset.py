import numpy as np
import pytest

from sudoku_ml.dataset.end_to_end import (
    create_end_to_end_dataset,
    create_end_to_end_train_test_split,
)


def test_dataset_contains_one_sample_per_solution_and_rate() -> None:
    dataset = create_end_to_end_dataset(
        num_solutions=3,
        removal_rates=(0.10, 0.20),
        random_seed=42,
    )

    assert dataset.X.shape == (6, 9, 9)
    assert dataset.y.shape == (6, 9, 9)
    assert dataset.empty_mask.shape == (6, 9, 9)
    assert dataset.removal_rates.shape == (6,)
    assert dataset.solution_ids.shape == (6,)
    assert dataset.empty_mask.dtype == np.bool_


def test_puzzle_clues_match_complete_targets() -> None:
    dataset = create_end_to_end_dataset(
        num_solutions=2,
        removal_rates=(0.10, 0.20),
        random_seed=42,
    )

    assert np.all(
        (dataset.X == 0)
        | (dataset.X == dataset.y)
    )
    assert np.all((dataset.y >= 1) & (dataset.y <= 9))
    assert np.array_equal(dataset.empty_mask, dataset.X == 0)


def test_dataset_uses_requested_removal_rates() -> None:
    dataset = create_end_to_end_dataset(
        num_solutions=2,
        removal_rates=(0.10, 0.20),
        random_seed=42,
    )

    empty_counts = dataset.empty_mask.sum(axis=(1, 2))

    assert np.array_equal(empty_counts, np.array([8, 16, 8, 16]))
    assert np.allclose(
        dataset.removal_rates,
        np.array([0.10, 0.20, 0.10, 0.20]),
    )


def test_dataset_is_reproducible() -> None:
    first = create_end_to_end_dataset(
        num_solutions=2,
        removal_rates=(0.10, 0.20),
        random_seed=42,
    )
    second = create_end_to_end_dataset(
        num_solutions=2,
        removal_rates=(0.10, 0.20),
        random_seed=42,
    )

    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.y, second.y)
    assert np.array_equal(first.solution_ids, second.solution_ids)


def test_inputs_and_targets_do_not_share_memory() -> None:
    dataset = create_end_to_end_dataset(
        num_solutions=2,
        removal_rates=(0.10,),
        random_seed=42,
    )

    assert not np.shares_memory(dataset.X, dataset.y)


def test_split_keeps_solution_families_separate() -> None:
    split = create_end_to_end_train_test_split(
        num_solutions=5,
        test_size=0.4,
        removal_rates=(0.10, 0.20),
        random_seed=42,
    )

    train_ids = set(split.train.solution_ids.tolist())
    test_ids = set(split.test.solution_ids.tolist())

    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) == 3
    assert len(test_ids) == 2
    assert split.train.X.shape[0] == 6
    assert split.test.X.shape[0] == 4


@pytest.mark.parametrize("num_solutions", (0, -1))
def test_dataset_rejects_invalid_solution_count(
    num_solutions: int,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        create_end_to_end_dataset(num_solutions=num_solutions)


@pytest.mark.parametrize("removal_rates", ((), (-0.1,), (1.0,)))
def test_dataset_rejects_invalid_removal_rates(
    removal_rates: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="[Rr]emoval rate"):
        create_end_to_end_dataset(removal_rates=removal_rates)


@pytest.mark.parametrize("test_size", (0.0, 1.0, -0.1, 1.1))
def test_split_rejects_invalid_test_size(test_size: float) -> None:
    with pytest.raises(ValueError, match="Test size"):
        create_end_to_end_train_test_split(
            num_solutions=2,
            test_size=test_size,
        )
