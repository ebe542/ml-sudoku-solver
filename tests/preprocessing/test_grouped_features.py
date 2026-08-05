import numpy as np

from sudoku_ml.dataset.generator import create_diverse_dataset
from sudoku_ml.preprocessing.features import (
    create_grouped_features_and_targets,
)


def test_grouped_features_have_matching_lengths() -> None:
    dataset = create_diverse_dataset(
        num_samples=5,
        removal_rate=0.5,
        random_seed=42,
    )

    features, targets, groups = create_grouped_features_and_targets(
        dataset
    )

    assert len(features) == len(targets)
    assert len(features) == len(groups)


def test_grouped_features_have_expected_shape() -> None:
    dataset = create_diverse_dataset(
        num_samples=5,
        removal_rate=0.5,
        random_seed=42,
    )

    features, targets, groups = create_grouped_features_and_targets(
        dataset
    )

    assert features.shape == (200, 118)
    assert targets.shape == (200,)
    assert groups.shape == (200,)


def test_group_identifiers_match_number_of_sudokus() -> None:
    dataset = create_diverse_dataset(
        num_samples=5,
        removal_rate=0.5,
        random_seed=42,
    )

    _, _, groups = create_grouped_features_and_targets(dataset)

    np.testing.assert_array_equal(
        np.unique(groups),
        np.arange(5),
    )


def test_each_sudoku_has_expected_number_of_samples() -> None:
    dataset = create_diverse_dataset(
        num_samples=5,
        removal_rate=0.5,
        random_seed=42,
    )

    _, _, groups = create_grouped_features_and_targets(dataset)

    unique_groups, counts = np.unique(
        groups,
        return_counts=True,
    )

    np.testing.assert_array_equal(
        unique_groups,
        np.arange(5),
    )

    np.testing.assert_array_equal(
        counts,
        np.full(5, 40),
    )


def test_grouped_features_are_reproducible() -> None:
    first_dataset = create_diverse_dataset(
        num_samples=5,
        removal_rate=0.5,
        random_seed=42,
    )

    second_dataset = create_diverse_dataset(
        num_samples=5,
        removal_rate=0.5,
        random_seed=42,
    )

    first = create_grouped_features_and_targets(first_dataset)
    second = create_grouped_features_and_targets(second_dataset)

    for first_array, second_array in zip(first, second):
        np.testing.assert_array_equal(
            first_array,
            second_array,
        )
