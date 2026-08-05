import numpy as np
import pytest

from sudoku_ml.evaluation.group_cross_validation import (
    evaluate_group_cross_validation,
)


def test_group_cross_validation_returns_one_score_per_fold() -> None:
    result = evaluate_group_cross_validation(
        num_solutions=10,
        n_splits=5,
        n_estimators=10,
        random_seed=42,
    )

    assert result.fold_accuracies.shape == (5,)


def test_cross_validation_accuracies_are_valid() -> None:
    result = evaluate_group_cross_validation(
        num_solutions=10,
        n_splits=5,
        n_estimators=10,
        random_seed=42,
    )

    assert np.all(result.fold_accuracies >= 0.0)
    assert np.all(result.fold_accuracies <= 1.0)


def test_cross_validation_summary_values_are_valid() -> None:
    result = evaluate_group_cross_validation(
        num_solutions=10,
        n_splits=5,
        n_estimators=10,
        random_seed=42,
    )

    assert 0.0 <= result.mean_accuracy <= 1.0
    assert result.standard_deviation >= 0.0
    assert result.minimum_accuracy <= result.maximum_accuracy


def test_group_cross_validation_is_reproducible() -> None:
    first = evaluate_group_cross_validation(
        num_solutions=10,
        n_splits=5,
        n_estimators=10,
        random_seed=42,
    )

    second = evaluate_group_cross_validation(
        num_solutions=10,
        n_splits=5,
        n_estimators=10,
        random_seed=42,
    )

    np.testing.assert_array_equal(
        first.fold_accuracies,
        second.fold_accuracies,
    )


def test_invalid_number_of_folds_is_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate_group_cross_validation(
            num_solutions=10,
            n_splits=1,
        )


def test_too_few_solutions_are_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate_group_cross_validation(
            num_solutions=4,
            n_splits=5,
        )
