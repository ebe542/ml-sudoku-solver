import numpy as np
import pytest

from sudoku_ml.data.split import create_train_test_split


def test_train_test_split_has_data() -> None:
    data = create_train_test_split(
        num_solutions=10,
        test_size=0.2,
        random_seed=42,
    )

    assert len(data.X_train) > 0
    assert len(data.X_test) > 0
    assert len(data.y_train) > 0
    assert len(data.y_test) > 0


def test_features_and_targets_have_matching_lengths() -> None:
    data = create_train_test_split(
        num_solutions=10,
        test_size=0.2,
        random_seed=42,
    )

    assert len(data.X_train) == len(data.y_train)
    assert len(data.X_test) == len(data.y_test)


def test_feature_shape_is_correct() -> None:
    data = create_train_test_split(
        num_solutions=10,
        test_size=0.2,
        random_seed=42,
    )

    assert data.X_train.shape[1] == 91
    assert data.X_test.shape[1] == 91


def test_split_is_reproducible() -> None:
    first = create_train_test_split(
        num_solutions=10,
        test_size=0.2,
        random_seed=42,
    )

    second = create_train_test_split(
        num_solutions=10,
        test_size=0.2,
        random_seed=42,
    )

    np.testing.assert_array_equal(first.X_train, second.X_train)
    np.testing.assert_array_equal(first.X_test, second.X_test)
    np.testing.assert_array_equal(first.y_train, second.y_train)
    np.testing.assert_array_equal(first.y_test, second.y_test)


def test_invalid_number_of_solutions_is_rejected() -> None:
    with pytest.raises(ValueError):
        create_train_test_split(num_solutions=1)


def test_invalid_test_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        create_train_test_split(
            num_solutions=10,
            test_size=1.0,
        )
