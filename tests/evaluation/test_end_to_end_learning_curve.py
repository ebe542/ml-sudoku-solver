import pytest

from sudoku_ml.dataset.end_to_end import (
    create_end_to_end_train_test_split,
)
from sudoku_ml.evaluation.end_to_end_learning_curve import (
    evaluate_end_to_end_learning_curve,
)


@pytest.fixture(scope="module")
def small_split():
    return create_end_to_end_train_test_split(
        num_solutions=12,
        test_size=0.25,
        removal_rates=(0.20,),
        random_seed=42,
    )


def test_learning_curve_returns_requested_points(small_split) -> None:
    result = evaluate_end_to_end_learning_curve(
        split=small_split,
        training_solution_counts=(3, 6),
        hidden_layer_sizes=(8,),
        max_iter=5,
        random_seed=42,
    )

    assert len(result.points) == 2
    assert [
        point.training_solution_count
        for point in result.points
    ] == [3, 6]
    assert [
        point.training_sample_count
        for point in result.points
    ] == [3, 6]

    for point in result.points:
        assert point.training_seconds >= 0.0
        assert 1 <= point.iterations <= 5
        assert point.initial_loss > 0.0
        assert point.final_loss > 0.0
        assert 0.0 <= point.training.empty_cell_accuracy <= 1.0
        assert 0.0 <= point.test.empty_cell_accuracy <= 1.0
        assert -1.0 <= point.empty_cell_accuracy_gap <= 1.0


def test_learning_curve_uses_fixed_test_partition(small_split) -> None:
    result = evaluate_end_to_end_learning_curve(
        split=small_split,
        training_solution_counts=(3, 6),
        hidden_layer_sizes=(8,),
        max_iter=5,
        random_seed=42,
    )

    assert {
        point.test.total_puzzles
        for point in result.points
    } == {len(small_split.test.X)}


def test_learning_curve_records_non_convergence(small_split) -> None:
    result = evaluate_end_to_end_learning_curve(
        split=small_split,
        training_solution_counts=(3,),
        hidden_layer_sizes=(8,),
        max_iter=1,
        random_seed=42,
    )

    assert result.points[0].converged is False


def test_learning_curve_rejects_empty_counts(small_split) -> None:
    with pytest.raises(ValueError, match="At least one"):
        evaluate_end_to_end_learning_curve(
            split=small_split,
            training_solution_counts=(),
        )


@pytest.mark.parametrize(
    "counts",
    ((0,), (-1,), (10,)),
)
def test_learning_curve_rejects_unavailable_counts(
    small_split,
    counts: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="positive and available"):
        evaluate_end_to_end_learning_curve(
            split=small_split,
            training_solution_counts=counts,
        )


@pytest.mark.parametrize(
    "counts",
    ((6, 3), (3, 3)),
)
def test_learning_curve_rejects_unsorted_or_duplicate_counts(
    small_split,
    counts: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="unique and ascending"):
        evaluate_end_to_end_learning_curve(
            split=small_split,
            training_solution_counts=counts,
        )
