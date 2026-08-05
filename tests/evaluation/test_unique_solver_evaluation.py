import pytest

from sudoku_ml.evaluation.unique_solver_evaluation import (
    evaluate_unique_removal_rates,
)


def test_unique_evaluation_matches_ground_truth() -> None:
    result = evaluate_unique_removal_rates(
        removal_rates=[0.3],
        num_training_solutions=10,
        num_evaluation_puzzles=2,
        n_estimators=5,
        training_seed=42,
        evaluation_seed=123,
    )

    item = result.results[0]

    assert item.removal_rate == pytest.approx(0.3)
    assert item.comparison.hybrid.valid_solution_rate == pytest.approx(1.0)
    assert item.comparison.classical.valid_solution_rate == pytest.approx(1.0)
    assert item.comparison.hybrid.matching_solution_rate == pytest.approx(1.0)
    assert item.comparison.classical.matching_solution_rate == pytest.approx(1.0)


def test_unique_evaluation_rejects_empty_rates() -> None:
    with pytest.raises(ValueError, match="At least one removal rate"):
        evaluate_unique_removal_rates(removal_rates=[])


@pytest.mark.parametrize("removal_rate", [0.0, 1.0, -0.1, 1.1])
def test_unique_evaluation_rejects_invalid_rate(
    removal_rate: float,
) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_unique_removal_rates(removal_rates=[removal_rate])


def test_unique_evaluation_rejects_invalid_puzzle_count() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        evaluate_unique_removal_rates(
            removal_rates=[0.5],
            num_evaluation_puzzles=0,
        )
