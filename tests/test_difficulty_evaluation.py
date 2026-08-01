import pytest

from sudoku_ml.evaluation.difficulty_evaluation import (
    RemovalRateResult,
    evaluate_removal_rates,
)
from sudoku_ml.evaluation.solver_comparison import SolverComparisonResult
from sudoku_ml.evaluation.solver_evaluation import SolverEvaluationResult


def test_evaluate_removal_rates_returns_one_result_per_rate() -> None:
    removal_rates = [0.02, 0.03]

    result = evaluate_removal_rates(
        removal_rates=removal_rates,
        num_training_solutions=10,
        num_evaluation_puzzles=2,
        n_estimators=5,
        training_seed=42,
        evaluation_seed=123,
    )

    assert len(result.results) == 2
    assert [item.removal_rate for item in result.results] == removal_rates

    for item in result.results:
        assert item.comparison.hybrid.total_puzzles == 2
        assert item.comparison.classical.total_puzzles == 2
        assert item.comparison.hybrid.valid_solution_rate == pytest.approx(1.0)
        assert item.comparison.classical.valid_solution_rate == pytest.approx(1.0)


def test_evaluate_removal_rates_rejects_empty_rate_list() -> None:
    with pytest.raises(ValueError, match="At least one removal rate"):
        evaluate_removal_rates(removal_rates=[])


@pytest.mark.parametrize("removal_rate", [0.0, 1.0, -0.1, 1.1])
def test_evaluate_removal_rates_rejects_invalid_rate(removal_rate: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_removal_rates(removal_rates=[removal_rate])


def test_evaluate_removal_rates_rejects_invalid_puzzle_count() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        evaluate_removal_rates(
            removal_rates=[0.5],
            num_evaluation_puzzles=0,
        )


def create_evaluation_result(runtime: float, backtracks: int) -> SolverEvaluationResult:
    return SolverEvaluationResult(
        total_puzzles=10,
        solved_puzzles=10,
        valid_solutions=10,
        total_runtime_seconds=runtime,
        deterministic_steps=100,
        ml_decisions=10,
        backtracks=backtracks,
    )


def test_removal_rate_result_calculates_comparison_metrics() -> None:
    result = RemovalRateResult(
        removal_rate=0.65,
        comparison=SolverComparisonResult(
            hybrid=create_evaluation_result(
                runtime=2.0,
                backtracks=40,
            ),
            classical=create_evaluation_result(
                runtime=0.5,
                backtracks=100,
            ),
        ),
    )

    assert result.backtrack_reduction == pytest.approx(0.6)
    assert result.runtime_ratio == pytest.approx(4.0)
