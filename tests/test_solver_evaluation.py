import pytest

from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.dataset.generator import create_diverse_dataset
from sudoku_ml.evaluation.solver_evaluation import (
    SolverEvaluationResult,
    evaluate_solver,
)
from sudoku_ml.model.random_forest import SudokuRandomForest
from sudoku_ml.solver import HybridSudokuSolver

def test_solver_evaluation_calculates_summary_metrics() -> None:
    result = SolverEvaluationResult(
        total_puzzles=10,
        solved_puzzles=8,
        valid_solutions=7,
        total_runtime_seconds=2.0,
        deterministic_steps=100,
        ml_decisions=40,
        backtracks=20,
    )

    assert result.solution_rate == pytest.approx(0.8)
    assert result.valid_solution_rate == pytest.approx(0.7)
    assert result.average_runtime_seconds == pytest.approx(0.2)
    assert result.average_backtracks == pytest.approx(2.0)
    assert result.average_ml_decisions == pytest.approx(4.0)
    assert result.average_generated_states == 0.0
    assert result.average_pruned_states == 0.0
    assert result.maximum_active_states == 0
    assert result.matching_solution_rate is None


def test_solver_evaluation_handles_empty_evaluation() -> None:
    result = SolverEvaluationResult(
        total_puzzles=0,
        solved_puzzles=0,
        valid_solutions=0,
        total_runtime_seconds=0.0,
        deterministic_steps=0,
        ml_decisions=0,
        backtracks=0,
    )

    assert result.solution_rate == 0.0
    assert result.valid_solution_rate == 0.0
    assert result.average_runtime_seconds == 0.0
    assert result.average_backtracks == 0.0
    assert result.average_ml_decisions == 0.0
    assert result.average_generated_states == 0.0
    assert result.average_pruned_states == 0.0
    assert result.maximum_active_states == 0
    assert result.matching_solution_rate is None


def test_evaluate_solver_evaluates_multiple_puzzles() -> None:
    training_data = create_train_test_split(
        num_solutions=20,
        random_seed=42,
    )

    model = SudokuRandomForest(
        n_estimators=20,
        random_seed=42,
    )
    model.fit(training_data)

    solver = HybridSudokuSolver(model)

    evaluation_dataset = create_diverse_dataset(
        num_samples=3,
        removal_rate=0.02,
        random_seed=123,
    )

    result = evaluate_solver(
        solver,
        evaluation_dataset.puzzles,
        evaluation_dataset.solutions,
    )

    assert result.total_puzzles == 3
    assert result.solved_puzzles == 3
    assert result.valid_solutions == 3
    assert result.solution_rate == pytest.approx(1.0)
    assert result.valid_solution_rate == pytest.approx(1.0)
    assert result.deterministic_steps == 3
    assert result.ml_decisions == 0
    assert result.backtracks == 0
    assert result.total_runtime_seconds >= 0.0
    assert result.matching_solutions == 3
    assert result.matching_solution_rate == pytest.approx(1.0)


def test_evaluate_solver_rejects_mismatched_solutions() -> None:
    training_data = create_train_test_split(
        num_solutions=10,
        random_seed=42,
    )
    model = SudokuRandomForest(n_estimators=5, random_seed=42)
    model.fit(training_data)

    solver = HybridSudokuSolver(model)

    with pytest.raises(ValueError, match="match the number of puzzles"):
        evaluate_solver(
            solver,
            training_data.X_test[:2],
            training_data.y_test[:1],
        )
