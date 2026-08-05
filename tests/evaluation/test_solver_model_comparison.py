import numpy as np
import pytest

from sudoku_ml.evaluation.solver_model_comparison import (
    compare_models_in_hybrid_solver,
    evaluate_solver_model_removal_rates,
)
from sudoku_ml.sudoku_generator import generate_solved_grid


class NamedProbabilityModel:
    """Provide a named model for solver comparison tests."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.classes = np.arange(1, 10)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        return np.full(
            (len(X), 9),
            1.0 / 9.0,
        )


def test_compare_models_uses_identical_puzzles() -> None:
    solution = generate_solved_grid(random_seed=42).values

    puzzle = solution.copy()
    puzzle[0, 0] = 0

    models = (
        NamedProbabilityModel("Model A"),
        NamedProbabilityModel("Model B"),
    )

    results = compare_models_in_hybrid_solver(
        models=models,
        puzzles=np.asarray([puzzle]),
        expected_solutions=np.asarray([solution]),
    )

    assert [result.name for result in results] == [
        "Model A",
        "Model B",
    ]

    for result in results:
        evaluation = result.evaluation

        assert evaluation.total_puzzles == 1
        assert evaluation.solution_rate == pytest.approx(1.0)
        assert (
            evaluation.valid_solution_rate
            == pytest.approx(1.0)
        )
        assert (
            evaluation.matching_solution_rate
            == pytest.approx(1.0)
        )
        assert evaluation.deterministic_steps == 1
        assert evaluation.ml_decisions == 0
        assert evaluation.backtracks == 0


def test_compare_models_rejects_mismatched_ground_truth() -> None:
    solution = generate_solved_grid(random_seed=42).values

    puzzle = solution.copy()
    puzzle[0, 0] = 0

    with pytest.raises(
        ValueError,
        match="match the number of puzzles"):
        compare_models_in_hybrid_solver(
            models=(NamedProbabilityModel("Model"),),
            puzzles=np.asarray([puzzle]),
            expected_solutions=np.empty(
                (0, 9, 9),
                dtype=int,
            ),
        )

def test_solver_model_evaluation_returns_each_model() -> None:
    evaluation = evaluate_solver_model_removal_rates(
        removal_rates=[0.3],
        num_training_solutions=20,
        num_evaluation_puzzles=2,
        test_size=0.2,
        n_estimators=5,
        training_seed=42,
        evaluation_seed=123,
    )

    assert len(evaluation.results) == 1

    rate_result = evaluation.results[0]

    assert rate_result.removal_rate == pytest.approx(0.3)
    assert [
        result.name
        for result in rate_result.models
    ] == [
        "Logistic Regression",
        "Random Forest",
        "Extra Trees",
        "Histogram Gradient Boosting",
    ]

    for result in rate_result.models:
        solver_result = result.evaluation

        assert solver_result.total_puzzles == 2
        assert solver_result.solution_rate == pytest.approx(
            1.0
        )
        assert (
            solver_result.valid_solution_rate
            == pytest.approx(1.0)
        )
        assert (
            solver_result.matching_solution_rate
            == pytest.approx(1.0)
        )


def test_solver_model_evaluation_rejects_empty_rates() -> None:
    with pytest.raises(
        ValueError,
        match="At least one removal rate"):
        evaluate_solver_model_removal_rates(
            removal_rates=[],
        )


@pytest.mark.parametrize(
    "removal_rate",
    [0.0, 1.0, -0.1, 1.1],
)
def test_solver_model_evaluation_rejects_invalid_rate(removal_rate: float) -> None:
    with pytest.raises(
        ValueError,
        match="between 0 and 1"):
        evaluate_solver_model_removal_rates(
            removal_rates=[removal_rate],
        )


def test_solver_model_evaluation_rejects_invalid_puzzle_count() -> None:
    with pytest.raises(
        ValueError,
        match="must be positive"):
        evaluate_solver_model_removal_rates(
            removal_rates=[0.5],
            num_evaluation_puzzles=0,
        )
