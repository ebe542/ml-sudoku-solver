from dataclasses import dataclass

import numpy as np

from sudoku_ml.analysis.model_comparison import create_comparison_models
from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.dataset.unique_generator import create_unique_dataset
from sudoku_ml.evaluation.solver_evaluation import (
    SolverEvaluationResult,
    evaluate_solver,
)
from sudoku_ml.model.protocol import NamedSudokuProbabilityModel
from sudoku_ml.solver import HybridSudokuSolver


@dataclass(frozen=True)
class SolverModelComparisonResult:
    """Store Hybrid solver results for one model."""

    name: str
    evaluation: SolverEvaluationResult


@dataclass(frozen=True)
class SolverModelRemovalRateResult:
    """Store model comparisons for one removal rate."""

    removal_rate: float
    models: tuple[
        SolverModelComparisonResult,
        ...,
    ]


@dataclass(frozen=True)
class SolverModelEvaluationResult:
    """Store solver model results across removal rates."""

    results: tuple[
        SolverModelRemovalRateResult,
        ...,
    ]


def compare_models_in_hybrid_solver(
    models: tuple[NamedSudokuProbabilityModel, ...],
    puzzles: np.ndarray,
    expected_solutions: np.ndarray,
) -> tuple[SolverModelComparisonResult, ...]:
    """Evaluate multiple models on identical puzzles."""
    results: list[SolverModelComparisonResult] = []

    for model in models:
        evaluation = evaluate_solver(
            HybridSudokuSolver(model),
            puzzles,
            expected_solutions,
        )

        results.append(
            SolverModelComparisonResult(
                name=model.name,
                evaluation=evaluation,
            )
        )

    return tuple(results)


def evaluate_solver_model_removal_rates(
    removal_rates: list[float],
    num_training_solutions: int = 100,
    num_evaluation_puzzles: int = 20,
    test_size: float = 0.2,
    n_estimators: int = 100,
    training_seed: int | None = 42,
    evaluation_seed: int | None = 123,
) -> SolverModelEvaluationResult:
    """Compare model-guided solvers across removal rates."""
    if not removal_rates:
        raise ValueError(
            "At least one removal rate is required."
        )

    if any(
        not 0.0 < removal_rate < 1.0
        for removal_rate in removal_rates
    ):
        raise ValueError(
            "Removal rates must be between 0 and 1."
        )

    if num_evaluation_puzzles <= 0:
        raise ValueError(
            "Number of evaluation puzzles must be positive."
        )

    removal_rate_results: list[
        SolverModelRemovalRateResult
    ] = []

    for removal_rate in removal_rates:
        training_data = create_train_test_split(
            num_solutions=num_training_solutions,
            test_size=test_size,
            removal_rate=removal_rate,
            random_seed=training_seed,
        )

        models = create_comparison_models(
            n_estimators=n_estimators,
            random_seed=training_seed,
        )

        for model in models:
            model.fit(
                training_data.X_train,
                training_data.y_train,
            )

        evaluation_dataset = create_unique_dataset(
            num_samples=num_evaluation_puzzles,
            removal_rate=removal_rate,
            random_seed=evaluation_seed,
        )

        model_results = (
            compare_models_in_hybrid_solver(
                models=models,
                puzzles=evaluation_dataset.puzzles,
                expected_solutions=(
                    evaluation_dataset.solutions
                ),
            )
        )

        removal_rate_results.append(
            SolverModelRemovalRateResult(
                removal_rate=removal_rate,
                models=model_results,
            )
        )

    return SolverModelEvaluationResult(
        results=tuple(removal_rate_results),
    )
