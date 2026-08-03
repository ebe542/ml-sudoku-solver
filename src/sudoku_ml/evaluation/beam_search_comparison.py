from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.dataset.unique_generator import create_unique_dataset
from sudoku_ml.evaluation.solver_evaluation import (
    SolverEvaluationResult,
    evaluate_solver,
)
from sudoku_ml.model.histogram_gradient_boosting import (
    SudokuHistogramGradientBoosting,
)
from sudoku_ml.model.protocol import SudokuProbabilityModel
from sudoku_ml.model.random_forest import SudokuRandomForest
from sudoku_ml.solver import (
    BeamSearchSudokuSolver,
    ClassicalSudokuSolver,
    GreedyMLSudokuSolver,
    HybridSudokuSolver,
)


@dataclass(frozen=True)
class SearchStrategyResult:
    """Store one model and search-strategy result."""

    model_name: str
    strategy_name: str
    beam_width: int | None
    evaluation: SolverEvaluationResult


@dataclass(frozen=True)
class BeamSearchRemovalRateResult:
    """Store search-strategy results for one removal rate."""

    removal_rate: float
    strategies: tuple[SearchStrategyResult, ...]


@dataclass(frozen=True)
class BeamSearchComparisonResult:
    """Store search-strategy results across removal rates."""

    results: tuple[BeamSearchRemovalRateResult, ...]


def _validate_beam_widths(beam_widths: Sequence[int]) -> None:
    if not beam_widths:
        raise ValueError("At least one beam width is required.")

    if any(
        not isinstance(beam_width, int) or beam_width <= 0
        for beam_width in beam_widths
    ):
        raise ValueError("Beam widths must be positive integers.")


def compare_search_strategies(
    models: Mapping[str, SudokuProbabilityModel],
    puzzles: np.ndarray,
    expected_solutions: np.ndarray,
    beam_widths: Sequence[int] = (2, 3, 4),
) -> tuple[SearchStrategyResult, ...]:
    """Compare Greedy, Beam, Hybrid, and classical solving."""
    if not models:
        raise ValueError("At least one model is required.")

    if len(puzzles) == 0:
        raise ValueError("At least one puzzle is required.")

    if len(puzzles) != len(expected_solutions):
        raise ValueError(
            "Expected solutions must match the number of puzzles."
        )

    _validate_beam_widths(beam_widths)

    results = [
        SearchStrategyResult(
            model_name="No model",
            strategy_name="Classical",
            beam_width=None,
            evaluation=evaluate_solver(
                ClassicalSudokuSolver(),
                puzzles,
                expected_solutions,
            ),
        )
    ]

    for model_name, model in models.items():
        results.append(
            SearchStrategyResult(
                model_name=model_name,
                strategy_name="Greedy",
                beam_width=1,
                evaluation=evaluate_solver(
                    GreedyMLSudokuSolver(model),
                    puzzles,
                    expected_solutions,
                ),
            )
        )

        for beam_width in beam_widths:
            results.append(
                SearchStrategyResult(
                    model_name=model_name,
                    strategy_name=f"Beam {beam_width}",
                    beam_width=beam_width,
                    evaluation=evaluate_solver(
                        BeamSearchSudokuSolver(
                            model,
                            beam_width=beam_width,
                        ),
                        puzzles,
                        expected_solutions,
                    ),
                )
            )

        results.append(
            SearchStrategyResult(
                model_name=model_name,
                strategy_name="Hybrid",
                beam_width=None,
                evaluation=evaluate_solver(
                    HybridSudokuSolver(model),
                    puzzles,
                    expected_solutions,
                ),
            )
        )

    return tuple(results)


def evaluate_beam_search_removal_rates(
    removal_rates: Sequence[float],
    beam_widths: Sequence[int] = (2, 3, 4),
    num_training_solutions: int = 100,
    num_evaluation_puzzles: int = 20,
    test_size: float = 0.2,
    model_iterations: int = 100,
    training_seed: int | None = 42,
    evaluation_seed: int | None = 123,
) -> BeamSearchComparisonResult:
    """Evaluate search strategies across puzzle removal rates."""
    if not removal_rates:
        raise ValueError("At least one removal rate is required.")

    if any(
        not 0.0 < removal_rate < 1.0
        for removal_rate in removal_rates
    ):
        raise ValueError("Removal rates must be between 0 and 1.")

    if num_evaluation_puzzles <= 0:
        raise ValueError(
            "Number of evaluation puzzles must be positive."
        )

    if model_iterations <= 0:
        raise ValueError("Model iterations must be positive.")

    _validate_beam_widths(beam_widths)
    removal_rate_results: list[BeamSearchRemovalRateResult] = []

    for removal_rate in removal_rates:
        training_data = create_train_test_split(
            num_solutions=num_training_solutions,
            test_size=test_size,
            removal_rate=removal_rate,
            random_seed=training_seed,
        )

        random_forest = SudokuRandomForest(
            n_estimators=model_iterations,
            random_seed=training_seed,
        )
        histogram_gradient_boosting = (
            SudokuHistogramGradientBoosting(
                max_iter=model_iterations,
                random_seed=training_seed,
            )
        )
        random_forest.fit(training_data)
        histogram_gradient_boosting.fit(training_data)

        dataset = create_unique_dataset(
            num_samples=num_evaluation_puzzles,
            removal_rate=removal_rate,
            random_seed=evaluation_seed,
        )

        strategies = compare_search_strategies(
            models={
                "Random Forest": random_forest,
                "Histogram Gradient Boosting": (
                    histogram_gradient_boosting
                ),
            },
            puzzles=dataset.puzzles,
            expected_solutions=dataset.solutions,
            beam_widths=beam_widths,
        )

        removal_rate_results.append(
            BeamSearchRemovalRateResult(
                removal_rate=removal_rate,
                strategies=strategies,
            )
        )

    return BeamSearchComparisonResult(
        results=tuple(removal_rate_results),
    )
