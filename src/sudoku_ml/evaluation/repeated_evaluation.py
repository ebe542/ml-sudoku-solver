from dataclasses import dataclass

import numpy as np

from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.dataset.unique_generator import create_unique_dataset
from sudoku_ml.evaluation.solver_comparison import compare_solvers
from sudoku_ml.model.random_forest import SudokuRandomForest
from sudoku_ml.solver import ClassicalSudokuSolver, HybridSudokuSolver


@dataclass(frozen=True)
class MetricSummary:
    """Store summary statistics for one metric across repeated runs."""

    mean: float
    standard_deviation: float
    minimum: float
    maximum: float

    @classmethod
    def from_values(cls, values: list[float]) -> "MetricSummary":
        """Create a metric summary from one or more values."""
        if not values:
            raise ValueError("At least one metric value is required.")

        array = np.asarray(values, dtype=float)

        return cls(
            mean=float(np.mean(array)),
            standard_deviation=float(np.std(array)),
            minimum=float(np.min(array)),
            maximum=float(np.max(array)),
        )


@dataclass(frozen=True)
class RepeatedRemovalRateResult:
    """Store repeated unique-puzzle results for one removal rate."""

    removal_rate: float
    hybrid_match_rate: MetricSummary
    classical_match_rate: MetricSummary
    hybrid_runtime_ms: MetricSummary
    classical_runtime_ms: MetricSummary
    runtime_ratio: MetricSummary
    hybrid_backtracks: MetricSummary
    classical_backtracks: MetricSummary
    backtrack_reduction: MetricSummary
    hybrid_ml_decisions: MetricSummary


@dataclass(frozen=True)
class RepeatedEvaluationResult:
    """Store repeated evaluations across removal rates and seeds."""

    evaluation_seeds: tuple[int, ...]
    results: tuple[RepeatedRemovalRateResult, ...]

    @property
    def run_count(self) -> int:
        """Return the number of evaluation runs per removal rate."""
        return len(self.evaluation_seeds)


def evaluate_repeated_unique_solvers(
    removal_rates: list[float],
    evaluation_seeds: list[int],
    num_training_solutions: int = 100,
    num_evaluation_puzzles: int = 10,
    test_size: float = 0.2,
    n_estimators: int = 100,
    training_seed: int | None = 42,
) -> RepeatedEvaluationResult:
    """Evaluate both solvers repeatedly on unique puzzle sets."""
    if not removal_rates:
        raise ValueError("At least one removal rate is required.")

    if any(not 0.0 < rate < 1.0 for rate in removal_rates):
        raise ValueError("Removal rates must be between 0 and 1.")

    if not evaluation_seeds:
        raise ValueError("At least one evaluation seed is required.")

    if num_evaluation_puzzles <= 0:
        raise ValueError("Number of evaluation puzzles must be positive.")

    results: list[RepeatedRemovalRateResult] = []

    for removal_rate in removal_rates:
        training_data = create_train_test_split(
            num_solutions=num_training_solutions,
            test_size=test_size,
            removal_rate=removal_rate,
            random_seed=training_seed,
        )

        model = SudokuRandomForest(
            n_estimators=n_estimators,
            random_seed=training_seed,
        )
        model.fit(training_data)

        hybrid_match_rates: list[float] = []
        classical_match_rates: list[float] = []
        hybrid_runtimes: list[float] = []
        classical_runtimes: list[float] = []
        runtime_ratios: list[float] = []
        hybrid_backtracks: list[float] = []
        classical_backtracks: list[float] = []
        backtrack_reductions: list[float] = []
        hybrid_ml_decisions: list[float] = []

        for evaluation_seed in evaluation_seeds:
            dataset = create_unique_dataset(
                num_samples=num_evaluation_puzzles,
                removal_rate=removal_rate,
                random_seed=evaluation_seed,
            )

            comparison = compare_solvers(
                HybridSudokuSolver(model),
                ClassicalSudokuSolver(),
                dataset.puzzles,
                dataset.solutions,
            )

            hybrid = comparison.hybrid
            classical = comparison.classical

            hybrid_match_rates.append(
                float(hybrid.matching_solution_rate)
            )
            classical_match_rates.append(
                float(classical.matching_solution_rate)
            )
            hybrid_runtimes.append(
                hybrid.average_runtime_seconds * 1000
            )
            classical_runtimes.append(
                classical.average_runtime_seconds * 1000
            )

            if classical.average_runtime_seconds == 0.0:
                runtime_ratios.append(0.0)
            else:
                runtime_ratios.append(
                    hybrid.average_runtime_seconds
                    / classical.average_runtime_seconds
                )

            hybrid_backtracks.append(hybrid.average_backtracks)
            classical_backtracks.append(classical.average_backtracks)

            if classical.backtracks == 0:
                backtrack_reductions.append(0.0)
            else:
                backtrack_reductions.append(
                    (classical.backtracks - hybrid.backtracks)
                    / classical.backtracks
                )

            hybrid_ml_decisions.append(hybrid.average_ml_decisions)

        results.append(
            RepeatedRemovalRateResult(
                removal_rate=removal_rate,
                hybrid_match_rate=MetricSummary.from_values(
                    hybrid_match_rates
                ),
                classical_match_rate=MetricSummary.from_values(
                    classical_match_rates
                ),
                hybrid_runtime_ms=MetricSummary.from_values(
                    hybrid_runtimes
                ),
                classical_runtime_ms=MetricSummary.from_values(
                    classical_runtimes
                ),
                runtime_ratio=MetricSummary.from_values(runtime_ratios),
                hybrid_backtracks=MetricSummary.from_values(
                    hybrid_backtracks
                ),
                classical_backtracks=MetricSummary.from_values(
                    classical_backtracks
                ),
                backtrack_reduction=MetricSummary.from_values(
                    backtrack_reductions
                ),
                hybrid_ml_decisions=MetricSummary.from_values(
                    hybrid_ml_decisions
                ),
            )
        )

    return RepeatedEvaluationResult(
        evaluation_seeds=tuple(evaluation_seeds),
        results=tuple(results),
    )
