from dataclasses import dataclass

from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.dataset.generator import create_diverse_dataset
from sudoku_ml.evaluation.solver_comparison import (
    SolverComparisonResult,
    compare_solvers,
)
from sudoku_ml.model.random_forest import SudokuRandomForest
from sudoku_ml.solver import ClassicalSudokuSolver, HybridSudokuSolver


@dataclass(frozen=True)
class RemovalRateResult:
    """Store a solver comparison for one removal rate."""

    removal_rate: float
    comparison: SolverComparisonResult

    @property
    def backtrack_reduction(self) -> float:
        """Return the relative backtrack reduction of the hybrid solver."""
        classical_backtracks = self.comparison.classical.backtracks

        if classical_backtracks == 0:
            return 0.0

        return (classical_backtracks - self.comparison.hybrid.backtracks) / classical_backtracks

    @property
    def runtime_ratio(self) -> float:
        """Return hybrid runtime divided by classical runtime."""
        classical_runtime = (
            self.comparison.classical.average_runtime_seconds
        )

        if classical_runtime == 0.0:
            return 0.0

        return (
            self.comparison.hybrid.average_runtime_seconds
            / classical_runtime
        )


@dataclass(frozen=True)
class DifficultyEvaluationResult:
    """Store solver comparisons across multiple removal rates."""

    results: tuple[RemovalRateResult, ...]


def evaluate_removal_rates(removal_rates: list[float], num_training_solutions: int = 100,
                           num_evaluation_puzzles: int = 20, test_size: float = 0.2,
                           n_estimators: int = 100, training_seed: int | None = 42,
                           evaluation_seed: int | None = 123) -> DifficultyEvaluationResult:
    """Compare both solver strategies at multiple removal rates."""
    if not removal_rates:
        raise ValueError("At least one removal rate is required.")

    if any(not 0.0 < rate < 1.0 for rate in removal_rates):
        raise ValueError("Removal rates must be between 0 and 1.")

    if num_evaluation_puzzles <= 0:
        raise ValueError("Number of evaluation puzzles must be positive.")

    results: list[RemovalRateResult] = []

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

        evaluation_dataset = create_diverse_dataset(
            num_samples=num_evaluation_puzzles,
            removal_rate=removal_rate,
            random_seed=evaluation_seed,
        )

        comparison = compare_solvers(
            HybridSudokuSolver(model),
            ClassicalSudokuSolver(),
            evaluation_dataset.puzzles,
        )

        results.append(
            RemovalRateResult(
                removal_rate=removal_rate,
                comparison=comparison,
            )
        )

    return DifficultyEvaluationResult(results=tuple(results))
