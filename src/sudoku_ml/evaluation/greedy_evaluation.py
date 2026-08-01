from dataclasses import dataclass

import numpy as np

from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.dataset.unique_generator import create_unique_dataset
from sudoku_ml.evaluation.solver_evaluation import (
    SolverEvaluationResult,
    evaluate_solver,
)
from sudoku_ml.model.random_forest import SudokuRandomForest
from sudoku_ml.solver import (
    GreedyMLSudokuSolver,
    HybridSudokuSolver,
)


@dataclass(frozen=True)
class GreedyComparisonResult:
    """Store Greedy ML and Hybrid ML evaluation results."""

    greedy: SolverEvaluationResult
    hybrid: SolverEvaluationResult

    @property
    def recovered_puzzles(self) -> int:
        """Return puzzles solved by Hybrid but not by Greedy."""
        greedy_matches = self.greedy.matching_solutions or 0
        hybrid_matches = self.hybrid.matching_solutions or 0

        return hybrid_matches - greedy_matches

    @property
    def greedy_failure_rate(self) -> float:
        """Return the proportion not solved exactly by Greedy ML."""
        match_rate = self.greedy.matching_solution_rate

        if match_rate is None:
            return 0.0

        return 1.0 - match_rate


@dataclass(frozen=True)
class GreedyRemovalRateResult:
    """Store one Greedy-vs-Hybrid removal-rate result."""

    removal_rate: float
    comparison: GreedyComparisonResult


@dataclass(frozen=True)
class GreedyRemovalRateEvaluation:
    """Store Greedy-vs-Hybrid results across removal rates."""

    results: tuple[GreedyRemovalRateResult, ...]


def compare_greedy_and_hybrid(model: SudokuRandomForest, puzzles: np.ndarray,
                              expected_solutions: np.ndarray) -> GreedyComparisonResult:
    """Evaluate Greedy and Hybrid ML on the same puzzles."""
    greedy_result = evaluate_solver(
        GreedyMLSudokuSolver(model),
        puzzles,
        expected_solutions,
    )

    hybrid_result = evaluate_solver(
        HybridSudokuSolver(model),
        puzzles,
        expected_solutions,
    )

    return GreedyComparisonResult(
        greedy=greedy_result,
        hybrid=hybrid_result,
    )

def evaluate_greedy_removal_rates(removal_rates: list[float], num_training_solutions: int = 100,
                                  num_evaluation_puzzles: int = 20, test_size: float = 0.2,
                                  n_estimators: int = 100, training_seed: int | None = 42,
                                  evaluation_seed: int | None = 123) -> GreedyRemovalRateEvaluation:
    """Evaluate Greedy and Hybrid ML across removal rates."""
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

    results: list[GreedyRemovalRateResult] = []

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

        evaluation_dataset = create_unique_dataset(
            num_samples=num_evaluation_puzzles,
            removal_rate=removal_rate,
            random_seed=evaluation_seed,
        )

        comparison = compare_greedy_and_hybrid(
            model,
            evaluation_dataset.puzzles,
            evaluation_dataset.solutions,
        )

        results.append(
            GreedyRemovalRateResult(
                removal_rate=removal_rate,
                comparison=comparison,
            )
        )

    return GreedyRemovalRateEvaluation(
        results=tuple(results)
    )
