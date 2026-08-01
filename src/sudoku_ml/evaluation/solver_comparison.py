from dataclasses import dataclass

import numpy as np

from sudoku_ml.evaluation.solver_evaluation import (
    SolverEvaluationResult,
    evaluate_solver,
)
from sudoku_ml.solver import (
    ClassicalSudokuSolver,
    HybridSudokuSolver,
)


@dataclass(frozen=True)
class SolverComparisonResult:
    """Store evaluation results for both solver strategies."""

    hybrid: SolverEvaluationResult
    classical: SolverEvaluationResult


def compare_solvers(hybrid_solver: HybridSudokuSolver, classical_solver: ClassicalSudokuSolver,
                    puzzles: np.ndarray) -> SolverComparisonResult:
    """Evaluate both solvers on the same Sudoku puzzles."""
    hybrid_result = evaluate_solver(
        hybrid_solver,
        puzzles,
    )

    classical_result = evaluate_solver(
        classical_solver,
        puzzles,
    )

    return SolverComparisonResult(
        hybrid=hybrid_result,
        classical=classical_result,
    )
