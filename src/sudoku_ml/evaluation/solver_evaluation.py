from dataclasses import dataclass
from time import perf_counter

import numpy as np

from sudoku_ml.grid import SudokuGrid
from sudoku_ml.solver import HybridSudokuSolver


@dataclass(frozen=True)
class SolverEvaluationResult:
    """Store aggregated results from evaluating the hybrid solver."""

    total_puzzles: int
    solved_puzzles: int
    valid_solutions: int
    total_runtime_seconds: float
    deterministic_steps: int
    ml_decisions: int
    backtracks: int
    matching_solutions: int | None = None

    @property
    def solution_rate(self) -> float:
        """Return the proportion of successfully solved puzzles."""
        if self.total_puzzles == 0:
            return 0.0

        return self.solved_puzzles / self.total_puzzles

    @property
    def valid_solution_rate(self) -> float:
        """Return the proportion of valid completed solutions."""
        if self.total_puzzles == 0:
            return 0.0

        return self.valid_solutions / self.total_puzzles

    @property
    def average_runtime_seconds(self) -> float:
        """Return the average runtime per evaluated puzzle."""
        if self.total_puzzles == 0:
            return 0.0

        return self.total_runtime_seconds / self.total_puzzles

    @property
    def average_backtracks(self) -> float:
        """Return the average number of backtracks per puzzle."""
        if self.total_puzzles == 0:
            return 0.0

        return self.backtracks / self.total_puzzles

    @property
    def average_ml_decisions(self) -> float:
        """Return the average number of ML decisions per puzzle."""
        if self.total_puzzles == 0:
            return 0.0

        return self.ml_decisions / self.total_puzzles

    @property
    def matching_solution_rate(self) -> float | None:
        """Return the rate of solutions matching supplied ground truth."""
        if self.matching_solutions is None:
            return None

        if self.total_puzzles == 0:
            return 0.0

        return self.matching_solutions / self.total_puzzles


def evaluate_solver(
    solver: HybridSudokuSolver,
    puzzles: np.ndarray,
    expected_solutions: np.ndarray | None = None,
) -> SolverEvaluationResult:
    """Evaluate the hybrid solver on multiple Sudoku puzzles."""
    if (
        expected_solutions is not None
        and len(expected_solutions) != len(puzzles)
    ):
        raise ValueError(
            "Expected solutions must match the number of puzzles."
        )

    solved_puzzles = 0
    valid_solutions = 0
    matching_solutions = 0 if expected_solutions is not None else None
    total_runtime_seconds = 0.0
    deterministic_steps = 0
    ml_decisions = 0
    backtracks = 0

    for puzzle_index, puzzle_values in enumerate(puzzles):
        puzzle = SudokuGrid(puzzle_values)

        start_time = perf_counter()
        solution = solver.solve(puzzle)
        total_runtime_seconds += perf_counter() - start_time

        if solution is not None:
            solved_puzzles += 1

            preserves_given_digits = np.all(
                (puzzle.values == 0)
                | (solution.values == puzzle.values)
            )

            if (solution.is_complete()
                and solution.is_valid()
                and preserves_given_digits):
                valid_solutions += 1

            if (
                matching_solutions is not None
                and np.array_equal(
                    solution.values,
                    expected_solutions[puzzle_index],
                )
            ):
                matching_solutions += 1

        deterministic_steps += solver.stats.deterministic_steps
        ml_decisions += solver.stats.ml_decisions
        backtracks += solver.stats.backtracks

    return SolverEvaluationResult(
        total_puzzles=len(puzzles),
        solved_puzzles=solved_puzzles,
        valid_solutions=valid_solutions,
        total_runtime_seconds=total_runtime_seconds,
        deterministic_steps=deterministic_steps,
        ml_decisions=ml_decisions,
        backtracks=backtracks,
        matching_solutions=matching_solutions,
    )
