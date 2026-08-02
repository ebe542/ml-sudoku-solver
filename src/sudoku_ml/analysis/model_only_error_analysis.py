from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from sudoku_ml.grid import SudokuGrid
from sudoku_ml.model.protocol import SudokuProbabilityModel
from sudoku_ml.solver import GreedyDecision, GreedyMLSudokuSolver


@dataclass(frozen=True)
class ModelOnlyDecisionError:
    """Describe the first decision that differs from ground truth."""

    step: int
    row: int
    column: int
    candidates: tuple[int, ...]
    ranked_candidates: tuple[int, ...]
    selected_digit: int
    correct_digit: int
    selected_confidence: float | None
    correct_digit_rank: int | None
    is_ml_decision: bool


@dataclass(frozen=True)
class ModelOnlyPuzzleResult:
    """Store the traced result for one puzzle."""

    exact_match: bool
    valid_solution: bool
    completed_solution: bool
    total_decisions: int
    ml_decisions: int
    correct_decisions_before_error: int
    first_error: ModelOnlyDecisionError | None


@dataclass(frozen=True)
class ModelOnlyModelResult:
    """Store Model-only puzzle results for one classifier."""

    name: str
    puzzles: tuple[ModelOnlyPuzzleResult, ...]

    @property
    def exact_solution_rate(self) -> float:
        """Return the proportion matching ground truth exactly."""
        return sum(result.exact_match for result in self.puzzles) / len(
            self.puzzles
        )

    @property
    def failure_rate(self) -> float:
        """Return the proportion that does not match ground truth."""
        return 1.0 - self.exact_solution_rate

    @property
    def average_correct_decisions_before_error(self) -> float | None:
        """Return mean correct steps before the first observed error."""
        failed_results = [
            result
            for result in self.puzzles
            if result.first_error is not None
        ]

        if not failed_results:
            return None

        return float(
            np.mean(
                [
                    result.correct_decisions_before_error
                    for result in failed_results
                ]
            )
        )

    @property
    def average_first_error_confidence(self) -> float | None:
        """Return mean model confidence for first ML errors."""
        confidences = [
            result.first_error.selected_confidence
            for result in self.puzzles
            if result.first_error is not None
            and result.first_error.selected_confidence is not None
        ]

        if not confidences:
            return None

        return float(np.mean(confidences))

    @property
    def average_correct_digit_rank(self) -> float | None:
        """Return mean rank of the correct digit at first errors."""
        ranks = [
            result.first_error.correct_digit_rank
            for result in self.puzzles
            if result.first_error is not None
            and result.first_error.correct_digit_rank is not None
        ]

        if not ranks:
            return None

        return float(np.mean(ranks))


def _validate_puzzle_and_solution(
    puzzle: np.ndarray,
    expected_solution: np.ndarray,
) -> None:
    if puzzle.shape != (9, 9) or expected_solution.shape != (9, 9):
        raise ValueError("Puzzles and solutions must have shape (9, 9).")

    puzzle_grid = SudokuGrid(puzzle)
    solution_grid = SudokuGrid(expected_solution)

    if not puzzle_grid.is_valid():
        raise ValueError("The puzzle must be a valid Sudoku grid.")

    if not solution_grid.is_valid() or not solution_grid.is_complete():
        raise ValueError("The expected solution must be valid and complete.")

    if not np.all((puzzle == 0) | (puzzle == expected_solution)):
        raise ValueError("Puzzle clues must match the expected solution.")


def _create_decision_error(
    decision: GreedyDecision,
    correct_digit: int,
) -> ModelOnlyDecisionError:
    try:
        correct_digit_rank = (
            decision.ranked_candidates.index(correct_digit) + 1
        )
    except ValueError:
        correct_digit_rank = None

    return ModelOnlyDecisionError(
        step=decision.step,
        row=decision.row,
        column=decision.column,
        candidates=decision.candidates,
        ranked_candidates=decision.ranked_candidates,
        selected_digit=decision.selected_digit,
        correct_digit=correct_digit,
        selected_confidence=decision.confidence,
        correct_digit_rank=correct_digit_rank,
        is_ml_decision=decision.is_ml_decision,
    )


def analyze_model_only_attempt(
    model: SudokuProbabilityModel,
    puzzle: np.ndarray,
    expected_solution: np.ndarray,
) -> ModelOnlyPuzzleResult:
    """Compare one Greedy ML solving trace with ground truth."""
    _validate_puzzle_and_solution(puzzle, expected_solution)

    solver = GreedyMLSudokuSolver(model)
    solution = solver.solve(SudokuGrid(puzzle))

    first_error: ModelOnlyDecisionError | None = None
    correct_decisions = 0

    for decision in solver.decision_trace:
        correct_digit = int(
            expected_solution[
                decision.row,
                decision.column,
            ]
        )

        if decision.selected_digit != correct_digit:
            first_error = _create_decision_error(
                decision,
                correct_digit,
            )
            break

        correct_decisions += 1

    completed_solution = (
        solution is not None
        and solution.is_complete()
    )
    valid_solution = (
        solution is not None
        and solution.is_valid()
    )
    exact_match = (
        solution is not None
        and np.array_equal(
            solution.values,
            expected_solution,
        )
    )

    return ModelOnlyPuzzleResult(
        exact_match=exact_match,
        valid_solution=valid_solution,
        completed_solution=completed_solution,
        total_decisions=len(solver.decision_trace),
        ml_decisions=solver.stats.ml_decisions,
        correct_decisions_before_error=correct_decisions,
        first_error=first_error,
    )


def compare_model_only_models(
    models: Mapping[str, SudokuProbabilityModel],
    puzzles: np.ndarray,
    expected_solutions: np.ndarray,
) -> tuple[ModelOnlyModelResult, ...]:
    """Analyze several models on identical puzzles and solutions."""
    if not models:
        raise ValueError("At least one model is required.")

    if len(puzzles) != len(expected_solutions):
        raise ValueError(
            "Expected solutions must match the number of puzzles."
        )

    if len(puzzles) == 0:
        raise ValueError("At least one puzzle is required.")

    return tuple(
        ModelOnlyModelResult(
            name=name,
            puzzles=tuple(
                analyze_model_only_attempt(
                    model,
                    puzzle,
                    expected_solution,
                )
                for puzzle, expected_solution in zip(
                    puzzles,
                    expected_solutions,
                )
            ),
        )
        for name, model in models.items()
    )
