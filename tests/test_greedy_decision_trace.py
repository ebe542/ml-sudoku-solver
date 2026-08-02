import numpy as np

from sudoku_ml.grid import SudokuGrid
from sudoku_ml.solver import GreedyMLSudokuSolver
from sudoku_ml.sudoku_generator import generate_solved_grid


class FixedProbabilityModel:
    """Return a fixed digit ranking for trace tests."""

    classes = np.arange(1, 10)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        probabilities = np.array(
            [0.40, 0.20, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03]
        )
        return np.tile(probabilities, (len(X), 1))


def test_trace_records_deterministic_placement() -> None:
    solved_grid = generate_solved_grid(random_seed=42)
    puzzle_values = solved_grid.values.copy()
    expected_digit = int(puzzle_values[0, 0])
    puzzle_values[0, 0] = 0

    solver = GreedyMLSudokuSolver(FixedProbabilityModel())
    solution = solver.solve(SudokuGrid(puzzle_values))

    assert solution is not None
    assert len(solver.decision_trace) == 1

    decision = solver.decision_trace[0]
    assert decision.step == 1
    assert decision.row == 0
    assert decision.column == 0
    assert decision.candidates == (expected_digit,)
    assert decision.ranked_candidates == (expected_digit,)
    assert decision.selected_digit == expected_digit
    assert decision.confidence is None
    assert decision.is_ml_decision is False


def test_trace_records_ranked_ml_decision() -> None:
    solver = GreedyMLSudokuSolver(FixedProbabilityModel())

    solver.solve(
        SudokuGrid(
            np.zeros((9, 9), dtype=int)
        )
    )

    first_decision = solver.decision_trace[0]

    assert first_decision.step == 1
    assert first_decision.row == 0
    assert first_decision.column == 0
    assert first_decision.candidates == tuple(range(1, 10))
    assert first_decision.ranked_candidates == tuple(range(1, 10))
    assert first_decision.selected_digit == 1
    assert first_decision.confidence == 0.40
    assert first_decision.is_ml_decision is True


def test_trace_contains_no_backtracking_placements() -> None:
    solver = GreedyMLSudokuSolver(FixedProbabilityModel())

    solver.solve(
        SudokuGrid(
            np.zeros((9, 9), dtype=int)
        )
    )

    assert len(solver.decision_trace) > 0
    assert solver.stats.backtracks == 0
    assert [
        decision.step
        for decision in solver.decision_trace
    ] == list(
        range(1, len(solver.decision_trace) + 1)
    )


def test_trace_is_reset_before_next_attempt() -> None:
    solver = GreedyMLSudokuSolver(FixedProbabilityModel())
    solver.solve(
        SudokuGrid(
            np.zeros((9, 9), dtype=int)
        )
    )

    solved_grid = generate_solved_grid(random_seed=42)
    puzzle_values = solved_grid.values.copy()
    puzzle_values[0, 0] = 0

    solver.solve(SudokuGrid(puzzle_values))

    assert len(solver.decision_trace) == 1
    assert solver.decision_trace[0].step == 1
