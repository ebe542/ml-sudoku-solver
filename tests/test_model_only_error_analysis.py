import numpy as np
import pytest

from sudoku_ml.analysis.model_only_error_analysis import (
    analyze_model_only_attempt,
    compare_model_only_models,
)


SOLUTION = np.array(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [4, 5, 6, 7, 8, 9, 1, 2, 3],
        [7, 8, 9, 1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6, 7, 8, 9, 1],
        [5, 6, 7, 8, 9, 1, 2, 3, 4],
        [8, 9, 1, 2, 3, 4, 5, 6, 7],
        [3, 4, 5, 6, 7, 8, 9, 1, 2],
        [6, 7, 8, 9, 1, 2, 3, 4, 5],
        [9, 1, 2, 3, 4, 5, 6, 7, 8],
    ],
    dtype=int,
)


class DescendingProbabilityModel:
    """Prefer larger digits for predictable error tests."""

    classes = np.arange(1, 10)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        probabilities = np.arange(1, 10, dtype=float)
        probabilities /= probabilities.sum()
        return np.tile(probabilities, (len(X), 1))


def test_analysis_reports_exact_deterministic_solution() -> None:
    puzzle = SOLUTION.copy()
    puzzle[0, 0] = 0

    result = analyze_model_only_attempt(
        DescendingProbabilityModel(),
        puzzle,
        SOLUTION,
    )

    assert result.exact_match is True
    assert result.valid_solution is True
    assert result.completed_solution is True
    assert result.total_decisions == 1
    assert result.ml_decisions == 0
    assert result.correct_decisions_before_error == 1
    assert result.first_error is None


def test_analysis_reports_first_wrong_ml_decision() -> None:
    result = analyze_model_only_attempt(
        DescendingProbabilityModel(),
        np.zeros((9, 9), dtype=int),
        SOLUTION,
    )

    error = result.first_error

    assert result.exact_match is False
    assert result.correct_decisions_before_error == 0
    assert error is not None
    assert error.step == 1
    assert error.row == 0
    assert error.column == 0
    assert error.selected_digit == 9
    assert error.correct_digit == 1
    assert error.correct_digit_rank == 9
    assert error.selected_confidence == pytest.approx(0.2)
    assert error.is_ml_decision is True


def test_model_summary_aggregates_error_metrics() -> None:
    puzzles = np.stack(
        [
            np.zeros((9, 9), dtype=int),
            np.zeros((9, 9), dtype=int),
        ]
    )
    solutions = np.stack([SOLUTION, SOLUTION])

    result = compare_model_only_models(
        models={"Descending": DescendingProbabilityModel()},
        puzzles=puzzles,
        expected_solutions=solutions,
    )[0]

    assert result.name == "Descending"
    assert result.exact_solution_rate == 0.0
    assert result.failure_rate == 1.0
    assert result.average_correct_decisions_before_error == 0.0
    assert result.average_first_error_confidence == pytest.approx(0.2)
    assert result.average_correct_digit_rank == 9.0


def test_model_summary_has_no_error_averages_for_success() -> None:
    puzzle = SOLUTION.copy()
    puzzle[0, 0] = 0

    result = compare_model_only_models(
        models={"Descending": DescendingProbabilityModel()},
        puzzles=np.stack([puzzle]),
        expected_solutions=np.stack([SOLUTION]),
    )[0]

    assert result.exact_solution_rate == 1.0
    assert result.average_correct_decisions_before_error is None
    assert result.average_first_error_confidence is None
    assert result.average_correct_digit_rank is None


def test_analysis_rejects_clues_that_differ_from_ground_truth() -> None:
    puzzle = SOLUTION.copy()
    puzzle[0, 0] = 2
    puzzle[0, 1] = 0
    puzzle[3, 0] = 0

    with pytest.raises(ValueError, match="clues must match"):
        analyze_model_only_attempt(
            DescendingProbabilityModel(),
            puzzle,
            SOLUTION,
        )


def test_comparison_rejects_mismatched_solution_count() -> None:
    with pytest.raises(ValueError, match="number of puzzles"):
        compare_model_only_models(
            models={"Descending": DescendingProbabilityModel()},
            puzzles=np.zeros((2, 9, 9), dtype=int),
            expected_solutions=np.stack([SOLUTION]),
        )


def test_comparison_rejects_empty_models() -> None:
    with pytest.raises(ValueError, match="At least one model"):
        compare_model_only_models(
            models={},
            puzzles=np.zeros((1, 9, 9), dtype=int),
            expected_solutions=np.stack([SOLUTION]),
        )


def test_comparison_rejects_empty_puzzles() -> None:
    with pytest.raises(ValueError, match="At least one puzzle"):
        compare_model_only_models(
            models={"Descending": DescendingProbabilityModel()},
            puzzles=np.empty((0, 9, 9), dtype=int),
            expected_solutions=np.empty((0, 9, 9), dtype=int),
        )
