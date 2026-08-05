import numpy as np
import pytest

from sudoku_ml.analysis.difficulty import (
    DifficultyLevel,
    analyze_puzzle_difficulty,
    classify_difficulty,
    summarize_puzzle_difficulties,
)
from sudoku_ml.grid import SudokuGrid
from sudoku_ml.sudoku_generator import generate_solved_grid


def create_single_empty_puzzle(cell_index: int = 0) -> SudokuGrid:
    solution = generate_solved_grid(random_seed=42)
    values = solution.values.copy()
    values.flat[cell_index] = 0
    return SudokuGrid(values)


@pytest.mark.parametrize(
    ("branching_decisions", "backtracks", "expected_level"),
    [
        (0, 0, DifficultyLevel.EASY),
        (1, 0, DifficultyLevel.MEDIUM),
        (4, 6, DifficultyLevel.MEDIUM),
        (5, 6, DifficultyLevel.HARD),
        (40, 60, DifficultyLevel.HARD),
        (40, 61, DifficultyLevel.EXPERT),
    ],
)
def test_classify_difficulty_uses_search_effort(
    branching_decisions: int,
    backtracks: int,
    expected_level: DifficultyLevel,
) -> None:
    assert (
        classify_difficulty(branching_decisions, backtracks)
        is expected_level
    )


def test_classify_difficulty_rejects_negative_metrics() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        classify_difficulty(-1, 0)


def test_analyze_easy_puzzle() -> None:
    puzzle = create_single_empty_puzzle()

    result = analyze_puzzle_difficulty(puzzle)

    assert result.given_cells == 80
    assert result.empty_cells == 1
    assert result.initial_single_candidates == 1
    assert result.initial_average_candidate_count == pytest.approx(1.0)
    assert result.deterministic_steps == 1
    assert result.branching_decisions == 0
    assert result.backtracks == 0
    assert result.difficulty_score == 0
    assert result.level is DifficultyLevel.EASY


def test_analyze_difficulty_does_not_modify_puzzle() -> None:
    puzzle = create_single_empty_puzzle()
    original = puzzle.values.copy()

    analyze_puzzle_difficulty(puzzle)

    assert np.array_equal(puzzle.values, original)


def test_analyze_difficulty_rejects_non_unique_puzzle() -> None:
    puzzle = SudokuGrid(np.zeros((9, 9), dtype=int))

    with pytest.raises(ValueError, match="exactly one solution"):
        analyze_puzzle_difficulty(puzzle)


def test_summarize_puzzle_difficulties() -> None:
    puzzles = np.asarray(
        [
            create_single_empty_puzzle(0).values,
            create_single_empty_puzzle(1).values,
        ]
    )

    result = summarize_puzzle_difficulties(puzzles)

    assert result.puzzle_count == 2
    assert result.average_given_cells == pytest.approx(80.0)
    assert result.average_initial_single_candidates == pytest.approx(1.0)
    assert result.average_initial_candidate_count == pytest.approx(1.0)
    assert result.average_deterministic_steps == pytest.approx(1.0)
    assert result.average_branching_decisions == pytest.approx(0.0)
    assert result.average_backtracks == pytest.approx(0.0)
    assert result.average_difficulty_score == pytest.approx(0.0)
    assert result.easy_count == 2
    assert result.medium_count == 0
    assert result.hard_count == 0
    assert result.expert_count == 0


def test_summarize_puzzle_difficulties_rejects_empty_collection() -> None:
    with pytest.raises(ValueError, match="At least one puzzle"):
        summarize_puzzle_difficulties(np.empty((0, 9, 9), dtype=int))
