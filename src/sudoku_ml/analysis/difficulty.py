from dataclasses import dataclass
from enum import Enum

import numpy as np

from sudoku_ml.grid import SudokuGrid
from sudoku_ml.preprocessing.constraints import get_candidates
from sudoku_ml.solution_counter import has_unique_solution
from sudoku_ml.solver import ClassicalSudokuSolver


class DifficultyLevel(str, Enum):
    """Represent a project-specific heuristic difficulty level."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass(frozen=True)
class PuzzleDifficulty:
    """Store structural and search-based puzzle difficulty metrics."""

    given_cells: int
    empty_cells: int
    initial_single_candidates: int
    initial_average_candidate_count: float
    deterministic_steps: int
    branching_decisions: int
    backtracks: int
    difficulty_score: int
    level: DifficultyLevel


@dataclass(frozen=True)
class DifficultyDatasetSummary:
    """Store average difficulty metrics for a puzzle collection."""

    puzzle_count: int
    average_given_cells: float
    average_initial_single_candidates: float
    average_initial_candidate_count: float
    average_deterministic_steps: float
    average_branching_decisions: float
    average_backtracks: float
    average_difficulty_score: float
    easy_count: int
    medium_count: int
    hard_count: int
    expert_count: int


def classify_difficulty(
    branching_decisions: int,
    backtracks: int,
) -> DifficultyLevel:
    """Classify search effort using project-specific thresholds."""
    if branching_decisions < 0 or backtracks < 0:
        raise ValueError("Difficulty metrics must be non-negative.")

    score = branching_decisions + backtracks

    if score == 0:
        return DifficultyLevel.EASY

    if score <= 10:
        return DifficultyLevel.MEDIUM

    if score <= 100:
        return DifficultyLevel.HARD

    return DifficultyLevel.EXPERT


def analyze_puzzle_difficulty(puzzle: SudokuGrid) -> PuzzleDifficulty:
    """Analyze one uniquely solvable puzzle without ML guidance."""
    if not puzzle.is_valid():
        raise ValueError("Difficulty analysis requires a valid puzzle.")

    if not has_unique_solution(puzzle):
        raise ValueError(
            "Difficulty analysis requires exactly one solution."
        )

    candidate_counts = [
        len(get_candidates(puzzle.values, row, column))
        for row, column in puzzle.empty_cells
    ]
    initial_single_candidates = sum(
        count == 1 for count in candidate_counts
    )
    initial_average_candidate_count = (
        float(np.mean(candidate_counts))
        if candidate_counts
        else 0.0
    )

    solver = ClassicalSudokuSolver()
    solution = solver.solve(puzzle)

    if solution is None:
        raise ValueError("Difficulty analysis requires a solvable puzzle.")

    difficulty_score = (
        solver.stats.branching_decisions
        + solver.stats.backtracks
    )

    return PuzzleDifficulty(
        given_cells=81 - len(puzzle.empty_cells),
        empty_cells=len(puzzle.empty_cells),
        initial_single_candidates=initial_single_candidates,
        initial_average_candidate_count=initial_average_candidate_count,
        deterministic_steps=solver.stats.deterministic_steps,
        branching_decisions=solver.stats.branching_decisions,
        backtracks=solver.stats.backtracks,
        difficulty_score=difficulty_score,
        level=classify_difficulty(
            solver.stats.branching_decisions,
            solver.stats.backtracks,
        ),
    )


def summarize_puzzle_difficulties(
    puzzles: np.ndarray,
) -> DifficultyDatasetSummary:
    """Summarize heuristic difficulty across multiple puzzles."""
    if len(puzzles) == 0:
        raise ValueError("At least one puzzle is required.")

    analyses = [
        analyze_puzzle_difficulty(SudokuGrid(values))
        for values in puzzles
    ]

    return DifficultyDatasetSummary(
        puzzle_count=len(analyses),
        average_given_cells=float(
            np.mean([item.given_cells for item in analyses])
        ),
        average_initial_single_candidates=float(
            np.mean(
                [item.initial_single_candidates for item in analyses]
            )
        ),
        average_initial_candidate_count=float(
            np.mean(
                [
                    item.initial_average_candidate_count
                    for item in analyses
                ]
            )
        ),
        average_deterministic_steps=float(
            np.mean([item.deterministic_steps for item in analyses])
        ),
        average_branching_decisions=float(
            np.mean([item.branching_decisions for item in analyses])
        ),
        average_backtracks=float(
            np.mean([item.backtracks for item in analyses])
        ),
        average_difficulty_score=float(
            np.mean([item.difficulty_score for item in analyses])
        ),
        easy_count=sum(
            item.level is DifficultyLevel.EASY for item in analyses
        ),
        medium_count=sum(
            item.level is DifficultyLevel.MEDIUM for item in analyses
        ),
        hard_count=sum(
            item.level is DifficultyLevel.HARD for item in analyses
        ),
        expert_count=sum(
            item.level is DifficultyLevel.EXPERT for item in analyses
        ),
    )
