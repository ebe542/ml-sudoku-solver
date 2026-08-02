from dataclasses import dataclass

import numpy as np

from sudoku_ml.grid import SudokuGrid
from sudoku_ml.model.protocol import SudokuProbabilityModel
from sudoku_ml.preprocessing.constraints import get_candidates
from sudoku_ml.preprocessing.features import create_feature_vector


@dataclass
class SolverStats:
    """Store statistics collected during one solving attempt."""

    deterministic_steps: int = 0
    ml_decisions: int = 0
    branching_decisions: int = 0
    backtracks: int = 0


class HybridSudokuSolver:
    """Solve Sudoku grids using constraints and ML-guided backtracking."""

    def __init__(self, model: SudokuProbabilityModel) -> None:
        self.model = model
        self.stats = SolverStats()

    def solve(self, puzzle: SudokuGrid) -> SudokuGrid | None:
        """Return a valid completed grid, or None if no solution exists."""
        self.stats = SolverStats()

        if not puzzle.is_valid():
            raise ValueError("The puzzle must be a valid Sudoku grid.")

        values = puzzle.values.copy()

        if not self._solve(values):
            return None

        return SudokuGrid(values)

    def _solve(self, grid: np.ndarray) -> bool:
        choice = self._select_cell(grid)
        if choice is None:
            return True

        row, column, candidates = choice
        if not candidates:
            return False

        for digit in self._rank_candidates(grid, row, column, candidates):
            grid[row, column] = digit
            if self._solve(grid):
                return True

            self.stats.backtracks += 1
            grid[row, column] = 0

        return False

    @staticmethod
    def _select_cell(grid: np.ndarray) -> tuple[int, int, set[int]] | None:
        """Choose the empty cell with the fewest valid candidates."""
        best: tuple[int, int, set[int]] | None = None

        for row, column in np.argwhere(grid == 0):
            candidates = get_candidates(grid, int(row), int(column))
            if not candidates:
                return int(row), int(column), candidates
            if best is None or len(candidates) < len(best[2]):
                best = int(row), int(column), candidates
                if len(candidates) == 1:
                    break

        return best

    def _rank_candidates(self, grid: np.ndarray, row: int, column: int, candidates: set[int]) -> list[int]:
        """Rank valid candidates using model probabilities."""
        if len(candidates) == 1:
            self.stats.deterministic_steps += 1
            return list(candidates)

        self.stats.ml_decisions += 1
        self.stats.branching_decisions += 1

        features = create_feature_vector(grid, row, column)[np.newaxis, :]
        probabilities = self.model.predict_probabilities(features)[0]
        probability_by_digit = dict(zip(self.model.classes, probabilities))

        return sorted(
            candidates,
            key=lambda digit: probability_by_digit.get(digit, 0.0),
            reverse=True,
        )


class GreedyMLSudokuSolver(HybridSudokuSolver):
    """Solve Sudoku grids with ML guidance and no backtracking."""

    def _solve(self, grid: np.ndarray) -> bool:
        while True:
            choice = self._select_cell(grid)

            if choice is None:
                return True

            row, column, candidates = choice

            if not candidates:
                return False

            ranked_candidates = self._rank_candidates(
                grid,
                row,
                column,
                candidates,
            )

            grid[row, column] = ranked_candidates[0]


class ClassicalSudokuSolver(HybridSudokuSolver):
    """Solve Sudoku grids using deterministic candidate ordering."""

    def __init__(self) -> None:
        self.stats = SolverStats()

    def _rank_candidates(self, grid: np.ndarray, row: int, column: int, candidates: set[int]) -> list[int]:
        """Return valid candidates in ascending numerical order."""
        if len(candidates) == 1:
            self.stats.deterministic_steps += 1
        else:
            self.stats.branching_decisions += 1

        return sorted(candidates)
