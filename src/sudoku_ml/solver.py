from dataclasses import dataclass
from math import log

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


@dataclass(frozen=True)
class GreedyDecision:
    """Describe one irreversible Greedy ML placement."""

    step: int
    row: int
    column: int
    candidates: tuple[int, ...]
    ranked_candidates: tuple[int, ...]
    selected_digit: int
    confidence: float | None
    is_ml_decision: bool


@dataclass
class BeamSearchStats(SolverStats):
    """Store statistics collected during Beam Search."""

    generated_states: int = 0
    pruned_states: int = 0
    max_active_states: int = 1


@dataclass(frozen=True)
class _BeamState:
    """Store one partial grid and its cumulative model score."""

    grid: np.ndarray
    score: float


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

    def _rank_candidates(
        self,
        grid: np.ndarray,
        row: int,
        column: int,
        candidates: set[int],
    ) -> list[int]:
        """Rank valid candidates using model probabilities."""
        ranked_candidates, _ = self._rank_candidates_with_confidence(
            grid,
            row,
            column,
            candidates,
        )

        return ranked_candidates

    def _rank_candidates_with_confidence(
        self,
        grid: np.ndarray,
        row: int,
        column: int,
        candidates: set[int],
    ) -> tuple[list[int], float | None]:
        """Rank candidates and return confidence in the first choice."""
        if len(candidates) == 1:
            self.stats.deterministic_steps += 1
            return list(candidates), None

        self.stats.ml_decisions += 1
        self.stats.branching_decisions += 1

        features = create_feature_vector(grid, row, column)[np.newaxis, :]
        probabilities = self.model.predict_probabilities(features)[0]
        probability_by_digit = dict(zip(self.model.classes, probabilities))

        ranked_candidates = sorted(
            candidates,
            key=lambda digit: probability_by_digit.get(digit, 0.0),
            reverse=True,
        )

        confidence = float(
            probability_by_digit.get(
                ranked_candidates[0],
                0.0,
            )
        )

        return ranked_candidates, confidence


class GreedyMLSudokuSolver(HybridSudokuSolver):
    """Solve Sudoku grids with ML guidance and no backtracking."""

    def __init__(self, model: SudokuProbabilityModel) -> None:
        super().__init__(model)
        self._decision_trace: list[GreedyDecision] = []

    @property
    def decision_trace(self) -> tuple[GreedyDecision, ...]:
        """Return placements from the most recent solving attempt."""
        return tuple(self._decision_trace)

    def solve(self, puzzle: SudokuGrid) -> SudokuGrid | None:
        """Solve a puzzle and reset its irreversible decision trace."""
        self._decision_trace = []
        return super().solve(puzzle)

    def _solve(self, grid: np.ndarray) -> bool:
        while True:
            choice = self._select_cell(grid)

            if choice is None:
                return True

            row, column, candidates = choice

            if not candidates:
                return False

            ranked_candidates, confidence = (
                self._rank_candidates_with_confidence(
                    grid,
                    row,
                    column,
                    candidates,
                )
            )

            selected_digit = ranked_candidates[0]

            self._decision_trace.append(
                GreedyDecision(
                    step=len(self._decision_trace) + 1,
                    row=row,
                    column=column,
                    candidates=tuple(sorted(candidates)),
                    ranked_candidates=tuple(ranked_candidates),
                    selected_digit=selected_digit,
                    confidence=confidence,
                    is_ml_decision=len(candidates) > 1,
                )
            )

            grid[row, column] = selected_digit


class BeamSearchSudokuSolver(HybridSudokuSolver):
    """Solve Sudoku grids while retaining several model-ranked paths."""

    def __init__(
        self,
        model: SudokuProbabilityModel,
        beam_width: int = 2,
    ) -> None:
        if not isinstance(beam_width, int):
            raise TypeError("Beam width must be an integer.")

        if beam_width <= 0:
            raise ValueError("Beam width must be positive.")

        super().__init__(model)
        self.beam_width = beam_width
        self.stats = BeamSearchStats()

    def solve(self, puzzle: SudokuGrid) -> SudokuGrid | None:
        """Return the highest-scoring valid completion in the beam."""
        self.stats = BeamSearchStats()

        if not puzzle.is_valid():
            raise ValueError("The puzzle must be a valid Sudoku grid.")

        active_states = [
            _BeamState(
                grid=puzzle.values.copy(),
                score=0.0,
            )
        ]

        while active_states:
            next_states: list[_BeamState] = []

            for state in active_states:
                choice = self._select_cell(state.grid)

                if choice is None:
                    return SudokuGrid(state.grid)

                row, column, candidates = choice

                if not candidates:
                    continue

                next_states.extend(
                    self._expand_state(
                        state,
                        row,
                        column,
                        candidates,
                    )
                )

            next_states.sort(
                key=lambda state: state.score,
                reverse=True,
            )

            discarded_states = max(
                0,
                len(next_states) - self.beam_width,
            )
            self.stats.pruned_states += discarded_states
            active_states = next_states[: self.beam_width]
            self.stats.max_active_states = max(
                self.stats.max_active_states,
                len(active_states),
            )

        return None

    def _expand_state(
        self,
        state: _BeamState,
        row: int,
        column: int,
        candidates: set[int],
    ) -> list[_BeamState]:
        if len(candidates) == 1:
            self.stats.deterministic_steps += 1
            digit = next(iter(candidates))

            return [
                self._create_child_state(
                    state,
                    row,
                    column,
                    digit,
                    state.score,
                )
            ]

        self.stats.ml_decisions += 1
        self.stats.branching_decisions += 1

        features = create_feature_vector(
            state.grid,
            row,
            column,
        )[np.newaxis, :]
        probabilities = self.model.predict_probabilities(features)[0]
        probability_by_digit = dict(
            zip(self.model.classes, probabilities)
        )

        children = [
            self._create_child_state(
                state,
                row,
                column,
                digit,
                state.score
                + log(
                    max(
                        float(probability_by_digit.get(digit, 0.0)),
                        np.finfo(float).tiny,
                    )
                ),
            )
            for digit in candidates
        ]

        return children

    def _create_child_state(
        self,
        state: _BeamState,
        row: int,
        column: int,
        digit: int,
        score: float,
    ) -> _BeamState:
        child_grid = state.grid.copy()
        child_grid[row, column] = digit
        self.stats.generated_states += 1

        return _BeamState(
            grid=child_grid,
            score=score,
        )


class ClassicalSudokuSolver(HybridSudokuSolver):
    """Solve Sudoku grids using deterministic candidate ordering."""

    def __init__(self) -> None:
        self.stats = SolverStats()

    def _rank_candidates(
        self,
        grid: np.ndarray,
        row: int,
        column: int,
        candidates: set[int],
    ) -> list[int]:
        """Return valid candidates in ascending numerical order."""
        if len(candidates) == 1:
            self.stats.deterministic_steps += 1
        else:
            self.stats.branching_decisions += 1

        return sorted(candidates)
