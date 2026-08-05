import numpy as np
import pytest

from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.grid import SudokuGrid
from sudoku_ml.model.random_forest import SudokuRandomForest
from sudoku_ml.solver import (
    BeamSearchSudokuSolver,
    GreedyMLSudokuSolver,
)
from sudoku_ml.sudoku_generator import generate_solved_grid


class FixedProbabilityModel:
    """Return stable probabilities for Beam Search tests."""

    classes = np.arange(1, 10)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        probabilities = np.array(
            [0.30, 0.20, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.02]
        )
        return np.tile(probabilities, (len(X), 1))


@pytest.fixture(scope="module")
def ambiguous_model() -> SudokuRandomForest:
    data = create_train_test_split(
        num_solutions=50,
        removal_rate=0.65,
        random_seed=42,
    )
    model = SudokuRandomForest(
        n_estimators=50,
        random_seed=42,
    )
    model.fit(data)
    return model


def test_beam_solver_completes_deterministic_puzzle() -> None:
    solved_grid = generate_solved_grid(random_seed=42)
    puzzle_values = solved_grid.values.copy()
    puzzle_values[0, 0] = 0

    solver = BeamSearchSudokuSolver(FixedProbabilityModel())
    solution = solver.solve(SudokuGrid(puzzle_values))

    assert solution is not None
    assert np.array_equal(solution.values, solved_grid.values)
    assert solver.stats.deterministic_steps == 1
    assert solver.stats.ml_decisions == 0
    assert solver.stats.generated_states == 1
    assert solver.stats.pruned_states == 0
    assert solver.stats.max_active_states == 1
    assert solver.stats.backtracks == 0


def test_beam_solver_does_not_modify_input() -> None:
    puzzle = SudokuGrid(np.zeros((9, 9), dtype=int))
    original_values = puzzle.values.copy()

    BeamSearchSudokuSolver(
        FixedProbabilityModel(),
        beam_width=2,
    ).solve(puzzle)

    assert np.array_equal(puzzle.values, original_values)


def test_beam_width_limits_active_states() -> None:
    solver = BeamSearchSudokuSolver(
        FixedProbabilityModel(),
        beam_width=2,
    )

    solver.solve(SudokuGrid(np.zeros((9, 9), dtype=int)))

    assert solver.stats.max_active_states == 2
    assert solver.stats.pruned_states > 0
    assert solver.stats.generated_states > 2
    assert solver.stats.backtracks == 0


def test_beam_search_recovers_known_greedy_failure(
    ambiguous_model: SudokuRandomForest,
) -> None:
    puzzle_text = (
        "650200040"
        "000603000"
        "100040200"
        "305000000"
        "068509003"
        "000301650"
        "006004000"
        "080090570"
        "500000068"
    )
    puzzle_values = np.array(
        [int(value) for value in puzzle_text],
        dtype=int,
    ).reshape(9, 9)
    puzzle = SudokuGrid(puzzle_values)

    greedy_solution = GreedyMLSudokuSolver(
        ambiguous_model
    ).solve(puzzle)
    beam_solver = BeamSearchSudokuSolver(
        ambiguous_model,
        beam_width=4,
    )
    beam_solution = beam_solver.solve(puzzle)

    assert greedy_solution is None
    assert beam_solution is not None
    assert beam_solution.is_complete()
    assert beam_solution.is_valid()
    assert beam_solver.stats.max_active_states == 4
    assert beam_solver.stats.backtracks == 0


@pytest.mark.parametrize("beam_width", [0, -1])
def test_beam_solver_rejects_non_positive_width(
    beam_width: int,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        BeamSearchSudokuSolver(
            FixedProbabilityModel(),
            beam_width=beam_width,
        )


def test_beam_solver_rejects_non_integer_width() -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        BeamSearchSudokuSolver(
            FixedProbabilityModel(),
            beam_width=2.5,
        )


def test_beam_solver_rejects_invalid_puzzle() -> None:
    values = np.zeros((9, 9), dtype=int)
    values[0, :2] = 1

    solver = BeamSearchSudokuSolver(FixedProbabilityModel())

    with pytest.raises(ValueError, match="valid Sudoku"):
        solver.solve(SudokuGrid(values))
