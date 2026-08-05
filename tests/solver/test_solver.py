import numpy as np
import pytest

from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.grid import SudokuGrid
from sudoku_ml.sudoku_generator import generate_solved_grid
from sudoku_ml.model.random_forest import SudokuRandomForest
from sudoku_ml.solver import (
    ClassicalSudokuSolver,
    GreedyMLSudokuSolver,
    HybridSudokuSolver,
)

@pytest.fixture(scope="module")
def trained_solver() -> HybridSudokuSolver:
    data = create_train_test_split(num_solutions=20, random_seed=42)
    model = SudokuRandomForest(n_estimators=20, random_seed=42)
    model.fit(data)
    return HybridSudokuSolver(model)

@pytest.fixture(scope="module")
def trained_solver() -> HybridSudokuSolver:
    data = create_train_test_split(
        num_solutions=20,
        random_seed=42,
    )
    model = SudokuRandomForest(
        n_estimators=20,
        random_seed=42,
    )
    model.fit(data)
    return HybridSudokuSolver(model)


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


def test_solver_completes_valid_puzzle(trained_solver: HybridSudokuSolver) -> None:
    puzzle = SudokuGrid(
        np.array(
            [
                [5, 3, 0, 0, 7, 0, 0, 0, 0],
                [6, 0, 0, 1, 9, 5, 0, 0, 0],
                [0, 9, 8, 0, 0, 0, 0, 6, 0],
                [8, 0, 0, 0, 6, 0, 0, 0, 3],
                [4, 0, 0, 8, 0, 3, 0, 0, 1],
                [7, 0, 0, 0, 2, 0, 0, 0, 6],
                [0, 6, 0, 0, 0, 0, 2, 8, 0],
                [0, 0, 0, 4, 1, 9, 0, 0, 5],
                [0, 0, 0, 0, 8, 0, 0, 7, 9],
            ]
        )
    )

    solution = trained_solver.solve(puzzle)

    assert solution is not None
    assert solution.is_complete()
    assert solution.is_valid()
    assert np.all((puzzle.values == 0) | (solution.values == puzzle.values))


def test_solver_does_not_modify_input(trained_solver: HybridSudokuSolver) -> None:
    puzzle = SudokuGrid(np.zeros((9, 9), dtype=int))
    original = puzzle.values.copy()

    trained_solver.solve(puzzle)

    assert np.array_equal(puzzle.values, original)


def test_solver_rejects_conflicting_puzzle(trained_solver: HybridSudokuSolver) -> None:
    values = np.zeros((9, 9), dtype=int)
    values[0, :2] = 1

    with pytest.raises(ValueError, match="valid Sudoku"):
        trained_solver.solve(SudokuGrid(values))

def test_solver_counts_deterministic_steps(trained_solver: HybridSudokuSolver) -> None:
    puzzle = SudokuGrid(
        np.array(
            [
                [0, 3, 4, 6, 7, 8, 9, 1, 2],
                [6, 7, 2, 1, 9, 5, 3, 4, 8],
                [1, 9, 8, 3, 4, 2, 5, 6, 7],
                [8, 5, 9, 7, 6, 1, 4, 2, 3],
                [4, 2, 6, 8, 5, 3, 7, 9, 1],
                [7, 1, 3, 9, 2, 4, 8, 5, 6],
                [9, 6, 1, 5, 3, 7, 2, 8, 4],
                [2, 8, 7, 4, 1, 9, 6, 3, 5],
                [3, 4, 5, 2, 8, 6, 1, 7, 9],
            ]
        )
    )

    solution = trained_solver.solve(puzzle)

    assert solution is not None
    assert trained_solver.stats.deterministic_steps == 1
    assert trained_solver.stats.ml_decisions == 0
    assert trained_solver.stats.backtracks == 0

def test_classical_solver_completes_puzzle_without_model() -> None:
    complete_grid = generate_solved_grid(random_seed=42)

    puzzle_values = complete_grid.values.copy()
    puzzle_values[0, 0] = 0

    solver = ClassicalSudokuSolver()
    solution = solver.solve(SudokuGrid(puzzle_values))

    assert solution is not None
    assert solution.is_complete()
    assert solution.is_valid()
    assert np.array_equal(solution.values, complete_grid.values)
    assert solver.stats.deterministic_steps == 1
    assert solver.stats.ml_decisions == 0
    assert solver.stats.backtracks == 0

def test_greedy_solver_completes_deterministic_puzzle(ambiguous_model: SudokuRandomForest) -> None:
    complete_grid = generate_solved_grid(random_seed=42)

    puzzle_values = complete_grid.values.copy()
    puzzle_values[0, 0] = 0

    solver = GreedyMLSudokuSolver(ambiguous_model)
    solution = solver.solve(SudokuGrid(puzzle_values))

    assert solution is not None
    assert solution.is_complete()
    assert solution.is_valid()
    assert solver.stats.deterministic_steps == 1
    assert solver.stats.ml_decisions == 0
    assert solver.stats.branching_decisions == 0
    assert solver.stats.backtracks == 0


def test_greedy_solver_cannot_recover_from_wrong_ml_choice(ambiguous_model: SudokuRandomForest) -> None:
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

    greedy_solver = GreedyMLSudokuSolver(
        ambiguous_model
    )
    hybrid_solver = HybridSudokuSolver(
        ambiguous_model
    )

    greedy_solution = greedy_solver.solve(puzzle)
    hybrid_solution = hybrid_solver.solve(puzzle)

    assert greedy_solution is None
    assert greedy_solver.stats.ml_decisions > 0
    assert greedy_solver.stats.branching_decisions > 0
    assert greedy_solver.stats.backtracks == 0

    assert hybrid_solution is not None
    assert hybrid_solution.is_complete()
    assert hybrid_solution.is_valid()
    assert hybrid_solver.stats.backtracks > 0
