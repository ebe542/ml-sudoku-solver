import numpy as np
import pytest

from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.grid import SudokuGrid
from sudoku_ml.model.random_forest import SudokuRandomForest
from sudoku_ml.solver import HybridSudokuSolver


@pytest.fixture(scope="module")
def trained_solver() -> HybridSudokuSolver:
    data = create_train_test_split(num_solutions=20, random_seed=42)
    model = SudokuRandomForest(n_estimators=20, random_seed=42)
    model.fit(data)
    return HybridSudokuSolver(model)


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
