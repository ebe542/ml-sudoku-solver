from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.dataset.generator import create_diverse_dataset
from sudoku_ml.evaluation.solver_comparison import compare_solvers
from sudoku_ml.model.random_forest import SudokuRandomForest
from sudoku_ml.solver import (
    ClassicalSudokuSolver,
    HybridSudokuSolver,
)


def test_compare_solvers_uses_same_puzzles() -> None:
    training_data = create_train_test_split(
        num_solutions=20,
        random_seed=42,
    )

    model = SudokuRandomForest(
        n_estimators=20,
        random_seed=42,
    )
    model.fit(training_data)

    hybrid_solver = HybridSudokuSolver(model)
    classical_solver = ClassicalSudokuSolver()

    evaluation_dataset = create_diverse_dataset(
        num_samples=3,
        removal_rate=0.02,
        random_seed=123,
    )

    result = compare_solvers(
        hybrid_solver,
        classical_solver,
        evaluation_dataset.puzzles,
    )

    assert result.hybrid.total_puzzles == 3
    assert result.classical.total_puzzles == 3

    assert result.hybrid.solved_puzzles == 3
    assert result.classical.solved_puzzles == 3

    assert result.hybrid.valid_solutions == 3
    assert result.classical.valid_solutions == 3

    assert result.hybrid.deterministic_steps == 3
    assert result.classical.deterministic_steps == 3

    assert result.hybrid.ml_decisions == 0
    assert result.classical.ml_decisions == 0

    assert result.hybrid.backtracks == 0
    assert result.classical.backtracks == 0
