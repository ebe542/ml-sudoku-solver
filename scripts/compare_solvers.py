from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.dataset.generator import create_diverse_dataset
from sudoku_ml.evaluation.solver_comparison import compare_solvers
from sudoku_ml.model.random_forest import SudokuRandomForest
from sudoku_ml.solver import (
    ClassicalSudokuSolver,
    HybridSudokuSolver,
)


def main() -> None:
    """Compare ML-guided and classical candidate ordering."""
    removal_rate = 0.65

    training_data = create_train_test_split(
        num_solutions=100,
        test_size=0.2,
        removal_rate=removal_rate,
        random_seed=42,
    )

    model = SudokuRandomForest(
        n_estimators=100,
        random_seed=42,
    )
    model.fit(training_data)

    hybrid_solver = HybridSudokuSolver(model)
    classical_solver = ClassicalSudokuSolver()

    evaluation_dataset = create_diverse_dataset(
        num_samples=20,
        removal_rate=removal_rate,
        random_seed=123,
    )

    result = compare_solvers(
        hybrid_solver,
        classical_solver,
        evaluation_dataset.puzzles,
    )

    print("ML-Guided vs Classical Sudoku Solver")
    print("------------------------------------")
    print(f"Removal rate: {removal_rate:.0%}")
    print(f"Puzzles:      {len(evaluation_dataset.puzzles)}")
    print()

    print(
        f"{'Metric':<25}"
        f"{'Hybrid':>15}"
        f"{'Classical':>15}"
    )
    print("-" * 55)

    print(
        f"{'Solution rate':<25}"
        f"{result.hybrid.solution_rate:>14.2%}"
        f"{result.classical.solution_rate:>14.2%}"
    )
    print(
        f"{'Valid solution rate':<25}"
        f"{result.hybrid.valid_solution_rate:>14.2%}"
        f"{result.classical.valid_solution_rate:>14.2%}"
    )
    print(
        f"{'Average runtime (ms)':<25}"
        f"{result.hybrid.average_runtime_seconds * 1000:>15.2f}"
        f"{result.classical.average_runtime_seconds * 1000:>15.2f}"
    )
    print(
        f"{'Deterministic steps':<25}"
        f"{result.hybrid.deterministic_steps:>15}"
        f"{result.classical.deterministic_steps:>15}"
    )
    print(
        f"{'Backtracks':<25}"
        f"{result.hybrid.backtracks:>15}"
        f"{result.classical.backtracks:>15}"
    )
    print(
        f"{'Average backtracks':<25}"
        f"{result.hybrid.average_backtracks:>15.2f}"
        f"{result.classical.average_backtracks:>15.2f}"
    )

    print()
    print(f"Hybrid ML decisions: {result.hybrid.ml_decisions}")


if __name__ == "__main__":
    main()
