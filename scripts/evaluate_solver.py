from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.dataset.generator import create_diverse_dataset
from sudoku_ml.evaluation.solver_evaluation import evaluate_solver
from sudoku_ml.model.random_forest import SudokuRandomForest
from sudoku_ml.solver import HybridSudokuSolver


def main() -> None:
    """Train the model and evaluate the hybrid solver."""
    training_data = create_train_test_split(
        num_solutions=100,
        test_size=0.2,
        removal_rate=0.5,
        random_seed=42,
    )

    model = SudokuRandomForest(
        n_estimators=100,
        random_seed=42,
    )
    model.fit(training_data)

    solver = HybridSudokuSolver(model)

    evaluation_dataset = create_diverse_dataset(
        num_samples=20,
        removal_rate=0.5,
        random_seed=123,
    )

    result = evaluate_solver(
        solver,
        evaluation_dataset.puzzles,
    )

    print("Hybrid Sudoku Solver Evaluation")
    print("--------------------------------")
    print(f"Puzzles evaluated:       {result.total_puzzles}")
    print(f"Puzzles solved:          {result.solved_puzzles}")
    print(f"Valid solutions:         {result.valid_solutions}")
    print(f"Solution rate:           {result.solution_rate:.2%}")
    print(f"Valid solution rate:     {result.valid_solution_rate:.2%}")
    print(
        "Average runtime:        "
        f"{result.average_runtime_seconds * 1000:.2f} ms"
    )
    print(f"Deterministic steps:     {result.deterministic_steps}")
    print(f"ML decisions:            {result.ml_decisions}")
    print(f"Backtracks:              {result.backtracks}")
    print(f"Average backtracks:      {result.average_backtracks:.2f}")


if __name__ == "__main__":
    main()
