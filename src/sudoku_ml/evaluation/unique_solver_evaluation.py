from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.dataset.unique_generator import create_unique_dataset
from sudoku_ml.evaluation.difficulty_evaluation import (
    DifficultyEvaluationResult,
    RemovalRateResult,
)
from sudoku_ml.evaluation.solver_comparison import compare_solvers
from sudoku_ml.model.random_forest import SudokuRandomForest
from sudoku_ml.solver import ClassicalSudokuSolver, HybridSudokuSolver


def evaluate_unique_removal_rates(
    removal_rates: list[float],
    num_training_solutions: int = 100,
    num_evaluation_puzzles: int = 10,
    test_size: float = 0.2,
    n_estimators: int = 100,
    training_seed: int | None = 42,
    evaluation_seed: int | None = 123,
) -> DifficultyEvaluationResult:
    """Compare both solvers on uniquely solvable Sudoku puzzles."""
    if not removal_rates:
        raise ValueError("At least one removal rate is required.")

    if any(not 0.0 < rate < 1.0 for rate in removal_rates):
        raise ValueError("Removal rates must be between 0 and 1.")

    if num_evaluation_puzzles <= 0:
        raise ValueError("Number of evaluation puzzles must be positive.")

    results: list[RemovalRateResult] = []

    for removal_rate in removal_rates:
        training_data = create_train_test_split(
            num_solutions=num_training_solutions,
            test_size=test_size,
            removal_rate=removal_rate,
            random_seed=training_seed,
        )

        model = SudokuRandomForest(
            n_estimators=n_estimators,
            random_seed=training_seed,
        )
        model.fit(training_data)

        evaluation_dataset = create_unique_dataset(
            num_samples=num_evaluation_puzzles,
            removal_rate=removal_rate,
            random_seed=evaluation_seed,
        )

        comparison = compare_solvers(
            HybridSudokuSolver(model),
            ClassicalSudokuSolver(),
            evaluation_dataset.puzzles,
            evaluation_dataset.solutions,
        )

        results.append(
            RemovalRateResult(
                removal_rate=removal_rate,
                comparison=comparison,
            )
        )

    return DifficultyEvaluationResult(results=tuple(results))
