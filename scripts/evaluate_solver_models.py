from sudoku_ml.evaluation.solver_model_comparison import (
    evaluate_solver_model_removal_rates,
)


def main() -> None:
    """Compare classifiers inside the Hybrid solver."""
    evaluation = evaluate_solver_model_removal_rates(
        removal_rates=[0.50, 0.60, 0.65],
        num_training_solutions=100,
        num_evaluation_puzzles=20,
        test_size=0.2,
        n_estimators=100,
        training_seed=42,
        evaluation_seed=123,
    )

    print("Classifier Comparison in Hybrid Sudoku Solver")
    print("---------------------------------------------")
    print()

    print("Solution Quality and Runtime")
    print(
        f"{'Removal':>8}"
        f" {'Model':<29}"
        f"{'Exact match':>13}"
        f"{'Valid':>10}"
        f"{'Runtime ms':>13}"
    )
    print("-" * 73)

    for rate_result in evaluation.results:
        for item in rate_result.models:
            result = item.evaluation

            print(
                f"{rate_result.removal_rate:>7.0%}"
                f" {item.name:<29}"
                f"{result.matching_solution_rate:>13.2%}"
                f"{result.valid_solution_rate:>10.2%}"
                f"{result.average_runtime_seconds * 1000:>13.2f}"
            )

    print()
    print("Search Effort")
    print(
        f"{'Removal':>8}"
        f" {'Model':<29}"
        f"{'Deterministic':>15}"
        f"{'ML decisions':>14}"
        f"{'Backtracks':>12}"
    )
    print("-" * 78)

    for rate_result in evaluation.results:
        for item in rate_result.models:
            result = item.evaluation

            average_deterministic_steps = (
                result.deterministic_steps
                / result.total_puzzles
            )

            print(
                f"{rate_result.removal_rate:>7.0%}"
                f" {item.name:<29}"
                f"{average_deterministic_steps:>15.2f}"
                f"{result.average_ml_decisions:>14.2f}"
                f"{result.average_backtracks:>12.2f}"
            )

    print()
    print(
        "Every classifier uses the same training data and "
        "unique evaluation puzzles."
    )
    print(
        "All values except solution rates are averages "
        "per puzzle."
    )
    print(
        "Runtime includes feature generation, model inference, "
        "and backtracking."
    )


if __name__ == "__main__":
    main()
