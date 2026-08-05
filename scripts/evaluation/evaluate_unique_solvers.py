from sudoku_ml.evaluation.unique_solver_evaluation import (
    evaluate_unique_removal_rates,
)


def main() -> None:
    """Compare both solvers on uniquely solvable Sudoku puzzles."""
    evaluation = evaluate_unique_removal_rates(
        removal_rates=[0.50, 0.60, 0.65],
        num_training_solutions=100,
        num_evaluation_puzzles=10,
        n_estimators=100,
        training_seed=42,
        evaluation_seed=123,
    )

    print("Unique-Solution Sudoku Evaluation")
    print("---------------------------------")
    print()
    print("Solution Quality and Runtime")
    print(
        f"{'Removal':>8}"
        f"{'Hybrid match':>14}"
        f"{'Classic match':>15}"
        f"{'Hybrid ms':>12}"
        f"{'Classic ms':>12}"
        f"{'Runtime ratio':>15}"
    )
    print("-" * 76)

    for item in evaluation.results:
        hybrid = item.comparison.hybrid
        classical = item.comparison.classical

        print(
            f"{item.removal_rate:>7.0%}"
            f"{hybrid.matching_solution_rate:>13.2%}"
            f"{classical.matching_solution_rate:>14.2%}"
            f"{hybrid.average_runtime_seconds * 1000:>12.2f}"
            f"{classical.average_runtime_seconds * 1000:>12.2f}"
            f"{item.runtime_ratio:>14.2f}x"
        )

    print()
    print("Search Effort")
    print(
        f"{'Removal':>8}"
        f"{'Hybrid BT':>12}"
        f"{'Classic BT':>12}"
        f"{'BT reduction':>14}"
        f"{'Avg ML decisions':>18}"
    )
    print("-" * 64)

    for item in evaluation.results:
        hybrid = item.comparison.hybrid
        classical = item.comparison.classical

        print(
            f"{item.removal_rate:>7.0%}"
            f"{hybrid.average_backtracks:>12.2f}"
            f"{classical.average_backtracks:>12.2f}"
            f"{item.backtrack_reduction:>13.2%}"
            f"{hybrid.average_ml_decisions:>18.2f}"
        )

    print()
    print("Match rate compares solver output with the unique ground truth.")
    print("Backtracks and ML decisions are averages per puzzle.")


if __name__ == "__main__":
    main()
