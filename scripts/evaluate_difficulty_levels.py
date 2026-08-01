from sudoku_ml.evaluation.difficulty_evaluation import evaluate_removal_rates


def main() -> None:
    """Compare both solvers across several puzzle removal rates."""
    removal_rates = [0.50, 0.60, 0.65, 0.70]

    evaluation = evaluate_removal_rates(
        removal_rates=removal_rates,
        num_training_solutions=100,
        num_evaluation_puzzles=20,
        n_estimators=100,
        training_seed=42,
        evaluation_seed=123,
    )

    print("Solver Comparison Across Removal Rates")
    print("--------------------------------------")

    print()
    print("Solution Quality and Runtime")
    print(
        f"{'Removal':>8}"
        f"{'Hybrid valid':>14}"
        f"{'Classic valid':>15}"
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
            f"{hybrid.valid_solution_rate:>13.2%}"
            f"{classical.valid_solution_rate:>14.2%}"
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
    print("Runtime ratio = hybrid runtime / classical runtime.")
    print("Backtracks and ML decisions are averages per puzzle.")
    print("Removal rate is a proxy for difficulty, not a formal rating.")


if __name__ == "__main__":
    main()
