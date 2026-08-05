from sudoku_ml.evaluation.greedy_evaluation import (
    evaluate_greedy_removal_rates,
)


def main() -> None:
    """Evaluate Greedy ML against Hybrid ML."""
    evaluation = evaluate_greedy_removal_rates(
        removal_rates=[0.50, 0.60, 0.65],
        num_training_solutions=100,
        num_evaluation_puzzles=20,
        n_estimators=100,
        training_seed=42,
        evaluation_seed=123,
    )

    print("Greedy ML vs Hybrid ML Sudoku Evaluation")
    print("----------------------------------------")
    print()

    print("Exact Solution Performance")
    print(
        f"{'Removal':>8}"
        f"{'Greedy match':>14}"
        f"{'Hybrid match':>14}"
        f"{'Greedy failure':>16}"
        f"{'Recovered':>12}"
    )
    print("-" * 64)

    for item in evaluation.results:
        comparison = item.comparison

        print(
            f"{item.removal_rate:>7.0%}"
            f"{comparison.greedy.matching_solution_rate:>13.2%}"
            f"{comparison.hybrid.matching_solution_rate:>13.2%}"
            f"{comparison.greedy_failure_rate:>15.2%}"
            f"{comparison.recovered_puzzles:>12}"
        )

    print()
    print("Runtime and Search Effort")
    print(
        f"{'Removal':>8}"
        f"{'Greedy ms':>12}"
        f"{'Hybrid ms':>12}"
        f"{'Greedy ML':>12}"
        f"{'Hybrid ML':>12}"
        f"{'Hybrid BT':>12}"
    )
    print("-" * 68)

    for item in evaluation.results:
        greedy = item.comparison.greedy
        hybrid = item.comparison.hybrid

        print(
            f"{item.removal_rate:>7.0%}"
            f"{greedy.average_runtime_seconds * 1000:>12.2f}"
            f"{hybrid.average_runtime_seconds * 1000:>12.2f}"
            f"{greedy.average_ml_decisions:>12.2f}"
            f"{hybrid.average_ml_decisions:>12.2f}"
            f"{hybrid.average_backtracks:>12.2f}"
        )

    print()
    print("Greedy ML never backtracks.")
    print(
        "Recovered counts puzzles solved by Hybrid "
        "but not by Greedy."
    )
    print(
        "All matches are checked against unique ground truth."
    )


if __name__ == "__main__":
    main()
