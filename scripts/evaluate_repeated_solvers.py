from sudoku_ml.evaluation.repeated_evaluation import (
    evaluate_repeated_unique_solvers,
)


def format_summary(mean: float, standard_deviation: float) -> str:
    """Format a mean and standard deviation for terminal output."""
    return f"{mean:.2f} +/- {standard_deviation:.2f}"


def main() -> None:
    """Evaluate both solvers repeatedly across evaluation seeds."""
    evaluation = evaluate_repeated_unique_solvers(
        removal_rates=[0.50, 0.60, 0.65],
        evaluation_seeds=[101, 123, 202],
        num_training_solutions=100,
        num_evaluation_puzzles=10,
        n_estimators=100,
        training_seed=42,
    )

    print("Repeated Unique-Solution Solver Evaluation")
    print("------------------------------------------")
    print(f"Runs per removal rate: {evaluation.run_count}")
    print(f"Evaluation seeds:      {evaluation.evaluation_seeds}")
    print()
    print("Values are mean +/- population standard deviation.")
    print()
    print("Solution Match and Runtime")
    print(
        f"{'Removal':>8}"
        f"{'Hybrid match %':>22}"
        f"{'Classic match %':>22}"
        f"{'Hybrid ms':>22}"
        f"{'Classic ms':>22}"
        f"{'Runtime ratio':>22}"
    )
    print("-" * 118)

    for item in evaluation.results:
        print(
            f"{item.removal_rate:>7.0%}"
            f"{format_summary(item.hybrid_match_rate.mean * 100, item.hybrid_match_rate.standard_deviation * 100):>22}"
            f"{format_summary(item.classical_match_rate.mean * 100, item.classical_match_rate.standard_deviation * 100):>22}"
            f"{format_summary(item.hybrid_runtime_ms.mean, item.hybrid_runtime_ms.standard_deviation):>22}"
            f"{format_summary(item.classical_runtime_ms.mean, item.classical_runtime_ms.standard_deviation):>22}"
            f"{format_summary(item.runtime_ratio.mean, item.runtime_ratio.standard_deviation):>22}"
        )

    print()
    print("Search Effort")
    print(
        f"{'Removal':>8}"
        f"{'Hybrid BT':>22}"
        f"{'Classic BT':>22}"
        f"{'BT reduction %':>22}"
        f"{'ML decisions':>22}"
    )
    print("-" * 96)

    for item in evaluation.results:
        print(
            f"{item.removal_rate:>7.0%}"
            f"{format_summary(item.hybrid_backtracks.mean, item.hybrid_backtracks.standard_deviation):>22}"
            f"{format_summary(item.classical_backtracks.mean, item.classical_backtracks.standard_deviation):>22}"
            f"{format_summary(item.backtrack_reduction.mean * 100, item.backtrack_reduction.standard_deviation * 100):>22}"
            f"{format_summary(item.hybrid_ml_decisions.mean, item.hybrid_ml_decisions.standard_deviation):>22}"
        )


if __name__ == "__main__":
    main()
