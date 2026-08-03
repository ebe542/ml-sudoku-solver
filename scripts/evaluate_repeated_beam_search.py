from sudoku_ml.evaluation.repeated_beam_search_comparison import (
    evaluate_repeated_beam_search,
)


def _summary(mean: float, standard_deviation: float) -> str:
    return f"{mean:.2f} +/- {standard_deviation:.2f}"


def main() -> None:
    """Repeat the search-strategy comparison across random seeds."""
    evaluation = evaluate_repeated_beam_search(
        removal_rates=(0.60, 0.65),
        random_seeds=(42, 123, 202),
        beam_widths=(2, 4),
        num_training_solutions=100,
        num_evaluation_puzzles=10,
        model_iterations=100,
    )

    print("Repeated Sudoku Beam Search Comparison")
    print("--------------------------------------")
    print(f"Runs per removal rate: {evaluation.run_count}")
    print(f"Random seeds:          {evaluation.random_seeds}")
    print()
    print("Values are mean +/- population standard deviation.")
    print()
    print("Solution Quality and Runtime")
    print(
        f"{'Removal':>8} {'Model':<28} {'Strategy':<10} "
        f"{'Exact %':>17} {'Runtime ms':>17}"
    )
    print("-" * 84)

    for rate_result in evaluation.results:
        for item in rate_result.strategies:
            print(
                f"{rate_result.removal_rate:>8.0%} "
                f"{item.model_name:<28} "
                f"{item.strategy_name:<10} "
                f"{_summary(item.exact_match_rate.mean * 100, item.exact_match_rate.standard_deviation * 100):>17} "
                f"{_summary(item.runtime_ms.mean, item.runtime_ms.standard_deviation):>17}"
            )

    print()
    print("Search Effort")
    print(
        f"{'Removal':>8} {'Model':<28} {'Strategy':<10} "
        f"{'ML decisions':>17} {'Backtracks':>17} "
        f"{'Generated':>17} {'Pruned':>17}"
    )
    print("-" * 121)

    for rate_result in evaluation.results:
        for item in rate_result.strategies:
            print(
                f"{rate_result.removal_rate:>8.0%} "
                f"{item.model_name:<28} "
                f"{item.strategy_name:<10} "
                f"{_summary(item.ml_decisions.mean, item.ml_decisions.standard_deviation):>17} "
                f"{_summary(item.backtracks.mean, item.backtracks.standard_deviation):>17} "
                f"{_summary(item.generated_states.mean, item.generated_states.standard_deviation):>17} "
                f"{_summary(item.pruned_states.mean, item.pruned_states.standard_deviation):>17}"
            )


if __name__ == "__main__":
    main()
