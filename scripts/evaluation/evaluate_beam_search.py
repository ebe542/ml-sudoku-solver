from sudoku_ml.evaluation.beam_search_comparison import (
    evaluate_beam_search_removal_rates,
)


def main() -> None:
    """Compare bounded Beam Search with existing solver strategies."""
    comparison = evaluate_beam_search_removal_rates(
        removal_rates=(0.50, 0.60, 0.65),
        beam_widths=(2, 3, 4),
        num_training_solutions=100,
        num_evaluation_puzzles=20,
        test_size=0.2,
        model_iterations=100,
        training_seed=42,
        evaluation_seed=123,
    )

    print("Sudoku Search Strategy Comparison")
    print("---------------------------------")
    print()
    print("Solution Quality and Runtime")
    print(
        f"{'Removal':>8} "
        f"{'Model':<28} "
        f"{'Strategy':<10} "
        f"{'Exact':>8} "
        f"{'Valid':>8} "
        f"{'Runtime ms':>11}"
    )
    print("-" * 83)

    for removal_result in comparison.results:
        for item in removal_result.strategies:
            evaluation = item.evaluation
            print(
                f"{removal_result.removal_rate:>8.0%} "
                f"{item.model_name:<28} "
                f"{item.strategy_name:<10} "
                f"{evaluation.matching_solution_rate:>8.2%} "
                f"{evaluation.valid_solution_rate:>8.2%} "
                f"{evaluation.average_runtime_seconds * 1000:>11.2f}"
            )

    print()
    print("Search Effort")
    print(
        f"{'Removal':>8} "
        f"{'Model':<28} "
        f"{'Strategy':<10} "
        f"{'ML':>7} "
        f"{'Backtracks':>10} "
        f"{'Generated':>10} "
        f"{'Pruned':>8} "
        f"{'Max active':>10}"
    )
    print("-" * 104)

    for removal_result in comparison.results:
        for item in removal_result.strategies:
            evaluation = item.evaluation
            print(
                f"{removal_result.removal_rate:>8.0%} "
                f"{item.model_name:<28} "
                f"{item.strategy_name:<10} "
                f"{evaluation.average_ml_decisions:>7.2f} "
                f"{evaluation.average_backtracks:>10.2f} "
                f"{evaluation.average_generated_states:>10.2f} "
                f"{evaluation.average_pruned_states:>8.2f} "
                f"{evaluation.maximum_active_states:>10}"
            )

    print()
    print("All strategies use identical unique-solution puzzles per rate.")
    print("Generated and pruned states apply only to Beam Search.")
    print("Runtime and count metrics are averages per puzzle except max active.")


if __name__ == "__main__":
    main()
