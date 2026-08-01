from sudoku_ml.analysis.difficulty import summarize_puzzle_difficulties
from sudoku_ml.dataset.unique_generator import create_unique_dataset


def main() -> None:
    """Analyze heuristic difficulty across unique puzzle datasets."""
    removal_rates = [0.50, 0.60, 0.65]

    print("Unique Sudoku Difficulty Analysis")
    print("---------------------------------")
    print()
    print(
        f"{'Removal':>8}"
        f"{'Clues':>9}"
        f"{'Singles':>10}"
        f"{'Candidates':>12}"
        f"{'Branches':>11}"
        f"{'Backtracks':>12}"
        f"{'Score':>9}"
    )
    print("-" * 71)

    summaries = []

    for removal_rate in removal_rates:
        dataset = create_unique_dataset(
            num_samples=10,
            removal_rate=removal_rate,
            random_seed=123,
        )
        summary = summarize_puzzle_difficulties(dataset.puzzles)
        summaries.append((removal_rate, summary))

        print(
            f"{removal_rate:>7.0%}"
            f"{summary.average_given_cells:>9.2f}"
            f"{summary.average_initial_single_candidates:>10.2f}"
            f"{summary.average_initial_candidate_count:>12.2f}"
            f"{summary.average_branching_decisions:>11.2f}"
            f"{summary.average_backtracks:>12.2f}"
            f"{summary.average_difficulty_score:>9.2f}"
        )

    print()
    print("Heuristic Level Distribution")
    print(
        f"{'Removal':>8}"
        f"{'Easy':>8}"
        f"{'Medium':>9}"
        f"{'Hard':>8}"
        f"{'Expert':>9}"
    )
    print("-" * 42)

    for removal_rate, summary in summaries:
        print(
            f"{removal_rate:>7.0%}"
            f"{summary.easy_count:>8}"
            f"{summary.medium_count:>9}"
            f"{summary.hard_count:>8}"
            f"{summary.expert_count:>9}"
        )

    print()
    print("Difficulty score = branching decisions + backtracks.")
    print("Levels are project-specific heuristics, not official ratings.")


if __name__ == "__main__":
    main()
