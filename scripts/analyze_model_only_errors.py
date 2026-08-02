from sudoku_ml.analysis.model_only_error_analysis import (
    compare_model_only_models,
)
from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.dataset.unique_generator import create_unique_dataset
from sudoku_ml.model.histogram_gradient_boosting import (
    SudokuHistogramGradientBoosting,
)
from sudoku_ml.model.random_forest import SudokuRandomForest


def _format_optional(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"

    return f"{value:.2f}{suffix}"


def main() -> None:
    """Compare first Greedy ML errors across models and removal rates."""
    removal_rates = [0.50, 0.60, 0.65]

    print("Model-only Sudoku Decision Error Analysis")
    print("-----------------------------------------")
    print()
    print(
        f"{'Removal':>8} "
        f"{'Model':<28} "
        f"{'Exact':>9} "
        f"{'Failure':>9} "
        f"{'Correct first':>13} "
        f"{'Error conf.':>11} "
        f"{'Correct rank':>12}"
    )
    print("-" * 98)

    for removal_rate in removal_rates:
        training_data = create_train_test_split(
            num_solutions=100,
            test_size=0.2,
            removal_rate=removal_rate,
            random_seed=42,
        )

        random_forest = SudokuRandomForest(
            n_estimators=100,
            random_seed=42,
        )
        histogram_gradient_boosting = (
            SudokuHistogramGradientBoosting(
                max_iter=100,
                random_seed=42,
            )
        )

        random_forest.fit(training_data)
        histogram_gradient_boosting.fit(training_data)

        dataset = create_unique_dataset(
            num_samples=20,
            removal_rate=removal_rate,
            random_seed=123,
        )

        results = compare_model_only_models(
            models={
                "Random Forest": random_forest,
                "Histogram Gradient Boosting": (
                    histogram_gradient_boosting
                ),
            },
            puzzles=dataset.puzzles,
            expected_solutions=dataset.solutions,
        )

        for result in results:
            confidence = result.average_first_error_confidence
            correct_steps_text = _format_optional(
                result.average_correct_decisions_before_error
            )
            confidence_text = (
                "n/a"
                if confidence is None
                else f"{confidence:.2%}"
            )

            print(
                f"{removal_rate:>8.0%} "
                f"{result.name:<28} "
                f"{result.exact_solution_rate:>8.2%} "
                f"{result.failure_rate:>8.2%} "
                f"{correct_steps_text:>13} "
                f"{confidence_text:>11} "
                f"{_format_optional(result.average_correct_digit_rank):>12}"
            )

    print()
    print("Correct first = average correct placements before the first error.")
    print("Error confidence is the raw probability of the wrong Top-1 digit.")
    print("Correct rank is the correct digit's average rank at first errors.")
    print("Greedy Model-only solving uses constraints but never backtracks.")


if __name__ == "__main__":
    main()
