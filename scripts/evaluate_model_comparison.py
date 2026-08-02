from sudoku_ml.analysis.model_comparison import (
    create_comparison_models,
    evaluate_models,
)
from sudoku_ml.data.split import create_train_test_split


def print_ranking_table(results, constrained: bool) -> None:
    """Print model-ranking metrics."""
    title = (
        "Candidate-Constrained Ranking"
        if constrained
        else "Raw Probability Ranking"
    )

    print(title)
    print(
        f"{'Model':<30}"
        f"{'Top-1':>10}"
        f"{'Top-2':>10}"
        f"{'Top-3':>10}"
        f"{'MRR':>10}"
    )
    print("-" * 70)

    for result in results:
        ranking = (
            result.candidate_constrained
            if constrained
            else result.raw
        )

        print(
            f"{result.name:<30}"
            f"{ranking.top_1_accuracy:>10.2%}"
            f"{ranking.top_2_accuracy:>10.2%}"
            f"{ranking.top_3_accuracy:>10.2%}"
            f"{ranking.mean_reciprocal_rank:>10.4f}"
        )


def print_probability_table(results, constrained: bool) -> None:
    """Print model probability-quality metrics."""
    title = (
        "Candidate-Constrained Probability Quality"
        if constrained
        else "Raw Probability Quality"
    )

    print(title)
    print(
        f"{'Model':<30}"
        f"{'Confidence':>12}"
        f"{'ECE':>12}"
        f"{'Log loss':>12}"
    )
    print("-" * 66)

    for result in results:
        ranking = (
            result.candidate_constrained
            if constrained
            else result.raw
        )

        print(
            f"{result.name:<30}"
            f"{ranking.mean_confidence:>12.2%}"
            f"{ranking.expected_calibration_error:>12.4f}"
            f"{ranking.log_loss:>12.4f}"
        )


def main() -> None:
    """Compare classifiers on identical Sudoku features."""
    data = create_train_test_split(
        num_solutions=100,
        test_size=0.2,
        removal_rate=0.5,
        random_seed=42,
    )

    models = create_comparison_models(
        n_estimators=100,
        random_seed=42,
    )

    results = evaluate_models(
        models=models,
        X_train=data.X_train,
        y_train=data.y_train,
        X_test=data.X_test,
        y_test=data.y_test,
    )

    print("Sudoku Classifier Comparison")
    print("----------------------------")
    print(f"Training samples:   {len(data.X_train)}")
    print(f"Evaluation samples: {len(data.X_test)}")
    print("Feature count:      118")
    print()

    print_ranking_table(
        results,
        constrained=False,
    )
    print()

    print_probability_table(
        results,
        constrained=False,
    )
    print()

    print_ranking_table(
        results,
        constrained=True,
    )
    print()

    print_probability_table(
        results,
        constrained=True,
    )
    print()

    print(
        "All classifiers use the same training and "
        "evaluation samples."
    )
    print(
        "Logistic Regression includes feature standardization."
    )
    print(
        "Candidate-constrained results exclude illegal "
        "Sudoku digits."
    )


if __name__ == "__main__":
    main()
