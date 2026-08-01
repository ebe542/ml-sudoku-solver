from sudoku_ml.analysis.feature_ablation import (
    evaluate_feature_ablation,
)
from sudoku_ml.data.split import create_train_test_split


def main() -> None:
    """Compare Random Forest models with different feature groups."""
    data = create_train_test_split(
        num_solutions=100,
        test_size=0.2,
        removal_rate=0.5,
        random_seed=42,
    )

    results = evaluate_feature_ablation(
        data.X_train,
        data.y_train,
        data.X_test,
        data.y_test,
        n_estimators=100,
        random_seed=42,
    )

    print("Sudoku Model Feature Ablation")
    print("-----------------------------")
    print(f"Training samples:   {len(data.X_train)}")
    print(f"Evaluation samples: {len(data.X_test)}")
    print()

    print("Ranking Performance")
    print(
        f"{'Feature configuration':<26}"
        f"{'Count':>8}"
        f"{'Top-1':>10}"
        f"{'Top-2':>10}"
        f"{'Top-3':>10}"
        f"{'MRR':>10}"
    )
    print("-" * 74)

    for result in results:
        ranking = result.ranking

        print(
            f"{result.name:<26}"
            f"{result.feature_count:>8}"
            f"{ranking.top_1_accuracy:>10.2%}"
            f"{ranking.top_2_accuracy:>10.2%}"
            f"{ranking.top_3_accuracy:>10.2%}"
            f"{ranking.mean_reciprocal_rank:>10.4f}"
        )

    print()
    print("Probability Quality")
    print(
        f"{'Feature configuration':<26}"
        f"{'Confidence':>12}"
        f"{'ECE':>12}"
        f"{'Log loss':>12}"
    )
    print("-" * 62)

    for result in results:
        ranking = result.ranking

        print(
            f"{result.name:<26}"
            f"{ranking.mean_confidence:>12.2%}"
            f"{ranking.expected_calibration_error:>12.4f}"
            f"{ranking.log_loss:>12.4f}"
        )

    print()
    print(
        "Each row represents a separately trained Random Forest."
    )
    print(
        "Later configurations include all preceding feature groups."
    )
    print(
        "Higher Top-k and MRR values are better."
    )
    print(
        "Lower ECE and log loss values are better."
    )


if __name__ == "__main__":
    main()
