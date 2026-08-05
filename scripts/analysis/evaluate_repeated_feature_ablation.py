from sudoku_ml.analysis.repeated_feature_ablation import (
    evaluate_repeated_feature_ablation,
)


def main() -> None:
    """Evaluate feature groups across rates and random seeds."""
    evaluation = evaluate_repeated_feature_ablation(
        removal_rates=[0.50, 0.60, 0.65],
        random_seeds=[101, 123, 202],
        num_solutions=100,
        test_size=0.2,
        n_estimators=100,
    )

    print("Repeated Sudoku Feature Ablation")
    print("--------------------------------")
    print(f"Runs per removal rate: {evaluation.run_count}")
    print()

    print("Ranking Performance")
    print(
        f"{'Removal':>8}"
        f"{'Features':>10}"
        f"{'Top-1 mean':>13}"
        f"{'Top-1 SD':>11}"
        f"{'Top-2':>10}"
        f"{'Top-3':>10}"
        f"{'MRR':>10}"
    )
    print("-" * 72)

    for rate_result in evaluation.results:
        for item in rate_result.configurations:
            print(
                f"{rate_result.removal_rate:>7.0%}"
                f"{item.feature_count:>10}"
                f"{item.top_1_accuracy.mean:>13.2%}"
                f"{item.top_1_accuracy.standard_deviation:>11.2%}"
                f"{item.top_2_accuracy.mean:>10.2%}"
                f"{item.top_3_accuracy.mean:>10.2%}"
                f"{item.mean_reciprocal_rank.mean:>10.4f}"
            )

    print()
    print("Probability Quality")
    print(
        f"{'Removal':>8}"
        f"{'Features':>10}"
        f"{'Confidence':>13}"
        f"{'ECE mean':>12}"
        f"{'ECE SD':>10}"
        f"{'Log loss':>12}"
    )
    print("-" * 65)

    for rate_result in evaluation.results:
        for item in rate_result.configurations:
            print(
                f"{rate_result.removal_rate:>7.0%}"
                f"{item.feature_count:>10}"
                f"{item.mean_confidence.mean:>13.2%}"
                f"{item.expected_calibration_error.mean:>12.4f}"
                f"{item.expected_calibration_error.standard_deviation:>10.4f}"
                f"{item.log_loss.mean:>12.4f}"
            )

    print()
    print(
        "Each result is the mean across independently generated "
        "training and test splits."
    )
    print(
        "Top-1 SD and ECE SD are population standard deviations."
    )
    print(
        "Feature counts 82, 91, and 118 represent cumulative "
        "feature groups."
    )


if __name__ == "__main__":
    main()
