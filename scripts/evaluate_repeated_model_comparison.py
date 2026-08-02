from sudoku_ml.analysis.repeated_model_comparison import (
    evaluate_repeated_model_comparison,
)


def main() -> None:
    """Compare classifiers across rates and seeds."""
    evaluation = evaluate_repeated_model_comparison(
        removal_rates=[0.50, 0.60, 0.65],
        random_seeds=[42, 123, 202],
        num_solutions=100,
        test_size=0.2,
        n_estimators=100,
    )

    print("Repeated Sudoku Classifier Comparison")
    print("-------------------------------------")
    print(f"Runs per removal rate: {evaluation.run_count}")
    print()

    print("Ranking Performance")
    print(
        f"{'Removal':>8}"
        f" {'Model':<28}"
        f"{'Raw Top-1':>11}"
        f"{'Valid Top-1':>13}"
        f"{'Top-1 SD':>11}"
        f"{'Valid Top-3':>13}"
        f"{'MRR':>10}"
    )
    print("-" * 96)

    for rate_result in evaluation.results:
        for item in rate_result.models:
            print(
                f"{rate_result.removal_rate:>7.0%}"
                f" {item.name:<29}"
                f"{item.raw_top_1.mean:>11.2%}"
                f"{item.constrained_top_1.mean:>13.2%}"
                f"{item.constrained_top_1.standard_deviation:>11.2%}"
                f"{item.constrained_top_3.mean:>13.2%}"
                f"{item.constrained_mrr.mean:>10.4f}"
            )

    print()
    print("Probability Quality")
    print(
        f"{'Removal':>8}"
        f" {'Model':<28}"
        f"{'Raw ECE':>10}"
        f"{'Valid ECE':>12}"
        f"{'Raw loss':>12}"
        f"{'Valid loss':>12}"
    )
    print("-" * 84)

    for rate_result in evaluation.results:
        for item in rate_result.models:
            print(
                f"{rate_result.removal_rate:>7.0%}"
                f" {item.name:<29}"
                f"{item.raw_ece.mean:>10.4f}"
                f"{item.constrained_ece.mean:>12.4f}"
                f"{item.raw_log_loss.mean:>12.4f}"
                f"{item.constrained_log_loss.mean:>12.4f}"
            )

    print()
    print("Training and Inference Runtime")
    print(
        f"{'Removal':>8}"
        f" {'Model':<28}"
        f"{'Train ms':>12}"
        f"{'Train SD':>11}"
        f"{'Infer ms':>12}"
        f"{'Infer SD':>11}"
    )
    print("-" * 84)

    for rate_result in evaluation.results:
        for item in rate_result.models:
            print(
                f"{rate_result.removal_rate:>7.0%}"
                f" {item.name:<29}"
                f"{item.training_ms.mean:>12.2f}"
                f"{item.training_ms.standard_deviation:>11.2f}"
                f"{item.inference_ms.mean:>12.2f}"
                f"{item.inference_ms.standard_deviation:>11.2f}"
            )

    print()
    print(
        "Valid metrics use candidate-constrained "
        "probabilities."
    )
    print(
        "Means and population standard deviations are "
        "calculated across seeds."
    )
    print(
        "Inference timing measures one probability prediction "
        "over the complete test set."
    )
    print(
        "Feature generation and metric calculation are excluded "
        "from inference timing."
    )


if __name__ == "__main__":
    main()
