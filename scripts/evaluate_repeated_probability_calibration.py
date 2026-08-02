from sudoku_ml.analysis.repeated_probability_calibration import (
    evaluate_repeated_probability_calibration,
)


def print_ranking_table(evaluation, constrained: bool) -> None:
    """Print summarized ranking metrics."""
    title = (
        "Candidate-Constrained Ranking"
        if constrained
        else "Raw Probability Ranking"
    )

    print(title)
    print(
        f"{'Removal':>8}"
        f"{'Method':>12}"
        f"{'Top-1':>10}"
        f"{'Top-1 SD':>11}"
        f"{'Top-2':>10}"
        f"{'Top-3':>10}"
        f"{'MRR':>10}"
    )
    print("-" * 71)

    for rate_result in evaluation.results:
        for item in rate_result.methods:
            if constrained:
                top_1 = item.constrained_top_1
                top_2 = item.constrained_top_2
                top_3 = item.constrained_top_3
                mrr = item.constrained_mrr
            else:
                top_1 = item.raw_top_1
                top_2 = item.raw_top_2
                top_3 = item.raw_top_3
                mrr = item.raw_mrr

            print(
                f"{rate_result.removal_rate:>7.0%}"
                f"{item.name:>12}"
                f"{top_1.mean:>10.2%}"
                f"{top_1.standard_deviation:>11.2%}"
                f"{top_2.mean:>10.2%}"
                f"{top_3.mean:>10.2%}"
                f"{mrr.mean:>10.4f}"
            )


def print_probability_table(evaluation, constrained: bool) -> None:
    """Print summarized probability-quality metrics."""
    title = (
        "Candidate-Constrained Probability Quality"
        if constrained
        else "Raw Probability Quality"
    )

    print(title)
    print(
        f"{'Removal':>8}"
        f"{'Method':>12}"
        f"{'Confidence':>13}"
        f"{'ECE':>10}"
        f"{'ECE SD':>10}"
        f"{'Log loss':>12}"
    )
    print("-" * 65)

    for rate_result in evaluation.results:
        for item in rate_result.methods:
            if constrained:
                confidence = item.constrained_confidence
                ece = item.constrained_ece
                loss = item.constrained_log_loss
            else:
                confidence = item.raw_confidence
                ece = item.raw_ece
                loss = item.raw_log_loss

            print(
                f"{rate_result.removal_rate:>7.0%}"
                f"{item.name:>12}"
                f"{confidence.mean:>13.2%}"
                f"{ece.mean:>10.4f}"
                f"{ece.standard_deviation:>10.4f}"
                f"{loss.mean:>12.4f}"
            )


def main() -> None:
    """Compare calibration methods across rates and seeds."""
    evaluation = (
        evaluate_repeated_probability_calibration(
            removal_rates=[0.50, 0.60, 0.65],
            random_seeds=[42, 123, 202],
            num_solutions=100,
            test_size=0.2,
            n_estimators=100,
        )
    )

    print("Repeated Sudoku Probability Calibration")
    print("---------------------------------------")
    print(f"Runs per removal rate: {evaluation.run_count}")
    print()

    print_ranking_table(
        evaluation,
        constrained=False,
    )
    print()

    print_probability_table(
        evaluation,
        constrained=False,
    )
    print()

    print_ranking_table(
        evaluation,
        constrained=True,
    )
    print()

    print_probability_table(
        evaluation,
        constrained=True,
    )
    print()

    print(
        "Every run uses separate training, calibration, "
        "and evaluation datasets."
    )
    print(
        "Means and population standard deviations are "
        "calculated across seeds."
    )
    print(
        "Lower ECE and log loss indicate better "
        "probability quality."
    )


if __name__ == "__main__":
    main()
