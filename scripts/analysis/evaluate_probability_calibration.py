from sudoku_ml.analysis.probability_calibration import (
    evaluate_probability_calibration,
)
from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.model.random_forest import SudokuRandomForest


def main() -> None:
    """Compare probability-calibration methods."""
    training_data = create_train_test_split(
        num_solutions=100,
        test_size=0.2,
        removal_rate=0.5,
        random_seed=42,
    )
    calibration_data = create_train_test_split(
        num_solutions=100,
        test_size=0.2,
        removal_rate=0.5,
        random_seed=123,
    )
    evaluation_data = create_train_test_split(
        num_solutions=100,
        test_size=0.2,
        removal_rate=0.5,
        random_seed=202,
    )

    model = SudokuRandomForest(
        n_estimators=100,
        random_seed=42,
    )
    model.fit(training_data)

    results = evaluate_probability_calibration(
        model,
        calibration_data.X_test,
        calibration_data.y_test,
        evaluation_data.X_test,
        evaluation_data.y_test,
    )

    print("Sudoku Probability Calibration")
    print("------------------------------")
    print(
        f"Training samples:    "
        f"{len(training_data.X_train)}"
    )
    print(
        f"Calibration samples: "
        f"{len(calibration_data.X_test)}"
    )
    print(
        f"Evaluation samples:  "
        f"{len(evaluation_data.X_test)}"
    )
    print()

    print("Raw Probability Ranking")
    print(
        f"{'Method':<12}"
        f"{'Top-1':>10}"
        f"{'Top-2':>10}"
        f"{'Top-3':>10}"
        f"{'MRR':>10}"
    )
    print("-" * 52)

    for result in results:
        ranking = result.raw

        print(
            f"{result.name:<12}"
            f"{ranking.top_1_accuracy:>10.2%}"
            f"{ranking.top_2_accuracy:>10.2%}"
            f"{ranking.top_3_accuracy:>10.2%}"
            f"{ranking.mean_reciprocal_rank:>10.4f}"
        )

    print()
    print("Raw Probability Quality")
    print(
        f"{'Method':<12}"
        f"{'Confidence':>12}"
        f"{'ECE':>12}"
        f"{'Log loss':>12}"
    )
    print("-" * 48)

    for result in results:
        ranking = result.raw

        print(
            f"{result.name:<12}"
            f"{ranking.mean_confidence:>12.2%}"
            f"{ranking.expected_calibration_error:>12.4f}"
            f"{ranking.log_loss:>12.4f}"
        )

    print()
    print("Candidate-Constrained Ranking")
    print(
        f"{'Method':<12}"
        f"{'Top-1':>10}"
        f"{'Top-2':>10}"
        f"{'Top-3':>10}"
        f"{'MRR':>10}"
    )
    print("-" * 52)

    for result in results:
        ranking = result.candidate_constrained

        print(
            f"{result.name:<12}"
            f"{ranking.top_1_accuracy:>10.2%}"
            f"{ranking.top_2_accuracy:>10.2%}"
            f"{ranking.top_3_accuracy:>10.2%}"
            f"{ranking.mean_reciprocal_rank:>10.4f}"
        )

    print()
    print("Candidate-Constrained Probability Quality")
    print(
        f"{'Method':<12}"
        f"{'Confidence':>12}"
        f"{'ECE':>12}"
        f"{'Log loss':>12}"
    )
    print("-" * 48)

    for result in results:
        ranking = result.candidate_constrained

        print(
            f"{result.name:<12}"
            f"{ranking.mean_confidence:>12.2%}"
            f"{ranking.expected_calibration_error:>12.4f}"
            f"{ranking.log_loss:>12.4f}"
        )

    print()
    print(
        "Training, calibration, and evaluation use "
        "independently generated Sudoku sets."
    )
    print(
        "Candidate-constrained results additionally exclude "
        "illegal Sudoku digits."
    )
    print(
        "Lower ECE and log loss indicate better "
        "probability quality."
    )


if __name__ == "__main__":
    main()
