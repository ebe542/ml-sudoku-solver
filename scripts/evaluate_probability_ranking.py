from sudoku_ml.analysis.probability_ranking import (
    analyze_probability_ranking,
)
from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.model.random_forest import SudokuRandomForest


def main() -> None:
    """Compare raw and candidate-constrained model probabilities."""
    data = create_train_test_split(
        num_solutions=100,
        test_size=0.2,
        removal_rate=0.5,
        random_seed=42,
    )

    model = SudokuRandomForest(
        n_estimators=100,
        random_seed=42,
    )
    model.fit(data)

    raw = analyze_probability_ranking(
        model,
        data.X_test,
        data.y_test,
    )
    constrained = analyze_probability_ranking(
        model,
        data.X_test,
        data.y_test,
        candidate_constrained=True,
    )

    print("Sudoku Model Probability Ranking")
    print("--------------------------------")
    print(f"Evaluation samples: {raw.sample_count}")
    print()

    print(
        f"{'Metric':<28}"
        f"{'Raw':>14}"
        f"{'Constrained':>16}"
    )
    print("-" * 58)

    print(
        f"{'Top-1 accuracy':<28}"
        f"{raw.top_1_accuracy:>13.2%}"
        f"{constrained.top_1_accuracy:>15.2%}"
    )
    print(
        f"{'Top-2 accuracy':<28}"
        f"{raw.top_2_accuracy:>13.2%}"
        f"{constrained.top_2_accuracy:>15.2%}"
    )
    print(
        f"{'Top-3 accuracy':<28}"
        f"{raw.top_3_accuracy:>13.2%}"
        f"{constrained.top_3_accuracy:>15.2%}"
    )
    print(
        f"{'Mean reciprocal rank':<28}"
        f"{raw.mean_reciprocal_rank:>14.4f}"
        f"{constrained.mean_reciprocal_rank:>16.4f}"
    )
    print(
        f"{'Mean confidence':<28}"
        f"{raw.mean_confidence:>13.2%}"
        f"{constrained.mean_confidence:>15.2%}"
    )
    print(
        f"{'Expected calibration error':<28}"
        f"{raw.expected_calibration_error:>14.4f}"
        f"{constrained.expected_calibration_error:>16.4f}"
    )
    print(
        f"{'Log loss':<28}"
        f"{raw.log_loss:>14.4f}"
        f"{constrained.log_loss:>16.4f}"
    )

    print()
    print(
        "Constrained probabilities exclude digits that violate "
        "Sudoku rules."
    )
    print(
        "Higher values are better for Top-k accuracy and MRR."
    )
    print(
        "Lower values are better for calibration error and log loss."
    )


if __name__ == "__main__":
    main()
