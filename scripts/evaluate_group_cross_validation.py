from sudoku_ml.analysis.group_cross_validation import (
    evaluate_group_cross_validation,
)


def main() -> None:
    result = evaluate_group_cross_validation(
        num_solutions=100,
        n_splits=5,
        removal_rate=0.5,
        n_estimators=100,
        random_seed=42,
    )

    print("Grouped cross-validation results:\n")

    for fold_number, accuracy in enumerate(result.fold_accuracies, start=1):
        print(f"Fold {fold_number}: {accuracy:.4f}")

    print(f"\nMean accuracy: {result.mean_accuracy:.4f}")
    print("Standard deviation: " 
        f"{result.standard_deviation:.4f}"
    )
    print(f"Minimum accuracy: {result.minimum_accuracy:.4f}")
    print(f"Maximum accuracy: {result.maximum_accuracy:.4f}")


if __name__ == "__main__":
    main()
