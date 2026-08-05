from sudoku_ml.dataset.end_to_end import (
    create_end_to_end_train_test_split,
)
from sudoku_ml.evaluation.end_to_end_learning_curve import (
    evaluate_end_to_end_learning_curve,
)


def main() -> None:
    """Evaluate MLP learning behavior across training-set sizes."""
    split = create_end_to_end_train_test_split(
        num_solutions=500,
        test_size=0.2,
        removal_rates=(0.50, 0.60, 0.65),
        random_seed=42,
    )
    result = evaluate_end_to_end_learning_curve(
        split=split,
        training_solution_counts=(50, 100, 200, 400),
        hidden_layer_sizes=(128, 64),
        max_iter=100,
        random_seed=42,
    )

    print("End-to-End Sudoku MLP Learning Curve")
    print("------------------------------------")
    print()
    print("Optimization and Empty-Cell Accuracy")
    print(
        f"{'Solutions':>10} {'Samples':>8} {'Seconds':>9} "
        f"{'Iter':>6} {'Conv.':>7} {'Initial loss':>13} "
        f"{'Final loss':>11} {'Train':>9} {'Test':>9} {'Gap':>9}"
    )
    print("-" * 107)

    for point in result.points:
        print(
            f"{point.training_solution_count:>10} "
            f"{point.training_sample_count:>8} "
            f"{point.training_seconds:>9.2f} "
            f"{point.iterations:>6} "
            f"{point.converged!s:>7} "
            f"{point.initial_loss:>13.4f} "
            f"{point.final_loss:>11.4f} "
            f"{point.training.empty_cell_accuracy:>8.2%} "
            f"{point.test.empty_cell_accuracy:>8.2%} "
            f"{point.empty_cell_accuracy_gap:>8.2%}"
        )

    print()
    print("Complete-Grid Quality")
    print(
        f"{'Solutions':>10} {'Train exact':>12} {'Test exact':>11} "
        f"{'Train valid':>12} {'Test valid':>11} "
        f"{'Train violations':>17} {'Test violations':>16}"
    )
    print("-" * 98)

    for point in result.points:
        print(
            f"{point.training_solution_count:>10} "
            f"{point.training.exact_solution_rate:>11.2%} "
            f"{point.test.exact_solution_rate:>10.2%} "
            f"{point.training.valid_solution_rate:>11.2%} "
            f"{point.test.valid_solution_rate:>10.2%} "
            f"{point.training.average_rule_violations:>17.2f} "
            f"{point.test.average_rule_violations:>16.2f}"
        )

    print()
    print("The test partition remains fixed for every learning-curve point.")
    print("Converged is false when scikit-learn emits ConvergenceWarning.")


if __name__ == "__main__":
    main()
