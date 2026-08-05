from sudoku_ml.dataset.end_to_end import (
    create_end_to_end_train_test_split,
)
from sudoku_ml.evaluation.end_to_end_evaluation import (
    evaluate_end_to_end_model,
)
from sudoku_ml.model.end_to_end_mlp import SudokuEndToEndMLP


def main() -> None:
    """Train and evaluate the direct full-grid MLP baseline."""
    split = create_end_to_end_train_test_split(
        num_solutions=500,
        test_size=0.2,
        removal_rates=(0.50, 0.60, 0.65),
        random_seed=42,
    )
    model = SudokuEndToEndMLP(
        hidden_layer_sizes=(128, 64),
        max_iter=100,
        random_seed=42,
    )
    model.fit(split.train)
    result = evaluate_end_to_end_model(model, split.test)

    print("End-to-End Sudoku MLP Baseline")
    print("------------------------------")
    print(f"Training samples:             {len(split.train.X)}")
    print(f"Test samples:                 {len(split.test.X)}")
    print(f"Empty-cell accuracy:          {result.empty_cell_accuracy:.2%}")
    print(
        "Empty-cell Top-3 accuracy:    "
        f"{result.empty_cell_top_3_accuracy:.2%}"
    )
    print(f"Exact solution rate:          {result.exact_solution_rate:.2%}")
    print(f"Valid solution rate:          {result.valid_solution_rate:.2%}")
    print(f"Clue preservation rate:       {result.clue_preservation_rate:.2%}")
    print(
        "Average incorrect empty cells: "
        f"{result.average_incorrect_empty_cells:.2f}"
    )
    print(
        "Average rule violations:       "
        f"{result.average_rule_violations:.2f}"
    )
    print()
    print("The model uses no candidate filtering or search.")


if __name__ == "__main__":
    main()
