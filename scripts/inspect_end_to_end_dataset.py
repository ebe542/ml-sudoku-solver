import numpy as np

from sudoku_ml.dataset.end_to_end import (
    create_end_to_end_train_test_split,
)


def _print_partition(name: str, X: np.ndarray, y: np.ndarray) -> None:
    empty_cells = np.sum(X == 0, axis=(1, 2))

    print(name)
    print("-" * len(name))
    print(f"Samples:             {len(X)}")
    print(f"Input shape:         {X.shape}")
    print(f"Target shape:        {y.shape}")
    print(f"Minimum empty cells: {int(empty_cells.min())}")
    print(f"Maximum empty cells: {int(empty_cells.max())}")
    print(f"Average empty cells: {float(empty_cells.mean()):.2f}")
    print()


def main() -> None:
    """Inspect the full-grid train/test dataset."""
    split = create_end_to_end_train_test_split(
        num_solutions=100,
        test_size=0.2,
        removal_rates=(0.50, 0.60, 0.65),
        random_seed=42,
    )

    print("End-to-End Sudoku Dataset")
    print("--------------------------")
    print()
    _print_partition("Training", split.train.X, split.train.y)
    _print_partition("Test", split.test.X, split.test.y)
    print(
        "Shared solution IDs:  "
        f"{len(set(split.train.solution_ids) & set(split.test.solution_ids))}"
    )
    print(
        "Inputs share target memory: "
        f"{np.shares_memory(split.train.X, split.train.y)}"
    )


if __name__ == "__main__":
    main()
