from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.model.random_forest import SudokuRandomForest


def main() -> None:
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

    accuracy = model.evaluate(data)

    print(f"Training samples: {len(data.X_train)}")
    print(f"Test samples: {len(data.X_test)}")
    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
