from pathlib import Path

from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.model.random_forest import SudokuRandomForest


MODEL_PATH = Path("models/sudoku_random_forest.joblib")


def main() -> None:
    """Train and save the Sudoku Random Forest model."""
    training_data = create_train_test_split(
        num_solutions=100,
        test_size=0.2,
        removal_rate=0.5,
        random_seed=42,
    )

    model = SudokuRandomForest(
        n_estimators=100,
        random_seed=42,
    )
    model.fit(training_data)

    test_accuracy = model.evaluate(training_data)
    model.save(MODEL_PATH)

    print("Sudoku Random Forest Training")
    print("-----------------------------")
    print(f"Training samples: {len(training_data.X_train)}")
    print(f"Test samples:     {len(training_data.X_test)}")
    print(f"Test accuracy:    {test_accuracy:.2%}")
    print(f"Model saved to:   {MODEL_PATH}")


if __name__ == "__main__":
    main()
