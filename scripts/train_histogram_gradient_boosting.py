from pathlib import Path

from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.model.histogram_gradient_boosting import (
    SudokuHistogramGradientBoosting,
)


MODEL_PATH = Path(
    "models/sudoku_histogram_gradient_boosting.joblib"
)


def main() -> None:
    """Train and save the Sudoku Histogram Gradient Boosting model."""
    training_data = create_train_test_split(
        num_solutions=100,
        test_size=0.2,
        removal_rate=0.5,
        random_seed=42,
    )

    model = SudokuHistogramGradientBoosting(
        max_iter=100,
        random_seed=42,
    )
    model.fit(training_data)

    test_accuracy = model.evaluate(training_data)
    model.save(MODEL_PATH)

    print("Sudoku Histogram Gradient Boosting Training")
    print("-------------------------------------------")
    print(f"Training samples: {len(training_data.X_train)}")
    print(f"Test samples:     {len(training_data.X_test)}")
    print(f"Test accuracy:    {test_accuracy:.2%}")
    print(f"Model saved to:   {MODEL_PATH}")


if __name__ == "__main__":
    main()
