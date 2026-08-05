import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.preprocessing.constraints import get_candidates
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

    predictions = model.predict(data.X_test)

    accuracy = accuracy_score(data.y_test, predictions)

    print(f"Test accuracy: {accuracy:.4f}")

    print("\nAccuracy per digit:")

    for digit in range(1, 10):
        mask = data.y_test == digit

        if np.any(mask):
            digit_accuracy = np.mean(predictions[mask] == digit)
            count = np.sum(mask)

            print(
                f"Digit {digit}: "
                f"{digit_accuracy:.4f} "
                f"({count} samples)"
            )

    matrix = confusion_matrix(data.y_test, predictions, labels=list(range(1, 10)))

    print("\nConfusion matrix:")
    print(matrix)

    print("\nAccuracy by number of candidates:")

    candidate_stats: dict[int, list[bool]] = {}

    for features, target, prediction in zip(data.X_test, data.y_test, predictions):
        grid = features[:81].reshape(9, 9).astype(int)
        cell_index = int(features[81])

        row = cell_index // 9
        column = cell_index % 9

        candidate_count = len(get_candidates(grid, row, column))

        candidate_stats.setdefault(candidate_count, []).append(
            prediction == target
        )

    for candidate_count in sorted(candidate_stats):
        results = candidate_stats[candidate_count]
        accuracy = np.mean(results)

        print(
            f"{candidate_count} candidates: "
            f"{accuracy:.4f} "
            f"({len(results)} samples)"
        )


if __name__ == "__main__":
    main()
