import numpy as np

from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.model.random_forest import SudokuRandomForest


FEATURE_GROUPS = {
    "Grid values": slice(0, 81),
    "Cell position": slice(81, 82),
    "Candidate indicators": slice(82, 91),
    "Row interactions": slice(91, 100),
    "Column interactions": slice(100, 109),
    "Block interactions": slice(109, 118)
}


def get_feature_name(index: int) -> str:
    """Return a readable name for a feature index."""
    if 0 <= index < 81:
        row = index // 9
        column = index % 9
        return f"grid_r{row + 1}_c{column + 1}"

    if index == 81:
        return "target_cell_index"

    if 82 <= index < 91:
        digit = index - 81
        return f"candidate_{digit}"

    if 91 <= index < 100:
        digit = index - 90
        return f"row_candidate_frequency_{digit}"

    if 100 <= index < 109:
        digit = index - 99
        return f"column_candidate_frequency_{digit}"

    if 109 <= index < 118:
        digit = index - 108
        return f"block_candidate_frequency_{digit}"

    raise ValueError(f"Unknown feature index: {index}")


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

    importance = model.model.feature_importances_

    if len(importance) != 118:
        raise RuntimeError(
            f"Expected 118 feature importances, received {len(importance)}."
        )

    print("Feature importance by group:\n")

    for group_name, feature_slice in FEATURE_GROUPS.items():
        group_importance = float(np.sum(importance[feature_slice]))
        print(f"{group_name:<24} {group_importance:.4f}")

    sorted_indices = np.argsort(importance)[::-1]

    print("\nTop 20 individual features:\n")

    for rank, index in enumerate(sorted_indices[:20], start=1):
        feature_name = get_feature_name(int(index))

        print(
            f"{rank:>2}. "
            f"{feature_name:<34} "
            f"{importance[index]:.6f}"
        )


if __name__ == "__main__":
    main()
