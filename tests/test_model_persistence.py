from pathlib import Path

import joblib
import numpy as np
import pytest

from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.model.random_forest import SudokuRandomForest


def test_saved_and_loaded_model_produces_same_predictions(tmp_path: Path) -> None:
    data = create_train_test_split(
        num_solutions=20,
        random_seed=42,
    )

    model = SudokuRandomForest(
        n_estimators=20,
        random_seed=42,
    )
    model.fit(data)

    predictions_before = model.predict(data.X_test)

    model_path = tmp_path / "models" / "sudoku_random_forest.joblib"
    model.save(model_path)

    loaded_model = SudokuRandomForest.load(model_path)
    predictions_after = loaded_model.predict(data.X_test)

    assert model_path.exists()
    assert np.array_equal(
        predictions_before,
        predictions_after,
    )
    assert np.array_equal(
        model.classes,
        loaded_model.classes,
    )


def test_load_rejects_unexpected_object(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid_model.joblib"
    joblib.dump({"not": "a random forest"}, invalid_path)

    with pytest.raises(TypeError, match="RandomForestClassifier"):
        SudokuRandomForest.load(invalid_path)
