import numpy as np

from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.model.random_forest import SudokuRandomForest


def test_model_can_be_trained() -> None:
    data = create_train_test_split(
        num_solutions=10,
        test_size=0.2,
        random_seed=42,
    )

    model = SudokuRandomForest(
        n_estimators=10, random_seed=42)

    model.fit(data)

    assert model.model is not None


def test_model_can_make_predictions() -> None:
    data = create_train_test_split(
        num_solutions=10,
        test_size=0.2,
        random_seed=42,
    )

    model = SudokuRandomForest(
        n_estimators=10, random_seed=42)

    model.fit(data)

    predictions = model.predict(data.X_test)

    assert len(predictions) == len(data.y_test)


def test_predictions_are_valid_sudoku_digits() -> None:
    data = create_train_test_split(
        num_solutions=10,
        test_size=0.2,
        random_seed=42,
    )

    model = SudokuRandomForest(
        n_estimators=10,
        random_seed=42,
    )

    model.fit(data)

    predictions = model.predict(data.X_test)

    assert np.all((predictions >= 1) & (predictions <= 9))


def test_model_can_be_evaluated() -> None:
    data = create_train_test_split(
        num_solutions=10,
        test_size=0.2,
        random_seed=42,
    )

    model = SudokuRandomForest(
        n_estimators=10,
        random_seed=42,
    )

    model.fit(data)

    accuracy = model.evaluate(data)

    assert 0.0 <= accuracy <= 1.0
