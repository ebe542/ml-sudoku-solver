import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sudoku_ml.data.split import MLDataSplit
from sklearn.metrics import accuracy_score

class SudokuRandomForest:
    """Random Forest baseline model for Sudoku digit prediction."""

    def __init__(self, n_estimators: int = 100, random_seed: int | None = None) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_seed,
            n_jobs=-1,
        )

    def fit_arrays(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model using feature and target arrays."""
        self.model.fit(X, y)

    def fit(self, data: MLDataSplit) -> None:
        """Train the model on the training portion of an ML data split."""
        self.fit_arrays(data.X_train, data.y_train)

    def predict(self, X):
        """Predict Sudoku digits for the provided features."""
        return self.model.predict(X)

    def evaluate(self, data: MLDataSplit) -> float:
        """Evaluate the model on the test data."""
        predictions = self.predict(data.X_test)

        return float(
            accuracy_score(
                data.y_test,
                predictions,
            )
        )
