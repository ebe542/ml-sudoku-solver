from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sudoku_ml.data.split import MLDataSplit


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

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for the provided features."""
        return self.model.predict_proba(X)

    @property
    def classes(self) -> np.ndarray:
        """Return the digit classes learned by the model."""
        return self.model.classes_

    def save(self, path: str | Path) -> None:
        """Save the trained Random Forest model to a file."""
        model_path = Path(path)
        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self.model,
            model_path
        )

    @classmethod
    def load(cls, path: str | Path) -> "SudokuRandomForest":
        """Load a Random Forest model from a file."""
        loaded_model = joblib.load(path)

        if not isinstance(loaded_model, RandomForestClassifier):
            raise TypeError(
                "The file does not contain a RandomForestClassifier."
            )

        instance = cls()
        instance.model = loaded_model

        return instance

    def evaluate(self, data: MLDataSplit) -> float:
        """Evaluate the model on the test data."""
        predictions = self.predict(data.X_test)

        return float(
            accuracy_score(
                data.y_test,
                predictions,
            )
        )
