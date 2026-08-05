import numpy as np

from sudoku_ml.model.protocol import (
    SudokuProbabilityModel,
)


class CompatibleProbabilityModel:
    """Implement the structural solver model interface."""

    def __init__(self) -> None:
        self.classes = np.arange(1, 10)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        return np.full(
            (len(X), 9),
            1.0 / 9.0,
        )


class IncompatibleModel:
    """Omit the probability prediction method."""

    def __init__(self) -> None:
        self.classes = np.arange(1, 10)


def test_probability_model_protocol_accepts_compatible_model() -> None:
    model = CompatibleProbabilityModel()

    assert isinstance(
        model,
        SudokuProbabilityModel,
    )


def test_probability_model_protocol_rejects_incomplete_model() -> None:
    model = IncompatibleModel()

    assert not isinstance(
        model,
        SudokuProbabilityModel,
    )
