from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SudokuProbabilityModel(Protocol):
    """Define the probability interface required by solvers."""

    @property
    def classes(self) -> np.ndarray:
        """Return the digit classes known by the model."""
        ...

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for feature samples."""
        ...


class NamedSudokuProbabilityModel(SudokuProbabilityModel, Protocol):
    """Define a named probability model for comparisons."""

    name: str
