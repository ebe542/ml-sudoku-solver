import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sudoku_ml.dataset.end_to_end import EndToEndDataset


class SudokuEndToEndMLP:
    """Predict every Sudoku cell from the incomplete full grid."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
        max_iter: int = 100,
        random_seed: int | None = None,
    ) -> None:
        self.model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                random_state=random_seed,
            ),
        )

    @staticmethod
    def _create_cell_features(
        grids: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Combine full grids with one-hot target-cell positions."""
        if grids.ndim != 3 or grids.shape[1:] != (9, 9):
            raise ValueError("Input grids must have shape (samples, 9, 9).")

        sample_count = len(grids)
        flattened_grids = grids.reshape(sample_count, 81)
        repeated_grids = np.repeat(flattened_grids, 81, axis=0)
        positions = np.tile(
            np.eye(81, dtype=np.float32),
            (sample_count, 1),
        )
        features = np.concatenate(
            [repeated_grids, positions],
            axis=1,
            dtype=np.float32,
        )

        if mask is None:
            return features

        if mask.shape != grids.shape:
            raise ValueError("Mask shape must match the input grids.")

        return features[mask.reshape(-1)]

    def fit(self, dataset: EndToEndDataset) -> None:
        """Train only on cells hidden in the input puzzles."""
        features = self._create_cell_features(
            dataset.X,
            dataset.empty_mask,
        )
        targets = dataset.y.reshape(-1)[dataset.empty_mask.reshape(-1)]
        self.model.fit(features, targets)

    @property
    def classes(self) -> np.ndarray:
        """Return the digit classes learned by the MLP."""
        return self.model.classes_

    @property
    def iterations(self) -> int:
        """Return the number of completed optimizer iterations."""
        return int(
            self.model.named_steps["mlpclassifier"].n_iter_
        )

    @property
    def loss_curve(self) -> tuple[float, ...]:
        """Return the training loss recorded after each iteration."""
        return tuple(
            float(value)
            for value in self.model.named_steps[
                "mlpclassifier"
            ].loss_curve_
        )

    def predict_probabilities(self, grids: np.ndarray) -> np.ndarray:
        """Return digit probabilities with shape (samples, 81, classes)."""
        features = self._create_cell_features(grids)
        probabilities = self.model.predict_proba(features)

        return probabilities.reshape(
            len(grids),
            81,
            len(self.classes),
        )

    def predict(
        self,
        grids: np.ndarray,
        preserve_clues: bool = True,
    ) -> np.ndarray:
        """Predict complete grids and optionally retain given digits."""
        probabilities = self.predict_probabilities(grids)
        predictions = self.classes[
            np.argmax(probabilities, axis=2)
        ].reshape(-1, 9, 9)

        if preserve_clues:
            predictions = np.where(
                grids == 0,
                predictions,
                grids,
            )

        return predictions.astype(np.int8)
