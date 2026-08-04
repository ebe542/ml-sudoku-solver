from dataclasses import dataclass

import numpy as np

from sudoku_ml.dataset.end_to_end import EndToEndDataset
from sudoku_ml.grid import SudokuGrid
from sudoku_ml.model.end_to_end_mlp import SudokuEndToEndMLP


@dataclass(frozen=True)
class EndToEndEvaluationResult:
    """Store full-grid prediction metrics."""

    total_puzzles: int
    empty_cells: int
    correct_empty_cells: int
    top_3_correct_empty_cells: int
    exact_solutions: int
    valid_solutions: int
    clue_preserving_solutions: int
    total_incorrect_empty_cells: int
    total_rule_violations: int

    @property
    def empty_cell_accuracy(self) -> float:
        return self.correct_empty_cells / self.empty_cells

    @property
    def empty_cell_top_3_accuracy(self) -> float:
        return self.top_3_correct_empty_cells / self.empty_cells

    @property
    def exact_solution_rate(self) -> float:
        return self.exact_solutions / self.total_puzzles

    @property
    def valid_solution_rate(self) -> float:
        return self.valid_solutions / self.total_puzzles

    @property
    def clue_preservation_rate(self) -> float:
        return self.clue_preserving_solutions / self.total_puzzles

    @property
    def average_incorrect_empty_cells(self) -> float:
        return self.total_incorrect_empty_cells / self.total_puzzles

    @property
    def average_rule_violations(self) -> float:
        return self.total_rule_violations / self.total_puzzles


def count_rule_violations(grid: np.ndarray) -> int:
    """Count rows, columns, and blocks containing duplicate digits."""
    units = [*grid, *grid.T]
    units.extend(
        grid[row : row + 3, column : column + 3].reshape(-1)
        for row in range(0, 9, 3)
        for column in range(0, 9, 3)
    )

    return sum(
        len(set(int(value) for value in unit)) != 9
        for unit in units
    )


def evaluate_end_to_end_model(
    model: SudokuEndToEndMLP,
    dataset: EndToEndDataset,
) -> EndToEndEvaluationResult:
    """Evaluate direct full-grid predictions without search or constraints."""
    probabilities = model.predict_probabilities(dataset.X)
    predictions = model.predict(dataset.X, preserve_clues=True)
    empty_mask = dataset.empty_mask
    empty_targets = dataset.y[empty_mask]
    empty_predictions = predictions[empty_mask]
    class_by_index = model.classes
    top_3_indices = np.argsort(probabilities, axis=2)[:, :, -3:]
    top_3_digits = class_by_index[top_3_indices].reshape(-1, 3)
    flat_mask = empty_mask.reshape(-1)
    empty_top_3_digits = top_3_digits[flat_mask]
    top_3_matches = np.any(
        empty_top_3_digits == empty_targets[:, np.newaxis],
        axis=1,
    )

    exact_solutions = int(
        np.all(predictions == dataset.y, axis=(1, 2)).sum()
    )
    valid_solutions = sum(
        SudokuGrid(prediction).is_valid()
        and SudokuGrid(prediction).is_complete()
        for prediction in predictions
    )
    clue_preserving_solutions = int(
        np.all(
            (dataset.X == 0) | (predictions == dataset.X),
            axis=(1, 2),
        ).sum()
    )

    return EndToEndEvaluationResult(
        total_puzzles=len(dataset.X),
        empty_cells=int(empty_mask.sum()),
        correct_empty_cells=int(
            np.sum(empty_predictions == empty_targets)
        ),
        top_3_correct_empty_cells=int(top_3_matches.sum()),
        exact_solutions=exact_solutions,
        valid_solutions=valid_solutions,
        clue_preserving_solutions=clue_preserving_solutions,
        total_incorrect_empty_cells=int(
            np.sum(empty_predictions != empty_targets)
        ),
        total_rule_violations=sum(
            count_rule_violations(prediction)
            for prediction in predictions
        ),
    )
