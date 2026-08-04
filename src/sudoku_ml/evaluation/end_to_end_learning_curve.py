import warnings
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from sudoku_ml.dataset.end_to_end import (
    EndToEndDataset,
    EndToEndDataSplit,
)
from sudoku_ml.evaluation.end_to_end_evaluation import (
    EndToEndEvaluationResult,
    evaluate_end_to_end_model,
)
from sudoku_ml.model.end_to_end_mlp import SudokuEndToEndMLP


@dataclass(frozen=True)
class EndToEndLearningCurvePoint:
    """Store training information and metrics for one dataset size."""

    training_solution_count: int
    training_sample_count: int
    training_seconds: float
    iterations: int
    converged: bool
    initial_loss: float
    final_loss: float
    training: EndToEndEvaluationResult
    test: EndToEndEvaluationResult

    @property
    def empty_cell_accuracy_gap(self) -> float:
        """Return training accuracy minus test accuracy."""
        return (
            self.training.empty_cell_accuracy
            - self.test.empty_cell_accuracy
        )


@dataclass(frozen=True)
class EndToEndLearningCurveResult:
    """Store MLP learning-curve points."""

    points: tuple[EndToEndLearningCurvePoint, ...]


def _select_solution_families(
    dataset: EndToEndDataset,
    solution_count: int,
) -> EndToEndDataset:
    selected_ids = np.unique(dataset.solution_ids)[:solution_count]
    sample_mask = np.isin(dataset.solution_ids, selected_ids)
    X = dataset.X[sample_mask].copy()

    return EndToEndDataset(
        X=X,
        y=dataset.y[sample_mask].copy(),
        empty_mask=X == 0,
        removal_rates=dataset.removal_rates[sample_mask].copy(),
        solution_ids=dataset.solution_ids[sample_mask].copy(),
    )


def evaluate_end_to_end_learning_curve(
    split: EndToEndDataSplit,
    training_solution_counts: tuple[int, ...],
    hidden_layer_sizes: tuple[int, ...] = (128, 64),
    max_iter: int = 100,
    random_seed: int | None = 42,
) -> EndToEndLearningCurveResult:
    """Evaluate MLP training and test behavior across dataset sizes."""
    if not training_solution_counts:
        raise ValueError(
            "At least one training solution count is required."
        )

    available_solution_count = len(
        np.unique(split.train.solution_ids)
    )

    if any(
        count <= 0 or count > available_solution_count
        for count in training_solution_counts
    ):
        raise ValueError(
            "Training solution counts must be positive and available."
        )

    if tuple(sorted(set(training_solution_counts))) != (
        training_solution_counts
    ):
        raise ValueError(
            "Training solution counts must be unique and ascending."
        )

    points: list[EndToEndLearningCurvePoint] = []

    for training_solution_count in training_solution_counts:
        training_dataset = _select_solution_families(
            split.train,
            training_solution_count,
        )
        model = SudokuEndToEndMLP(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_seed=random_seed,
        )

        start_time = perf_counter()
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(training_dataset)
        training_seconds = perf_counter() - start_time
        convergence_warning = any(
            issubclass(warning.category, ConvergenceWarning)
            for warning in caught_warnings
        )
        loss_curve = model.loss_curve

        points.append(
            EndToEndLearningCurvePoint(
                training_solution_count=training_solution_count,
                training_sample_count=len(training_dataset.X),
                training_seconds=training_seconds,
                iterations=model.iterations,
                converged=not convergence_warning,
                initial_loss=loss_curve[0],
                final_loss=loss_curve[-1],
                training=evaluate_end_to_end_model(
                    model,
                    training_dataset,
                ),
                test=evaluate_end_to_end_model(
                    model,
                    split.test,
                ),
            )
        )

    return EndToEndLearningCurveResult(points=tuple(points))
