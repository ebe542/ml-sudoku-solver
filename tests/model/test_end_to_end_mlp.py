import numpy as np
import pytest

from sudoku_ml.dataset.end_to_end import (
    create_end_to_end_train_test_split,
)
from sudoku_ml.evaluation.end_to_end_evaluation import (
    count_rule_violations,
    evaluate_end_to_end_model,
)
from sudoku_ml.model.end_to_end_mlp import SudokuEndToEndMLP
from sudoku_ml.sudoku_generator import generate_solved_grid


pytestmark = pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)


@pytest.fixture(scope="module")
def trained_model_and_split():
    split = create_end_to_end_train_test_split(
        num_solutions=12,
        test_size=0.25,
        removal_rates=(0.20,),
        random_seed=42,
    )
    model = SudokuEndToEndMLP(
        hidden_layer_sizes=(8,),
        max_iter=20,
        random_seed=42,
    )
    model.fit(split.train)
    return model, split


def test_model_returns_full_grid_probabilities(
    trained_model_and_split,
) -> None:
    model, split = trained_model_and_split
    probabilities = model.predict_probabilities(split.test.X)

    assert probabilities.shape == (len(split.test.X), 81, 9)
    assert np.allclose(probabilities.sum(axis=2), 1.0)
    assert np.array_equal(model.classes, np.arange(1, 10))


def test_model_predicts_complete_digit_grids(
    trained_model_and_split,
) -> None:
    model, split = trained_model_and_split
    predictions = model.predict(split.test.X)

    assert predictions.shape == split.test.y.shape
    assert np.all((predictions >= 1) & (predictions <= 9))


def test_model_preserves_given_clues(trained_model_and_split) -> None:
    model, split = trained_model_and_split
    predictions = model.predict(split.test.X)

    assert np.all(
        (split.test.X == 0)
        | (predictions == split.test.X)
    )


def test_evaluation_metrics_are_bounded(
    trained_model_and_split,
) -> None:
    model, split = trained_model_and_split
    result = evaluate_end_to_end_model(model, split.test)

    assert 0.0 <= result.empty_cell_accuracy <= 1.0
    assert 0.0 <= result.empty_cell_top_3_accuracy <= 1.0
    assert 0.0 <= result.exact_solution_rate <= 1.0
    assert 0.0 <= result.valid_solution_rate <= 1.0
    assert result.clue_preservation_rate == 1.0
    assert result.average_incorrect_empty_cells >= 0.0
    assert result.average_rule_violations >= 0.0


def test_rule_violations_are_zero_for_valid_solution() -> None:
    solution = generate_solved_grid(random_seed=42)

    assert count_rule_violations(solution.values) == 0


def test_rule_violations_count_invalid_units() -> None:
    solution = generate_solved_grid(random_seed=42)
    values = solution.values.copy()
    values[0, 0] = values[0, 1]

    assert count_rule_violations(values) == 3


def test_model_rejects_invalid_grid_shape() -> None:
    model = SudokuEndToEndMLP(
        hidden_layer_sizes=(8,),
        max_iter=5,
        random_seed=42,
    )

    with pytest.raises(ValueError, match="shape"):
        model.predict_probabilities(np.zeros((2, 81)))
