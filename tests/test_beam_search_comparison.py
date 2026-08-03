import numpy as np
import pytest

from sudoku_ml.dataset.unique_generator import create_unique_dataset
from sudoku_ml.evaluation.beam_search_comparison import (
    compare_search_strategies,
    evaluate_beam_search_removal_rates,
)


class FixedProbabilityModel:
    """Return stable probabilities for comparison tests."""

    classes = np.arange(1, 10)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        probabilities = np.full(9, 1.0 / 9.0)
        return np.tile(probabilities, (len(X), 1))


def test_comparison_returns_all_strategies() -> None:
    dataset = create_unique_dataset(
        num_samples=2,
        removal_rate=0.02,
        random_seed=123,
    )

    results = compare_search_strategies(
        models={
            "Model A": FixedProbabilityModel(),
            "Model B": FixedProbabilityModel(),
        },
        puzzles=dataset.puzzles,
        expected_solutions=dataset.solutions,
        beam_widths=(2, 3),
    )

    assert len(results) == 9
    assert results[0].model_name == "No model"
    assert results[0].strategy_name == "Classical"
    assert [
        result.strategy_name
        for result in results[1:5]
    ] == ["Greedy", "Beam 2", "Beam 3", "Hybrid"]

    for result in results:
        assert result.evaluation.total_puzzles == 2
        assert result.evaluation.matching_solution_rate == 1.0


def test_comparison_collects_beam_statistics() -> None:
    dataset = create_unique_dataset(
        num_samples=1,
        removal_rate=0.02,
        random_seed=123,
    )

    results = compare_search_strategies(
        models={"Model": FixedProbabilityModel()},
        puzzles=dataset.puzzles,
        expected_solutions=dataset.solutions,
        beam_widths=(2,),
    )
    beam_result = next(
        result
        for result in results
        if result.strategy_name == "Beam 2"
    )

    assert beam_result.beam_width == 2
    assert beam_result.evaluation.generated_states == 1
    assert beam_result.evaluation.pruned_states == 0
    assert beam_result.evaluation.maximum_active_states == 1


def test_removal_rate_evaluation_returns_both_models() -> None:
    result = evaluate_beam_search_removal_rates(
        removal_rates=(0.02,),
        beam_widths=(2,),
        num_training_solutions=10,
        num_evaluation_puzzles=1,
        model_iterations=5,
        training_seed=42,
        evaluation_seed=123,
    )

    assert len(result.results) == 1
    assert len(result.results[0].strategies) == 7
    assert {
        item.model_name
        for item in result.results[0].strategies
    } == {
        "No model",
        "Random Forest",
        "Histogram Gradient Boosting",
    }


@pytest.mark.parametrize(
    ("models", "puzzles", "solutions", "message"),
    [
        ({}, np.zeros((1, 9, 9)), np.zeros((1, 9, 9)), "model"),
        (
            {"Model": FixedProbabilityModel()},
            np.empty((0, 9, 9)),
            np.empty((0, 9, 9)),
            "puzzle",
        ),
        (
            {"Model": FixedProbabilityModel()},
            np.zeros((2, 9, 9)),
            np.zeros((1, 9, 9)),
            "number of puzzles",
        ),
    ],
)
def test_comparison_rejects_invalid_collections(
    models,
    puzzles: np.ndarray,
    solutions: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compare_search_strategies(
            models=models,
            puzzles=puzzles,
            expected_solutions=solutions,
        )


@pytest.mark.parametrize("beam_widths", [(), (0,), (2.5,)])
def test_comparison_rejects_invalid_beam_widths(
    beam_widths,
) -> None:
    dataset = create_unique_dataset(
        num_samples=1,
        removal_rate=0.02,
        random_seed=123,
    )

    with pytest.raises(ValueError, match="[Bb]eam width"):
        compare_search_strategies(
            models={"Model": FixedProbabilityModel()},
            puzzles=dataset.puzzles,
            expected_solutions=dataset.solutions,
            beam_widths=beam_widths,
        )
