import pytest

from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.dataset.unique_generator import create_unique_dataset
from sudoku_ml.evaluation.greedy_evaluation import (
    compare_greedy_and_hybrid,
    evaluate_greedy_removal_rates,
)
from sudoku_ml.model.random_forest import SudokuRandomForest


@pytest.fixture(scope="module")
def greedy_evaluation_model() -> SudokuRandomForest:
    data = create_train_test_split(
        num_solutions=50,
        removal_rate=0.65,
        random_seed=42,
    )

    model = SudokuRandomForest(
        n_estimators=50,
        random_seed=42,
    )
    model.fit(data)

    return model


def test_hybrid_recovers_puzzles_greedy_cannot_solve(greedy_evaluation_model: SudokuRandomForest) -> None:
    dataset = create_unique_dataset(
        num_samples=3,
        removal_rate=0.65,
        random_seed=123,
    )

    result = compare_greedy_and_hybrid(
        greedy_evaluation_model,
        dataset.puzzles,
        dataset.solutions,
    )

    assert result.hybrid.matching_solution_rate == pytest.approx(1.0)
    assert result.greedy.matching_solution_rate < 1.0

    assert result.recovered_puzzles > 0
    assert result.recovered_puzzles == (
        result.hybrid.matching_solutions
        - result.greedy.matching_solutions
    )

    assert result.greedy_failure_rate == pytest.approx(
        1.0 - result.greedy.matching_solution_rate
    )

    assert result.greedy.backtracks == 0
    assert result.hybrid.backtracks > 0


def test_greedy_comparison_rejects_mismatched_ground_truth(
    greedy_evaluation_model: SudokuRandomForest) -> None:
    dataset = create_unique_dataset(
        num_samples=2,
        removal_rate=0.3,
        random_seed=123,
    )

    with pytest.raises(
        ValueError,
        match="match the number of puzzles",
    ):
        compare_greedy_and_hybrid(
            greedy_evaluation_model,
            dataset.puzzles,
            dataset.solutions[:1],
        )

def test_evaluate_greedy_removal_rates_returns_each_rate() -> None:
    removal_rates = [0.02, 0.3]

    result = evaluate_greedy_removal_rates(
        removal_rates=removal_rates,
        num_training_solutions=10,
        num_evaluation_puzzles=2,
        n_estimators=5,
        training_seed=42,
        evaluation_seed=123,
    )

    assert len(result.results) == 2
    assert [
        item.removal_rate
        for item in result.results
    ] == removal_rates

    for item in result.results:
        assert item.comparison.greedy.total_puzzles == 2
        assert item.comparison.hybrid.total_puzzles == 2
        assert (
            item.comparison.hybrid.matching_solution_rate
            == pytest.approx(1.0)
        )
        assert (
            0.0
            <= item.comparison.greedy.matching_solution_rate
            <= 1.0
        )


def test_evaluate_greedy_removal_rates_rejects_empty_rates() -> None:
    with pytest.raises(
        ValueError,
        match="At least one removal rate"):
        evaluate_greedy_removal_rates(
            removal_rates=[]
        )


@pytest.mark.parametrize(
    "removal_rate",
    [0.0, 1.0, -0.1, 1.1],
)
def test_evaluate_greedy_removal_rates_rejects_invalid_rate(removal_rate: float) -> None:
    with pytest.raises(
        ValueError,
        match="between 0 and 1"):
        evaluate_greedy_removal_rates(
            removal_rates=[removal_rate]
        )


def test_evaluate_greedy_removal_rates_rejects_invalid_count() -> None:
    with pytest.raises(
        ValueError,
        match="must be positive"):
        evaluate_greedy_removal_rates(
            removal_rates=[0.5],
            num_evaluation_puzzles=0,
        )
