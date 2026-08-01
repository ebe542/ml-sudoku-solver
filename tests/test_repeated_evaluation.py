import pytest

from sudoku_ml.evaluation.repeated_evaluation import (
    MetricSummary,
    evaluate_repeated_unique_solvers,
)


def test_metric_summary_calculates_statistics() -> None:
    summary = MetricSummary.from_values([1.0, 2.0, 3.0])

    assert summary.mean == pytest.approx(2.0)
    assert summary.standard_deviation == pytest.approx(0.81649658)
    assert summary.minimum == pytest.approx(1.0)
    assert summary.maximum == pytest.approx(3.0)


def test_metric_summary_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="At least one metric value"):
        MetricSummary.from_values([])


def test_repeated_evaluation_returns_statistics_per_rate() -> None:
    result = evaluate_repeated_unique_solvers(
        removal_rates=[0.3],
        evaluation_seeds=[101, 202],
        num_training_solutions=10,
        num_evaluation_puzzles=2,
        n_estimators=5,
        training_seed=42,
    )

    assert result.run_count == 2
    assert result.evaluation_seeds == (101, 202)
    assert len(result.results) == 1

    item = result.results[0]

    assert item.removal_rate == pytest.approx(0.3)
    assert item.hybrid_match_rate.mean == pytest.approx(1.0)
    assert item.classical_match_rate.mean == pytest.approx(1.0)
    assert item.hybrid_runtime_ms.minimum >= 0.0
    assert item.classical_runtime_ms.minimum >= 0.0
    assert item.hybrid_backtracks.minimum >= 0.0
    assert item.classical_backtracks.minimum >= 0.0


def test_repeated_evaluation_rejects_empty_rates() -> None:
    with pytest.raises(ValueError, match="At least one removal rate"):
        evaluate_repeated_unique_solvers([], [101])


def test_repeated_evaluation_rejects_empty_seeds() -> None:
    with pytest.raises(ValueError, match="At least one evaluation seed"):
        evaluate_repeated_unique_solvers([0.5], [])


@pytest.mark.parametrize("removal_rate", [0.0, 1.0, -0.1, 1.1])
def test_repeated_evaluation_rejects_invalid_rate(removal_rate: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_repeated_unique_solvers([removal_rate], [101])


def test_repeated_evaluation_rejects_invalid_puzzle_count() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        evaluate_repeated_unique_solvers(
            [0.5],
            [101],
            num_evaluation_puzzles=0,
        )
