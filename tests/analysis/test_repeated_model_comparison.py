import pytest

from sudoku_ml.analysis.repeated_model_comparison import (
    evaluate_repeated_model_comparison,
)


def test_repeated_model_comparison_returns_all_models() -> None:
    evaluation = evaluate_repeated_model_comparison(
        removal_rates=[0.3],
        random_seeds=[42],
        num_solutions=20,
        test_size=0.2,
        n_estimators=5,
    )

    assert evaluation.run_count == 1
    assert evaluation.random_seeds == (42,)
    assert len(evaluation.results) == 1

    rate_result = evaluation.results[0]

    assert rate_result.removal_rate == pytest.approx(0.3)
    assert [
        item.name
        for item in rate_result.models
    ] == [
        "Logistic Regression",
        "Random Forest",
        "Extra Trees",
        "Histogram Gradient Boosting",
    ]

    for item in rate_result.models:
        assert item.training_ms.mean >= 0.0
        assert item.inference_ms.mean >= 0.0
        assert 0.0 <= item.raw_top_1.mean <= 1.0
        assert 0.0 <= item.raw_top_3.mean <= 1.0
        assert 0.0 <= item.raw_mrr.mean <= 1.0
        assert item.raw_log_loss.mean >= 0.0
        assert (
            0.0
            <= item.constrained_top_1.mean
            <= 1.0
        )
        assert item.constrained_log_loss.mean >= 0.0


def test_repeated_model_comparison_rejects_empty_rates() -> None:
    with pytest.raises(
        ValueError,
        match="At least one removal rate",
    ):
        evaluate_repeated_model_comparison(
            removal_rates=[],
            random_seeds=[42],
        )


def test_repeated_model_comparison_rejects_empty_seeds() -> None:
    with pytest.raises(
        ValueError,
        match="At least one random seed",
    ):
        evaluate_repeated_model_comparison(
            removal_rates=[0.5],
            random_seeds=[],
        )


@pytest.mark.parametrize(
    "removal_rate",
    [0.0, 1.0, -0.1, 1.1],
)
def test_repeated_model_comparison_rejects_invalid_rate(removal_rate: float) -> None:
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        evaluate_repeated_model_comparison(
            removal_rates=[removal_rate],
            random_seeds=[42],
        )
