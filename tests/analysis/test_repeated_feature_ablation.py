import pytest

from sudoku_ml.analysis.repeated_feature_ablation import (
    evaluate_repeated_feature_ablation,
)


def test_repeated_feature_ablation_returns_all_results() -> None:
    evaluation = evaluate_repeated_feature_ablation(
        removal_rates=[0.3],
        random_seeds=[101, 202],
        num_solutions=10,
        test_size=0.2,
        n_estimators=5,
    )

    assert evaluation.run_count == 2
    assert evaluation.random_seeds == (101, 202)
    assert len(evaluation.results) == 1

    rate_result = evaluation.results[0]

    assert rate_result.removal_rate == pytest.approx(0.3)
    assert len(rate_result.configurations) == 3

    assert [
        item.feature_count
        for item in rate_result.configurations
    ] == [82, 91, 118]

    for item in rate_result.configurations:
        assert 0.0 <= item.top_1_accuracy.mean <= 1.0
        assert 0.0 <= item.top_2_accuracy.mean <= 1.0
        assert 0.0 <= item.top_3_accuracy.mean <= 1.0
        assert 0.0 <= item.mean_reciprocal_rank.mean <= 1.0
        assert item.log_loss.mean >= 0.0


def test_repeated_feature_ablation_rejects_empty_rates() -> None:
    with pytest.raises(
        ValueError,
        match="At least one removal rate",
    ):
        evaluate_repeated_feature_ablation(
            removal_rates=[],
            random_seeds=[42],
        )


def test_repeated_feature_ablation_rejects_empty_seeds() -> None:
    with pytest.raises(
        ValueError,
        match="At least one random seed",
    ):
        evaluate_repeated_feature_ablation(
            removal_rates=[0.5],
            random_seeds=[],
        )


@pytest.mark.parametrize(
    "removal_rate",
    [0.0, 1.0, -0.1, 1.1],
)
def test_repeated_feature_ablation_rejects_invalid_rate(removal_rate: float) -> None:
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        evaluate_repeated_feature_ablation(
            removal_rates=[removal_rate],
            random_seeds=[42],
        )


def test_repeated_feature_ablation_is_reproducible() -> None:
    arguments = {
        "removal_rates": [0.3],
        "random_seeds": [42],
        "num_solutions": 10,
        "test_size": 0.2,
        "n_estimators": 5,
    }

    first = evaluate_repeated_feature_ablation(**arguments)
    second = evaluate_repeated_feature_ablation(**arguments)

    first_result = first.results[0].configurations
    second_result = second.results[0].configurations

    for first_item, second_item in zip(
        first_result,
        second_result):
        assert (
            first_item.top_1_accuracy.mean
            == pytest.approx(
                second_item.top_1_accuracy.mean
            )
        )
        assert (
            first_item.mean_reciprocal_rank.mean
            == pytest.approx(
                second_item.mean_reciprocal_rank.mean
            )
        )
