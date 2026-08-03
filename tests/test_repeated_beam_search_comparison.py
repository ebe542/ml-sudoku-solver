import pytest

from sudoku_ml.evaluation.repeated_beam_search_comparison import (
    evaluate_repeated_beam_search,
)


def test_repeated_evaluation_returns_summaries() -> None:
    result = evaluate_repeated_beam_search(
        removal_rates=(0.02,),
        random_seeds=(42, 123),
        beam_widths=(2,),
        num_training_solutions=10,
        num_evaluation_puzzles=1,
        model_iterations=5,
    )

    assert result.run_count == 2
    assert result.random_seeds == (42, 123)
    assert len(result.results) == 1
    assert len(result.results[0].strategies) == 7

    for strategy in result.results[0].strategies:
        assert strategy.exact_match_rate.mean == pytest.approx(1.0)
        assert strategy.runtime_ms.minimum >= 0.0
        assert strategy.generated_states.minimum >= 0.0
        assert strategy.pruned_states.minimum >= 0.0


def test_repeated_evaluation_rejects_empty_seeds() -> None:
    with pytest.raises(
        ValueError,
        match="At least one random seed",
    ):
        evaluate_repeated_beam_search(
            removal_rates=(0.60,),
            random_seeds=(),
        )


def test_repeated_evaluation_rejects_empty_rates() -> None:
    with pytest.raises(ValueError, match="At least one removal rate"):
        evaluate_repeated_beam_search(
            removal_rates=(),
            random_seeds=(42,),
        )


@pytest.mark.parametrize("removal_rate", (0.0, 1.0, -0.1, 1.1))
def test_repeated_evaluation_rejects_invalid_rate(
    removal_rate: float,
) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_repeated_beam_search(
            removal_rates=(removal_rate,),
            random_seeds=(42,),
        )
