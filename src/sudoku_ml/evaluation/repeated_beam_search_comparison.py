from dataclasses import dataclass

from sudoku_ml.evaluation.beam_search_comparison import (
    evaluate_beam_search_removal_rates,
)
from sudoku_ml.evaluation.repeated_evaluation import MetricSummary


@dataclass(frozen=True)
class RepeatedSearchStrategyResult:
    """Store summarized metrics for one model and search strategy."""

    model_name: str
    strategy_name: str
    beam_width: int | None
    exact_match_rate: MetricSummary
    valid_solution_rate: MetricSummary
    runtime_ms: MetricSummary
    ml_decisions: MetricSummary
    backtracks: MetricSummary
    generated_states: MetricSummary
    pruned_states: MetricSummary


@dataclass(frozen=True)
class RepeatedBeamRemovalRateResult:
    """Store repeated strategy results for one removal rate."""

    removal_rate: float
    strategies: tuple[RepeatedSearchStrategyResult, ...]


@dataclass(frozen=True)
class RepeatedBeamSearchResult:
    """Store repeated Beam comparisons across removal rates."""

    random_seeds: tuple[int, ...]
    results: tuple[RepeatedBeamRemovalRateResult, ...]

    @property
    def run_count(self) -> int:
        """Return the number of runs per removal rate."""
        return len(self.random_seeds)


def _create_storage() -> dict[
    tuple[str, str, int | None],
    dict[str, list[float]],
]:
    return {}


def _append_run(storage, strategy) -> None:
    key = (
        strategy.model_name,
        strategy.strategy_name,
        strategy.beam_width,
    )
    values = storage.setdefault(
        key,
        {
            "exact": [],
            "valid": [],
            "runtime": [],
            "ml": [],
            "backtracks": [],
            "generated": [],
            "pruned": [],
        },
    )
    evaluation = strategy.evaluation
    values["exact"].append(float(evaluation.matching_solution_rate))
    values["valid"].append(evaluation.valid_solution_rate)
    values["runtime"].append(
        evaluation.average_runtime_seconds * 1_000
    )
    values["ml"].append(evaluation.average_ml_decisions)
    values["backtracks"].append(evaluation.average_backtracks)
    values["generated"].append(evaluation.average_generated_states)
    values["pruned"].append(evaluation.average_pruned_states)


def _summarize_strategy(key, values) -> RepeatedSearchStrategyResult:
    model_name, strategy_name, beam_width = key
    return RepeatedSearchStrategyResult(
        model_name=model_name,
        strategy_name=strategy_name,
        beam_width=beam_width,
        exact_match_rate=MetricSummary.from_values(values["exact"]),
        valid_solution_rate=MetricSummary.from_values(values["valid"]),
        runtime_ms=MetricSummary.from_values(values["runtime"]),
        ml_decisions=MetricSummary.from_values(values["ml"]),
        backtracks=MetricSummary.from_values(values["backtracks"]),
        generated_states=MetricSummary.from_values(values["generated"]),
        pruned_states=MetricSummary.from_values(values["pruned"]),
    )


def evaluate_repeated_beam_search(
    removal_rates: tuple[float, ...],
    random_seeds: tuple[int, ...],
    beam_widths: tuple[int, ...] = (2, 4),
    num_training_solutions: int = 100,
    num_evaluation_puzzles: int = 10,
    test_size: float = 0.2,
    model_iterations: int = 100,
) -> RepeatedBeamSearchResult:
    """Repeat the Beam comparison across independent random seeds."""
    if not removal_rates:
        raise ValueError("At least one removal rate is required.")

    if any(not 0.0 < rate < 1.0 for rate in removal_rates):
        raise ValueError("Removal rates must be between 0 and 1.")

    if not random_seeds:
        raise ValueError("At least one random seed is required.")

    results: list[RepeatedBeamRemovalRateResult] = []

    for removal_rate in removal_rates:
        storage = _create_storage()

        for random_seed in random_seeds:
            run = evaluate_beam_search_removal_rates(
                removal_rates=(removal_rate,),
                beam_widths=beam_widths,
                num_training_solutions=num_training_solutions,
                num_evaluation_puzzles=num_evaluation_puzzles,
                test_size=test_size,
                model_iterations=model_iterations,
                training_seed=random_seed,
                evaluation_seed=random_seed + 10_000,
            )

            for strategy in run.results[0].strategies:
                _append_run(storage, strategy)

        results.append(
            RepeatedBeamRemovalRateResult(
                removal_rate=removal_rate,
                strategies=tuple(
                    _summarize_strategy(key, values)
                    for key, values in storage.items()
                ),
            )
        )

    return RepeatedBeamSearchResult(
        random_seeds=random_seeds,
        results=tuple(results),
    )
