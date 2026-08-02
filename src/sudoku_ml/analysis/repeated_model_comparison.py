from dataclasses import dataclass

from sudoku_ml.analysis.model_comparison import (
    COMPARISON_MODEL_NAMES,
    create_comparison_models,
    evaluate_models_with_timing,
)
from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.evaluation.repeated_evaluation import MetricSummary


METRIC_NAMES = (
    "training_ms",
    "inference_ms",
    "raw_top_1",
    "raw_top_2",
    "raw_top_3",
    "raw_mrr",
    "raw_confidence",
    "raw_ece",
    "raw_log_loss",
    "constrained_top_1",
    "constrained_top_2",
    "constrained_top_3",
    "constrained_mrr",
    "constrained_confidence",
    "constrained_ece",
    "constrained_log_loss",
)


@dataclass(frozen=True)
class RepeatedModelResult:
    """Store summarized metrics for one classifier."""

    name: str
    training_ms: MetricSummary
    inference_ms: MetricSummary

    raw_top_1: MetricSummary
    raw_top_2: MetricSummary
    raw_top_3: MetricSummary
    raw_mrr: MetricSummary
    raw_confidence: MetricSummary
    raw_ece: MetricSummary
    raw_log_loss: MetricSummary

    constrained_top_1: MetricSummary
    constrained_top_2: MetricSummary
    constrained_top_3: MetricSummary
    constrained_mrr: MetricSummary
    constrained_confidence: MetricSummary
    constrained_ece: MetricSummary
    constrained_log_loss: MetricSummary


@dataclass(frozen=True)
class RepeatedModelRemovalRateResult:
    """Store model results for one removal rate."""

    removal_rate: float
    models: tuple[
        RepeatedModelResult,
        ...,
    ]


@dataclass(frozen=True)
class RepeatedModelComparisonResult:
    """Store repeated classifier comparisons."""

    random_seeds: tuple[int, ...]
    results: tuple[
        RepeatedModelRemovalRateResult,
        ...,
    ]

    @property
    def run_count(self) -> int:
        """Return the number of runs per removal rate."""
        return len(self.random_seeds)


def create_metric_storage() -> dict[str, dict[str, list[float]]]:
    """Create empty metric lists for every classifier."""
    return {
        model_name: {
            metric_name: []
            for metric_name in METRIC_NAMES
        }
        for model_name in COMPARISON_MODEL_NAMES
    }


def append_model_result(storage: dict[str, dict[str, list[float]]], result) -> None:
    """Append one timed model result to metric storage."""
    values = storage[result.name]
    raw = result.raw
    constrained = result.candidate_constrained

    values["training_ms"].append(
        result.training_seconds * 1_000
    )
    values["inference_ms"].append(
        result.inference_seconds * 1_000
    )

    values["raw_top_1"].append(
        raw.top_1_accuracy
    )
    values["raw_top_2"].append(
        raw.top_2_accuracy
    )
    values["raw_top_3"].append(
        raw.top_3_accuracy
    )
    values["raw_mrr"].append(
        raw.mean_reciprocal_rank
    )
    values["raw_confidence"].append(
        raw.mean_confidence
    )
    values["raw_ece"].append(
        raw.expected_calibration_error
    )
    values["raw_log_loss"].append(
        raw.log_loss
    )

    values["constrained_top_1"].append(
        constrained.top_1_accuracy
    )
    values["constrained_top_2"].append(
        constrained.top_2_accuracy
    )
    values["constrained_top_3"].append(
        constrained.top_3_accuracy
    )
    values["constrained_mrr"].append(
        constrained.mean_reciprocal_rank
    )
    values["constrained_confidence"].append(
        constrained.mean_confidence
    )
    values["constrained_ece"].append(
        constrained.expected_calibration_error
    )
    values["constrained_log_loss"].append(
        constrained.log_loss
    )


def summarize_model(name: str, values: dict[str, list[float]]) -> RepeatedModelResult:
    """Create metric summaries for one classifier."""
    return RepeatedModelResult(
        name=name,
        training_ms=MetricSummary.from_values(
            values["training_ms"]
        ),
        inference_ms=MetricSummary.from_values(
            values["inference_ms"]
        ),
        raw_top_1=MetricSummary.from_values(
            values["raw_top_1"]
        ),
        raw_top_2=MetricSummary.from_values(
            values["raw_top_2"]
        ),
        raw_top_3=MetricSummary.from_values(
            values["raw_top_3"]
        ),
        raw_mrr=MetricSummary.from_values(
            values["raw_mrr"]
        ),
        raw_confidence=MetricSummary.from_values(
            values["raw_confidence"]
        ),
        raw_ece=MetricSummary.from_values(
            values["raw_ece"]
        ),
        raw_log_loss=MetricSummary.from_values(
            values["raw_log_loss"]
        ),
        constrained_top_1=MetricSummary.from_values(
            values["constrained_top_1"]
        ),
        constrained_top_2=MetricSummary.from_values(
            values["constrained_top_2"]
        ),
        constrained_top_3=MetricSummary.from_values(
            values["constrained_top_3"]
        ),
        constrained_mrr=MetricSummary.from_values(
            values["constrained_mrr"]
        ),
        constrained_confidence=(
            MetricSummary.from_values(
                values["constrained_confidence"]
            )
        ),
        constrained_ece=MetricSummary.from_values(
            values["constrained_ece"]
        ),
        constrained_log_loss=(
            MetricSummary.from_values(
                values["constrained_log_loss"]
            )
        ),
    )


def evaluate_repeated_model_comparison(removal_rates: list[float], random_seeds: list[int],
                                       num_solutions: int = 100, test_size: float = 0.2,
                                       n_estimators: int = 100) -> RepeatedModelComparisonResult:
    """Compare classifiers across removal rates and seeds."""
    if not removal_rates:
        raise ValueError(
            "At least one removal rate is required."
        )

    if any(
        not 0.0 < rate < 1.0
        for rate in removal_rates
    ):
        raise ValueError(
            "Removal rates must be between 0 and 1."
        )

    if not random_seeds:
        raise ValueError(
            "At least one random seed is required."
        )

    removal_rate_results: list[
        RepeatedModelRemovalRateResult
    ] = []

    for removal_rate in removal_rates:
        metric_storage = create_metric_storage()

        for random_seed in random_seeds:
            data = create_train_test_split(
                num_solutions=num_solutions,
                test_size=test_size,
                removal_rate=removal_rate,
                random_seed=random_seed,
            )

            models = create_comparison_models(
                n_estimators=n_estimators,
                random_seed=random_seed,
            )

            run_results = evaluate_models_with_timing(
                models=models,
                X_train=data.X_train,
                y_train=data.y_train,
                X_test=data.X_test,
                y_test=data.y_test,
            )

            for result in run_results:
                append_model_result(
                    metric_storage,
                    result,
                )

        model_results = tuple(
            summarize_model(
                name,
                metric_storage[name],
            )
            for name in COMPARISON_MODEL_NAMES
        )

        removal_rate_results.append(
            RepeatedModelRemovalRateResult(
                removal_rate=removal_rate,
                models=model_results,
            )
        )

    return RepeatedModelComparisonResult(
        random_seeds=tuple(random_seeds),
        results=tuple(removal_rate_results),
    )
