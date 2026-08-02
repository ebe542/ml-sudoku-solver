from dataclasses import dataclass

from sudoku_ml.analysis.probability_calibration import (
    evaluate_probability_calibration,
)
from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.evaluation.repeated_evaluation import MetricSummary
from sudoku_ml.model.random_forest import SudokuRandomForest


CALIBRATION_METHOD_NAMES = (
    "Raw",
    "Sigmoid",
    "Isotonic",
)


@dataclass(frozen=True)
class RepeatedCalibrationMethodResult:
    """Store summarized metrics for one calibration method."""

    name: str

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
class RepeatedCalibrationRemovalRateResult:
    """Store calibration results for one removal rate."""

    removal_rate: float
    methods: tuple[
        RepeatedCalibrationMethodResult,
        ...,
    ]


@dataclass(frozen=True)
class RepeatedProbabilityCalibrationResult:
    """Store repeated calibration results."""

    random_seeds: tuple[int, ...]
    results: tuple[
        RepeatedCalibrationRemovalRateResult,
        ...,
    ]

    @property
    def run_count(self) -> int:
        """Return the number of runs per removal rate."""
        return len(self.random_seeds)


def create_metric_storage() -> dict[str, dict[str, list[float]]]:
    """Create empty metric lists for every method."""
    metric_names = (
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

    return {
        method_name: {
            metric_name: []
            for metric_name in metric_names
        }
        for method_name in CALIBRATION_METHOD_NAMES
    }


def evaluate_repeated_probability_calibration(removal_rates: list[float], random_seeds: list[int],
                                              num_solutions: int = 100, test_size: float = 0.2,
                                              n_estimators: int = 100) -> RepeatedProbabilityCalibrationResult:
    """Repeat calibration analysis across rates and seeds."""
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
        RepeatedCalibrationRemovalRateResult
    ] = []

    for removal_rate in removal_rates:
        metric_storage = create_metric_storage()

        for random_seed in random_seeds:
            training_data = create_train_test_split(
                num_solutions=num_solutions,
                test_size=test_size,
                removal_rate=removal_rate,
                random_seed=random_seed,
            )
            calibration_data = create_train_test_split(
                num_solutions=num_solutions,
                test_size=test_size,
                removal_rate=removal_rate,
                random_seed=random_seed + 10_000,
            )
            evaluation_data = create_train_test_split(
                num_solutions=num_solutions,
                test_size=test_size,
                removal_rate=removal_rate,
                random_seed=random_seed + 20_000,
            )

            model = SudokuRandomForest(
                n_estimators=n_estimators,
                random_seed=random_seed,
            )
            model.fit(training_data)

            run_results = evaluate_probability_calibration(
                model,
                calibration_data.X_test,
                calibration_data.y_test,
                evaluation_data.X_test,
                evaluation_data.y_test,
            )

            for result in run_results:
                raw = result.raw
                constrained = result.candidate_constrained
                values = metric_storage[result.name]

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

        method_results = []

        for method_name in CALIBRATION_METHOD_NAMES:
            values = metric_storage[method_name]

            method_results.append(
                RepeatedCalibrationMethodResult(
                    name=method_name,
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
                    constrained_top_1=(
                        MetricSummary.from_values(
                            values["constrained_top_1"]
                        )
                    ),
                    constrained_top_2=(
                        MetricSummary.from_values(
                            values["constrained_top_2"]
                        )
                    ),
                    constrained_top_3=(
                        MetricSummary.from_values(
                            values["constrained_top_3"]
                        )
                    ),
                    constrained_mrr=(
                        MetricSummary.from_values(
                            values["constrained_mrr"]
                        )
                    ),
                    constrained_confidence=(
                        MetricSummary.from_values(
                            values[
                                "constrained_confidence"
                            ]
                        )
                    ),
                    constrained_ece=(
                        MetricSummary.from_values(
                            values["constrained_ece"]
                        )
                    ),
                    constrained_log_loss=(
                        MetricSummary.from_values(
                            values[
                                "constrained_log_loss"
                            ]
                        )
                    ),
                )
            )

        removal_rate_results.append(
            RepeatedCalibrationRemovalRateResult(
                removal_rate=removal_rate,
                methods=tuple(method_results),
            )
        )

    return RepeatedProbabilityCalibrationResult(
        random_seeds=tuple(random_seeds),
        results=tuple(removal_rate_results),
    )

                      