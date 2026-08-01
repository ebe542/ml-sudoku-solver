from dataclasses import dataclass

from sudoku_ml.analysis.feature_ablation import (
    FEATURE_CONFIGURATIONS,
    evaluate_feature_ablation,
)
from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.evaluation.repeated_evaluation import MetricSummary


@dataclass(frozen=True)
class RepeatedFeatureConfigurationResult:
    """Store summarized metrics for one feature configuration."""

    name: str
    feature_count: int
    top_1_accuracy: MetricSummary
    top_2_accuracy: MetricSummary
    top_3_accuracy: MetricSummary
    mean_reciprocal_rank: MetricSummary
    mean_confidence: MetricSummary
    expected_calibration_error: MetricSummary
    log_loss: MetricSummary


@dataclass(frozen=True)
class RepeatedFeatureRemovalRateResult:
    """Store feature results for one removal rate."""

    removal_rate: float
    configurations: tuple[
        RepeatedFeatureConfigurationResult,
        ...,
    ]


@dataclass(frozen=True)
class RepeatedFeatureAblationResult:
    """Store repeated feature-ablation results."""

    random_seeds: tuple[int, ...]
    results: tuple[
        RepeatedFeatureRemovalRateResult,
        ...,
    ]

    @property
    def run_count(self) -> int:
        """Return the number of runs per removal rate."""
        return len(self.random_seeds)


def evaluate_repeated_feature_ablation(removal_rates: list[float], random_seeds: list[int],
                                       num_solutions: int = 100, test_size: float = 0.2,
                                       n_estimators: int = 100) -> RepeatedFeatureAblationResult:
    """Repeat feature ablation across removal rates and seeds."""
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
        RepeatedFeatureRemovalRateResult
    ] = []

    for removal_rate in removal_rates:
        metric_values = {
            feature_count: {
                "top_1_accuracy": [],
                "top_2_accuracy": [],
                "top_3_accuracy": [],
                "mean_reciprocal_rank": [],
                "mean_confidence": [],
                "expected_calibration_error": [],
                "log_loss": [],
            }
            for _, feature_count in FEATURE_CONFIGURATIONS
        }

        for random_seed in random_seeds:
            data = create_train_test_split(
                num_solutions=num_solutions,
                test_size=test_size,
                removal_rate=removal_rate,
                random_seed=random_seed,
            )

            run_results = evaluate_feature_ablation(
                data.X_train,
                data.y_train,
                data.X_test,
                data.y_test,
                n_estimators=n_estimators,
                random_seed=random_seed,
            )

            for result in run_results:
                ranking = result.ranking
                values = metric_values[
                    result.feature_count
                ]

                values["top_1_accuracy"].append(
                    ranking.top_1_accuracy
                )
                values["top_2_accuracy"].append(
                    ranking.top_2_accuracy
                )
                values["top_3_accuracy"].append(
                    ranking.top_3_accuracy
                )
                values["mean_reciprocal_rank"].append(
                    ranking.mean_reciprocal_rank
                )
                values["mean_confidence"].append(
                    ranking.mean_confidence
                )
                values[
                    "expected_calibration_error"
                ].append(
                    ranking.expected_calibration_error
                )
                values["log_loss"].append(
                    ranking.log_loss
                )

        configuration_results = []

        for name, feature_count in FEATURE_CONFIGURATIONS:
            values = metric_values[feature_count]

            configuration_results.append(
                RepeatedFeatureConfigurationResult(
                    name=name,
                    feature_count=feature_count,
                    top_1_accuracy=MetricSummary.from_values(
                        values["top_1_accuracy"]
                    ),
                    top_2_accuracy=MetricSummary.from_values(
                        values["top_2_accuracy"]
                    ),
                    top_3_accuracy=MetricSummary.from_values(
                        values["top_3_accuracy"]
                    ),
                    mean_reciprocal_rank=(
                        MetricSummary.from_values(
                            values[
                                "mean_reciprocal_rank"
                            ]
                        )
                    ),
                    mean_confidence=MetricSummary.from_values(
                        values["mean_confidence"]
                    ),
                    expected_calibration_error=(
                        MetricSummary.from_values(
                            values[
                                "expected_calibration_error"
                            ]
                        )
                    ),
                    log_loss=MetricSummary.from_values(
                        values["log_loss"]
                    ),
                )
            )

        removal_rate_results.append(
            RepeatedFeatureRemovalRateResult(
                removal_rate=removal_rate,
                configurations=tuple(
                    configuration_results
                ),
            )
        )

    return RepeatedFeatureAblationResult(
        random_seeds=tuple(random_seeds),
        results=tuple(removal_rate_results),
    )
