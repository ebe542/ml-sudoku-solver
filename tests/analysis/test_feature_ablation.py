import numpy as np
import pytest

from sudoku_ml.analysis.feature_ablation import (
    FEATURE_CONFIGURATIONS,
    evaluate_feature_ablation,
    select_features,
)


def test_feature_configurations_cover_all_feature_groups() -> None:
    assert FEATURE_CONFIGURATIONS == (
        ("Grid + position", 82),
        ("Candidate indicators", 91),
        ("Candidate interactions", 118),
    )


@pytest.mark.parametrize(
    ("feature_count", "expected_shape"),
    [
        (82, (3, 82)),
        (91, (3, 91)),
        (118, (3, 118)),
    ],
)
def test_select_features_returns_requested_columns(
    feature_count: int,
    expected_shape: tuple[int, int],
) -> None:
    X = np.zeros((3, 118))

    result = select_features(X, feature_count)

    assert result.shape == expected_shape


def test_select_features_preserves_values() -> None:
    X = np.arange(236).reshape(2, 118)

    result = select_features(X, 82)

    assert np.array_equal(result, X[:, :82])


def test_select_features_rejects_too_many_columns() -> None:
    X = np.zeros((2, 91))

    with pytest.raises(
        ValueError,
        match="does not contain 118 features",
    ):
        select_features(X, 118)


def test_select_features_rejects_invalid_feature_count() -> None:
    X = np.zeros((2, 118))

    with pytest.raises(
        ValueError,
        match="Unsupported feature count",
    ):
        select_features(X, 100)

def test_feature_ablation_evaluates_all_configurations() -> None:
    rng = np.random.default_rng(42)

    X_train = rng.random((18, 118))
    y_train = np.tile(
        np.arange(1, 10),
        2,
    )
    X_test = rng.random((9, 118))
    y_test = np.arange(1, 10)

    results = evaluate_feature_ablation(
        X_train,
        y_train,
        X_test,
        y_test,
        n_estimators=5,
        random_seed=42,
    )

    assert len(results) == 3
    assert [
        result.name
        for result in results ] == [
        name
        for name, _ in FEATURE_CONFIGURATIONS
    ]
    assert [
        result.feature_count
        for result in results
    ] == [82, 91, 118]


def test_feature_ablation_reports_ranking_metrics() -> None:
    rng = np.random.default_rng(42)

    X_train = rng.random((18, 118))
    y_train = np.tile(
        np.arange(1, 10),
        2,
    )
    X_test = rng.random((9, 118))
    y_test = np.arange(1, 10)

    results = evaluate_feature_ablation(
        X_train,
        y_train,
        X_test,
        y_test,
        n_estimators=5,
        random_seed=42,
    )

    for result in results:
        assert result.ranking.sample_count == 9
        assert 0.0 <= result.ranking.top_1_accuracy <= 1.0
        assert 0.0 <= result.ranking.top_2_accuracy <= 1.0
        assert 0.0 <= result.ranking.top_3_accuracy <= 1.0
        assert 0.0 <= result.ranking.mean_reciprocal_rank <= 1.0
