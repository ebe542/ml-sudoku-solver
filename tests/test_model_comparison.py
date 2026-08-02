import numpy as np
import pytest

from sudoku_ml.analysis.model_comparison import (
    COMPARISON_MODEL_NAMES,
    create_comparison_models,
    evaluate_models,
)


class FakeComparisonModel:
    """Provide controlled probabilities for comparison tests."""

    def __init__(self) -> None:
        self.name = "Fake Model"
        self.classes = np.arange(1, 10)
        self.was_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.was_fitted = True

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        probabilities = np.array(
            [
                0.40,
                0.20,
                0.10,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
            ]
        )

        return np.tile(
            probabilities,
            (len(X), 1),
        )


def test_comparison_contains_expected_models() -> None:
    assert COMPARISON_MODEL_NAMES == (
        "Logistic Regression",
        "Random Forest",
        "Extra Trees",
        "Histogram Gradient Boosting",
    )


def test_comparison_models_return_probabilities() -> None:
    rng = np.random.default_rng(42)

    X_train = rng.random((27, 118))
    y_train = np.tile(
        np.arange(1, 10),
        3,
    )
    X_test = rng.random((3, 118))

    models = create_comparison_models(
        n_estimators=5,
        random_seed=42,
    )

    assert len(models) == 4

    for model in models:
        model.fit(
            X_train,
            y_train,
        )

        probabilities = model.predict_probabilities(
            X_test
        )

        assert probabilities.shape == (3, 9)
        assert np.all(probabilities >= 0.0)
        assert np.all(probabilities <= 1.0)
        assert np.allclose(
            np.sum(probabilities, axis=1),
            1.0,
        )
        assert np.array_equal(
            model.classes,
            np.arange(1, 10),
        )


def test_comparison_models_reject_invalid_estimator_count() -> None:
    with pytest.raises(
        ValueError,
        match="must be positive"):
        create_comparison_models(
            n_estimators=0,
            random_seed=42,
        )

def test_evaluate_models_reports_both_probability_modes() -> None:
    model = FakeComparisonModel()

    X_train = np.zeros((2, 118))
    y_train = np.array([1, 2])

    X_test = np.zeros((2, 118))
    X_test[:, 82:91] = 1.0
    y_test = np.array([1, 2])

    results = evaluate_models(
        models=(model,),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    assert model.was_fitted
    assert len(results) == 1
    assert results[0].name == "Fake Model"
    assert results[0].raw.sample_count == 2
    assert (
        results[0]
        .candidate_constrained
        .sample_count
        == 2
    )
    assert results[0].raw.top_1_accuracy == pytest.approx(
        0.5
    )
