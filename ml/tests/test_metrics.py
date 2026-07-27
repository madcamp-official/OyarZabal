import numpy as np
import pytest
from oyarzabal.metrics import (
    bootstrap_log_loss_gain,
    evaluate_diagnostics,
    evaluate_probabilities,
    validate_probability_matrix,
)


def test_metrics_include_top3_and_probability_quality() -> None:
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.3, 0.5],
            [0.3, 0.4, 0.3],
        ]
    )
    metrics = evaluate_probabilities([0, 2, 0], probabilities, labels=[0, 1, 2])
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["top3Accuracy"] == 1
    assert metrics["logLoss"] > 0


def test_probability_rows_are_normalized() -> None:
    values = validate_probability_matrix(np.array([[2.0, 1.0]]))
    assert np.allclose(values, [[2 / 3, 1 / 3]])


def test_negative_probability_fails() -> None:
    with pytest.raises(ValueError, match="negative"):
        validate_probability_matrix(np.array([[1.1, -0.1]]))


def test_diagnostics_expose_class_recall_and_prediction_distribution() -> None:
    diagnostics = evaluate_diagnostics(
        [0, 0, 1, 2],
        np.array(
            [
                [0.8, 0.1, 0.1],
                [0.6, 0.3, 0.1],
                [0.7, 0.2, 0.1],
                [0.1, 0.2, 0.7],
            ]
        ),
        labels=[0, 1, 2],
        names=["A", "B", "C"],
    )
    assert diagnostics["actualDistribution"] == {"A": 0.5, "B": 0.25, "C": 0.25}
    assert diagnostics["predictedDistribution"] == {"A": 0.75, "B": 0.0, "C": 0.25}
    assert diagnostics["perClass"]["B"]["recall"] == 0
    assert diagnostics["zeroRecallClasses"] == ["B"]
    assert diagnostics["majorityPredictionGap"] == pytest.approx(0.25)
    assert diagnostics["classShareError"] == {
        "A": pytest.approx(0.25),
        "B": pytest.approx(-0.25),
        "C": pytest.approx(0),
    }
    assert diagnostics["maxClassShareError"] == pytest.approx(0.25)
    assert diagnostics["totalVariationDistance"] == pytest.approx(0.25)
    assert diagnostics["classCalibrationError"] == {
        "A": pytest.approx(0.05),
        "B": pytest.approx(-0.05),
        "C": pytest.approx(0),
    }
    assert diagnostics["maxClassCalibrationError"] == pytest.approx(0.05)


def test_probability_calibration_error_detects_hidden_argmax_error() -> None:
    diagnostics = evaluate_diagnostics(
        [0, 1, 0, 1],
        np.array(
            [
                [0.90, 0.10],
                [0.49, 0.51],
                [0.90, 0.10],
                [0.49, 0.51],
            ]
        ),
        labels=[0, 1],
        names=["A", "B"],
    )

    assert diagnostics["maxClassShareError"] == 0
    assert diagnostics["totalVariationDistance"] == 0
    assert diagnostics["maxClassCalibrationError"] > 0


def test_game_bootstrap_is_grouped_reproducible_and_detects_gain() -> None:
    actual = np.array([0, 1, 0, 1])
    reference = np.full((4, 2), 0.5)
    candidate = np.array(
        [[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]]
    )

    first = bootstrap_log_loss_gain(
        [10, 10, 20, 20],
        actual,
        reference,
        candidate,
        samples=50,
        seed=7,
    )
    second = bootstrap_log_loss_gain(
        [10, 10, 20, 20],
        actual,
        reference,
        candidate,
        samples=50,
        seed=7,
    )

    assert first == second
    assert first["games"] == 2
    assert first["improvementProbability"] > 0.98
    assert first["meanGain"] > 0
