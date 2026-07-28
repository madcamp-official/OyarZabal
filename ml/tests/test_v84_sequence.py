import numpy as np
import pytest
from oyarzabal.v83_sequence import SequenceObjective, mild_class_weights
from oyarzabal.v84 import _select
from oyarzabal.v84_sequence import (
    DistributionSafeTransform,
    calibration_candidates,
    fit_class_conditional_shrinkage,
    select_safe_calibration,
)


def _probabilities() -> tuple[np.ndarray, np.ndarray]:
    global_probabilities = np.array(
        [
            [0.35, 0.20, 0.18, 0.10, 0.12, 0.05],
            [0.20, 0.30, 0.20, 0.10, 0.15, 0.05],
            [0.25, 0.20, 0.30, 0.10, 0.10, 0.05],
            [0.20, 0.20, 0.15, 0.25, 0.15, 0.05],
            [0.25, 0.20, 0.15, 0.10, 0.25, 0.05],
            [0.25, 0.20, 0.15, 0.10, 0.10, 0.20],
        ],
        dtype=float,
    )
    return np.arange(6), global_probabilities


def test_sequence_objective_and_weights_are_bounded() -> None:
    objective = SequenceObjective(
        soft_target_strength=0.25,
        focal_gamma=1.0,
        group_balanced=True,
    )
    weights = mild_class_weights(
        np.array([0] * 100 + [1] * 50 + [2] * 30 + [3] * 20 + [4] * 10 + [5] * 5),
        classes=6,
    )

    assert objective.soft_target_strength == 0.25
    assert weights.mean() == pytest.approx(1)
    assert weights.min() >= 0.75
    assert weights.max() <= 1.5


def test_class_conditional_shrinkage_is_bounded_and_probability_safe() -> None:
    actual, global_probabilities = _probabilities()
    family_delta = np.array([[0.4, -0.2, -0.2]] * len(actual))
    child_delta = np.array([[[0.2, -0.2], [0.1, -0.1], [0.1, -0.1]]] * len(actual))
    shrinkage = fit_class_conditional_shrinkage(
        actual,
        global_probabilities,
        family_delta,
        child_delta,
        np.ones(6),
    )
    transform = DistributionSafeTransform(
        scale=0.25,
        family_shrinkage=shrinkage[:3],
        child_shrinkage=shrinkage[3:],
        calibration_mode="identity",
        calibration_parameters=np.array([]),
    )

    adjusted = transform.apply(global_probabilities, family_delta, child_delta)

    assert np.all((shrinkage >= 0) & (shrinkage <= 1))
    np.testing.assert_allclose(adjusted.sum(axis=1), 1)


def test_scale_zero_is_exact_global_fallback() -> None:
    _, global_probabilities = _probabilities()
    transform = DistributionSafeTransform.identity(scale=0)
    family_delta = np.ones((len(global_probabilities), 3))
    child_delta = np.ones((len(global_probabilities), 3, 2))

    adjusted = transform.apply(global_probabilities, family_delta, child_delta)

    np.testing.assert_array_equal(adjusted, global_probabilities)


def test_safe_calibration_rejects_argmax_distribution_collapse() -> None:
    actual, global_probabilities = _probabilities()
    unsafe = global_probabilities.copy()
    unsafe[:, 0] += unsafe[:, 1:].sum(axis=1)
    unsafe[:, 1:] = 0
    candidates = {
        **calibration_candidates(actual, global_probabilities),
        "unsafe": (unsafe, np.array([])),
    }

    selected = select_safe_calibration(
        actual,
        global_probabilities,
        candidates,
    )

    assert selected[0] != "unsafe"


def test_selection_keeps_half_of_safety_tolerance_in_reserve() -> None:
    reference = {
        "accuracy": 0.50,
        "familyAccuracy": 0.60,
        "hierarchicalAccuracy": 0.55,
        "macroF1": 0.46,
        "totalVariationDistance": 0.08,
        "maxClassCalibrationError": 0.05,
    }

    def candidate(
        objective: str,
        log_loss: float,
        macro_f1: float,
    ) -> dict[str, object]:
        metrics = {
            **reference,
            "logLoss": log_loss,
            "macroF1": macro_f1,
        }
        assessment = {"accepted": True, "metrics": metrics}
        return {
            "spec": {
                "objective": objective,
                "shrinkage": "none",
                "scale": 0.1,
            },
            "normal": assessment,
            "physicalDropStress": assessment,
        }

    selected = _select(
        {
            "borderline": candidate("NLL", 1.07, 0.457),
            "robust": candidate("FOCAL_1", 1.08, 0.458),
        },
        reference,
    )

    assert selected is not None
    assert selected.objective == "FOCAL_1"
