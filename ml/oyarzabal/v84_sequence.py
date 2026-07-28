"""Distribution-safe post-processing for the V8.4 sequence residual."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .metrics import evaluate_diagnostics, validate_probability_matrix
from .taxonomy import PITCH_GROUP_FAMILY_LABELS, PITCH_GROUPS
from .v83_sequence import (
    apply_hierarchical_calibration,
    fit_hierarchical_calibration,
    hierarchical_residual_probabilities,
)


def _metrics(actual: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    return evaluate_diagnostics(
        actual,
        probabilities,
        labels=range(len(PITCH_GROUPS)),
        names=[str(group) for group in PITCH_GROUPS],
        family_labels=PITCH_GROUP_FAMILY_LABELS,
    )


def _temperature_probabilities(
    probabilities: np.ndarray,
    log_temperature: float,
) -> np.ndarray:
    logits = np.log(validate_probability_matrix(probabilities))
    logits /= np.exp(log_temperature)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return validate_probability_matrix(values / values.sum(axis=1, keepdims=True))


def calibration_candidates(
    actual: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fit calibration candidates on the supplied validation rows only."""
    truth = np.asarray(actual, dtype=int)
    values = validate_probability_matrix(probabilities)
    rows = np.arange(len(truth))

    def temperature_loss(parameter: np.ndarray) -> float:
        adjusted = _temperature_probabilities(values, float(parameter[0]))
        return float(-np.log(adjusted[rows, truth]).mean())

    temperature = minimize(
        temperature_loss,
        np.zeros(1),
        method="L-BFGS-B",
        bounds=[(-1.5, 1.5)],
    )
    if not temperature.success:
        raise RuntimeError(f"temperature calibration failed: {temperature.message}")
    hierarchical = fit_hierarchical_calibration(truth, values)
    return {
        "identity": (values.copy(), np.array([], dtype=float)),
        "temperature": (
            _temperature_probabilities(values, float(temperature.x[0])),
            np.asarray(temperature.x, dtype=float),
        ),
        "hierarchical": (
            apply_hierarchical_calibration(values, hierarchical.copy()),
            hierarchical,
        ),
    }


def select_safe_calibration(
    actual: np.ndarray,
    reference_probabilities: np.ndarray,
    candidates: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[str, np.ndarray]:
    """Choose the lowest validation Log Loss without degrading safety metrics."""
    reference = _metrics(actual, reference_probabilities)
    passing = []
    for name, (probabilities, parameters) in candidates.items():
        metrics = _metrics(actual, probabilities)
        major_zero = [
            label
            for label in metrics["zeroRecallClasses"]
            if metrics["actualDistribution"][label] >= 0.05
        ]
        safe = (
            metrics["accuracy"] >= reference["accuracy"] - 0.005
            and metrics["familyAccuracy"] >= reference["familyAccuracy"] - 0.005
            and metrics["hierarchicalAccuracy"]
            >= reference["hierarchicalAccuracy"] - 0.005
            and metrics["macroF1"] >= reference["macroF1"] - 0.005
            and metrics["totalVariationDistance"]
            <= reference["totalVariationDistance"] + 0.005
            and metrics["maxClassCalibrationError"]
            <= reference["maxClassCalibrationError"] + 0.005
            and not major_zero
        )
        if safe:
            passing.append((metrics["logLoss"], name, parameters))
    if not passing:
        return "identity", np.array([], dtype=float)
    _, name, parameters = min(passing, key=lambda item: item[0])
    return name, np.asarray(parameters, dtype=float)


def fit_class_conditional_shrinkage(
    actual: np.ndarray,
    global_probabilities: np.ndarray,
    family_delta: np.ndarray,
    child_delta: np.ndarray,
    group_weights: np.ndarray,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Fit bounded family/child residual multipliers on validation rows."""
    truth = np.asarray(actual, dtype=int)
    weights = np.asarray(group_weights, dtype=float)
    if weights.shape != (6,):
        raise ValueError("group weights must contain six values")
    rows = np.arange(len(truth))

    def objective(parameters: np.ndarray) -> float:
        probabilities = hierarchical_residual_probabilities(
            global_probabilities,
            family_delta * parameters[:3],
            child_delta * parameters[3:].reshape(3, 2),
            scale,
        )
        nll = -np.log(probabilities[rows, truth])
        sample_weights = weights[truth]
        balanced = (nll * sample_weights).sum() / sample_weights.sum()
        regularization = 0.002 * np.square(parameters - 1).mean()
        return float(0.8 * nll.mean() + 0.2 * balanced + regularization)

    fitted = minimize(
        objective,
        np.ones(9),
        method="L-BFGS-B",
        bounds=[(0.0, 1.0)] * 9,
    )
    if not fitted.success:
        raise RuntimeError(f"class shrinkage failed: {fitted.message}")
    return np.asarray(fitted.x, dtype=float)


@dataclass(frozen=True)
class DistributionSafeTransform:
    scale: float
    family_shrinkage: np.ndarray
    child_shrinkage: np.ndarray
    calibration_mode: str
    calibration_parameters: np.ndarray

    @classmethod
    def identity(cls, *, scale: float) -> DistributionSafeTransform:
        return cls(
            scale=scale,
            family_shrinkage=np.ones(3),
            child_shrinkage=np.ones(6),
            calibration_mode="identity",
            calibration_parameters=np.array([], dtype=float),
        )

    def apply(
        self,
        global_probabilities: np.ndarray,
        family_delta: np.ndarray,
        child_delta: np.ndarray,
    ) -> np.ndarray:
        values = hierarchical_residual_probabilities(
            global_probabilities,
            family_delta * self.family_shrinkage,
            child_delta * self.child_shrinkage.reshape(3, 2),
            self.scale,
        )
        if self.scale == 0 or self.calibration_mode == "identity":
            return values
        if self.calibration_mode == "temperature":
            return _temperature_probabilities(
                values,
                float(self.calibration_parameters[0]),
            )
        if self.calibration_mode == "hierarchical":
            return apply_hierarchical_calibration(
                values,
                self.calibration_parameters.copy(),
            )
        raise ValueError(f"unknown calibration mode: {self.calibration_mode}")
