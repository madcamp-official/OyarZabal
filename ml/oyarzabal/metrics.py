"""Small, consistent multiclass metrics for replay artifacts."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)


def validate_probability_matrix(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("probabilities must be a non-empty 2D matrix")
    if not np.isfinite(values).all():
        raise ValueError("probabilities contain non-finite values")
    if (values < 0).any():
        raise ValueError("probabilities contain negative values")
    totals = values.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("a probability row has zero mass")
    return values / totals[:, None]


def evaluate_probabilities(
    actual: Sequence[int],
    probabilities: np.ndarray,
    *,
    labels: Sequence[int],
    family_labels: Sequence[int] | None = None,
) -> dict[str, float | int]:
    values = validate_probability_matrix(probabilities)
    y = np.asarray(actual, dtype=int)
    if len(y) != len(values):
        raise ValueError("actual and probability row counts differ")
    if family_labels is not None and len(family_labels) != len(labels):
        raise ValueError("family labels and labels differ in length")
    if len(y) == 0:
        result: dict[str, float | int] = {
            "n": 0,
            "accuracy": 0.0,
            "top3Accuracy": 0.0,
            "macroF1": 0.0,
            "logLoss": 0.0,
        }
        if family_labels is not None:
            result.update(
                {
                    "familyAccuracy": 0.0,
                    "hierarchicalAccuracy": 0.0,
                }
            )
        return result

    predicted = values.argmax(axis=1)
    top_k = min(3, values.shape[1])
    top_indices = np.argpartition(values, -top_k, axis=1)[:, -top_k:]
    top3 = float(
        np.mean([truth in row for truth, row in zip(y, top_indices, strict=True)])
    )
    result = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "top3Accuracy": top3,
        "macroF1": float(
            f1_score(
                y,
                predicted,
                labels=list(labels),
                average="macro",
                zero_division=0,
            )
        ),
        "logLoss": float(log_loss(y, values, labels=list(labels))),
    }
    if family_labels is not None:
        family_by_label = dict(zip(labels, family_labels, strict=True))
        actual_families = np.array([family_by_label[label] for label in y])
        predicted_families = np.array(
            [family_by_label[label] for label in predicted]
        )
        family_accuracy = float(
            accuracy_score(actual_families, predicted_families)
        )
        result["familyAccuracy"] = family_accuracy
        result["hierarchicalAccuracy"] = (
            float(result["accuracy"]) + family_accuracy
        ) / 2
    return result


def evaluate_diagnostics(
    actual: Sequence[int],
    probabilities: np.ndarray,
    *,
    labels: Sequence[int],
    names: Sequence[str],
    family_labels: Sequence[int] | None = None,
) -> dict[str, object]:
    """Return aggregate and per-class diagnostics for model selection and UI."""
    label_values = list(labels)
    if len(label_values) != len(names):
        raise ValueError("labels and names differ in length")
    values = validate_probability_matrix(probabilities)
    y = np.asarray(actual, dtype=int)
    aggregate = evaluate_probabilities(
        y,
        values,
        labels=label_values,
        family_labels=family_labels,
    )
    if len(y) == 0:
        zeros = dict.fromkeys(names, 0.0)
        return {
            **aggregate,
            "actualDistribution": zeros,
            "predictedDistribution": zeros,
            "classShareError": zeros,
            "maxClassShareError": 0.0,
            "totalVariationDistance": 0.0,
            "classCalibrationError": zeros,
            "maxClassCalibrationError": 0.0,
            "perClass": {
                name: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
                for name in names
            },
            "zeroRecallClasses": [],
            "majorityPredictionGap": 0.0,
        }
    predicted = values.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y,
        predicted,
        labels=label_values,
        zero_division=0,
    )
    actual_counts = np.array([(y == label).sum() for label in label_values])
    predicted_counts = np.array([(predicted == label).sum() for label in label_values])
    total = max(1, len(y))
    actual_shares = actual_counts / total
    predicted_shares = predicted_counts / total
    probability_shares = values.mean(axis=0)
    share_errors = predicted_shares - actual_shares
    calibration_errors = probability_shares - actual_shares
    actual_distribution = {
        name: float(count / total)
        for name, count in zip(names, actual_counts, strict=True)
    }
    predicted_distribution = {
        name: float(count / total)
        for name, count in zip(names, predicted_counts, strict=True)
    }
    per_class = {
        name: {
            "precision": float(class_precision),
            "recall": float(class_recall),
            "f1": float(class_f1),
            "support": int(class_support),
        }
        for name, class_precision, class_recall, class_f1, class_support in zip(
            names, precision, recall, f1, support, strict=True
        )
    }
    zero_recall = [
        name
        for name, class_support, class_recall in zip(
            names, support, recall, strict=True
        )
        if class_support > 0 and class_recall == 0
    ]
    return {
        **aggregate,
        "actualDistribution": actual_distribution,
        "predictedDistribution": predicted_distribution,
        "classShareError": {
            name: float(error)
            for name, error in zip(names, share_errors, strict=True)
        },
        "maxClassShareError": float(np.abs(share_errors).max(initial=0)),
        "totalVariationDistance": float(np.abs(share_errors).sum() / 2),
        "classCalibrationError": {
            name: float(error)
            for name, error in zip(names, calibration_errors, strict=True)
        },
        "maxClassCalibrationError": float(
            np.abs(calibration_errors).max(initial=0)
        ),
        "perClass": per_class,
        "zeroRecallClasses": zero_recall,
        "majorityPredictionGap": max(
            0.0,
            float(predicted_counts.max() / total - actual_counts.max() / total),
        ),
    }


def bootstrap_log_loss_gain(
    game_ids: Sequence[object],
    actual: Sequence[int],
    reference_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    *,
    samples: int = 500,
    seed: int = 737,
) -> dict[str, float | int]:
    """Paired game bootstrap of reference minus candidate per-pitch log loss."""
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    reference = validate_probability_matrix(reference_probabilities)
    candidate = validate_probability_matrix(candidate_probabilities)
    labels = np.asarray(actual, dtype=int)
    games = np.asarray(game_ids)
    if (
        len(labels) != len(reference)
        or candidate.shape != reference.shape
        or len(games) != len(labels)
        or not len(labels)
    ):
        raise ValueError("bootstrap inputs are empty or misaligned")
    if labels.min() < 0 or labels.max() >= reference.shape[1]:
        raise ValueError("bootstrap labels are outside probability classes")

    positions = np.arange(len(labels))
    gains = np.log(np.clip(candidate[positions, labels], 1e-12, 1))
    gains -= np.log(np.clip(reference[positions, labels], 1e-12, 1))
    unique_games, inverse = np.unique(games, return_inverse=True)
    game_gain = np.bincount(inverse, weights=gains)
    game_count = np.bincount(inverse).astype(float)
    random = np.random.default_rng(seed)
    sampled = random.integers(
        0,
        len(unique_games),
        size=(samples, len(unique_games)),
    )
    bootstrap_gain = game_gain[sampled].sum(axis=1) / game_count[sampled].sum(axis=1)
    return {
        "games": int(len(unique_games)),
        "samples": int(samples),
        "meanGain": float(gains.mean()),
        "improvementProbability": float(
            (np.count_nonzero(bootstrap_gain > 0) + 0.5) / (samples + 1)
        ),
        "ciLower": float(np.quantile(bootstrap_gain, 0.025)),
        "ciUpper": float(np.quantile(bootstrap_gain, 0.975)),
    }


def entropy(probabilities: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0)
