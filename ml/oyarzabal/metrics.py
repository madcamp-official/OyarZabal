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
) -> dict[str, float | int]:
    values = validate_probability_matrix(probabilities)
    y = np.asarray(actual, dtype=int)
    if len(y) != len(values):
        raise ValueError("actual and probability row counts differ")
    if len(y) == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "top3Accuracy": 0.0,
            "macroF1": 0.0,
            "logLoss": 0.0,
        }

    predicted = values.argmax(axis=1)
    top_k = min(3, values.shape[1])
    top_indices = np.argpartition(values, -top_k, axis=1)[:, -top_k:]
    top3 = float(
        np.mean([truth in row for truth, row in zip(y, top_indices, strict=True)])
    )
    return {
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


def evaluate_diagnostics(
    actual: Sequence[int],
    probabilities: np.ndarray,
    *,
    labels: Sequence[int],
    names: Sequence[str],
) -> dict[str, object]:
    """Return aggregate and per-class diagnostics for model selection and UI."""
    label_values = list(labels)
    if len(label_values) != len(names):
        raise ValueError("labels and names differ in length")
    values = validate_probability_matrix(probabilities)
    y = np.asarray(actual, dtype=int)
    aggregate = evaluate_probabilities(y, values, labels=label_values)
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
        "perClass": per_class,
        "zeroRecallClasses": zero_recall,
        "majorityPredictionGap": max(
            0.0,
            float(predicted_counts.max() / total - actual_counts.max() / total),
        ),
    }


def entropy(probabilities: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0)
