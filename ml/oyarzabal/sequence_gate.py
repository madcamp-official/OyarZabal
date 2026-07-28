"""ID-free utility gate for the V8.1 sequence expert."""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .metrics import hierarchical_top_indices, validate_probability_matrix
from .sequence import SequenceExamples, blend_log_probabilities
from .taxonomy import PITCH_GROUP_FAMILY_LABELS


def _entropy(values: np.ndarray) -> np.ndarray:
    return -(values * np.log(np.clip(values, 1e-12, 1))).sum(axis=1)


def _margin(values: np.ndarray) -> np.ndarray:
    ordered = np.partition(values, -2, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    midpoint = (left + right) / 2
    return 0.5 * (
        (left * np.log(np.clip(left / midpoint, 1e-12, None))).sum(axis=1)
        + (right * np.log(np.clip(right / midpoint, 1e-12, None))).sum(axis=1)
    )


def gate_feature_frame(
    examples: SequenceExamples,
    indices: np.ndarray,
    global_probabilities: np.ndarray,
    sequence_probabilities: np.ndarray,
) -> pd.DataFrame:
    """Build context-only features; entity IDs and targets are absent."""
    global_values = validate_probability_matrix(global_probabilities)
    sequence_values = validate_probability_matrix(sequence_probabilities)
    if (
        len(indices) != len(global_values)
        or sequence_values.shape != global_values.shape
    ):
        raise ValueError("sequence gate rows and probabilities are misaligned")

    history = examples.history_indices[indices]
    valid = history >= 0
    safe = np.maximum(history, 0)
    groups = examples.source_categorical[safe, 0].copy()
    groups[~valid] = 0
    valid_length = valid.sum(axis=1)
    history_numeric = examples.source_numeric[safe]
    observed = np.isfinite(history_numeric) & valid[:, :, None]
    denominator = np.maximum(
        valid_length * history_numeric.shape[2],
        1,
    )
    missing_rate = 1 - observed.sum(axis=(1, 2)) / denominator
    diversity = np.column_stack(
        [(groups == group).any(axis=1) for group in range(2, 8)]
    ).sum(axis=1)
    reversed_groups = groups[:, ::-1]
    last = reversed_groups[:, :1]
    streak = np.cumprod(reversed_groups == last, axis=1).sum(axis=1)
    streak[valid_length == 0] = 0

    global_top = hierarchical_top_indices(
        global_values,
        PITCH_GROUP_FAMILY_LABELS,
    )
    sequence_top = hierarchical_top_indices(
        sequence_values,
        PITCH_GROUP_FAMILY_LABELS,
    )
    family_labels = np.asarray(PITCH_GROUP_FAMILY_LABELS)
    repertoire = examples.repertoire_context[indices]
    if repertoire.shape[1] != 18:
        raise ValueError("sequence gate requires season/recent repertoire context")
    rows = np.arange(len(indices))
    season_top = repertoire[rows, sequence_top]
    recent_top = repertoire[rows, 6 + sequence_top]

    return pd.DataFrame(
        {
            "global_entropy": _entropy(global_values),
            "sequence_entropy": _entropy(sequence_values),
            "global_margin": _margin(global_values),
            "sequence_margin": _margin(sequence_values),
            "top1_agreement": global_top == sequence_top,
            "family_agreement": family_labels[global_top]
            == family_labels[sequence_top],
            "js_divergence": _js_divergence(global_values, sequence_values),
            "sequence_length": valid_length,
            "history_missing_rate": missing_rate,
            "recent_diversity": diversity,
            "recent_streak": streak,
            "balls": examples.current_categorical[indices, 0] - 1,
            "strikes": examples.current_categorical[indices, 1] - 1,
            "same_plate_appearance": examples.history_flags[indices, -1, 0],
            "top1_season_rate": season_top,
            "top1_recent_rate": recent_top,
            "top1_repertoire_delta": recent_top - season_top,
        },
        dtype=float,
    )


def gate_targets(
    actual: np.ndarray,
    global_probabilities: np.ndarray,
    sequence_probabilities: np.ndarray,
    max_blend: float,
) -> np.ndarray:
    candidate = blend_log_probabilities(
        global_probabilities,
        sequence_probabilities,
        max_blend,
    )
    global_values = validate_probability_matrix(global_probabilities)
    labels = np.asarray(actual, dtype=int)
    rows = np.arange(len(labels))
    gain = np.log(np.clip(candidate[rows, labels], 1e-12, 1))
    gain -= np.log(np.clip(global_values[rows, labels], 1e-12, 1))
    return (gain > 0).astype(np.int8)


@dataclass
class FittedSequenceGate:
    model: XGBClassifier
    device: str
    feature_names: tuple[str, ...]


def fit_sequence_gate(features: pd.DataFrame, labels: np.ndarray) -> FittedSequenceGate:
    if len(features) != len(labels) or len(np.unique(labels)) != 2:
        raise ValueError("sequence gate needs aligned rows and both classes")
    for device in ("cuda", "cpu"):
        model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10,
            tree_method="hist",
            device=device,
            eval_metric="logloss",
            n_jobs=8,
            random_state=737,
        )
        try:
            model.fit(features, labels, verbose=False)
            return FittedSequenceGate(model, device, tuple(features.columns))
        except Exception:
            if device == "cpu":
                raise
            gc.collect()
    raise AssertionError("unreachable")


def predict_sequence_gate(
    fitted: FittedSequenceGate,
    features: pd.DataFrame,
) -> np.ndarray:
    if tuple(features.columns) != fitted.feature_names:
        raise ValueError("sequence gate feature schema changed")
    probabilities = np.asarray(fitted.model.predict_proba(features), dtype=float)
    return probabilities[:, 1]


def discrete_gate_scales(
    probabilities: np.ndarray,
    max_blend: float,
    threshold: float,
) -> np.ndarray:
    """Map utility confidence to 0/half/full; full starts 0.2 above use."""
    if threshold not in {0.4, 0.5, 0.6, 0.7}:
        raise ValueError("unsupported sequence gate threshold")
    values = np.asarray(probabilities, dtype=float)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("gate probabilities must be between zero and one")
    scales = np.zeros_like(values)
    scales[values >= threshold] = max_blend / 2
    scales[values + 1e-12 >= min(1.0, threshold + 0.2)] = max_blend
    return scales
