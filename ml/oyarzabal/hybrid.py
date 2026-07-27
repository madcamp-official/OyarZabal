"""Validation and routing for global/specialist probability blends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import evaluate_diagnostics, validate_probability_matrix
from .taxonomy import PITCH_GROUPS


@dataclass(frozen=True)
class RegistryEntry:
    pitcher_id: int
    enabled: bool
    specialist_weight: float
    model: str
    data_cutoff: str | None = None
    reason: str | None = None
    temperature: float = 1.0
    spec: str | None = None
    logit_bias: tuple[float, ...] | None = None
    personalizer_strength: float | None = None

    @property
    def global_weight(self) -> float:
        return 1 - self.specialist_weight


def serialize_registry_entry(
    entry: RegistryEntry, *, name: str
) -> dict[str, object]:
    return {
        "pitcherId": entry.pitcher_id,
        "name": name,
        "enabled": entry.enabled,
        "model": entry.model,
        "dataCutoff": entry.data_cutoff,
        "globalWeight": entry.global_weight,
        "specialistWeight": entry.specialist_weight,
        "temperature": entry.temperature,
        "spec": entry.spec,
        "logitBias": list(entry.logit_bias) if entry.logit_bias is not None else None,
        "personalizerStrength": entry.personalizer_strength,
        "reason": entry.reason,
    }


def blend_probabilities(
    global_probabilities: np.ndarray,
    specialist_probabilities: np.ndarray,
    specialist_weight: float,
) -> np.ndarray:
    if not 0 <= specialist_weight <= 1:
        raise ValueError("specialist weight must be between zero and one")
    global_values = validate_probability_matrix(global_probabilities)
    specialist_values = validate_probability_matrix(specialist_probabilities)
    if global_values.shape != specialist_values.shape:
        raise ValueError("global and specialist probability shapes differ")
    return validate_probability_matrix(
        (1 - specialist_weight) * global_values
        + specialist_weight * specialist_values
    )


def fit_logit_bias(
    actual: np.ndarray,
    global_probabilities: np.ndarray,
    *,
    prior_strength: float,
) -> np.ndarray:
    """Estimate a shrunk per-class intercept from past out-of-fold predictions."""
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")
    probabilities = validate_probability_matrix(global_probabilities)
    labels = np.asarray(actual, dtype=int)
    if len(labels) != len(probabilities) or not len(labels):
        raise ValueError("personalizer calibration rows are empty or misaligned")
    classes = probabilities.shape[1]
    actual_counts = np.bincount(labels, minlength=classes).astype(float)
    expected_share = probabilities.mean(axis=0)
    posterior_share = (
        actual_counts + prior_strength * expected_share
    ) / (len(labels) + prior_strength)
    bias = np.log(np.clip(posterior_share, 1e-12, 1))
    bias -= np.log(np.clip(expected_share, 1e-12, 1))
    return bias - bias.mean()


def apply_logit_bias(
    probabilities: np.ndarray, logit_bias: Sequence[float]
) -> np.ndarray:
    values = validate_probability_matrix(probabilities)
    bias = np.asarray(logit_bias, dtype=float)
    if bias.shape != (values.shape[1],) or not np.isfinite(bias).all():
        raise ValueError("logit_bias does not match probability classes")
    logits = np.log(np.clip(values, 1e-12, 1)) + bias
    logits -= logits.max(axis=1, keepdims=True)
    adjusted = np.exp(logits)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def personalizer_passes(
    global_metrics: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
) -> bool:
    major_zero_recall = [
        name
        for name in candidate_metrics["zeroRecallClasses"]
        if candidate_metrics["actualDistribution"][name] >= 0.05
    ]
    return (
        candidate_metrics["logLoss"] < global_metrics["logLoss"]
        and candidate_metrics["macroF1"] > global_metrics["macroF1"]
        and candidate_metrics["accuracy"] >= global_metrics["accuracy"] - 0.005
        and not major_zero_recall
        and candidate_metrics["majorityPredictionGap"] <= 0.20
    )


def select_personalizer_strength(
    calibration_actual: np.ndarray,
    calibration_probabilities: np.ndarray,
    evaluation_actual: np.ndarray,
    evaluation_probabilities: np.ndarray,
    *,
    strengths: Sequence[float] = (250, 500, 1_000),
) -> dict[str, object]:
    """Fit on earlier rows and select shrinkage on a later validation slice."""
    global_metrics = _diagnostics(evaluation_actual, evaluation_probabilities)
    candidates = []
    for strength in strengths:
        bias = fit_logit_bias(
            calibration_actual,
            calibration_probabilities,
            prior_strength=float(strength),
        )
        metrics = _diagnostics(
            evaluation_actual,
            apply_logit_bias(evaluation_probabilities, bias),
        )
        major_zero_recall = [
            name
            for name in metrics["zeroRecallClasses"]
            if metrics["actualDistribution"][name] >= 0.05
        ]
        accepted = personalizer_passes(global_metrics, metrics)
        candidates.append(
            {
                "personalizerStrength": float(strength),
                "effectiveWeight": float(
                    len(calibration_actual) / (len(calibration_actual) + strength)
                ),
                "logitBias": bias.tolist(),
                "metrics": metrics,
                "majorZeroRecallClasses": major_zero_recall,
                "accepted": accepted,
            }
        )

    eligible = [candidate for candidate in candidates if candidate["accepted"]]
    selected = (
        min(
            eligible,
            key=lambda candidate: (
                candidate["metrics"]["logLoss"],
                -candidate["metrics"]["macroF1"],
            ),
        )
        if eligible
        else {
            "personalizerStrength": None,
            "effectiveWeight": 0.0,
            "logitBias": [0.0] * evaluation_probabilities.shape[1],
            "metrics": global_metrics,
            "majorZeroRecallClasses": [],
            "accepted": False,
        }
    )
    return {
        **selected,
        "globalMetrics": global_metrics,
        "candidates": candidates,
    }


def _diagnostics(actual: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    return evaluate_diagnostics(
        actual,
        probabilities,
        labels=range(probabilities.shape[1]),
        names=[str(group) for group in PITCH_GROUPS[: probabilities.shape[1]]],
    )


def select_blend_weight(
    actual: np.ndarray,
    global_probabilities: np.ndarray,
    specialist_probabilities: np.ndarray,
    *,
    weights: Sequence[float] = (0, 0.25, 0.5, 0.75, 1),
) -> dict[str, object]:
    global_metrics = _diagnostics(actual, global_probabilities)
    candidates = []
    for weight in weights:
        probabilities = blend_probabilities(
            global_probabilities, specialist_probabilities, weight
        )
        metrics = _diagnostics(actual, probabilities)
        major_zero_recall = [
            name
            for name in metrics["zeroRecallClasses"]
            if metrics["actualDistribution"][name] >= 0.05
        ]
        accepted = (
            weight > 0
            and metrics["logLoss"] < global_metrics["logLoss"]
            and metrics["macroF1"] > global_metrics["macroF1"]
            and metrics["accuracy"] >= global_metrics["accuracy"] - 0.005
            and not major_zero_recall
            and metrics["majorityPredictionGap"] <= 0.20
        )
        candidates.append(
            {
                "specialistWeight": float(weight),
                "metrics": metrics,
                "majorZeroRecallClasses": major_zero_recall,
                "accepted": accepted,
            }
        )

    eligible = [candidate for candidate in candidates if candidate["accepted"]]
    selected = (
        min(
            eligible,
            key=lambda candidate: (
                candidate["metrics"]["logLoss"],
                -candidate["metrics"]["macroF1"],
            ),
        )
        if eligible
        else next(
            candidate for candidate in candidates if candidate["specialistWeight"] == 0
        )
    )
    return {
        **selected,
        "globalMetrics": global_metrics,
        "candidates": candidates,
    }


def specialist_eligibility(
    rows: pd.DataFrame,
    *,
    min_pitches: int = 5_000,
    min_recent_pitches: int = 500,
    min_pitch_groups: int = 3,
    min_group_share: float = 0.05,
) -> dict[str, object]:
    if rows.empty:
        return {"eligible": False, "reasons": ["pitch_count"], "pitchCount": 0}
    latest_year = int(rows["game_date"].dt.year.max())
    recent_count = int(rows["game_date"].dt.year.eq(latest_year).sum())
    shares = rows["target"].value_counts(normalize=True)
    repertoire_count = int((shares >= min_group_share).sum())
    reasons = []
    if len(rows) < min_pitches:
        reasons.append("pitch_count")
    if recent_count < min_recent_pitches:
        reasons.append("recent_pitch_count")
    if repertoire_count < min_pitch_groups:
        reasons.append("repertoire")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "pitchCount": int(len(rows)),
        "recentYear": latest_year,
        "recentPitchCount": recent_count,
        "repertoirePitchGroups": repertoire_count,
    }


def blend_by_pitcher(
    pitcher_ids: np.ndarray,
    global_probabilities: np.ndarray,
    specialist_probabilities: Mapping[int, np.ndarray],
    registry: Mapping[int, RegistryEntry],
) -> tuple[np.ndarray, list[str]]:
    global_values = validate_probability_matrix(global_probabilities)
    if len(pitcher_ids) != len(global_values):
        raise ValueError("pitcher IDs and global probabilities differ in length")
    output = global_values.copy()
    sources = ["global"] * len(output)
    for pitcher_id, entry in registry.items():
        positions = np.flatnonzero(np.asarray(pitcher_ids) == pitcher_id)
        if not entry.enabled or not len(positions):
            continue
        specialist = specialist_probabilities.get(pitcher_id)
        if specialist is None:
            raise ValueError(f"missing specialist probabilities for {pitcher_id}")
        output[positions] = blend_probabilities(
            global_values[positions],
            specialist,
            entry.specialist_weight,
        )
        for position in positions:
            sources[int(position)] = "hybrid"
    return output, sources


def personalize_by_pitcher(
    pitcher_ids: np.ndarray,
    global_probabilities: np.ndarray,
    registry: Mapping[int, RegistryEntry],
) -> tuple[np.ndarray, list[str]]:
    global_values = validate_probability_matrix(global_probabilities)
    if len(pitcher_ids) != len(global_values):
        raise ValueError("pitcher IDs and global probabilities differ in length")
    output = global_values.copy()
    sources = ["global"] * len(output)
    for pitcher_id, entry in registry.items():
        positions = np.flatnonzero(np.asarray(pitcher_ids) == pitcher_id)
        if not entry.enabled or not len(positions):
            continue
        if entry.logit_bias is None:
            raise ValueError(f"missing personalizer bias for {pitcher_id}")
        output[positions] = apply_logit_bias(
            global_values[positions], entry.logit_bias
        )
        for position in positions:
            sources[int(position)] = "hybrid"
    return output, sources
