"""Pooled contextual correction on top of frozen Global probabilities."""

from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from .metrics import evaluate_diagnostics, validate_probability_matrix
from .taxonomy import PITCH_GROUPS

RESIDUAL_FEATURES = (
    "pitcher_id",
    "count_bucket",
    "stand",
    "pa_prev_pitch_1",
)
RESIDUAL_SCALES = (0.25, 0.5, 0.75, 1.0)


@dataclass
class FittedResidual:
    encoder: OneHotEncoder
    model: XGBClassifier
    tree_count: int
    device: str


def count_bucket(rows: pd.DataFrame) -> pd.Series:
    balls = rows["balls"].astype(int)
    strikes = rows["strikes"].astype(int)
    values = np.select(
        [
            balls.eq(3) & strikes.eq(2),
            balls.eq(0) & strikes.eq(0),
            strikes.gt(balls),
            balls.gt(strikes),
        ],
        ["full", "first", "pitcher_ahead", "batter_ahead"],
        default="even",
    )
    return pd.Series(values, index=rows.index, dtype="object")


def residual_feature_frame(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pitcher_id": rows["pitcher_id"].astype(str),
            "count_bucket": count_bucket(rows),
            "stand": rows["stand"].fillna("UNKNOWN").astype(str),
            "pa_prev_pitch_1": rows["pa_prev_pitch_1"]
            .fillna("UNKNOWN")
            .astype(str),
        },
        index=rows.index,
    )


def _base_margin(probabilities: np.ndarray) -> np.ndarray:
    values = validate_probability_matrix(probabilities)
    margin = np.log(np.clip(values, 1e-12, 1.0))
    return margin - margin.mean(axis=1, keepdims=True)


def _model(
    *,
    device: str,
    n_estimators: int,
    early_stopping: bool,
) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(PITCH_GROUPS),
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=20,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=5.0,
        tree_method="hist",
        device=device,
        eval_metric="mlogloss",
        early_stopping_rounds=50 if early_stopping else None,
        n_jobs=8,
        random_state=737,
    )


def _fit(
    x_train: Any,
    labels: np.ndarray,
    base_margin: np.ndarray,
    *,
    n_estimators: int,
    x_tuning: Any | None = None,
    tuning_labels: np.ndarray | None = None,
    tuning_margin: np.ndarray | None = None,
) -> tuple[XGBClassifier, str]:
    early_stopping = x_tuning is not None
    for device in ("cuda", "cpu"):
        model = _model(
            device=device,
            n_estimators=n_estimators,
            early_stopping=early_stopping,
        )
        arguments: dict[str, object] = {
            "base_margin": base_margin,
            "verbose": False,
        }
        if early_stopping:
            arguments.update(
                {
                    "eval_set": [(x_tuning, tuning_labels)],
                    "base_margin_eval_set": [tuning_margin],
                }
            )
        try:
            model.fit(x_train, labels, **arguments)
            return model, device
        except Exception:
            if device == "cpu":
                raise
            gc.collect()
    raise AssertionError("unreachable")


def _best_tree_count(model: XGBClassifier) -> int:
    best_iteration = getattr(model, "best_iteration", None)
    return int(best_iteration + 1) if best_iteration is not None else model.n_estimators


def train_final_residual(
    rows: pd.DataFrame,
    global_probabilities: np.ndarray,
    *,
    n_estimators: int,
) -> FittedResidual:
    if len(rows) != len(global_probabilities) or rows.empty:
        raise ValueError("residual training rows are empty or misaligned")
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=20,
        sparse_output=True,
    )
    transformed = encoder.fit_transform(residual_feature_frame(rows))
    model, device = _fit(
        transformed,
        rows["target"].to_numpy(),
        _base_margin(global_probabilities),
        n_estimators=n_estimators,
    )
    return FittedResidual(encoder, model, n_estimators, device)


def train_residual_with_tuning(
    rows: pd.DataFrame,
    global_probabilities: np.ndarray,
    *,
    n_estimators: int = 600,
) -> FittedResidual:
    if len(rows) != len(global_probabilities) or rows.empty:
        raise ValueError("residual tuning rows are empty or misaligned")
    dates = np.array(sorted(rows["game_date"].dt.normalize().unique()))
    if len(dates) < 2:
        raise ValueError("residual tuning requires at least two game dates")
    cutoff = dates[max(1, min(len(dates) - 1, int(len(dates) * 0.8)))]
    core_mask = rows["game_date"].dt.normalize().to_numpy() < cutoff
    tuning_mask = ~core_mask
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=20,
        sparse_output=True,
    )
    core = encoder.fit_transform(residual_feature_frame(rows.loc[core_mask]))
    tuning = encoder.transform(residual_feature_frame(rows.loc[tuning_mask]))
    model, _ = _fit(
        core,
        rows.loc[core_mask, "target"].to_numpy(),
        _base_margin(global_probabilities[core_mask]),
        n_estimators=n_estimators,
        x_tuning=tuning,
        tuning_labels=rows.loc[tuning_mask, "target"].to_numpy(),
        tuning_margin=_base_margin(global_probabilities[tuning_mask]),
    )
    tree_count = _best_tree_count(model)
    del model, encoder, core, tuning
    gc.collect()
    return train_final_residual(
        rows,
        global_probabilities,
        n_estimators=tree_count,
    )


def predict_correction(fitted: FittedResidual, rows: pd.DataFrame) -> np.ndarray:
    transformed = fitted.encoder.transform(residual_feature_frame(rows))
    zero_margin = np.zeros((len(rows), len(PITCH_GROUPS)), dtype=float)
    correction = np.asarray(
        fitted.model.predict(
            transformed,
            output_margin=True,
            base_margin=zero_margin,
        ),
        dtype=float,
    )
    if correction.shape != zero_margin.shape or not np.isfinite(correction).all():
        raise ValueError("residual correction has invalid shape or values")
    return correction - correction.mean(axis=1, keepdims=True)


def apply_correction(
    global_probabilities: np.ndarray,
    correction: np.ndarray,
    scale: float,
) -> np.ndarray:
    if not 0 <= scale <= 1:
        raise ValueError("residual scale must be between zero and one")
    global_values = validate_probability_matrix(global_probabilities)
    residual_values = np.asarray(correction, dtype=float)
    if residual_values.shape != global_values.shape:
        raise ValueError("global probabilities and residual correction differ")
    logits = _base_margin(global_values) + scale * residual_values
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return validate_probability_matrix(
        probabilities / probabilities.sum(axis=1, keepdims=True)
    )


def diagnostics(actual: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    return evaluate_diagnostics(
        actual,
        probabilities,
        labels=range(len(PITCH_GROUPS)),
        names=[str(group) for group in PITCH_GROUPS],
    )


def residual_passes(
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
        and candidate_metrics["accuracy"] >= global_metrics["accuracy"] - 0.005
        and candidate_metrics["macroF1"] >= global_metrics["macroF1"] - 0.005
        and not major_zero_recall
        and candidate_metrics["majorityPredictionGap"] <= 0.20
    )


def select_residual_scale(
    actual: np.ndarray,
    global_probabilities: np.ndarray,
    correction: np.ndarray,
    *,
    scales: Sequence[float] = RESIDUAL_SCALES,
) -> dict[str, object]:
    global_metrics = diagnostics(actual, global_probabilities)
    candidates = []
    for scale in scales:
        metrics = diagnostics(
            actual,
            apply_correction(global_probabilities, correction, float(scale)),
        )
        candidates.append(
            {
                "scale": float(scale),
                "metrics": metrics,
                "accepted": residual_passes(global_metrics, metrics),
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
        else None
    )
    return {
        "accepted": selected is not None,
        "scale": float(selected["scale"]) if selected else 0.0,
        "metrics": selected["metrics"] if selected else global_metrics,
        "globalMetrics": global_metrics,
        "candidates": candidates,
    }


def pitcher_residual_passes(
    actual: np.ndarray,
    global_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    *,
    min_support: int = 300,
) -> tuple[bool, dict[str, object], dict[str, object]]:
    global_metrics = diagnostics(actual, global_probabilities)
    candidate_metrics = diagnostics(actual, candidate_probabilities)
    return (
        len(actual) >= min_support
        and pitcher_metrics_pass(global_metrics, candidate_metrics),
        global_metrics,
        candidate_metrics,
    )


def pitcher_metrics_pass(
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
        and candidate_metrics["accuracy"] >= global_metrics["accuracy"] - 0.005
        and not major_zero_recall
    )


def provisional_scale(
    gap_days: int,
    *,
    half_life_days: int = 365,
    cap: float = 0.25,
) -> float:
    if gap_days < 0 or half_life_days <= 0 or not 0 <= cap <= 1:
        raise ValueError("invalid provisional decay parameters")
    return min(cap, float(np.exp(-np.log(2) * gap_days / half_life_days)))
