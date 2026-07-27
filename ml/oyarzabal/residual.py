"""Pooled contextual correction on top of frozen Global probabilities."""

from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from .metrics import (
    bootstrap_log_loss_gain,
    evaluate_diagnostics,
    validate_probability_matrix,
)
from .taxonomy import PITCH_GROUP_FAMILY_LABELS, PITCH_GROUPS

RESIDUAL_FEATURES = (
    "pitcher_id",
    "count_bucket",
    "stand",
    "pa_prev_pitch_1",
)
RESIDUAL_SCALES = (0.25, 0.5, 0.75, 1.0)
SAFE_SCALE_MULTIPLIERS = tuple(value / 20 for value in range(1, 21))
GATE_CATEGORICAL_FEATURES = (
    "count_bucket",
    "stand",
    "p_throws",
    "pa_prev_pitch_1",
)
GATE_NUMERIC_FEATURES = (
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "base_1",
    "base_2",
    "base_3",
    "score_diff",
    "game_pitch_count",
    "n_thruorder_pitcher",
    "pitcher_days_since_prev_game",
    "pitch_number",
    "career_support",
    "count_support",
    "stand_support",
    "transition_support",
    "global_entropy",
    "global_top1",
    "global_top1_margin",
    "global_reference_js",
    "global_reference_top1_disagreement",
)


@dataclass
class FittedResidual:
    encoder: OneHotEncoder
    model: XGBClassifier
    tree_count: int
    device: str


@dataclass
class FittedGate:
    preprocessor: ColumnTransformer
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


def gate_feature_frame(
    rows: pd.DataFrame,
    global_probabilities: np.ndarray,
    correction: np.ndarray,
) -> pd.DataFrame:
    global_values = validate_probability_matrix(global_probabilities)
    residual_values = np.asarray(correction, dtype=float)
    if len(rows) != len(global_values) or residual_values.shape != global_values.shape:
        raise ValueError("gate rows and model outputs are misaligned")
    reference = apply_correction(global_values, residual_values, 0.5)
    midpoint = (global_values + reference) / 2
    js = 0.5 * (
        np.where(
            global_values > 0,
            global_values * np.log(global_values / midpoint),
            0,
        ).sum(axis=1)
        + np.where(
            reference > 0,
            reference * np.log(reference / midpoint),
            0,
        ).sum(axis=1)
    )
    sorted_probabilities = np.sort(global_values, axis=1)
    values: dict[str, object] = {
        "count_bucket": count_bucket(rows),
        "stand": rows.get("stand", pd.Series("UNKNOWN", index=rows.index)),
        "p_throws": rows.get("p_throws", pd.Series("UNKNOWN", index=rows.index)),
        "pa_prev_pitch_1": rows.get(
            "pa_prev_pitch_1", pd.Series("UNKNOWN", index=rows.index)
        ),
        "global_entropy": -np.sum(
            np.where(global_values > 0, global_values * np.log(global_values), 0),
            axis=1,
        ),
        "global_top1": global_values.max(axis=1),
        "global_top1_margin": (
            sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
        ),
        "global_reference_js": js,
        "global_reference_top1_disagreement": (
            global_values.argmax(axis=1) != reference.argmax(axis=1)
        ).astype(float),
    }
    for name in GATE_NUMERIC_FEATURES:
        if name in values:
            continue
        values[name] = rows.get(name, pd.Series(np.nan, index=rows.index))
    frame = pd.DataFrame(values, index=rows.index)
    for name in GATE_CATEGORICAL_FEATURES:
        frame[name] = frame[name].fillna("UNKNOWN").astype(str)
    for name in GATE_NUMERIC_FEATURES:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame[[*GATE_NUMERIC_FEATURES, *GATE_CATEGORICAL_FEATURES]]


def gate_targets(
    actual: np.ndarray,
    global_probabilities: np.ndarray,
    correction: np.ndarray,
) -> np.ndarray:
    global_values = validate_probability_matrix(global_probabilities)
    reference = apply_correction(global_values, correction, 0.5)
    labels = np.asarray(actual, dtype=int)
    if len(labels) != len(global_values):
        raise ValueError("gate target rows are misaligned")
    positions = np.arange(len(labels))
    utility = np.log(np.clip(reference[positions, labels], 1e-12, 1))
    utility -= np.log(np.clip(global_values[positions, labels], 1e-12, 1))
    return (utility > 0).astype("int8")


def _gate_preprocessor() -> ColumnTransformer:
    categorical = Pipeline(
        [
            ("missing", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median"),
                list(GATE_NUMERIC_FEATURES),
            ),
            ("categorical", categorical, list(GATE_CATEGORICAL_FEATURES)),
        ]
    )


def _gate_model(
    *,
    device: str,
    n_estimators: int,
    early_stopping: bool,
) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=n_estimators,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=10.0,
        tree_method="hist",
        device=device,
        eval_metric="logloss",
        early_stopping_rounds=50 if early_stopping else None,
        n_jobs=8,
        random_state=737,
    )


def train_gate(
    rows: pd.DataFrame,
    global_probabilities: np.ndarray,
    correction: np.ndarray,
    *,
    n_estimators: int,
    tuning: tuple[pd.DataFrame, np.ndarray, np.ndarray] | None = None,
) -> FittedGate:
    labels = gate_targets(
        rows["target"].to_numpy(),
        global_probabilities,
        correction,
    )
    if len(np.unique(labels)) < 2:
        raise ValueError("gate training requires both utility classes")
    preprocessor = _gate_preprocessor()
    train_x = preprocessor.fit_transform(
        gate_feature_frame(rows, global_probabilities, correction)
    )
    tune_x = None
    tune_labels = None
    if tuning is not None:
        tune_rows, tune_global, tune_correction = tuning
        tune_x = preprocessor.transform(
            gate_feature_frame(tune_rows, tune_global, tune_correction)
        )
        tune_labels = gate_targets(
            tune_rows["target"].to_numpy(),
            tune_global,
            tune_correction,
        )
    for device in ("cuda", "cpu"):
        model = _gate_model(
            device=device,
            n_estimators=n_estimators,
            early_stopping=tuning is not None,
        )
        arguments: dict[str, object] = {"verbose": False}
        if tuning is not None:
            arguments["eval_set"] = [(tune_x, tune_labels)]
        try:
            model.fit(train_x, labels, **arguments)
            best_iteration = getattr(model, "best_iteration", None)
            tree_count = (
                int(best_iteration + 1)
                if best_iteration is not None
                else n_estimators
            )
            return FittedGate(preprocessor, model, tree_count, device)
        except Exception:
            if device == "cpu":
                raise
            gc.collect()
    raise AssertionError("unreachable")


def predict_context_gate(
    fitted: FittedGate,
    rows: pd.DataFrame,
    global_probabilities: np.ndarray,
    correction: np.ndarray,
) -> np.ndarray:
    transformed = fitted.preprocessor.transform(
        gate_feature_frame(rows, global_probabilities, correction)
    )
    probabilities = np.asarray(fitted.model.predict_proba(transformed), dtype=float)
    if probabilities.shape != (len(rows), 2):
        raise ValueError("gate probabilities have invalid shape")
    return probabilities[:, 1]


def hard_safety_mask(rows: pd.DataFrame) -> tuple[np.ndarray, list[str | None]]:
    required = ("count_support", "stand_support", "transition_support")
    missing = [name for name in required if name not in rows]
    if missing:
        raise ValueError(f"missing hard safety support: {missing}")
    count = pd.to_numeric(rows["count_support"], errors="coerce").to_numpy()
    stand = pd.to_numeric(rows["stand_support"], errors="coerce").to_numpy()
    transition = pd.to_numeric(
        rows["transition_support"], errors="coerce"
    ).to_numpy()
    previous = (
        rows.get("pa_prev_pitch_1", pd.Series("UNKNOWN", index=rows.index))
        .fillna("UNKNOWN")
        .astype(str)
        .to_numpy()
    )
    reasons: list[str | None] = []
    for count_value, stand_value, transition_value, previous_value in zip(
        count, stand, transition, previous, strict=True
    ):
        if not np.isfinite([count_value, stand_value, transition_value]).all():
            reasons.append("invalid_support")
        elif count_value < 20:
            reasons.append("count_support")
        elif stand_value < 20:
            reasons.append("stand_support")
        elif previous_value != "UNKNOWN" and transition_value < 20:
            reasons.append("transition_support")
        else:
            reasons.append(None)
    return np.array([reason is None for reason in reasons]), reasons


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
    scale: float | np.ndarray,
) -> np.ndarray:
    global_values = validate_probability_matrix(global_probabilities)
    residual_values = np.asarray(correction, dtype=float)
    if residual_values.shape != global_values.shape:
        raise ValueError("global probabilities and residual correction differ")
    scale_values = np.asarray(scale, dtype=float)
    if scale_values.ndim == 0:
        if not 0 <= float(scale_values) <= 1:
            raise ValueError("residual scale must be between zero and one")
        scaled = float(scale_values) * residual_values
    elif scale_values.shape == (len(global_values),):
        if (
            not np.isfinite(scale_values).all()
            or (scale_values < 0).any()
            or (scale_values > 1).any()
        ):
            raise ValueError("residual scales must be between zero and one")
        scaled = scale_values[:, None] * residual_values
    else:
        raise ValueError("residual scales do not match probability rows")
    logits = _base_margin(global_values) + scaled
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return validate_probability_matrix(
        probabilities / probabilities.sum(axis=1, keepdims=True)
    )


def reliability_score(
    support: int,
    p_all: float,
    p_recent: float,
) -> float:
    if support < 0 or not 0 <= p_all <= 1 or not 0 <= p_recent <= 1:
        raise ValueError("invalid reliability inputs")
    return float(
        0.5
        * support
        / (support + 1_000)
        * min(p_all, p_recent)
    )


def compute_pitcher_reliability(
    rows: pd.DataFrame,
    global_probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
    *,
    recent_days: int = 90,
    samples: int = 500,
    seed: int = 737,
) -> dict[int, dict[str, float | int]]:
    if recent_days <= 0:
        raise ValueError("recent_days must be positive")
    global_values = validate_probability_matrix(global_probabilities)
    reference_values = validate_probability_matrix(reference_probabilities)
    if len(rows) != len(global_values) or reference_values.shape != global_values.shape:
        raise ValueError("reliability rows and probabilities are misaligned")
    if rows.empty:
        return {}

    result: dict[int, dict[str, float | int]] = {}
    dates = pd.to_datetime(rows["game_date"], errors="raise")
    ids = rows["pitcher_id"].to_numpy(dtype=int)
    actual = rows["target"].to_numpy(dtype=int)
    games = rows["game_pk"].to_numpy()
    for pitcher_id in sorted(set(ids)):
        positions = np.flatnonzero(ids == pitcher_id)
        latest = dates.iloc[positions].max().normalize()
        recent_cutoff = latest - pd.Timedelta(days=recent_days - 1)
        recent_positions = positions[
            dates.iloc[positions].dt.normalize().to_numpy() >= recent_cutoff
        ]
        all_bootstrap = bootstrap_log_loss_gain(
            games[positions],
            actual[positions],
            global_values[positions],
            reference_values[positions],
            samples=samples,
            seed=seed + int(pitcher_id),
        )
        if len(recent_positions):
            recent_bootstrap = bootstrap_log_loss_gain(
                games[recent_positions],
                actual[recent_positions],
                global_values[recent_positions],
                reference_values[recent_positions],
                samples=samples,
                seed=seed + int(pitcher_id) + 1,
            )
            p_recent = float(recent_bootstrap["improvementProbability"])
        else:
            recent_bootstrap = None
            p_recent = 0.0
        p_all = float(all_bootstrap["improvementProbability"])
        support = int(len(positions))
        result[int(pitcher_id)] = {
            "n": support,
            "supportCoefficient": float(support / (support + 1_000)),
            "pAll": p_all,
            "pRecent": p_recent,
            "reliability": reliability_score(support, p_all, p_recent),
            "recentSupport": int(len(recent_positions)),
            "allMeanGain": float(all_bootstrap["meanGain"]),
            "recentMeanGain": (
                float(recent_bootstrap["meanGain"])
                if recent_bootstrap is not None
                else 0.0
            ),
        }
    return result


def effective_residual_scale(
    reliability: float,
    context_gate: float,
    *,
    hard_safety_pass: bool,
) -> float:
    if not 0 <= reliability <= 0.5 or not 0 <= context_gate <= 1:
        raise ValueError("invalid dynamic scale inputs")
    return float(reliability * context_gate) if hard_safety_pass else 0.0


def _js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    midpoint = (left + right) / 2
    left_term = np.where(left > 0, left * np.log(left / midpoint), 0)
    right_term = np.where(right > 0, right * np.log(right / midpoint), 0)
    return 0.5 * (left_term.sum(axis=1) + right_term.sum(axis=1))


def apply_dynamic_correction(
    global_probabilities: np.ndarray,
    correction: np.ndarray,
    scales: np.ndarray,
    *,
    js_cap: float = 0.05,
    probability_shift_cap: float = 0.20,
    iterations: int = 24,
) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
    global_values = validate_probability_matrix(global_probabilities)
    residual_values = np.asarray(correction, dtype=float)
    requested = np.asarray(scales, dtype=float)
    if residual_values.shape != global_values.shape or requested.shape != (
        len(global_values),
    ):
        raise ValueError("dynamic correction inputs are misaligned")
    if (
        not np.isfinite(residual_values).all()
        or not np.isfinite(requested).all()
        or (requested < 0).any()
        or (requested > 0.5).any()
    ):
        raise ValueError("dynamic correction inputs are invalid")
    if js_cap <= 0 or probability_shift_cap <= 0 or iterations <= 0:
        raise ValueError("dynamic correction caps must be positive")

    def probabilities(values: np.ndarray) -> np.ndarray:
        logits = _base_margin(global_values) + values[:, None] * residual_values
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    initial = probabilities(requested)
    js_failed = _js_divergence(global_values, initial) > js_cap
    shift_failed = (
        np.abs(initial - global_values).max(axis=1) > probability_shift_cap
    )
    capped = js_failed | shift_failed
    low = np.zeros_like(requested)
    high = requested.copy()
    low[~capped] = requested[~capped]
    for _ in range(iterations):
        middle = (low + high) / 2
        candidate = probabilities(middle)
        failed = (_js_divergence(global_values, candidate) > js_cap) | (
            np.abs(candidate - global_values).max(axis=1)
            > probability_shift_cap
        )
        low = np.where(failed, low, middle)
        high = np.where(failed, middle, high)
    applied = np.where(capped, low, requested)
    output = validate_probability_matrix(probabilities(applied))
    reasons = [
        (
            "js_and_probability_shift"
            if js and shift
            else "js"
            if js
            else "probability_shift"
            if shift
            else None
        )
        for js, shift in zip(js_failed, shift_failed, strict=True)
    ]
    return output, applied, reasons


def diagnostics(actual: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    return evaluate_diagnostics(
        actual,
        probabilities,
        labels=range(len(PITCH_GROUPS)),
        names=[str(group) for group in PITCH_GROUPS],
        family_labels=PITCH_GROUP_FAMILY_LABELS,
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
        and candidate_metrics["hierarchicalAccuracy"]
        >= global_metrics["hierarchicalAccuracy"] - 0.005
        and not major_zero_recall
        and candidate_metrics["maxClassShareError"] <= 0.20
        and candidate_metrics["totalVariationDistance"] <= 0.20
        and candidate_metrics["maxClassCalibrationError"] <= 0.10
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
        and candidate_metrics["macroF1"] >= global_metrics["macroF1"] - 0.005
        and candidate_metrics["hierarchicalAccuracy"]
        >= global_metrics["hierarchicalAccuracy"] - 0.005
        and not major_zero_recall
        and candidate_metrics.get("maxClassShareError", float("inf")) <= 0.20
        and candidate_metrics.get("totalVariationDistance", float("inf")) <= 0.20
        and candidate_metrics.get("maxClassCalibrationError", float("inf"))
        <= 0.10
    )


def relative_pitcher_failure_reasons(
    global_metrics: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
) -> list[str]:
    reasons = []
    if candidate_metrics["logLoss"] >= global_metrics["logLoss"]:
        reasons.append("log_loss_not_improved")
    if candidate_metrics["accuracy"] < global_metrics["accuracy"] - 0.005:
        reasons.append("accuracy_drop_gt_0.5pp")
    if candidate_metrics["macroF1"] < global_metrics["macroF1"] - 0.005:
        reasons.append("macro_f1_drop_gt_0.5pp")
    if (
        candidate_metrics["hierarchicalAccuracy"]
        < global_metrics["hierarchicalAccuracy"] - 0.005
    ):
        reasons.append("hierarchical_accuracy_drop_gt_0.5pp")
    global_major_zero = {
        name
        for name in global_metrics["zeroRecallClasses"]
        if global_metrics["actualDistribution"][name] >= 0.05
    }
    candidate_major_zero = {
        name
        for name in candidate_metrics["zeroRecallClasses"]
        if candidate_metrics["actualDistribution"][name] >= 0.05
    }
    if candidate_major_zero - global_major_zero:
        reasons.append("new_major_zero_recall")
    for name, limit, reason in (
        ("maxClassShareError", 0.20, "share_error_worsened"),
        ("totalVariationDistance", 0.20, "tvd_worsened"),
        ("maxClassCalibrationError", 0.10, "calibration_error_worsened"),
    ):
        if (
            candidate_metrics[name] > limit
            and candidate_metrics[name] > global_metrics[name] + 1e-12
        ):
            reasons.append(reason)
    return reasons


def maximum_safe_scale_multiplier(
    actual: np.ndarray,
    global_probabilities: np.ndarray,
    correction: np.ndarray,
    base_scales: np.ndarray,
    *,
    multipliers: Sequence[float] = SAFE_SCALE_MULTIPLIERS,
) -> dict[str, object]:
    global_values = validate_probability_matrix(global_probabilities)
    residual_values = np.asarray(correction, dtype=float)
    scales = np.asarray(base_scales, dtype=float)
    if (
        len(actual) != len(global_values)
        or residual_values.shape != global_values.shape
        or scales.shape != (len(global_values),)
    ):
        raise ValueError("safe scale inputs are misaligned")
    candidates = sorted({float(value) for value in multipliers}, reverse=True)
    if not candidates or candidates[-1] <= 0 or candidates[0] > 1:
        raise ValueError("safe scale multipliers must be in (0, 1]")
    global_metrics = diagnostics(actual, global_values)
    last_reasons = ["log_loss_not_improved"]
    for multiplier in candidates:
        probabilities = apply_correction(
            global_values,
            residual_values,
            scales * multiplier,
        )
        metrics = diagnostics(actual, probabilities)
        reasons = relative_pitcher_failure_reasons(global_metrics, metrics)
        if not reasons:
            return {
                "maxSafeMultiplier": multiplier,
                "globalMetrics": global_metrics,
                "metrics": metrics,
                "failureReasons": [],
            }
        last_reasons = reasons
    return {
        "maxSafeMultiplier": 0.0,
        "globalMetrics": global_metrics,
        "metrics": global_metrics,
        "failureReasons": last_reasons,
    }


def provisional_scale(
    gap_days: int,
    *,
    base_reliability: float = 0.5,
    half_life_days: int = 365,
    cap: float = 0.15,
) -> float:
    if (
        gap_days < 0
        or half_life_days <= 0
        or not 0 <= cap <= 0.5
        or not 0 <= base_reliability <= 0.5
    ):
        raise ValueError("invalid provisional decay parameters")
    return float(
        min(cap, base_reliability)
        * np.exp(-np.log(2) * gap_days / half_life_days)
    )
