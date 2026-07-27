"""Train the frozen Global model and V7 incremental residual registry."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import traceback
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from .features import prepare_pitch_rows
from .hybrid import (
    RegistryEntry,
    apply_logit_bias,
    apply_reliability_gated_residual,
    fit_logit_bias,
    serialize_registry_entry,
    specialist_eligibility,
)
from .metrics import evaluate_diagnostics
from .modeling import (
    CandidateSpec,
    predict_candidate,
    train_candidate_with_tuning,
    train_final_candidate,
)
from .residual import (
    apply_correction,
    compute_pitcher_reliability,
    gate_targets,
    maximum_safe_scale_multiplier,
    pitcher_residual_passes,
    predict_context_gate,
    predict_correction,
    residual_passes,
    train_final_residual,
    train_gate,
    train_residual_with_tuning,
)
from .resources import assert_safe, snapshot
from .taxonomy import (
    PITCH_GROUP_FAMILY_LABELS,
    PITCH_GROUPS,
    family_names,
    group_families,
)

PILOT_PITCHERS = {
    543037: "Gerrit Cole",
    656427: "Jack Flaherty",
    595014: "Blake Treinen",
    592454: "Tommy Kahnle",
    641482: "Nestor Cortes",
}
GLOBAL_SPEC = CandidateSpec("global", "sqrt", 6, 8)
GLOBAL_TEMPERATURE = 1.0465
CALIBRATION_WEIGHTS = (0.0, 0.25, 0.5, 0.75)
MIN_2024_EVALUATION_PITCHES = 100
MIN_2025_EVALUATION_PITCHES = 300


def global_specs() -> tuple[CandidateSpec, ...]:
    return tuple(
        CandidateSpec("global", weight_mode, depth, 8)
        for weight_mode in ("none", "light", "sqrt")
        for depth in (4, 6)
    )


def specialist_specs() -> tuple[CandidateSpec, ...]:
    return tuple(
        CandidateSpec("global", weight_mode, depth, 8)
        for weight_mode in ("none", "sqrt")
        for depth in (3, 4)
    )


def validation_folds(
    rows: pd.DataFrame, evaluation_years: tuple[int, ...] = (2023, 2024, 2025)
) -> list[tuple[int, pd.Series, pd.Series]]:
    years = rows["game_date"].dt.year
    folds = []
    for year in evaluation_years:
        train = years < year
        evaluation = years == year
        if not train.any() or not evaluation.any():
            raise ValueError(f"missing train or evaluation rows for {year}")
        train_end = rows.loc[train, "game_date"].max()
        evaluation_start = rows.loc[evaluation, "game_date"].min()
        if train_end >= evaluation_start:
            raise AssertionError(f"validation fold overlaps for {year}")
        folds.append((year, train, evaluation))
    return folds


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle)
    temporary.replace(path)


def _load_statcast(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Statcast shards in {directory}")
    frames = [pd.read_parquet(path) for path in files]
    return prepare_pitch_rows(frames)


def _metrics(actual: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    return evaluate_diagnostics(
        actual,
        probabilities,
        labels=range(len(PITCH_GROUPS)),
        names=[str(group) for group in PITCH_GROUPS],
        family_labels=PITCH_GROUP_FAMILY_LABELS,
    )


def _calibration_bias(
    actual: np.ndarray,
    probabilities: np.ndarray,
    effective_weight: float,
) -> np.ndarray:
    if effective_weight == 0:
        return np.zeros(probabilities.shape[1], dtype=float)
    prior_strength = len(actual) * (1 - effective_weight) / effective_weight
    return fit_logit_bias(
        actual,
        probabilities,
        prior_strength=prior_strength,
    )


def _calibration_passes(
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
        and candidate_metrics["totalVariationDistance"]
        <= global_metrics["totalVariationDistance"]
        and candidate_metrics["maxClassCalibrationError"]
        <= global_metrics["maxClassCalibrationError"]
    )


def _select_calibration(
    actual_2023: np.ndarray,
    probabilities_2023: np.ndarray,
    actual_2024: np.ndarray,
    probabilities_2024: np.ndarray,
    actual_2025: np.ndarray,
    probabilities_2025: np.ndarray,
) -> dict[str, object]:
    base_2024 = _metrics(actual_2024, probabilities_2024)
    candidates = []
    for weight in CALIBRATION_WEIGHTS:
        bias = _calibration_bias(actual_2023, probabilities_2023, weight)
        candidate = apply_logit_bias(probabilities_2024, bias)
        metrics = _metrics(actual_2024, candidate)
        candidates.append(
            {
                "effectiveWeight": weight,
                "bias": bias.tolist(),
                "metrics": metrics,
                "accepted": weight > 0
                and _calibration_passes(base_2024, metrics),
            }
        )
    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    selected = (
        min(
            accepted,
            key=lambda candidate: (
                candidate["metrics"]["logLoss"],
                -candidate["metrics"]["macroF1"],
            ),
        )
        if accepted
        else candidates[0]
    )
    weight = float(selected["effectiveWeight"])
    combined_actual = np.concatenate([actual_2023, actual_2024])
    combined_probabilities = np.concatenate(
        [probabilities_2023, probabilities_2024]
    )
    validation_bias = _calibration_bias(
        combined_actual,
        combined_probabilities,
        weight,
    )
    validation_probabilities = apply_logit_bias(
        probabilities_2025,
        validation_bias,
    )
    validation_global = _metrics(actual_2025, probabilities_2025)
    validation_metrics = _metrics(actual_2025, validation_probabilities)
    validation_passed = weight > 0 and _calibration_passes(
        validation_global,
        validation_metrics,
    )
    if not validation_passed:
        weight = 0.0
        validation_bias = np.zeros(probabilities_2025.shape[1], dtype=float)
        validation_probabilities = probabilities_2025
        validation_metrics = validation_global
    return {
        "effectiveWeight": weight,
        "selectionGlobalMetrics": base_2024,
        "selection": selected,
        "candidates": candidates,
        "validationPassed": validation_passed,
        "validationGlobalMetrics": validation_global,
        "validationMetrics": validation_metrics,
        "validationBias": validation_bias.tolist(),
    }


def _date_partitions(
    rows: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = np.array(sorted(rows["game_date"].dt.normalize().unique()))
    if len(dates) < 10:
        raise ValueError("gate training requires at least ten game dates")
    first = dates[max(1, min(len(dates) - 2, int(len(dates) * 0.7)))]
    second = dates[max(2, min(len(dates) - 1, int(len(dates) * 0.8)))]
    normalized = rows["game_date"].dt.normalize().to_numpy()
    training = normalized < first
    tuning = (normalized >= first) & (normalized < second)
    evaluation = normalized >= second
    if not training.any() or not tuning.any() or not evaluation.any():
        raise ValueError("gate chronological partitions are empty")
    return training, tuning, evaluation


def _routing_registry(
    reliability: Mapping[int, Mapping[str, float | int]],
) -> dict[int, RegistryEntry]:
    return {
        pitcher_id: RegistryEntry(
            pitcher_id=pitcher_id,
            enabled=True,
            specialist_weight=0,
            model="pooled-residual.pkl",
            status="full",
            reliability=float(values["reliability"]),
            reliability_components=values,
        )
        for pitcher_id, values in reliability.items()
    }


def _pitcher_name(
    rows: pd.DataFrame,
    pitcher_id: int,
    pilot_pitchers: Mapping[int, str],
) -> str:
    if pitcher_id in pilot_pitchers:
        return pilot_pitchers[pitcher_id]
    if "player_name" in rows:
        values = rows.loc[
            rows["pitcher_id"].eq(pitcher_id),
            "player_name",
        ].dropna()
        if not values.empty:
            return str(values.iloc[-1])
    return f"선수 #{pitcher_id}"


def _reason(
    *,
    aggregate_passed: bool,
    support_2024: int,
    support_2025: int,
    validation_passed: bool,
    test_passed: bool,
) -> str:
    if not aggregate_passed:
        return "aggregate_gate_failed"
    if support_2024 < MIN_2024_EVALUATION_PITCHES:
        return "insufficient_2024_support"
    if not validation_passed:
        return "2024_player_gate_failed"
    if support_2025 and support_2025 < MIN_2025_EVALUATION_PITCHES:
        return "insufficient_2025_support"
    if support_2025 and not test_passed:
        return "2025_player_gate_failed"
    if support_2025 == 0:
        return "missing_2025_rows"
    return "inactive"


def _routing_summary(values: list[dict[str, object]]) -> dict[str, object]:
    scales = np.asarray(
        [float(value["effectiveScale"]) for value in values],
        dtype=float,
    )
    return {
        "rows": int(len(values)),
        "nonZero": int(np.count_nonzero(scales > 0)),
        "scale": {
            "p10": float(np.quantile(scales, 0.10)),
            "p50": float(np.quantile(scales, 0.50)),
            "p90": float(np.quantile(scales, 0.90)),
        },
        "capReasons": dict(
            sorted(
                Counter(
                    str(value["capReason"])
                    for value in values
                    if value["capReason"] is not None
                ).items()
            )
        ),
        "hardGateReasons": dict(
            sorted(
                Counter(
                    str(value["hardGateReason"])
                    for value in values
                    if value["hardGateReason"] is not None
                ).items()
            )
        ),
    }


def _gate_metrics(actual: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "logLoss": float(log_loss(actual, probabilities, labels=[0, 1])),
        "brierScore": float(brier_score_loss(actual, probabilities)),
        "positiveRate": float(np.mean(actual)),
        "meanProbability": float(np.mean(probabilities)),
    }


def _safe_scale_analysis(
    actual: np.ndarray,
    global_probabilities: np.ndarray,
    correction: np.ndarray,
    routing: list[dict[str, object]],
    positions: np.ndarray,
    *,
    min_support: int,
) -> dict[str, object]:
    support = int(len(positions))
    if support < min_support:
        return {
            "support": support,
            "maxSafeMultiplier": 0.0,
            "failureReasons": ["insufficient_support"],
            "metrics": None,
        }
    base_scales = np.asarray(
        [float(routing[position]["effectiveScale"]) for position in positions]
    )
    result = maximum_safe_scale_multiplier(
        actual[positions],
        global_probabilities[positions],
        correction[positions],
        base_scales,
    )
    multiplier = float(result["maxSafeMultiplier"])
    return {
        "support": support,
        "maxSafeMultiplier": multiplier,
        "failureReasons": result["failureReasons"],
        "metrics": result["metrics"],
        "effectiveScale": {
            "baseMean": float(np.mean(base_scales)),
            "safeMean": float(np.mean(base_scales * multiplier)),
            "safeMaximum": float(np.max(base_scales * multiplier)),
        },
    }


def train_hybrid(
    rows: pd.DataFrame,
    model_directory: Path,
    *,
    pilot_pitchers: Mapping[int, str] = PILOT_PITCHERS,
) -> dict[str, object]:
    folds = validation_folds(rows)
    global_predictions: dict[int, np.ndarray] = {}
    global_actual: dict[int, np.ndarray] = {}
    global_trees: list[int] = []
    evaluations: dict[int, pd.DataFrame] = {}
    pre_2025 = rows[rows["game_date"].dt.year < 2025]
    pitcher_ids = [int(value) for value in sorted(pre_2025["pitcher_id"].unique())]
    eligibility = {
        pitcher_id: specialist_eligibility(
            pre_2025[pre_2025["pitcher_id"] == pitcher_id]
        )
        for pitcher_id in pitcher_ids
    }
    pool = tuple(
        pitcher_id
        for pitcher_id, result in eligibility.items()
        if result["eligible"]
    )
    pool_set = set(pool)
    names = {
        pitcher_id: _pitcher_name(rows, pitcher_id, pilot_pitchers)
        for pitcher_id in pool
    }

    for year, train_mask, evaluation_mask in folds:
        assert_safe(snapshot(Path.cwd()))
        train = rows[train_mask]
        evaluation = rows[evaluation_mask]
        evaluations[year] = evaluation
        global_actual[year] = evaluation["target"].to_numpy()
        model, trees = train_candidate_with_tuning(train, GLOBAL_SPEC)
        global_predictions[year] = predict_candidate(
            model,
            evaluation,
            temperature=GLOBAL_TEMPERATURE,
        )
        global_trees.append(trees)
        del model
        gc.collect()

    calibration = _select_calibration(
        global_actual[2023],
        global_predictions[2023],
        global_actual[2024],
        global_predictions[2024],
        global_actual[2025],
        global_predictions[2025],
    )
    calibration_weight = float(calibration["effectiveWeight"])
    bias_2023 = _calibration_bias(
        global_actual[2023],
        global_predictions[2023],
        calibration_weight,
    )
    bias_2024 = _calibration_bias(
        np.concatenate([global_actual[2023], global_actual[2024]]),
        np.concatenate([global_predictions[2023], global_predictions[2024]]),
        calibration_weight,
    )
    calibrated_global = {
        2023: apply_logit_bias(global_predictions[2023], bias_2023),
        2024: apply_logit_bias(global_predictions[2024], bias_2023),
        2025: apply_logit_bias(global_predictions[2025], bias_2024),
    }

    pooled_rows = {
        year: evaluation[evaluation["pitcher_id"].isin(pool_set)]
        .copy()
        .reset_index(drop=True)
        for year, evaluation in evaluations.items()
    }
    pooled_positions = {
        year: np.flatnonzero(
            evaluations[year]["pitcher_id"].isin(pool_set).to_numpy()
        )
        for year in pooled_rows
    }
    pooled_global = {
        year: calibrated_global[year][pooled_positions[year]]
        for year in pooled_rows
    }

    residual_2023 = train_residual_with_tuning(
        pooled_rows[2023],
        pooled_global[2023],
    )
    residual_tree_count = residual_2023.tree_count
    correction_2024 = predict_correction(residual_2023, pooled_rows[2024])
    reference_2024 = apply_correction(
        pooled_global[2024],
        correction_2024,
        0.5,
    )
    del residual_2023
    gc.collect()

    train_mask, tune_mask, evaluation_mask = _date_partitions(pooled_rows[2024])
    fitted_gate = train_gate(
        pooled_rows[2024].loc[train_mask],
        pooled_global[2024][train_mask],
        correction_2024[train_mask],
        n_estimators=1_000,
        tuning=(
            pooled_rows[2024].loc[tune_mask],
            pooled_global[2024][tune_mask],
            correction_2024[tune_mask],
        ),
    )
    gate_tree_count = fitted_gate.tree_count
    del fitted_gate
    gc.collect()

    development_mask = train_mask | tune_mask
    gate_2024 = train_gate(
        pooled_rows[2024].loc[development_mask],
        pooled_global[2024][development_mask],
        correction_2024[development_mask],
        n_estimators=gate_tree_count,
    )
    reliability_2024_development = compute_pitcher_reliability(
        pooled_rows[2024].loc[development_mask],
        pooled_global[2024][development_mask],
        reference_2024[development_mask],
    )
    context_gate_2024 = predict_context_gate(
        gate_2024,
        pooled_rows[2024].loc[evaluation_mask],
        pooled_global[2024][evaluation_mask],
        correction_2024[evaluation_mask],
    )
    candidate_2024, _, routing_2024 = apply_reliability_gated_residual(
        pooled_rows[2024].loc[evaluation_mask],
        pooled_global[2024][evaluation_mask],
        correction_2024[evaluation_mask],
        context_gate_2024,
        _routing_registry(reliability_2024_development),
    )
    evaluation_rows_2024 = (
        pooled_rows[2024].loc[evaluation_mask].copy().reset_index(drop=True)
    )
    evaluation_global_2024 = pooled_global[2024][evaluation_mask]
    aggregate_2024_global = _metrics(
        evaluation_rows_2024["target"].to_numpy(),
        evaluation_global_2024,
    )
    aggregate_2024_candidate = _metrics(
        evaluation_rows_2024["target"].to_numpy(),
        candidate_2024,
    )
    aggregate_2024_passed = residual_passes(
        aggregate_2024_global,
        aggregate_2024_candidate,
    )
    gate_metrics_2024 = _gate_metrics(
        gate_targets(
            evaluation_rows_2024["target"].to_numpy(),
            evaluation_global_2024,
            correction_2024[evaluation_mask],
        ),
        context_gate_2024,
    )
    del gate_2024
    gc.collect()

    residual_2024 = train_final_residual(
        pd.concat([pooled_rows[2023], pooled_rows[2024]], ignore_index=True),
        np.concatenate([pooled_global[2023], pooled_global[2024]]),
        n_estimators=residual_tree_count,
    )
    correction_2025 = predict_correction(residual_2024, pooled_rows[2025])
    reference_2025 = apply_correction(
        pooled_global[2025],
        correction_2025,
        0.5,
    )
    gate_2025 = train_gate(
        pooled_rows[2024],
        pooled_global[2024],
        correction_2024,
        n_estimators=gate_tree_count,
    )
    reliability_2025 = compute_pitcher_reliability(
        pooled_rows[2024],
        pooled_global[2024],
        reference_2024,
    )
    context_gate_2025 = predict_context_gate(
        gate_2025,
        pooled_rows[2025],
        pooled_global[2025],
        correction_2025,
    )
    candidate_2025, _, routing_2025 = apply_reliability_gated_residual(
        pooled_rows[2025],
        pooled_global[2025],
        correction_2025,
        context_gate_2025,
        _routing_registry(reliability_2025),
    )
    aggregate_2025_global = _metrics(
        pooled_rows[2025]["target"].to_numpy(),
        pooled_global[2025],
    )
    aggregate_2025_candidate = _metrics(
        pooled_rows[2025]["target"].to_numpy(),
        candidate_2025,
    )
    aggregate_2025_passed = residual_passes(
        aggregate_2025_global,
        aggregate_2025_candidate,
    )
    gate_metrics_2025 = _gate_metrics(
        gate_targets(
            pooled_rows[2025]["target"].to_numpy(),
            pooled_global[2025],
            correction_2025,
        ),
        context_gate_2025,
    )
    aggregate_passed = aggregate_2024_passed and aggregate_2025_passed
    del gate_2025
    gc.collect()

    reliability_final = compute_pitcher_reliability(
        pd.concat([pooled_rows[2024], pooled_rows[2025]], ignore_index=True),
        np.concatenate([pooled_global[2024], pooled_global[2025]]),
        np.concatenate([reference_2024, reference_2025]),
    )
    validations: dict[str, object] = {}
    tiers: dict[int, str] = {}
    scale_multipliers: dict[int, float] = {}
    stale_ids: set[int] = set()
    recommended_tiers: Counter[str] = Counter()
    for pitcher_id in pool:
        positions_2024 = np.flatnonzero(
            evaluation_rows_2024["pitcher_id"].to_numpy() == pitcher_id
        )
        positions_2025 = np.flatnonzero(
            pooled_rows[2025]["pitcher_id"].to_numpy() == pitcher_id
        )
        if len(positions_2024):
            validation_passed, validation_global, validation_candidate = (
                pitcher_residual_passes(
                    evaluation_rows_2024["target"].to_numpy()[positions_2024],
                    evaluation_global_2024[positions_2024],
                    candidate_2024[positions_2024],
                    min_support=MIN_2024_EVALUATION_PITCHES,
                )
            )
        else:
            validation_passed = False
            validation_global = None
            validation_candidate = None
        if len(positions_2025):
            test_passed, test_global, test_candidate = pitcher_residual_passes(
                pooled_rows[2025]["target"].to_numpy()[positions_2025],
                pooled_global[2025][positions_2025],
                candidate_2025[positions_2025],
                min_support=MIN_2025_EVALUATION_PITCHES,
            )
        else:
            test_passed = False
            test_global = None
            test_candidate = None
        safe_2024 = _safe_scale_analysis(
            evaluation_rows_2024["target"].to_numpy(),
            evaluation_global_2024,
            correction_2024[evaluation_mask],
            routing_2024,
            positions_2024,
            min_support=MIN_2024_EVALUATION_PITCHES,
        )
        safe_2025 = _safe_scale_analysis(
            pooled_rows[2025]["target"].to_numpy(),
            pooled_global[2025],
            correction_2025,
            routing_2025,
            positions_2025,
            min_support=MIN_2025_EVALUATION_PITCHES,
        )
        safe_multiplier_2024 = float(safe_2024["maxSafeMultiplier"])
        safe_multiplier_2025 = float(safe_2025["maxSafeMultiplier"])
        conservative_multiplier = min(
            safe_multiplier_2024,
            safe_multiplier_2025,
        )
        if aggregate_passed and validation_passed and test_passed:
            recommended_tier = "full"
            registry_multiplier = 1.0
        elif aggregate_passed and conservative_multiplier > 0:
            recommended_tier = "limited"
            registry_multiplier = conservative_multiplier
        elif (
            aggregate_passed
            and not len(positions_2025)
            and safe_multiplier_2024 > 0
        ):
            recommended_tier = "limited"
            registry_multiplier = safe_multiplier_2024
            stale_ids.add(pitcher_id)
        else:
            recommended_tier = "shadow"
            registry_multiplier = 0.0
        tiers[pitcher_id] = recommended_tier
        scale_multipliers[pitcher_id] = registry_multiplier
        recommended_tiers[recommended_tier] += 1
        validations[str(pitcher_id)] = {
            "pitcherId": pitcher_id,
            "name": names[pitcher_id],
            "eligibility": eligibility[pitcher_id],
            "support": {
                "2024Evaluation": int(len(positions_2024)),
                "2025": int(len(positions_2025)),
            },
            "reliability": reliability_final.get(pitcher_id),
            "validationPassed": validation_passed,
            "testPassed": test_passed,
            "validationGlobalMetrics": validation_global,
            "validationMetrics": validation_candidate,
            "testGlobalMetrics": test_global,
            "testMetrics": test_candidate,
            "safeScale": {
                "2024": safe_2024,
                "2025": safe_2025,
                "conservativeMultiplier": conservative_multiplier,
                "recommendedTier": recommended_tier,
                "registryMultiplier": registry_multiplier,
                "stale": pitcher_id in stale_ids,
            },
        }

    registry: dict[int, RegistryEntry] = {}
    for pitcher_id in pool:
        record = validations[str(pitcher_id)]
        support_2024 = int(record["support"]["2024Evaluation"])
        support_2025 = int(record["support"]["2025"])
        status = tiers[pitcher_id]
        if status == "full":
            reason = None
        elif status == "limited":
            reason = (
                "stale_latest_season"
                if pitcher_id in stale_ids
                else "incremental_gate_only"
            )
        else:
            reason = _reason(
                aggregate_passed=aggregate_passed,
                support_2024=support_2024,
                support_2025=support_2025,
                validation_passed=bool(record["validationPassed"]),
                test_passed=bool(record["testPassed"]),
            )
        pitcher_rows = rows[rows["pitcher_id"] == pitcher_id]
        reliability = reliability_final.get(
            pitcher_id,
            {
                "n": 0,
                "supportCoefficient": 0.0,
                "pAll": 0.0,
                "pRecent": 0.0,
                "reliability": 0.0,
                "recentSupport": 0,
            },
        )
        enabled = status in {"full", "limited"}
        safe_scale = record["safeScale"]
        registry[pitcher_id] = RegistryEntry(
            pitcher_id=pitcher_id,
            enabled=enabled,
            specialist_weight=0,
            model="pooled-residual.pkl" if enabled else "",
            data_cutoff=pitcher_rows["game_date"].max().date().isoformat(),
            reason=reason,
            spec="v7-incremental-residual" if enabled else None,
            status=status,
            reliability=float(reliability["reliability"]),
            reliability_components=reliability,
            support={
                "2024Evaluation": support_2024,
                "2025": support_2025,
            },
            scale_multiplier=scale_multipliers[pitcher_id],
            stale=pitcher_id in stale_ids,
            incremental_validation={
                "2024": {
                    "support": safe_scale["2024"]["support"],
                    "maxSafeMultiplier": safe_scale["2024"][
                        "maxSafeMultiplier"
                    ],
                    "failureReasons": safe_scale["2024"]["failureReasons"],
                },
                "2025": {
                    "support": safe_scale["2025"]["support"],
                    "maxSafeMultiplier": safe_scale["2025"][
                        "maxSafeMultiplier"
                    ],
                    "failureReasons": safe_scale["2025"]["failureReasons"],
                },
                "strict2024Passed": bool(record["validationPassed"]),
                "strict2025Passed": bool(record["testPassed"]),
            },
        )

    final_actual = np.concatenate(
        [global_actual[year] for year in (2023, 2024, 2025)]
    )
    final_global_oof = np.concatenate(
        [global_predictions[year] for year in (2023, 2024, 2025)]
    )
    final_bias = _calibration_bias(
        final_actual,
        final_global_oof,
        calibration_weight,
    )
    final_pooled_global = {
        year: apply_logit_bias(
            global_predictions[year][pooled_positions[year]],
            final_bias,
        )
        for year in (2023, 2024, 2025)
    }
    residual_training_rows = pd.concat(
        [pooled_rows[year] for year in (2023, 2024, 2025)],
        ignore_index=True,
    )
    final_residual = train_final_residual(
        residual_training_rows,
        np.concatenate(
            [final_pooled_global[year] for year in (2023, 2024, 2025)]
        ),
        n_estimators=residual_tree_count,
    )
    _atomic_pickle(model_directory / "pooled-residual.pkl", final_residual)
    del final_residual, residual_2024
    gc.collect()

    final_gate = train_gate(
        pd.concat([pooled_rows[2024], pooled_rows[2025]], ignore_index=True),
        np.concatenate([pooled_global[2024], pooled_global[2025]]),
        np.concatenate([correction_2024, correction_2025]),
        n_estimators=gate_tree_count,
    )
    _atomic_pickle(model_directory / "context-gate.pkl", final_gate)
    del final_gate
    gc.collect()

    global_tree_count = int(np.median(global_trees))
    global_model = train_final_candidate(rows, GLOBAL_SPEC, global_tree_count)
    _atomic_pickle(model_directory / "global.pkl", global_model)
    del global_model
    gc.collect()

    result = {
        "schemaVersion": 6,
        "modelVersion": "V7",
        "deploymentStatus": "shadow",
        "pitchGroups": [str(group) for group in PITCH_GROUPS],
        "pitchFamilies": family_names(),
        "pitchGroupFamilies": group_families(),
        "generatedAt": datetime.now(UTC).isoformat(),
        "dataCutoff": rows["game_date"].max().date().isoformat(),
        "rows": len(rows),
        "global": {
            "model": "global.pkl",
            "selectedSpec": GLOBAL_SPEC.name,
            "treeCount": global_tree_count,
            "temperature": GLOBAL_TEMPERATURE,
            "features": "id-independent",
            "metrics": _metrics(
                global_actual[2025],
                calibrated_global[2025],
            ),
            "calibration": {
                **calibration,
                "type": "class-wise-logit-bias",
                "bias": final_bias.tolist(),
            },
        },
        "residual": {
            "model": "pooled-residual.pkl",
            "features": [
                "pitcher_id",
                "count_bucket",
                "stand",
                "pa_prev_pitch_1",
            ],
            "trainingPool": len(pool),
            "treeCount": residual_tree_count,
            "referenceScale": 0.5,
            "formula": (
                "0.5 * n/(n+1000) * min(pAll,pRecent) * contextGate "
                "* registryScaleMultiplier"
            ),
            "recentDays": 90,
            "bootstrapSamples": 500,
            "jsCap": 0.05,
            "probabilityShiftCap": 0.20,
            "selectionGlobalMetrics": aggregate_2024_global,
            "selectionMetrics": aggregate_2024_candidate,
            "selectionPassed": aggregate_2024_passed,
            "testGlobalMetrics": aggregate_2025_global,
            "testMetrics": aggregate_2025_candidate,
            "testPassed": aggregate_2025_passed,
            "fullCount": recommended_tiers["full"],
            "limitedCount": recommended_tiers["limited"],
            "shadowCount": recommended_tiers["shadow"],
            "enabledCount": (
                recommended_tiers["full"] + recommended_tiers["limited"]
            ),
            "safeScaleAnalysis": {
                "method": (
                    "largest 0.05-step multiplier of per-pitch V7 "
                    "scale that improves log loss without new relative harm"
                ),
                "selectionYears": [2024, 2025],
                "recommendedCounts": dict(sorted(recommended_tiers.items())),
                "productionRoutingChanged": True,
            },
            "routing2024": _routing_summary(routing_2024),
            "routing2025": _routing_summary(routing_2025),
        },
        "gate": {
            "model": "context-gate.pkl",
            "objective": "binary:logistic",
            "target": "global_to_reference_log_probability_gain_gt_zero",
            "treeCount": gate_tree_count,
            "identityFeatures": False,
            "trainingYears": [2024, 2025],
            "selectionMetrics": gate_metrics_2024,
            "testMetrics": gate_metrics_2025,
        },
        "specialists": {
            str(pitcher_id): serialize_registry_entry(
                entry,
                name=names[pitcher_id],
            )
            for pitcher_id, entry in registry.items()
        },
        "validation": validations,
    }
    _atomic_json(model_directory / "registry.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/statcast"))
    parser.add_argument("--models", type=Path, default=Path("models/v7"))
    parser.add_argument("--runs", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args()
    run = args.runs / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True, exist_ok=False)
    before = snapshot(Path.cwd())
    assert_safe(before)
    _atomic_json(run / "resources-before.json", asdict(before))
    try:
        rows = _load_statcast(args.data)
        result = train_hybrid(rows, args.models)
        _atomic_json(run / "result.json", result)
        _atomic_json(run / "resources-after.json", asdict(snapshot(Path.cwd())))
    except Exception as error:
        _atomic_json(
            run / "error.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
