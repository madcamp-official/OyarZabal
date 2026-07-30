"""Train and evaluate the V8.5 Sequence-adjusted Pitcher Personalizer."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .features import prepare_pitch_rows
from .metrics import (
    bootstrap_log_loss_gain,
    hierarchical_top_indices,
    validate_probability_matrix,
)
from .modeling import predict_candidate, train_candidate_with_tuning
from .residual import (
    RESIDUAL_SCALES,
    SAFE_SCALE_MULTIPLIERS,
    V85_RESIDUAL_FEATURES,
    apply_correction,
    apply_dynamic_correction,
    compute_pitcher_reliability,
    diagnostics,
    hard_safety_mask,
    predict_context_gate,
    predict_correction,
    train_final_residual,
    train_gate,
    train_residual_with_tuning,
)
from .resources import assert_safe, snapshot
from .sequence import SequenceExampleBuilder, SequenceVocabulary
from .taxonomy import PITCH_GROUP_FAMILY_LABELS, PITCH_GROUPS
from .training import GLOBAL_SPEC, GLOBAL_TEMPERATURE, _date_partitions
from .v8 import _fingerprint, _load_raw
from .v83 import _examples_for_context
from .v83_sequence import load_v83_expert, predict_v83_deltas
from .v84 import (
    CONFIRMATION_SEEDS,
    CandidateSpec,
    ObjectiveConfig,
    _fold,
)
from .v84_sequence import DistributionSafeTransform

FIXED_OBJECTIVE = ObjectiveConfig("FOCAL_1", 0.0, 1.0, True)
FIXED_SPEC = CandidateSpec("FOCAL_1", "none", 0.25)
OOF_YEARS = (2023, 2024, 2025)
MIN_PLAYER_SUPPORT = 100
SELECTION_TOLERANCE = 0.005
CONFIRMATION_TOLERANCE = 0.0025
JS_CAP = 0.05
PROBABILITY_SHIFT_CAP = 0.20
PITCHER_GAIN_SHARE_CAP = 0.50
CLASS_GAIN_SHARE_CAP = 0.70
REPRODUCTION_TOLERANCE = 0.0005


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _major_zero(metrics: dict[str, object]) -> set[str]:
    return {
        name
        for name in metrics["zeroRecallClasses"]
        if metrics["actualDistribution"][name] >= 0.05
    }


def _failure_reasons(
    base: dict[str, object],
    candidate: dict[str, object],
    *,
    tolerance: float,
) -> list[str]:
    reasons = []
    if candidate["logLoss"] >= base["logLoss"]:
        reasons.append("log_loss_not_improved")
    for key, label in (
        ("accuracy", "exact"),
        ("familyAccuracy", "family"),
        ("hierarchicalAccuracy", "hierarchical"),
        ("macroF1", "macro_f1"),
    ):
        if candidate[key] < base[key] - tolerance:
            reasons.append(f"{label}_drop_gt_{tolerance:g}")
    if candidate["totalVariationDistance"] > (
        base["totalVariationDistance"] + tolerance
    ):
        reasons.append(f"tvd_worse_gt_{tolerance:g}")
    if candidate["maxClassCalibrationError"] > (
        base["maxClassCalibrationError"] + tolerance
    ):
        reasons.append(f"calibration_worse_gt_{tolerance:g}")
    if _major_zero(candidate) - _major_zero(base):
        reasons.append("new_major_zero_recall")
    return reasons


def _player_failure_reasons(
    base: dict[str, object],
    candidate: dict[str, object],
) -> list[str]:
    reasons = []
    if candidate["logLoss"] >= base["logLoss"]:
        reasons.append("log_loss_not_improved")
    if candidate["accuracy"] < base["accuracy"] - 0.005:
        reasons.append("exact_drop_gt_0.005")
    if candidate["macroF1"] < base["macroF1"] - 0.005:
        reasons.append("macro_f1_drop_gt_0.005")
    if _major_zero(candidate) - _major_zero(base):
        reasons.append("new_major_zero_recall")
    for key, label in (
        ("totalVariationDistance", "tvd_worsened"),
        ("maxClassShareError", "share_error_worsened"),
        ("maxClassCalibrationError", "calibration_worsened"),
    ):
        if candidate[key] > base[key] + 1e-12:
            reasons.append(label)
    return reasons


def _assessment(
    rows: pd.DataFrame,
    base: np.ndarray,
    candidate: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, object]:
    actual = rows["target"].to_numpy(dtype=int)
    base_metrics = diagnostics(actual, base)
    metrics = diagnostics(actual, candidate)
    bootstrap = bootstrap_log_loss_gain(
        rows["game_pk"].to_numpy(),
        actual,
        base,
        candidate,
    )
    reasons = _failure_reasons(base_metrics, metrics, tolerance=tolerance)
    if bootstrap["ciLower"] <= 0:
        reasons.append("bootstrap_ci_lower_not_positive")
    return {
        "rows": len(rows),
        "baseMetrics": base_metrics,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "failureReasons": reasons,
        "accepted": not reasons,
    }


def _source_registry(path: Path) -> tuple[tuple[int, ...], dict[int, str], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    specialists = payload.get("specialists")
    if not isinstance(specialists, dict) or not specialists:
        raise ValueError("source registry has no pitcher pool")
    ids = tuple(sorted(int(value) for value in specialists))
    names = {
        int(pitcher_id): str(entry.get("name", f"선수 #{pitcher_id}"))
        for pitcher_id, entry in specialists.items()
    }
    return ids, names, _hash_file(path)


def _expected_metrics(path: Path) -> dict[int, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        year: {
            key: float(
                payload["folds"][str(year)]["selectedMetrics"][key]
            )
            for key in (
                "accuracy",
                "familyAccuracy",
                "hierarchicalAccuracy",
                "macroF1",
                "logLoss",
                "totalVariationDistance",
            )
        }
        for year in (2024, 2025)
    }


def _generate_oof(
    raw: pd.DataFrame,
    rows: pd.DataFrame,
    year: int,
    *,
    epochs: int,
    batch_size: int,
) -> dict[str, object]:
    fold = _fold(
        raw,
        rows,
        year,
        (FIXED_OBJECTIVE,),
        (FIXED_SPEC,),
        seeds=CONFIRMATION_SEEDS,
        epochs=epochs,
        batch_size=batch_size,
        keep_probabilities=True,
    )
    evaluation_rows = (
        rows.loc[rows["game_date"].dt.year.eq(year)]
        .copy()
        .reset_index(drop=True)
    )
    candidate = fold["candidates"][FIXED_SPEC.key]
    base = validate_probability_matrix(candidate.pop("probabilities"))
    global_probabilities = validate_probability_matrix(
        fold.pop("globalProbabilities")
    )
    if (
        len(evaluation_rows) != len(base)
        or fold["rowFingerprint"] != _fingerprint(evaluation_rows)
    ):
        raise AssertionError(f"V8.5 OOF alignment failed for {year}")
    result = {
        "year": year,
        "rows": evaluation_rows,
        "base": base.astype(np.float32),
        "global": global_probabilities.astype(np.float32),
        "metrics": candidate["normal"]["metrics"],
        "globalMetrics": fold["globalMetrics"],
        "rowFingerprint": fold["rowFingerprint"],
        "transforms": candidate["transforms"],
        "epochs": fold["tuningEpochs"]["FOCAL_1"],
    }
    del fold
    gc.collect()
    return result


def _assert_reproduction(
    folds: dict[int, dict[str, object]],
    expected: dict[int, dict[str, float]],
) -> dict[str, object]:
    differences = {}
    for year, metrics in expected.items():
        differences[str(year)] = {
            key: float(folds[year]["metrics"][key] - value)
            for key, value in metrics.items()
        }
        if any(
            abs(value) > REPRODUCTION_TOLERANCE
            for value in differences[str(year)].values()
        ):
            raise AssertionError(f"V8.4 reproduction failed for {year}")
    return {
        "tolerance": REPRODUCTION_TOLERANCE,
        "differences": differences,
        "passed": True,
    }


def _pool_positions(
    rows: pd.DataFrame,
    pitcher_ids: tuple[int, ...],
) -> np.ndarray:
    return np.flatnonzero(rows["pitcher_id"].isin(pitcher_ids).to_numpy())


def _raw_candidate(
    rows: pd.DataFrame,
    base: np.ndarray,
    correction: np.ndarray,
    pool_mask: np.ndarray,
    scale: float,
) -> np.ndarray:
    output = base.copy()
    output[pool_mask] = apply_correction(
        base[pool_mask],
        correction[pool_mask],
        scale,
    )
    return output


def _select_common_scale(
    rows: pd.DataFrame,
    base: np.ndarray,
    correction: np.ndarray,
) -> dict[str, object]:
    actual = rows["target"].to_numpy(dtype=int)
    base_metrics = diagnostics(actual, base)
    candidates = []
    for scale in RESIDUAL_SCALES:
        probabilities = apply_correction(base, correction, scale)
        metrics = diagnostics(actual, probabilities)
        reasons = []
        if metrics["logLoss"] >= base_metrics["logLoss"]:
            reasons.append("log_loss_not_improved")
        for key, label in (
            ("accuracy", "exact"),
            ("familyAccuracy", "family"),
            ("hierarchicalAccuracy", "hierarchical"),
            ("macroF1", "macro_f1"),
        ):
            if metrics[key] < base_metrics[key] - SELECTION_TOLERANCE:
                reasons.append(f"{label}_drop_gt_0.005")
        if _major_zero(metrics) - _major_zero(base_metrics):
            reasons.append("new_major_zero_recall")
        if metrics["maxClassShareError"] > 0.20:
            reasons.append("share_error_gt_0.20")
        if metrics["totalVariationDistance"] > 0.20:
            reasons.append("tvd_gt_0.20")
        if metrics["maxClassCalibrationError"] > 0.10:
            reasons.append("calibration_error_gt_0.10")
        candidates.append(
            {
                "scale": float(scale),
                "metrics": metrics,
                "failureReasons": reasons,
                "accepted": not reasons,
            }
        )
    improving = [
        item
        for item in candidates
        if item["metrics"]["logLoss"] < base_metrics["logLoss"]
    ]
    safe = [item for item in candidates if item["accepted"]]
    selected = (
        min(safe, key=lambda item: item["metrics"]["logLoss"])
        if safe
        else None
    )
    return {
        "rawImproved": bool(improving),
        "accepted": selected is not None,
        "scale": float(selected["scale"]) if selected else 0.0,
        "baseMetrics": base_metrics,
        "candidates": candidates,
    }


def _oof_cache(
    cache_directory: Path,
    raw: pd.DataFrame,
    rows: pd.DataFrame,
    year: int,
    *,
    epochs: int,
    batch_size: int,
) -> dict[str, object]:
    cache_directory.mkdir(parents=True, exist_ok=True)
    arrays_path = cache_directory / f"{year}.npz"
    metadata_path = cache_directory / f"{year}.json"
    evaluation_rows = (
        rows.loc[rows["game_date"].dt.year.eq(year)]
        .copy()
        .reset_index(drop=True)
    )
    fingerprint = _fingerprint(evaluation_rows)
    if arrays_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("rowFingerprint") == fingerprint
            and metadata.get("spec") == FIXED_SPEC.key
            and metadata.get("seeds") == list(CONFIRMATION_SEEDS)
            and metadata.get("epochsLimit") == epochs
        ):
            with np.load(arrays_path) as arrays:
                base = validate_probability_matrix(arrays["base"])
                global_probabilities = validate_probability_matrix(
                    arrays["global_probabilities"]
                )
            if len(base) == len(evaluation_rows):
                return {
                    "year": year,
                    "rows": evaluation_rows,
                    "base": base.astype(np.float32),
                    "global": global_probabilities.astype(np.float32),
                    "metrics": diagnostics(
                        evaluation_rows["target"].to_numpy(dtype=int),
                        base,
                    ),
                    "globalMetrics": diagnostics(
                        evaluation_rows["target"].to_numpy(dtype=int),
                        global_probabilities,
                    ),
                    "rowFingerprint": fingerprint,
                    "epochs": metadata["tuningEpochs"],
                    "cache": {
                        "reused": True,
                        "arraysHash": _hash_file(arrays_path),
                    },
                }
    fold = _generate_oof(
        raw,
        rows,
        year,
        epochs=epochs,
        batch_size=batch_size,
    )
    np.savez(
        arrays_path,
        base=fold["base"],
        global_probabilities=fold["global"],
    )
    metadata_path.write_text(
        json.dumps(
            {
                "rowFingerprint": fold["rowFingerprint"],
                "spec": FIXED_SPEC.key,
                "seeds": list(CONFIRMATION_SEEDS),
                "epochsLimit": epochs,
                "tuningEpochs": fold["epochs"],
            },
            indent=2,
        )
        + "\n"
    )
    fold["cache"] = {
        "reused": False,
        "arraysHash": _hash_file(arrays_path),
    }
    return fold


def _reliability_array(
    rows: pd.DataFrame,
    reliability: dict[int, dict[str, float | int]],
) -> np.ndarray:
    return np.asarray(
        [
            float(reliability.get(int(pitcher_id), {}).get("reliability", 0))
            for pitcher_id in rows["pitcher_id"]
        ],
        dtype=float,
    )


def _scale_inputs(
    rows: pd.DataFrame,
    reliability: dict[int, dict[str, float | int]],
    gate: np.ndarray,
    pool: set[int],
) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
    safety, reasons = hard_safety_mask(rows)
    in_pool = rows["pitcher_id"].isin(pool).to_numpy()
    reliability_values = _reliability_array(rows, reliability)
    scales = (
        safety.astype(float)
        * in_pool.astype(float)
        * reliability_values
        * np.asarray(gate, dtype=float)
    )
    return scales, reliability_values, reasons


def _safe_registry(
    rows: pd.DataFrame,
    base: np.ndarray,
    normalized_correction: np.ndarray,
    base_scales: np.ndarray,
    pitcher_ids: tuple[int, ...],
    names: dict[int, str],
) -> dict[int, dict[str, object]]:
    ids = rows["pitcher_id"].to_numpy(dtype=int)
    actual = rows["target"].to_numpy(dtype=int)
    registry = {}
    for pitcher_id in pitcher_ids:
        positions = np.flatnonzero(ids == pitcher_id)
        support = int(len(positions))
        if support < MIN_PLAYER_SUPPORT:
            registry[pitcher_id] = {
                "pitcherId": pitcher_id,
                "name": names[pitcher_id],
                "status": "shadow",
                "safeAlpha": 0.0,
                "support": support,
                "failureReasons": ["insufficient_support"],
            }
            continue
        base_metrics = diagnostics(actual[positions], base[positions])
        accepted = None
        last_reasons = ["log_loss_not_improved"]
        last_metrics = base_metrics
        for alpha in sorted(SAFE_SCALE_MULTIPLIERS, reverse=True):
            candidate, _, _ = apply_dynamic_correction(
                base[positions],
                normalized_correction[positions],
                base_scales[positions] * alpha,
                js_cap=JS_CAP,
                probability_shift_cap=PROBABILITY_SHIFT_CAP,
            )
            metrics = diagnostics(actual[positions], candidate)
            reasons = _player_failure_reasons(base_metrics, metrics)
            if not reasons:
                accepted = (float(alpha), metrics)
                break
            last_reasons, last_metrics = reasons, metrics
        alpha = accepted[0] if accepted else 0.0
        registry[pitcher_id] = {
            "pitcherId": pitcher_id,
            "name": names[pitcher_id],
            "status": (
                "full"
                if alpha == 1
                else "limited"
                if alpha > 0
                else "shadow"
            ),
            "safeAlpha": alpha,
            "support": support,
            "baseMetrics": base_metrics,
            "metrics": accepted[1] if accepted else last_metrics,
            "failureReasons": [] if accepted else last_reasons,
        }
    return registry


def _apply_final(
    rows: pd.DataFrame,
    base: np.ndarray,
    normalized_correction: np.ndarray,
    base_scales: np.ndarray,
    registry: dict[int, dict[str, object]],
    reliability_values: np.ndarray,
    gate: np.ndarray,
    hard_reasons: list[str | None],
) -> tuple[np.ndarray, dict[str, object], np.ndarray]:
    ids = rows["pitcher_id"].to_numpy(dtype=int)
    alpha = np.asarray(
        [
            float(registry.get(int(pitcher_id), {}).get("safeAlpha", 0))
            for pitcher_id in ids
        ]
    )
    requested = base_scales * alpha
    output, applied, cap_reasons = apply_dynamic_correction(
        base,
        normalized_correction,
        requested,
        js_cap=JS_CAP,
        probability_shift_cap=PROBABILITY_SHIFT_CAP,
    )
    inactive = requested == 0
    output[inactive] = base[inactive]
    if not np.array_equal(output[inactive], base[inactive]):
        raise AssertionError("V8.5 shadow fallback changed V8.4 probabilities")
    validate_probability_matrix(output)
    nonzero = applied > 0
    routing = {
        "interventionRows": int(nonzero.sum()),
        "interventionRate": float(nonzero.mean()),
        "effectiveScale": {
            "mean": float(applied.mean()),
            "median": float(np.median(applied)),
            "p90": float(np.quantile(applied, 0.9)),
        },
        "hardSafetyBlocked": int(
            sum(reason is not None for reason in hard_reasons)
        ),
        "shadowBlocked": int(
            sum(
                registry.get(int(pitcher_id), {}).get("status", "shadow")
                == "shadow"
                for pitcher_id in ids
            )
        ),
        "capCounts": {
            reason: cap_reasons.count(reason)
            for reason in (
                "js",
                "probability_shift",
                "js_and_probability_shift",
            )
        },
        "meanReliability": float(reliability_values.mean()),
        "meanContextGate": float(np.mean(gate)),
        "fallbackExact": True,
    }
    return output, routing, applied


def _stage_table(
    rows: pd.DataFrame,
    base: np.ndarray,
    correction: np.ndarray,
    common_scale: float,
    reliability: dict[int, dict[str, float | int]],
    gate: np.ndarray,
    registry: dict[int, dict[str, object]],
    pool: set[int],
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    pool_mask = rows["pitcher_id"].isin(pool).to_numpy()
    normalized = correction * common_scale
    ones = np.ones(len(rows))
    base_scales_c, reliability_values, hard_reasons = _scale_inputs(
        rows,
        reliability,
        ones,
        pool,
    )
    base_scales_d, _, _ = _scale_inputs(rows, reliability, gate, pool)
    b = _raw_candidate(rows, base, correction, pool_mask, common_scale)
    c = apply_correction(base, normalized, base_scales_c)
    c[base_scales_c == 0] = base[base_scales_c == 0]
    d = apply_correction(base, normalized, base_scales_d)
    d[base_scales_d == 0] = base[base_scales_d == 0]
    e, routing, applied = _apply_final(
        rows,
        base,
        normalized,
        base_scales_d,
        registry,
        reliability_values,
        gate,
        hard_reasons,
    )
    actual = rows["target"].to_numpy(dtype=int)
    stages = {
        name: diagnostics(actual, probabilities)
        for name, probabilities in (
            ("A_v84_base", base),
            ("B_raw_residual", b),
            ("C_reliability", c),
            ("D_context_gate", d),
            ("E_safe_alpha_caps", e),
        )
    }
    stages["E_safe_alpha_caps"]["routing"] = routing
    return stages, e, applied


def _cohort_assessment(
    rows: pd.DataFrame,
    base: np.ndarray,
    candidate: np.ndarray,
    applied: np.ndarray,
) -> dict[str, object]:
    positions = np.flatnonzero(applied > 0)
    if not len(positions):
        return {
            "rows": 0,
            "failureReasons": ["no_personalizer_interventions"],
            "accepted": False,
        }
    actual = rows["target"].to_numpy(dtype=int)[positions]
    base_metrics = diagnostics(actual, base[positions])
    metrics = diagnostics(actual, candidate[positions])
    reasons = []
    if metrics["logLoss"] >= base_metrics["logLoss"]:
        reasons.append("applied_cohort_log_loss_not_improved")
    return {
        "rows": int(len(positions)),
        "baseMetrics": base_metrics,
        "metrics": metrics,
        "failureReasons": reasons,
        "accepted": not reasons,
    }


def _gain_concentration(
    rows: pd.DataFrame,
    base: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, object]:
    actual = rows["target"].to_numpy(dtype=int)
    positions = np.arange(len(rows))
    gains = np.log(
        np.clip(candidate[positions, actual], 1e-12, 1)
    ) - np.log(np.clip(base[positions, actual], 1e-12, 1))
    positive = np.clip(gains, 0, None)
    total = float(positive.sum())

    def maximum_share(groups: np.ndarray) -> float:
        if total <= 0:
            return 1.0
        return float(
            max(positive[groups == value].sum() for value in set(groups))
            / total
        )

    pitcher_share = maximum_share(rows["pitcher_id"].to_numpy(dtype=int))
    class_share = maximum_share(actual)
    return {
        "largestPitcherPositiveGainShare": pitcher_share,
        "largestClassPositiveGainShare": class_share,
        "pitcherCap": PITCHER_GAIN_SHARE_CAP,
        "classCap": CLASS_GAIN_SHARE_CAP,
        "accepted": (
            pitcher_share <= PITCHER_GAIN_SHARE_CAP
            and class_share <= CLASS_GAIN_SHARE_CAP
        ),
    }


def _cancellation(
    rows: pd.DataFrame,
    global_probabilities: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, object]:
    actual = rows["target"].to_numpy(dtype=int)
    positions = np.arange(len(rows))
    sequence_utility = np.log(
        np.clip(base[positions, actual], 1e-12, 1)
    ) - np.log(
        np.clip(global_probabilities[positions, actual], 1e-12, 1)
    )
    personal_utility = np.log(
        np.clip(candidate[positions, actual], 1e-12, 1)
    ) - np.log(np.clip(base[positions, actual], 1e-12, 1))
    quadrant = np.select(
        [
            (sequence_utility > 0) & (personal_utility > 0),
            (sequence_utility > 0) & (personal_utility <= 0),
            (sequence_utility <= 0) & (personal_utility > 0),
        ],
        [
            "sequence_help_personal_help",
            "sequence_help_personal_hurt",
            "sequence_hurt_personal_repair",
        ],
        default="both_hurt",
    )
    sequence_delta = np.log(np.clip(base, 1e-12, 1)) - np.log(
        np.clip(global_probabilities, 1e-12, 1)
    )
    personal_delta = np.log(np.clip(candidate, 1e-12, 1)) - np.log(
        np.clip(base, 1e-12, 1)
    )
    denominator = np.linalg.norm(sequence_delta, axis=1) * np.linalg.norm(
        personal_delta, axis=1
    )
    cosine = np.divide(
        np.sum(sequence_delta * personal_delta, axis=1),
        denominator,
        out=np.zeros(len(rows)),
        where=denominator > 0,
    )
    global_top = hierarchical_top_indices(
        global_probabilities,
        PITCH_GROUP_FAMILY_LABELS,
    )
    base_top = hierarchical_top_indices(base, PITCH_GROUP_FAMILY_LABELS)
    final_top = hierarchical_top_indices(candidate, PITCH_GROUP_FAMILY_LABELS)
    sequence_changed = base_top != global_top
    reverted = sequence_changed & (final_top == global_top)

    def breakdown(groups: np.ndarray) -> dict[str, object]:
        return {
            str(value): {
                "rows": int(np.sum(groups == value)),
                "sequenceUtility": float(sequence_utility[groups == value].sum()),
                "personalUtility": float(personal_utility[groups == value].sum()),
            }
            for value in sorted(set(groups))
        }

    return {
        "quadrants": {
            name: int(np.sum(quadrant == name))
            for name in (
                "sequence_help_personal_help",
                "sequence_help_personal_hurt",
                "sequence_hurt_personal_repair",
                "both_hurt",
            )
        },
        "sequenceTop1ChangedRows": int(sequence_changed.sum()),
        "sequenceTop1RevertedRows": int(reverted.sum()),
        "sequenceTop1RevertedRate": float(
            reverted.sum() / max(1, sequence_changed.sum())
        ),
        "meanLogitDeltaCosine": float(np.mean(cosine)),
        "cancelledUtility": float(
            np.minimum(
                np.clip(sequence_utility, 0, None),
                np.clip(-personal_utility, 0, None),
            ).sum()
        ),
        "repairedUtility": float(
            np.minimum(
                np.clip(-sequence_utility, 0, None),
                np.clip(personal_utility, 0, None),
            ).sum()
        ),
        "byClass": breakdown(actual),
        "byPitcher": breakdown(rows["pitcher_id"].to_numpy(dtype=int)),
    }


def _pitch_mix_drift(rows: pd.DataFrame) -> dict[str, float]:
    season_columns = [f"season_rate_{group}" for group in PITCH_GROUPS]
    recent_columns = [f"recent_100_rate_{group}" for group in PITCH_GROUPS]
    if not set(season_columns + recent_columns).issubset(rows.columns):
        return {"rows": 0, "meanJs": 0.0, "p90Js": 0.0}
    season = rows[season_columns].to_numpy(dtype=float)
    recent = rows[recent_columns].to_numpy(dtype=float)
    midpoint = (season + recent) / 2
    js = 0.5 * (
        np.where(season > 0, season * np.log(season / midpoint), 0).sum(axis=1)
        + np.where(recent > 0, recent * np.log(recent / midpoint), 0).sum(axis=1)
    )
    finite = js[np.isfinite(js)]
    return {
        "rows": int(len(finite)),
        "meanJs": float(np.mean(finite)) if len(finite) else 0.0,
        "p90Js": float(np.quantile(finite, 0.9)) if len(finite) else 0.0,
    }


def _registry_summary(registry: dict[int, dict[str, object]]) -> dict[str, int]:
    return {
        status: sum(entry["status"] == status for entry in registry.values())
        for status in ("full", "limited", "shadow")
    }


def _public_registry(
    registry: dict[int, dict[str, object]],
    reliability: dict[int, dict[str, float | int]],
) -> dict[str, object]:
    return {
        str(pitcher_id): {
            **entry,
            "reliability": reliability.get(pitcher_id, {}),
        }
        for pitcher_id, entry in registry.items()
    }


def _joint_registry(
    first: dict[int, dict[str, object]],
    second: dict[int, dict[str, object]],
    names: dict[int, str],
) -> dict[int, dict[str, object]]:
    result = {}
    for pitcher_id in sorted(set(first) | set(second)):
        alpha = min(
            float(first.get(pitcher_id, {}).get("safeAlpha", 0)),
            float(second.get(pitcher_id, {}).get("safeAlpha", 0)),
        )
        result[pitcher_id] = {
            "pitcherId": pitcher_id,
            "name": names[pitcher_id],
            "status": (
                "full"
                if alpha == 1
                else "limited"
                if alpha > 0
                else "shadow"
            ),
            "safeAlpha": alpha,
            "support": {
                "2024": int(first.get(pitcher_id, {}).get("support", 0)),
                "2025": int(second.get(pitcher_id, {}).get("support", 0)),
            },
            "validationEvidence": {
                "2024": first.get(pitcher_id, {}),
                "2025": second.get(pitcher_id, {}),
            },
            "failureReasons": sorted(
                set(first.get(pitcher_id, {}).get("failureReasons", []))
                | set(second.get(pitcher_id, {}).get("failureReasons", []))
            ),
        }
    return result


def _v84_retrospective_base(
    training_raw: pd.DataFrame,
    holdout_directory: Path,
    refit_model_directory: Path,
    holdout_artifact_path: Path,
    *,
    batch_size: int,
    include_v9: bool = False,
) -> dict[str, object]:
    expected = json.loads(holdout_artifact_path.read_text())
    holdout_raw = _load_raw(holdout_directory)
    required_holdout_columns = {
        "fielder_2",
        "pfx_x",
        "pfx_z",
        "release_extension",
        "release_pos_x",
        "release_pos_z",
        "release_spin_rate",
    }
    missing_holdout_columns = required_holdout_columns - set(holdout_raw)
    if missing_holdout_columns:
        raise ValueError(
            "V8.4 retrospective requires extended Statcast columns: "
            f"{sorted(missing_holdout_columns)}"
        )
    holdout_raw["game_date"] = pd.to_datetime(
        holdout_raw["game_date"],
        errors="coerce",
    )
    if (
        training_raw["game_date"].dt.year.max() > 2025
        or set(holdout_raw["game_date"].dropna().dt.year.unique()) != {2026}
    ):
        raise ValueError("2026 retrospective date boundary failed")
    combined_raw = pd.concat([training_raw, holdout_raw], ignore_index=True)
    combined_rows = prepare_pitch_rows(
        [combined_raw],
        include_v83=True,
        include_v9=include_v9,
    )
    training_rows = combined_rows[
        combined_rows["game_date"].dt.year < 2026
    ]
    evaluation_rows = (
        combined_rows[combined_rows["game_date"].dt.year == 2026]
        .copy()
        .reset_index(drop=True)
    )
    global_model, trees = train_candidate_with_tuning(
        training_rows,
        GLOBAL_SPEC,
    )
    global_all = predict_candidate(
        global_model,
        combined_rows,
        GLOBAL_TEMPERATURE,
    ).astype(np.float32)
    del global_model
    gc.collect()

    with (refit_model_directory / "sequence-vocabulary.pkl").open("rb") as handle:
        vocabulary: SequenceVocabulary = pickle.load(handle)
    examples = SequenceExampleBuilder().build(combined_raw, vocabulary)
    if len(examples) != len(combined_rows):
        raise AssertionError("2026 V8.4 sequence alignment failed")
    examples = _examples_for_context(
        examples.with_repertoire(combined_rows),
        combined_rows,
        "catcher",
    )
    evaluation = np.flatnonzero(
        examples.target_dates.astype("datetime64[Y]").astype(int) + 1970
        == 2026
    )
    if len(evaluation) != len(evaluation_rows):
        raise AssertionError("2026 V8.4 evaluation rows differ")
    transforms = expected["candidate"]["transforms"]
    members = []
    checkpoint_hashes = {}
    for index, transform_payload in enumerate(transforms, 1):
        checkpoint = refit_model_directory / f"sequence-seed-{index}.pt"
        checkpoint_hashes[checkpoint.name] = _hash_file(checkpoint)
        fitted = load_v83_expert(checkpoint, device="cuda")
        family, child = predict_v83_deltas(
            fitted,
            examples,
            evaluation,
            global_all,
            batch_size=batch_size,
        )
        transform = DistributionSafeTransform(
            scale=float(transform_payload["scale"]),
            family_shrinkage=np.asarray(
                transform_payload["familyShrinkage"],
                dtype=float,
            ),
            child_shrinkage=np.asarray(
                transform_payload["childShrinkage"],
                dtype=float,
            ),
            calibration_mode=str(transform_payload["calibrationMode"]),
            calibration_parameters=np.asarray(
                transform_payload["calibrationParameters"],
                dtype=float,
            ),
        )
        members.append(
            transform.apply(global_all[evaluation], family, child)
        )
        del fitted, family, child
        gc.collect()
    base = validate_probability_matrix(np.mean(members, axis=0)).astype(
        np.float32
    )
    global_probabilities = global_all[evaluation]
    fingerprint = _fingerprint(evaluation_rows)
    metrics = diagnostics(
        evaluation_rows["target"].to_numpy(dtype=int),
        base,
    )
    expected_metrics = expected["candidate"]["normal"]["metrics"]
    if (
        fingerprint != expected["holdoutRowFingerprint"]
        or trees != expected["globalTrees"]
        or any(
            abs(metrics[key] - expected_metrics[key])
            > REPRODUCTION_TOLERANCE
            for key in (
                "accuracy",
                "familyAccuracy",
                "hierarchicalAccuracy",
                "macroF1",
                "logLoss",
                "totalVariationDistance",
            )
        )
    ):
        raise AssertionError("2026 V8.4 retrospective reproduction failed")
    return {
        "rows": evaluation_rows,
        "base": base,
        "global": global_probabilities,
        "metadata": {
            "trainingRange": expected["trainingRange"],
            "holdoutRange": expected["holdoutRange"],
            "holdoutRows": len(evaluation_rows),
            "rowFingerprint": fingerprint,
            "globalTrees": trees,
            "v84ArtifactHash": _hash_file(holdout_artifact_path),
            "v84ModelHashes": checkpoint_hashes,
            "vocabularyHash": _hash_file(
                refit_model_directory / "sequence-vocabulary.pkl"
            ),
            "2026UsedForTraining": False,
            "2026UsedForSelection": False,
            "historicallyObserved": True,
        },
    }


def _report(result: dict[str, object]) -> str:
    lines = [
        "# V8.5 Sequence-adjusted Pitcher Personalizer",
        "",
        f"- 판정: `{result['status']}`",
        f"- 최초 실패 단계: `{result.get('firstFailureStage', 'none')}`",
        f"- 제품 active: `{result['activeModel']}`",
        "",
    ]
    for year, evaluation in result.get("evaluations", {}).items():
        lines.extend(
            [
                f"## {year}",
                "",
                "| 단계 | Exact | Family | Hierarchical | Macro F1 | Log Loss | TVD |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, metrics in evaluation["stages"].items():
            lines.append(
                f"| {name} | {metrics['accuracy']:.2%} | "
                f"{metrics['familyAccuracy']:.2%} | "
                f"{metrics['hierarchicalAccuracy']:.2%} | "
                f"{metrics['macroF1']:.2%} | {metrics['logLoss']:.5f} | "
                f"{metrics['totalVariationDistance']:.2%} |"
            )
        assessment = evaluation.get("assessment", {})
        bootstrap = assessment.get("bootstrap", {})
        routing = evaluation["stages"].get("E_safe_alpha_caps", {}).get(
            "routing", {}
        )
        lines.extend(
            [
                "",
                f"- 판정: `{evaluation['accepted']}`",
                (
                    "- Log Loss 개선 및 경기 bootstrap 95% CI: "
                    f"`{bootstrap.get('meanGain', 0):.6f}` / "
                    f"`[{bootstrap.get('ciLower', 0):.6f}, "
                    f"{bootstrap.get('ciUpper', 0):.6f}]`"
                ),
                (
                    "- Personalizer 개입: "
                    f"`{routing.get('interventionRows', 0):,}`구 "
                    f"(`{routing.get('interventionRate', 0):.2%}`), "
                    "effective scale 평균/중앙/p90 "
                    f"`{routing.get('effectiveScale', {}).get('mean', 0):.4f}` / "
                    f"`{routing.get('effectiveScale', {}).get('median', 0):.4f}` / "
                    f"`{routing.get('effectiveScale', {}).get('p90', 0):.4f}`"
                ),
                "",
            ]
        )
    registry = result.get("registrySummary")
    if registry:
        lines.extend(
            [
                "## 최종 Registry",
                "",
                f"- full `{registry['full']}`명",
                f"- limited `{registry['limited']}`명",
                f"- shadow `{registry['shadow']}`명",
                "",
            ]
        )
    cancellation = result.get("evaluations", {}).get("2025", {}).get(
        "cancellation"
    )
    if cancellation:
        quadrants = cancellation["quadrants"]
        lines.extend(
            [
                "## 2025 Sequence 취소·복구 진단",
                "",
                (
                    "- Sequence 도움 / Personalizer 도움: "
                    f"`{quadrants['sequence_help_personal_help']:,}`구"
                ),
                (
                    "- Sequence 도움 / Personalizer 악화: "
                    f"`{quadrants['sequence_help_personal_hurt']:,}`구"
                ),
                (
                    "- Sequence 악화 / Personalizer 복구: "
                    f"`{quadrants['sequence_hurt_personal_repair']:,}`구"
                ),
                f"- 둘 다 악화: `{quadrants['both_hurt']:,}`구",
                (
                    "- Sequence Top-1 되돌림률: "
                    f"`{cancellation['sequenceTop1RevertedRate']:.3%}`"
                ),
                "",
            ]
        )
    retrospective = result.get("retrospective2026")
    if retrospective:
        retrospective_bootstrap = retrospective["diagnosticAssessment"][
            "bootstrap"
        ]
        retrospective_routing = retrospective["stages"][
            "E_safe_alpha_caps"
        ]["routing"]
        lines.extend(
            [
                "## 2026 Retrospective",
                "",
                "| 모델 | Exact | Family | Hierarchical | Macro F1 | Log Loss | TVD |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in ("A_v84_base", "E_safe_alpha_caps"):
            metrics = retrospective["stages"][name]
            lines.append(
                f"| {name} | {metrics['accuracy']:.2%} | "
                f"{metrics['familyAccuracy']:.2%} | "
                f"{metrics['hierarchicalAccuracy']:.2%} | "
                f"{metrics['macroF1']:.2%} | {metrics['logLoss']:.5f} | "
                f"{metrics['totalVariationDistance']:.2%} |"
            )
        lines.extend(
            [
                "",
                (
                    "- Log Loss 개선 및 경기 bootstrap 95% CI: "
                    f"`{retrospective_bootstrap['meanGain']:.6f}` / "
                    f"`[{retrospective_bootstrap['ciLower']:.6f}, "
                    f"{retrospective_bootstrap['ciUpper']:.6f}]`"
                ),
                (
                    "- Personalizer 개입: "
                    f"`{retrospective_routing['interventionRows']:,}`구 "
                    f"(`{retrospective_routing['interventionRate']:.2%}`)"
                ),
                "",
                (
                    "이 구간은 이미 관찰된 historical retrospective이며 "
                    "학습·선택에 사용하지 않았다."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## 해석",
            "",
            str(result["conclusion"]),
            "",
            "V7.2 active와 replay 라우팅은 변경하지 않았다.",
            "",
        ]
    )
    return "\n".join(lines)


def _save_result(
    result: dict[str, object],
    artifact_directory: Path,
    report_path: Path,
) -> None:
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(result))


def run(
    data_directory: Path,
    source_registry_path: Path,
    v84_artifact_path: Path,
    artifact_directory: Path,
    report_path: Path,
    *,
    epochs: int = 6,
    batch_size: int = 8192,
) -> dict[str, object]:
    before = asdict(snapshot(Path.cwd()))
    assert_safe(snapshot(Path.cwd()))
    raw = _load_raw(data_directory)
    raw["game_date"] = pd.to_datetime(raw["game_date"], errors="coerce")
    rows = prepare_pitch_rows([raw], include_v83=True)
    pitcher_ids, names, source_registry_hash = _source_registry(
        source_registry_path
    )
    expected = _expected_metrics(v84_artifact_path)
    configuration = {
        "model": "V8.5",
        "base": FIXED_SPEC.key,
        "sequenceArchitectureFrozen": True,
        "residualFeatures": list(V85_RESIDUAL_FEATURES),
        "residualScales": list(RESIDUAL_SCALES),
        "safeAlphaGrid": [0.0, *SAFE_SCALE_MULTIPLIERS],
        "effectiveScale": (
            "hardSafety * safeAlpha[pitcher] * "
            "reliability[pitcher] * contextGate[pitch]"
        ),
        "jsCap": JS_CAP,
        "classProbabilityShiftCap": PROBABILITY_SHIFT_CAP,
        "minimumPlayerSupport": MIN_PLAYER_SUPPORT,
        "selectionTolerance": SELECTION_TOLERANCE,
        "confirmationTolerance": CONFIRMATION_TOLERANCE,
        "seeds": list(CONFIRMATION_SEEDS),
        "epochs": epochs,
        "batchSize": batch_size,
        "pitcherPoolSourceHash": source_registry_hash,
        "v84ArtifactHash": _hash_file(v84_artifact_path),
        "registryMultiplier": None,
    }
    result: dict[str, object] = {
        "schemaVersion": "8.5",
        "modelVersion": "V8.5-sequence-adjusted-pitcher-personalizer",
        "generatedAt": datetime.now(UTC).isoformat(),
        "status": "running",
        "deploymentStatus": "shadow-only",
        "activeModel": "V7.2",
        "configuration": configuration,
        "configurationHash": _hash_json(configuration),
        "poolSize": len(pitcher_ids),
        "oof": {},
        "evaluations": {},
        "resourceBefore": before,
    }

    folds = {}
    try:
        for year in OOF_YEARS:
            folds[year] = _oof_cache(
                Path("models/v8.5-oof"),
                raw,
                rows,
                year,
                epochs=epochs,
                batch_size=batch_size,
            )
            result["oof"][str(year)] = {
                "rows": len(folds[year]["rows"]),
                "rowFingerprint": folds[year]["rowFingerprint"],
                "metrics": folds[year]["metrics"],
                "globalMetrics": folds[year]["globalMetrics"],
                "epochs": folds[year]["epochs"],
                "cache": folds[year]["cache"],
            }
            print(f"V8.5 OOF {year} complete", flush=True)
        result["v84Reproduction"] = _assert_reproduction(folds, expected)
    except (AssertionError, ValueError) as error:
        result.update(
            {
                "status": "invalid-experiment",
                "firstFailureStage": "oof-reproduction",
                "conclusion": str(error),
                "resourceAfter": asdict(snapshot(Path.cwd())),
            }
        )
        _save_result(result, artifact_directory, report_path)
        return result

    evaluations = {
        year: folds[year]["rows"].reset_index(drop=True)
        for year in OOF_YEARS
    }
    base = {year: folds[year]["base"] for year in OOF_YEARS}
    pool = set(pitcher_ids)
    pool_positions = {
        year: _pool_positions(evaluations[year], pitcher_ids)
        for year in OOF_YEARS
    }
    pooled_rows = {
        year: evaluations[year].iloc[pool_positions[year]].reset_index(drop=True)
        for year in OOF_YEARS
    }
    pooled_base = {
        year: base[year][pool_positions[year]]
        for year in OOF_YEARS
    }

    residual_2023 = train_residual_with_tuning(
        pooled_rows[2023],
        pooled_base[2023],
        feature_names=V85_RESIDUAL_FEATURES,
    )
    tree_count = residual_2023.tree_count
    correction_2024 = predict_correction(residual_2023, evaluations[2024])
    pooled_correction_2024 = correction_2024[pool_positions[2024]]
    training, tuning, registry_evaluation = _date_partitions(pooled_rows[2024])
    common_scale = _select_common_scale(
        pooled_rows[2024].loc[tuning].reset_index(drop=True),
        pooled_base[2024][tuning],
        pooled_correction_2024[tuning],
    )
    result["commonScaleSelection"] = common_scale
    if not common_scale["rawImproved"] or not common_scale["accepted"]:
        result.update(
            {
                "status": "rejected",
                "firstFailureStage": (
                    "B_raw_residual"
                    if not common_scale["rawImproved"]
                    else "common_scale"
                ),
                "conclusion": (
                    "2024 중간 구간에서 V8.4보다 안전하게 개선되는 "
                    "pooled Residual scale이 없었다."
                ),
                "resourceAfter": asdict(snapshot(Path.cwd())),
            }
        )
        _save_result(result, artifact_directory, report_path)
        return result
    scale = float(common_scale["scale"])

    tuned_gate = train_gate(
        pooled_rows[2024].loc[training],
        pooled_base[2024][training],
        pooled_correction_2024[training],
        n_estimators=1_000,
        tuning=(
            pooled_rows[2024].loc[tuning],
            pooled_base[2024][tuning],
            pooled_correction_2024[tuning],
        ),
        reference_scale=scale,
    )
    gate_tree_count = tuned_gate.tree_count
    development = training | tuning
    gate_2024 = train_gate(
        pooled_rows[2024].loc[development],
        pooled_base[2024][development],
        pooled_correction_2024[development],
        n_estimators=gate_tree_count,
        reference_scale=scale,
    )
    reference_2024 = apply_correction(
        pooled_base[2024],
        pooled_correction_2024,
        scale,
    )
    reliability_2024 = compute_pitcher_reliability(
        pooled_rows[2024].loc[development].reset_index(drop=True),
        pooled_base[2024][development],
        reference_2024[development],
    )
    evaluation_start = pooled_rows[2024].loc[
        registry_evaluation, "game_date"
    ].min()
    final_2024_mask = (
        evaluations[2024]["game_date"] >= evaluation_start
    ).to_numpy()
    final_rows_2024 = (
        evaluations[2024].loc[final_2024_mask].reset_index(drop=True)
    )
    final_base_2024 = base[2024][final_2024_mask]
    final_correction_2024 = correction_2024[final_2024_mask]
    context_gate_2024 = predict_context_gate(
        gate_2024,
        final_rows_2024,
        final_base_2024,
        final_correction_2024,
        reference_scale=scale,
    )
    base_scales_2024, reliability_values_2024, hard_reasons_2024 = (
        _scale_inputs(
            final_rows_2024,
            reliability_2024,
            context_gate_2024,
            pool,
        )
    )
    registry_2024 = _safe_registry(
        final_rows_2024,
        final_base_2024,
        final_correction_2024 * scale,
        base_scales_2024,
        pitcher_ids,
        names,
    )
    stages_2024, final_2024, applied_2024 = _stage_table(
        final_rows_2024,
        final_base_2024,
        final_correction_2024,
        scale,
        reliability_2024,
        context_gate_2024,
        registry_2024,
        pool,
    )
    assessment_2024 = _assessment(
        final_rows_2024,
        final_base_2024,
        final_2024,
        tolerance=SELECTION_TOLERANCE,
    )
    cohort_2024 = _cohort_assessment(
        final_rows_2024,
        final_base_2024,
        final_2024,
        applied_2024,
    )
    accepted_2024 = assessment_2024["accepted"] and cohort_2024["accepted"]
    result["evaluations"]["2024"] = {
        "range": {
            "start": final_rows_2024["game_date"].min().date().isoformat(),
            "end": final_rows_2024["game_date"].max().date().isoformat(),
        },
        "stages": stages_2024,
        "assessment": assessment_2024,
        "appliedCohort": cohort_2024,
        "pitchMixDriftDiagnostic": _pitch_mix_drift(final_rows_2024),
        "registrySummary": _registry_summary(registry_2024),
        "accepted": accepted_2024,
    }
    if not accepted_2024:
        result.update(
            {
                "status": "rejected",
                "firstFailureStage": "E_2024_safety",
                "registry": _public_registry(
                    registry_2024,
                    reliability_2024,
                ),
                "conclusion": (
                    "E가 2024 최종 구간 안전 게이트를 통과하지 못해 "
                    "2025를 열지 않았다."
                ),
                "resourceAfter": asdict(snapshot(Path.cwd())),
            }
        )
        _save_result(result, artifact_directory, report_path)
        return result

    residual_2025 = train_final_residual(
        pd.concat([pooled_rows[2023], pooled_rows[2024]], ignore_index=True),
        np.concatenate([pooled_base[2023], pooled_base[2024]]),
        n_estimators=tree_count,
        feature_names=V85_RESIDUAL_FEATURES,
    )
    correction_2025 = predict_correction(residual_2025, evaluations[2025])
    gate_2025 = train_gate(
        pooled_rows[2024],
        pooled_base[2024],
        pooled_correction_2024,
        n_estimators=gate_tree_count,
        reference_scale=scale,
    )
    reliability_2025 = compute_pitcher_reliability(
        pooled_rows[2024],
        pooled_base[2024],
        reference_2024,
    )
    context_gate_2025 = predict_context_gate(
        gate_2025,
        evaluations[2025],
        base[2025],
        correction_2025,
        reference_scale=scale,
    )
    stages_2025, final_2025, applied_2025 = _stage_table(
        evaluations[2025],
        base[2025],
        correction_2025,
        scale,
        reliability_2025,
        context_gate_2025,
        registry_2024,
        pool,
    )
    assessment_2025 = _assessment(
        evaluations[2025],
        base[2025],
        final_2025,
        tolerance=CONFIRMATION_TOLERANCE,
    )
    cohort_2025 = _cohort_assessment(
        evaluations[2025],
        base[2025],
        final_2025,
        applied_2025,
    )
    concentration = _gain_concentration(
        evaluations[2025],
        base[2025],
        final_2025,
    )
    global_metrics = diagnostics(
        evaluations[2025]["target"].to_numpy(dtype=int),
        folds[2025]["global"],
    )
    final_metrics = stages_2025["E_safe_alpha_caps"]
    global_reasons = []
    if final_metrics["accuracy"] < global_metrics["accuracy"] - 0.005:
        global_reasons.append("exact_below_global_gt_0.005")
    if final_metrics["macroF1"] < global_metrics["macroF1"] - 0.005:
        global_reasons.append("macro_f1_below_global_gt_0.005")
    accepted_2025 = bool(
        assessment_2025["accepted"]
        and cohort_2025["accepted"]
        and concentration["accepted"]
        and not global_reasons
    )
    result["evaluations"]["2025"] = {
        "range": {
            "start": evaluations[2025]["game_date"].min().date().isoformat(),
            "end": evaluations[2025]["game_date"].max().date().isoformat(),
        },
        "stages": stages_2025,
        "assessment": assessment_2025,
        "appliedCohort": cohort_2025,
        "gainConcentration": concentration,
        "globalComparison": {
            "globalMetrics": global_metrics,
            "failureReasons": global_reasons,
        },
        "pitchMixDriftDiagnostic": _pitch_mix_drift(evaluations[2025]),
        "cancellation": _cancellation(
            evaluations[2025],
            folds[2025]["global"],
            base[2025],
            final_2025,
        ),
        "accepted": accepted_2025,
    }
    result["modelMetadata"] = {
        "residualTreeCount": tree_count,
        "gateTreeCount": gate_tree_count,
        "commonResidualScale": scale,
        "residualFeatures": list(V85_RESIDUAL_FEATURES),
        "gateUsesPitcherId": False,
        "registryHasMultiplier": False,
    }
    if not accepted_2025:
        result.update(
            {
                "status": "rejected",
                "firstFailureStage": "2025_locked_confirmation",
                "conclusion": (
                    "고정된 V8.5가 2025 confirmation 조건을 모두 "
                    "통과하지 못했다. 사후 튜닝과 2026 평가는 수행하지 않았다."
                ),
                "resourceAfter": asdict(snapshot(Path.cwd())),
            }
        )
        _save_result(result, artifact_directory, report_path)
        return result

    base_scales_2025, _, _ = _scale_inputs(
        evaluations[2025],
        reliability_2025,
        context_gate_2025,
        pool,
    )
    registry_2025 = _safe_registry(
        evaluations[2025],
        base[2025],
        correction_2025 * scale,
        base_scales_2025,
        pitcher_ids,
        names,
    )
    final_registry = _joint_registry(
        registry_2024,
        registry_2025,
        names,
    )
    final_training_rows = pd.concat(
        [pooled_rows[year] for year in OOF_YEARS],
        ignore_index=True,
    )
    final_training_base = np.concatenate(
        [pooled_base[year] for year in OOF_YEARS]
    )
    final_residual = train_final_residual(
        final_training_rows,
        final_training_base,
        n_estimators=tree_count,
        feature_names=V85_RESIDUAL_FEATURES,
    )
    final_gate_rows = pd.concat(
        [pooled_rows[2024], pooled_rows[2025]],
        ignore_index=True,
    )
    final_gate_base = np.concatenate(
        [pooled_base[2024], pooled_base[2025]]
    )
    final_gate_correction = np.concatenate(
        [pooled_correction_2024, correction_2025[pool_positions[2025]]]
    )
    final_gate = train_gate(
        final_gate_rows,
        final_gate_base,
        final_gate_correction,
        n_estimators=gate_tree_count,
        reference_scale=scale,
    )
    final_reference = apply_correction(
        final_gate_base,
        final_gate_correction,
        scale,
    )
    final_reliability = compute_pitcher_reliability(
        final_gate_rows,
        final_gate_base,
        final_reference,
    )
    result["registry"] = _public_registry(
        final_registry,
        final_reliability,
    )
    result["registrySummary"] = _registry_summary(final_registry)

    model_directory = Path("models/v8.5")
    model_directory.mkdir(parents=True, exist_ok=True)
    with (model_directory / "pooled-residual.pkl").open("wb") as handle:
        pickle.dump(final_residual, handle)
    with (model_directory / "context-gate.pkl").open("wb") as handle:
        pickle.dump(final_gate, handle)
    (model_directory / "registry.json").write_text(
        json.dumps(result["registry"], ensure_ascii=False, indent=2) + "\n"
    )
    model_hashes = {
        path.name: _hash_file(path)
        for path in (
            model_directory / "pooled-residual.pkl",
            model_directory / "context-gate.pkl",
            model_directory / "registry.json",
        )
    }
    result["modelMetadata"]["hashes"] = model_hashes

    result.update(
        {
            "status": "research-passed",
            "deploymentStatus": "prospective-pending",
            "conclusion": (
                "V8.4 위 Personalizer가 2024·2025 안전 조건을 통과했다. "
                "V7.2 active는 유지하며 2026 retrospective와 prospective "
                "freeze는 후속 고정 진단으로만 수행한다."
            ),
            "prospectiveFreeze": {
                "freezeDate": datetime.now(UTC).date().isoformat(),
                "startDate": (
                    pd.Timestamp(datetime.now(UTC).date()) + pd.Timedelta(days=1)
                ).date().isoformat(),
                "minimumDays": 30,
                "minimumMlbPitches": 100_000,
                "minimumPersonalizerInterventions": 15_000,
                "looks": 1,
                "activeModel": "V7.2",
                "configurationHash": result["configurationHash"],
                "modelHashes": model_hashes,
                "registryHash": model_hashes["registry.json"],
            },
            "retrospectiveStatus": "pending",
            "resourceAfter": asdict(snapshot(Path.cwd())),
        }
    )
    _save_result(result, artifact_directory, report_path)
    return result


def run_retrospective(
    data_directory: Path,
    holdout_directory: Path,
    v84_refit_model_directory: Path,
    v84_holdout_artifact_path: Path,
    artifact_directory: Path,
    report_path: Path,
    *,
    batch_size: int = 8192,
) -> dict[str, object]:
    result_path = artifact_directory / "result.json"
    result = json.loads(result_path.read_text())
    if result.get("status") != "research-passed":
        raise ValueError("2026 retrospective requires passed 2025 confirmation")
    model_directory = Path("models/v8.5")
    registry_path = model_directory / "registry.json"
    public_registry = json.loads(registry_path.read_text())
    registry = {
        int(pitcher_id): {
            "pitcherId": int(pitcher_id),
            "status": entry["status"],
            "safeAlpha": float(entry["safeAlpha"]),
        }
        for pitcher_id, entry in public_registry.items()
    }
    reliability = {
        int(pitcher_id): entry.get("reliability", {})
        for pitcher_id, entry in public_registry.items()
    }
    pool = set(registry)
    with (model_directory / "pooled-residual.pkl").open("rb") as handle:
        final_residual = pickle.load(handle)
    with (model_directory / "context-gate.pkl").open("rb") as handle:
        final_gate = pickle.load(handle)
    training_raw = _load_raw(data_directory)
    training_raw["game_date"] = pd.to_datetime(
        training_raw["game_date"],
        errors="coerce",
    )
    retrospective_base = _v84_retrospective_base(
        training_raw,
        holdout_directory,
        v84_refit_model_directory,
        v84_holdout_artifact_path,
        batch_size=batch_size,
    )
    rows = retrospective_base["rows"]
    base = retrospective_base["base"]
    correction = predict_correction(final_residual, rows)
    common_scale = float(result["modelMetadata"]["commonResidualScale"])
    gate = predict_context_gate(
        final_gate,
        rows,
        base,
        correction,
        reference_scale=common_scale,
    )
    stages, final, applied = _stage_table(
        rows,
        base,
        correction,
        common_scale,
        reliability,
        gate,
        registry,
        pool,
    )
    result["retrospective2026"] = {
        **retrospective_base["metadata"],
        "stages": stages,
        "diagnosticAssessment": _assessment(
            rows,
            base,
            final,
            tolerance=CONFIRMATION_TOLERANCE,
        ),
        "appliedCohort": _cohort_assessment(
            rows,
            base,
            final,
            applied,
        ),
        "gainConcentration": _gain_concentration(rows, base, final),
        "cancellation": _cancellation(
            rows,
            retrospective_base["global"],
            base,
            final,
        ),
        "selectionImpact": "none-retrospective-only",
    }
    model_hashes = {
        path.name: _hash_file(path)
        for path in (
            model_directory / "pooled-residual.pkl",
            model_directory / "context-gate.pkl",
            registry_path,
        )
    }
    result["registry"] = public_registry
    result["registrySummary"] = _registry_summary(registry)
    result["modelMetadata"]["hashes"] = model_hashes
    result["prospectiveFreeze"] = {
        "freezeDate": datetime.now(UTC).date().isoformat(),
        "startDate": (
            pd.Timestamp(datetime.now(UTC).date()) + pd.Timedelta(days=1)
        ).date().isoformat(),
        "minimumDays": 30,
        "minimumMlbPitches": 100_000,
        "minimumPersonalizerInterventions": 15_000,
        "looks": 1,
        "activeModel": "V7.2",
        "configurationHash": result["configurationHash"],
        "modelHashes": model_hashes,
        "registryHash": model_hashes["registry.json"],
        "retrospectiveFingerprint": retrospective_base["metadata"][
            "rowFingerprint"
        ],
    }
    result["retrospectiveStatus"] = "complete"
    result["resourceAfterRetrospective"] = asdict(snapshot(Path.cwd()))
    _save_result(result, artifact_directory, report_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--v84-artifact", type=Path, required=True)
    parser.add_argument("--holdout-data", type=Path)
    parser.add_argument("--v84-refit-models", type=Path)
    parser.add_argument("--v84-holdout-artifact", type=Path)
    parser.add_argument("--retrospective-only", action="store_true")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts/v8.5-sequence-personalizer"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/2026-07-29-v8.5-sequence-personalizer.md"),
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    if args.retrospective_only:
        if any(
            value is None
            for value in (
                args.holdout_data,
                args.v84_refit_models,
                args.v84_holdout_artifact,
            )
        ):
            parser.error("retrospective inputs are required")
        result = run_retrospective(
            args.data,
            args.holdout_data,
            args.v84_refit_models,
            args.v84_holdout_artifact,
            args.artifacts,
            args.report,
            batch_size=args.batch_size,
        )
    else:
        result = run(
            args.data,
            args.source_registry,
            args.v84_artifact,
            args.artifacts,
            args.report,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "firstFailureStage": result.get("firstFailureStage"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
