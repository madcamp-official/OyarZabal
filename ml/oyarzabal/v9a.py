"""Evaluate an ID-free current-game state expert on the frozen V8.4 base."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .features import (
    V9_GAME_STATE_FEATURES,
    V9_PHYSICAL_FEATURES,
    V9_STRATEGY_FEATURES,
    prepare_pitch_rows,
)
from .metrics import bootstrap_log_loss_gain, validate_probability_matrix
from .residual import (
    _base_margin,
    _best_tree_count,
    _fit,
    apply_correction,
    diagnostics,
)
from .resources import assert_safe, snapshot
from .taxonomy import PITCH_GROUPS
from .v8 import _fingerprint, _load_raw

OOF_YEARS = (2023, 2024, 2025)
FEATURE_SETS = {
    "strategy": V9_STRATEGY_FEATURES,
    "physical": V9_PHYSICAL_FEATURES,
    "combined": V9_GAME_STATE_FEATURES,
}
SCALES = (0.1, 0.25, 0.5)
SAFETY_TOLERANCE = 0.0025
REPRODUCTION_TOLERANCE = 0.0005
EXPECTED_OOF_SPEC = "FOCAL_1:none:0.25"
EXPECTED_OOF_SEEDS = (8401, 8402, 8403)
PRACTICAL_LOG_LOSS_GAIN = 0.005
PRACTICAL_ACCURACY_GAIN = 0.004
FINGERPRINT_COLUMNS = (
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitcher_id",
    "target",
)


@dataclass
class FittedGameStateExpert:
    feature_names: tuple[str, ...]
    model: XGBClassifier
    tree_count: int
    device: str


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


def game_state_matrix(
    rows: pd.DataFrame,
    feature_names: tuple[str, ...],
    *,
    drop_physical: bool = False,
) -> np.ndarray:
    missing = set(feature_names) - set(rows)
    if missing:
        raise ValueError(f"missing game-state features: {sorted(missing)}")
    matrix = rows.loc[:, feature_names].to_numpy(dtype=np.float32, copy=True)
    if drop_physical:
        physical = set(V9_PHYSICAL_FEATURES)
        for index, name in enumerate(feature_names):
            if name in physical:
                matrix[:, index] = np.nan
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def _fit_final(
    rows: pd.DataFrame,
    base: np.ndarray,
    feature_names: tuple[str, ...],
    tree_count: int,
) -> FittedGameStateExpert:
    model, device = _fit(
        game_state_matrix(rows, feature_names),
        rows["target"].to_numpy(dtype=int),
        _base_margin(base),
        n_estimators=tree_count,
    )
    return FittedGameStateExpert(feature_names, model, tree_count, device)


def train_game_state_expert(
    rows: pd.DataFrame,
    base: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    n_estimators: int = 1_500,
) -> FittedGameStateExpert:
    if rows.empty or len(rows) != len(base):
        raise ValueError("game-state training inputs are empty or misaligned")
    dates = np.array(sorted(rows["game_date"].dt.normalize().unique()))
    if len(dates) < 2:
        raise ValueError("game-state tuning requires at least two dates")
    cutoff = dates[max(1, min(len(dates) - 1, int(len(dates) * 0.8)))]
    core = rows["game_date"].dt.normalize().to_numpy() < cutoff
    tuning = ~core
    model, _ = _fit(
        game_state_matrix(rows.loc[core], feature_names),
        rows.loc[core, "target"].to_numpy(dtype=int),
        _base_margin(base[core]),
        n_estimators=n_estimators,
        x_tuning=game_state_matrix(rows.loc[tuning], feature_names),
        tuning_labels=rows.loc[tuning, "target"].to_numpy(dtype=int),
        tuning_margin=_base_margin(base[tuning]),
    )
    tree_count = _best_tree_count(model)
    del model
    gc.collect()
    return _fit_final(rows, base, feature_names, tree_count)


def predict_game_state_correction(
    fitted: FittedGameStateExpert,
    rows: pd.DataFrame,
    *,
    drop_physical: bool = False,
) -> np.ndarray:
    matrix = game_state_matrix(
        rows,
        fitted.feature_names,
        drop_physical=drop_physical,
    )
    zero_margin = np.zeros((len(rows), len(PITCH_GROUPS)), dtype=float)
    correction = np.asarray(
        fitted.model.predict(
            matrix,
            output_margin=True,
            base_margin=zero_margin,
        ),
        dtype=float,
    )
    if correction.shape != zero_margin.shape or not np.isfinite(correction).all():
        raise ValueError("game-state correction has invalid shape or values")
    return correction - correction.mean(axis=1, keepdims=True)


def _major_zero(metrics: dict[str, object]) -> set[str]:
    return {
        name
        for name in metrics["zeroRecallClasses"]
        if metrics["actualDistribution"][name] >= 0.05
    }


def _failure_reasons(
    base: dict[str, object],
    candidate: dict[str, object],
    bootstrap: dict[str, object],
) -> list[str]:
    reasons = []
    if candidate["logLoss"] >= base["logLoss"]:
        reasons.append("log_loss_not_improved")
    if bootstrap["ciLower"] <= 0:
        reasons.append("bootstrap_ci_lower_not_positive")
    for key, label in (
        ("accuracy", "exact"),
        ("familyAccuracy", "family"),
        ("hierarchicalAccuracy", "hierarchical"),
        ("macroF1", "macro_f1"),
    ):
        if candidate[key] < base[key] - SAFETY_TOLERANCE:
            reasons.append(f"{label}_drop_gt_0.25pp")
    if candidate["totalVariationDistance"] > (
        base["totalVariationDistance"] + SAFETY_TOLERANCE
    ):
        reasons.append("tvd_worse_gt_0.25pp")
    if candidate["maxClassCalibrationError"] > (
        base["maxClassCalibrationError"] + SAFETY_TOLERANCE
    ):
        reasons.append("calibration_worse_gt_0.25pp")
    if _major_zero(candidate) - _major_zero(base):
        reasons.append("new_major_zero_recall")
    return reasons


def _assessment(
    rows: pd.DataFrame,
    base: np.ndarray,
    candidate: np.ndarray,
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
    reasons = _failure_reasons(base_metrics, metrics, bootstrap)
    return {
        "baseMetrics": base_metrics,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "failureReasons": reasons,
        "accepted": not reasons,
    }


def _feature_importance(
    fitted: FittedGameStateExpert,
) -> list[dict[str, float | str]]:
    raw = fitted.model.get_booster().get_score(importance_type="gain")
    ranked = []
    for key, gain in raw.items():
        if not key.startswith("f"):
            continue
        index = int(key[1:])
        if index >= len(fitted.feature_names):
            continue
        ranked.append(
            {
                "feature": fitted.feature_names[index],
                "gain": float(gain),
            }
        )
    return sorted(ranked, key=lambda item: item["gain"], reverse=True)[:25]


def _expected_v84_metrics(path: Path) -> dict[int, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        year: {
            key: float(payload["folds"][str(year)]["selectedMetrics"][key])
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


def _load_folds(
    data_directory: Path,
    oof_directory: Path,
    expected_metrics_path: Path,
) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    raw = _load_raw(data_directory)
    raw["game_date"] = pd.to_datetime(raw["game_date"], errors="coerce")
    rows = prepare_pitch_rows([raw], include_v9=True)
    del raw
    gc.collect()
    expected = _expected_v84_metrics(expected_metrics_path)
    folds = {}
    reproduction = {}
    columns = [
        *FINGERPRINT_COLUMNS,
        *dict.fromkeys(V9_GAME_STATE_FEATURES),
    ]
    for year in OOF_YEARS:
        metadata_path = oof_directory / f"{year}.json"
        arrays_path = oof_directory / f"{year}.npz"
        if not metadata_path.exists() or not arrays_path.exists():
            raise FileNotFoundError(f"missing frozen V8.4 OOF cache for {year}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("spec") != EXPECTED_OOF_SPEC
            or metadata.get("seeds") != list(EXPECTED_OOF_SEEDS)
        ):
            raise AssertionError(f"unexpected V8.4 OOF model contract for {year}")
        evaluation_rows = (
            rows.loc[rows["game_date"].dt.year.eq(year), columns]
            .copy()
            .reset_index(drop=True)
        )
        fingerprint = _fingerprint(evaluation_rows)
        if fingerprint != metadata.get("rowFingerprint"):
            raise AssertionError(f"V8.4 OOF fingerprint mismatch for {year}")
        with np.load(arrays_path) as arrays:
            base = validate_probability_matrix(arrays["base"]).astype(np.float32)
        if len(base) != len(evaluation_rows):
            raise AssertionError(f"V8.4 OOF row count mismatch for {year}")
        metrics = diagnostics(
            evaluation_rows["target"].to_numpy(dtype=int),
            base,
        )
        differences = {}
        if year in expected:
            differences = {
                key: float(metrics[key] - expected[year][key])
                for key in expected[year]
            }
            if max(abs(value) for value in differences.values()) > (
                REPRODUCTION_TOLERANCE
            ):
                raise AssertionError(f"V8.4 metrics failed to reproduce for {year}")
        folds[year] = {
            "rows": evaluation_rows,
            "base": base,
            "metrics": metrics,
            "rowFingerprint": fingerprint,
            "cacheHash": _hash_file(arrays_path),
        }
        reproduction[str(year)] = {
            "rows": len(evaluation_rows),
            "rowFingerprint": fingerprint,
            "spec": metadata["spec"],
            "seeds": metadata["seeds"],
            "cacheHash": _hash_file(arrays_path),
            "metadataHash": _hash_file(metadata_path),
            "differences": differences,
        }
    del rows
    gc.collect()
    return folds, reproduction


def _evaluate_feature_set(
    name: str,
    feature_names: tuple[str, ...],
    training: dict[str, object],
    evaluation: dict[str, object],
) -> tuple[dict[str, object], FittedGameStateExpert]:
    assert_safe(snapshot(Path.cwd()))
    fitted = train_game_state_expert(
        training["rows"],
        training["base"],
        feature_names,
    )
    correction = predict_game_state_correction(fitted, evaluation["rows"])
    stress_correction = predict_game_state_correction(
        fitted,
        evaluation["rows"],
        drop_physical=True,
    )
    candidates = []
    for scale in SCALES:
        candidate = apply_correction(evaluation["base"], correction, scale)
        stress = apply_correction(
            evaluation["base"],
            stress_correction,
            scale,
        )
        normal_assessment = _assessment(
            evaluation["rows"],
            evaluation["base"],
            candidate,
        )
        stress_assessment = _assessment(
            evaluation["rows"],
            evaluation["base"],
            stress,
        )
        candidates.append(
            {
                "scale": scale,
                "normal": normal_assessment,
                "physicalDropStress": stress_assessment,
                "accepted": (
                    normal_assessment["accepted"]
                    and stress_assessment["accepted"]
                ),
            }
        )
    return (
        {
            "featureSet": name,
            "featureCount": len(feature_names),
            "treeCount": fitted.tree_count,
            "device": fitted.device,
            "missingRate": float(
                evaluation["rows"].loc[:, feature_names].isna().mean().mean()
            ),
            "featureImportance": _feature_importance(fitted),
            "candidates": candidates,
        },
        fitted,
    )


def _select_candidate(
    evaluations: list[dict[str, object]],
) -> dict[str, object] | None:
    accepted = [
        {
            "featureSet": evaluation["featureSet"],
            "featureCount": evaluation["featureCount"],
            "treeCount": evaluation["treeCount"],
            "device": evaluation["device"],
            "missingRate": evaluation["missingRate"],
            "featureImportance": evaluation["featureImportance"],
            **candidate,
        }
        for evaluation in evaluations
        for candidate in evaluation["candidates"]
        if candidate["accepted"]
    ]
    return (
        min(accepted, key=lambda item: item["normal"]["metrics"]["logLoss"])
        if accepted
        else None
    )


def _report(result: dict[str, object]) -> str:
    lines = [
        "# V9-A Game-State Expert",
        "",
        f"- 판정: `{result['status']}`",
        f"- 제품 active: `{result['activeModel']}`",
        f"- 최초 실패 단계: `{result.get('firstFailureStage', 'none')}`",
        "",
    ]
    selection = result.get("selection2024")
    if selection:
        lines.extend(
            [
                "## 2024 후보 선택",
                "",
                (
                    "| Feature set | Scale | Exact | Family | Macro F1 | "
                    "Log Loss | TVD | 통과 |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        base = selection["baseMetrics"]
        lines.append(
            f"| V8.4 base | 0 | {base['accuracy']:.2%} | "
            f"{base['familyAccuracy']:.2%} | {base['macroF1']:.2%} | "
            f"{base['logLoss']:.5f} | "
            f"{base['totalVariationDistance']:.2%} | — |"
        )
        for evaluation in selection["evaluations"]:
            for candidate in evaluation["candidates"]:
                metrics = candidate["normal"]["metrics"]
                lines.append(
                    f"| {evaluation['featureSet']} | {candidate['scale']:.2f} | "
                    f"{metrics['accuracy']:.2%} | "
                    f"{metrics['familyAccuracy']:.2%} | "
                    f"{metrics['macroF1']:.2%} | "
                    f"{metrics['logLoss']:.5f} | "
                    f"{metrics['totalVariationDistance']:.2%} | "
                    f"{candidate['accepted']} |"
                )
        selected = selection.get("selected")
        lines.extend(
            [
                "",
                (
                    "- 선택: "
                    f"`{selected['featureSet']} × {selected['scale']}`"
                    if selected
                    else "- 선택: 없음"
                ),
                "",
            ]
        )
    confirmation = result.get("confirmation2025")
    if confirmation:
        base = confirmation["assessment"]["baseMetrics"]
        candidate = confirmation["assessment"]["metrics"]
        bootstrap = confirmation["assessment"]["bootstrap"]
        exact_delta = 100 * (candidate["accuracy"] - base["accuracy"])
        family_delta = 100 * (
            candidate["familyAccuracy"] - base["familyAccuracy"]
        )
        macro_delta = 100 * (candidate["macroF1"] - base["macroF1"])
        lines.extend(
            [
                "## 2025 Locked Confirmation",
                "",
                "| 모델 | Exact | Family | Hierarchical | Macro F1 | Log Loss | TVD |",
                "|---|---:|---:|---:|---:|---:|---:|",
                (
                    f"| V8.4 | {base['accuracy']:.2%} | "
                    f"{base['familyAccuracy']:.2%} | "
                    f"{base['hierarchicalAccuracy']:.2%} | "
                    f"{base['macroF1']:.2%} | {base['logLoss']:.5f} | "
                    f"{base['totalVariationDistance']:.2%} |"
                ),
                (
                    f"| V9-A | {candidate['accuracy']:.2%} | "
                    f"{candidate['familyAccuracy']:.2%} | "
                    f"{candidate['hierarchicalAccuracy']:.2%} | "
                    f"{candidate['macroF1']:.2%} | "
                    f"{candidate['logLoss']:.5f} | "
                    f"{candidate['totalVariationDistance']:.2%} |"
                ),
                "",
                (
                    "- paired-game Log Loss gain 95% CI: "
                    f"`[{bootstrap['ciLower']:.6f}, "
                    f"{bootstrap['ciUpper']:.6f}]`"
                ),
                (
                    "- 변화: Exact "
                    f"`{exact_delta:+.2f}%p`, Family "
                    f"`{family_delta:+.2f}%p`, Macro F1 "
                    f"`{macro_delta:+.2f}%p`, "
                    "Log Loss "
                    f"`{candidate['logLoss'] - base['logLoss']:+.6f}`"
                ),
                f"- 안전 게이트: `{confirmation['accepted']}`",
                (
                    "- 실용적 최소 효과: "
                    f"`{confirmation['practicalTargetMet']}`"
                ),
                "",
            ]
        )
    retrospective = result.get("retrospective2026")
    if retrospective:
        comparison = retrospective["comparison"]
        lines.extend(
            [
                "## 2026 Retrospective",
                "",
                (
                    f"- 학습 범위: `{retrospective['trainingRange']['start']}` ~ "
                    f"`{retrospective['trainingRange']['end']}`"
                ),
                (
                    f"- 평가 범위: `{retrospective['holdoutRange']['start']}` ~ "
                    f"`{retrospective['holdoutRange']['end']}`"
                ),
                f"- 공통 표본: `{retrospective['holdoutRows']:,}`구",
                f"- row fingerprint: `{retrospective['rowFingerprint']}`",
                "- 2026 학습·선택 사용: `false`",
                "",
                (
                    "| 모델 | Exact | Family | Hierarchical | Macro F1 | "
                    "Log Loss | TVD |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in ("V8.4", "V8.5", "V9-A"):
            metrics = comparison[name]
            lines.append(
                f"| {name} | {metrics['accuracy']:.2%} | "
                f"{metrics['familyAccuracy']:.2%} | "
                f"{metrics['hierarchicalAccuracy']:.2%} | "
                f"{metrics['macroF1']:.2%} | {metrics['logLoss']:.5f} | "
                f"{metrics['totalVariationDistance']:.2%} |"
            )
        bootstrap = retrospective["assessment"]["bootstrap"]
        v85_metrics = comparison["V8.5"]
        v9_metrics = comparison["V9-A"]
        exact_delta = 100 * (v9_metrics["accuracy"] - v85_metrics["accuracy"])
        family_delta = 100 * (
            v9_metrics["familyAccuracy"] - v85_metrics["familyAccuracy"]
        )
        macro_delta = 100 * (
            v9_metrics["macroF1"] - v85_metrics["macroF1"]
        )
        log_loss_delta = v9_metrics["logLoss"] - v85_metrics["logLoss"]
        lines.extend(
            [
                "",
                (
                    "- V9-A − V8.5: Exact "
                    f"`{exact_delta:+.2f}%p`, Family "
                    f"`{family_delta:+.2f}%p`, Macro F1 "
                    f"`{macro_delta:+.2f}%p`, Log Loss "
                    f"`{log_loss_delta:+.6f}`"
                ),
                (
                    "- V9-A의 V8.4 대비 paired-game Log Loss gain 95% CI: "
                    f"`[{bootstrap['ciLower']:.6f}, "
                    f"{bootstrap['ciUpper']:.6f}]`"
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
            "## 판단",
            "",
            str(result["conclusion"]),
            "",
            "V7.2 active와 replay 라우팅은 변경하지 않았다.",
            "",
        ]
    )
    return "\n".join(lines)


def _comparison_metrics(
    v85_artifact_path: Path,
    *,
    expected_fingerprint: str,
) -> dict[str, object]:
    payload = json.loads(v85_artifact_path.read_text(encoding="utf-8"))
    retrospective = payload.get("retrospective2026")
    if (
        payload.get("retrospectiveStatus") != "complete"
        or not retrospective
        or retrospective.get("rowFingerprint") != expected_fingerprint
    ):
        raise ValueError("V8.5 retrospective fingerprint does not match V9-A")
    return retrospective["stages"]["E_safe_alpha_caps"]


def _save(
    result: dict[str, object],
    artifact_directory: Path,
    report_path: Path,
) -> None:
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(result), encoding="utf-8")


def run_retrospective(
    data_directory: Path,
    oof_directory: Path,
    v84_artifact_path: Path,
    holdout_directory: Path,
    v84_refit_model_directory: Path,
    v84_holdout_artifact_path: Path,
    v85_artifact_path: Path,
    artifact_directory: Path,
    report_path: Path,
    *,
    batch_size: int = 8192,
) -> dict[str, object]:
    result_path = artifact_directory / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "research-passed":
        raise ValueError("2026 retrospective requires passed 2025 confirmation")
    selected = result["selection2024"]["selected"]
    if selected is None:
        raise ValueError("V9-A has no frozen candidate")

    folds, reproduction = _load_folds(
        data_directory,
        oof_directory,
        v84_artifact_path,
    )
    training_rows = pd.concat(
        [folds[year]["rows"] for year in OOF_YEARS],
        ignore_index=True,
    )
    training_base = np.concatenate(
        [folds[year]["base"] for year in OOF_YEARS],
    )
    fitted = _fit_final(
        training_rows,
        training_base,
        FEATURE_SETS[str(selected["featureSet"])],
        int(selected["treeCount"]),
    )
    del folds, training_rows, training_base
    gc.collect()
    assert_safe(snapshot(Path.cwd()))

    training_raw = _load_raw(data_directory)
    training_raw["game_date"] = pd.to_datetime(
        training_raw["game_date"],
        errors="coerce",
    )
    from .v85 import _v84_retrospective_base

    retrospective_base = _v84_retrospective_base(
        training_raw,
        holdout_directory,
        v84_refit_model_directory,
        v84_holdout_artifact_path,
        batch_size=batch_size,
        include_v9=True,
    )
    del training_raw
    gc.collect()

    rows = retrospective_base["rows"]
    base = retrospective_base["base"]
    correction = predict_game_state_correction(fitted, rows)
    final = apply_correction(base, correction, float(selected["scale"]))
    assessment = _assessment(rows, base, final)
    fingerprint = str(retrospective_base["metadata"]["rowFingerprint"])
    v85_metrics = _comparison_metrics(
        v85_artifact_path,
        expected_fingerprint=fingerprint,
    )

    model_directory = Path("models/v9a-2026")
    model_directory.mkdir(parents=True, exist_ok=True)
    model_path = model_directory / "game-state-expert.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(fitted, handle)
    result["retrospective2026"] = {
        **retrospective_base["metadata"],
        "oofTrainingYears": list(OOF_YEARS),
        "oofTrainingRows": int(
            sum(item["rows"] for item in reproduction.values())
        ),
        "selectedFeatureSet": selected["featureSet"],
        "scale": selected["scale"],
        "treeCount": selected["treeCount"],
        "modelHash": _hash_file(model_path),
        "oofReproduction": reproduction,
        "assessment": assessment,
        "comparison": {
            "V8.4": assessment["baseMetrics"],
            "V8.5": v85_metrics,
            "V9-A": assessment["metrics"],
        },
        "selectionImpact": "none-retrospective-only",
    }
    result["retrospectiveStatus"] = "complete"
    result["resourceAfterRetrospective"] = asdict(snapshot(Path.cwd()))
    _save(result, artifact_directory, report_path)
    return result


def run(
    data_directory: Path,
    oof_directory: Path,
    v84_artifact_path: Path,
    artifact_directory: Path,
    report_path: Path,
) -> dict[str, object]:
    before = asdict(snapshot(Path.cwd()))
    assert_safe(snapshot(Path.cwd()))
    configuration = {
        "model": "V9-A",
        "base": "V8.4 FOCAL_1:none:0.25 3-seed OOF",
        "featureSets": {
            name: list(features) for name, features in FEATURE_SETS.items()
        },
        "scales": list(SCALES),
        "safetyTolerance": SAFETY_TOLERANCE,
        "selectionYear": 2024,
        "confirmationYear": 2025,
        "pitcherIdUsed": False,
        "currentPitchPhysicalUsed": False,
        "2026UsedForSelection": False,
    }
    result: dict[str, object] = {
        "schemaVersion": "9.0",
        "modelVersion": "V9-A-game-state-expert",
        "generatedAt": datetime.now(UTC).isoformat(),
        "status": "running",
        "activeModel": "V7.2",
        "configuration": configuration,
        "configurationHash": _hash_json(configuration),
        "resourceBefore": before,
    }
    try:
        folds, reproduction = _load_folds(
            data_directory,
            oof_directory,
            v84_artifact_path,
        )
    except Exception as error:
        result.update(
            {
                "status": "invalid-experiment",
                "firstFailureStage": "oof_alignment_or_reproduction",
                "conclusion": str(error),
                "resourceAfter": asdict(snapshot(Path.cwd())),
            }
        )
        _save(result, artifact_directory, report_path)
        return result
    result["v84Reproduction"] = reproduction

    evaluations = []
    fitted_by_name = {}
    for name, feature_names in FEATURE_SETS.items():
        evaluation, fitted = _evaluate_feature_set(
            name,
            feature_names,
            folds[2023],
            folds[2024],
        )
        evaluations.append(evaluation)
        fitted_by_name[name] = fitted
        print(f"V9-A 2024: {name} complete", flush=True)
    selected = _select_candidate(evaluations)
    result["selection2024"] = {
        "baseMetrics": folds[2024]["metrics"],
        "evaluations": evaluations,
        "selected": (
            {
                key: value
                for key, value in selected.items()
                if key not in {"normal", "physicalDropStress"}
            }
            if selected
            else None
        ),
    }
    if selected is None:
        result.update(
            {
                "status": "rejected",
                "firstFailureStage": "2024_selection",
                "conclusion": (
                    "V8.4 대비 정상·physical-drop 안전 게이트를 모두 "
                    "통과한 Game-State 후보가 없어 2025를 평가하지 않았다."
                ),
                "resourceAfter": asdict(snapshot(Path.cwd())),
            }
        )
        _save(result, artifact_directory, report_path)
        return result

    selected_features = FEATURE_SETS[str(selected["featureSet"])]
    training_rows = pd.concat(
        [
            folds[2023]["rows"],
            folds[2024]["rows"],
        ],
        ignore_index=True,
    )
    training_base = np.concatenate(
        [folds[2023]["base"], folds[2024]["base"]],
    )
    del fitted_by_name
    gc.collect()
    assert_safe(snapshot(Path.cwd()))
    final = _fit_final(
        training_rows,
        training_base,
        selected_features,
        int(selected["treeCount"]),
    )
    correction_2025 = predict_game_state_correction(
        final,
        folds[2025]["rows"],
    )
    candidate_2025 = apply_correction(
        folds[2025]["base"],
        correction_2025,
        float(selected["scale"]),
    )
    assessment_2025 = _assessment(
        folds[2025]["rows"],
        folds[2025]["base"],
        candidate_2025,
    )
    stress_correction_2025 = predict_game_state_correction(
        final,
        folds[2025]["rows"],
        drop_physical=True,
    )
    stress_2025 = _assessment(
        folds[2025]["rows"],
        folds[2025]["base"],
        apply_correction(
            folds[2025]["base"],
            stress_correction_2025,
            float(selected["scale"]),
        ),
    )
    practical = (
        assessment_2025["baseMetrics"]["logLoss"]
        - assessment_2025["metrics"]["logLoss"]
        >= PRACTICAL_LOG_LOSS_GAIN
        or assessment_2025["metrics"]["accuracy"]
        - assessment_2025["baseMetrics"]["accuracy"]
        >= PRACTICAL_ACCURACY_GAIN
    )
    accepted = assessment_2025["accepted"] and stress_2025["accepted"]
    result["confirmation2025"] = {
        "selectedFeatureSet": selected["featureSet"],
        "scale": selected["scale"],
        "treeCount": selected["treeCount"],
        "assessment": assessment_2025,
        "physicalDropStress": stress_2025,
        "practicalTargetMet": practical,
        "accepted": accepted,
    }
    model_directory = Path("models/v9a")
    model_directory.mkdir(parents=True, exist_ok=True)
    model_path = model_directory / "game-state-expert.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(final, handle)
    result["modelMetadata"] = {
        "path": str(model_path),
        "hash": _hash_file(model_path),
        "featureNames": list(selected_features),
        "treeCount": final.tree_count,
        "device": final.device,
        "featureImportance": _feature_importance(final),
    }
    result.update(
        {
            "status": "research-passed" if accepted else "rejected",
            "firstFailureStage": (
                "none" if accepted else "2025_locked_confirmation"
            ),
            "deploymentStatus": (
                "research-passed"
                if accepted and practical
                else "shadow-only"
            ),
            "conclusion": (
                "V9-A가 2024 선택과 2025 고정 안전 게이트를 통과했다. "
                + (
                    "당일 physical 피처는 strategy 단독보다 독립적인 이득을 "
                    "만들지 못했다. "
                    if selected["featureSet"] == "strategy"
                    else ""
                )
                + (
                    "실용적 최소 효과도 충족해 후속 prospective 후보로 동결한다."
                    if practical
                    else (
                        "다만 실용적 최소 효과는 충족하지 못해 연구 결과만 "
                        "보존하고 제품 후보로 승격하지 않는다."
                    )
                )
                if accepted
                else (
                    "동결된 V9-A가 2025 안전 조건을 통과하지 못했다. "
                    "사후 scale·feature 재튜닝은 수행하지 않았다."
                )
            ),
            "resourceAfter": asdict(snapshot(Path.cwd())),
        }
    )
    _save(result, artifact_directory, report_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=Path("data/raw/statcast-v8"),
    )
    parser.add_argument(
        "--oof-directory",
        type=Path,
        default=Path("models/v8.5-oof"),
    )
    parser.add_argument(
        "--v84-artifact",
        type=Path,
        default=Path("artifacts/v8.4-distribution-safe-sequence/result.json"),
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=Path("artifacts/v9a-game-state"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/2026-07-29-v9a-game-state.md"),
    )
    parser.add_argument("--holdout-data", type=Path)
    parser.add_argument("--v84-refit-models", type=Path)
    parser.add_argument("--v84-holdout-artifact", type=Path)
    parser.add_argument("--v85-artifact", type=Path)
    parser.add_argument("--retrospective-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8192)
    arguments = parser.parse_args()
    if arguments.retrospective_only:
        if any(
            value is None
            for value in (
                arguments.holdout_data,
                arguments.v84_refit_models,
                arguments.v84_holdout_artifact,
                arguments.v85_artifact,
            )
        ):
            parser.error("retrospective inputs are required")
        result = run_retrospective(
            arguments.data_directory,
            arguments.oof_directory,
            arguments.v84_artifact,
            arguments.holdout_data,
            arguments.v84_refit_models,
            arguments.v84_holdout_artifact,
            arguments.v85_artifact,
            arguments.artifact_directory,
            arguments.report,
            batch_size=arguments.batch_size,
        )
    else:
        result = run(
            arguments.data_directory,
            arguments.oof_directory,
            arguments.v84_artifact,
            arguments.artifact_directory,
            arguments.report,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
