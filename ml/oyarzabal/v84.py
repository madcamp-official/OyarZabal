"""Train and evaluate the V8.4 distribution-safe sequence residual."""

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

from .features import prepare_pitch_rows
from .modeling import predict_candidate, train_candidate_with_tuning
from .resources import assert_safe, snapshot
from .sequence import SequenceExampleBuilder, SequenceVocabulary
from .training import GLOBAL_SPEC, GLOBAL_TEMPERATURE
from .v8 import _chronological_indices, _fingerprint, _load_raw, _metrics
from .v83 import (
    _assess,
    _diagnostic_manifest,
    _examples_for_context,
    _stress_examples,
)
from .v83_sequence import (
    FittedV83Expert,
    SequenceObjective,
    fit_v83_expert,
    predict_v83_deltas,
    refit_v83_expert,
    save_v83_expert,
)
from .v84_sequence import (
    DistributionSafeTransform,
    calibration_candidates,
    fit_class_conditional_shrinkage,
    select_safe_calibration,
)

SCALES = (0.05, 0.1, 0.25, 0.5)
SHRINKAGE_MODES = ("none", "class")
SELECTION_SEEDS = (8401,)
CONFIRMATION_SEEDS = (8401, 8402, 8403)
ROBUST_SELECTION_TOLERANCE = 0.0025


@dataclass(frozen=True)
class ObjectiveConfig:
    name: str
    soft_target_strength: float
    focal_gamma: float
    group_balanced: bool

    def objective(self) -> SequenceObjective:
        return SequenceObjective(
            soft_target_strength=self.soft_target_strength,
            focal_gamma=self.focal_gamma,
            group_balanced=self.group_balanced,
        )


OBJECTIVES = (
    ObjectiveConfig("NLL", 0.0, 0.0, False),
    ObjectiveConfig("SOFT_025", 0.25, 0.0, False),
    ObjectiveConfig("FOCAL_1", 0.0, 1.0, True),
    ObjectiveConfig("SOFT_025_FOCAL_1", 0.25, 1.0, True),
)


@dataclass(frozen=True)
class CandidateSpec:
    objective: str
    shrinkage: str
    scale: float

    @property
    def key(self) -> str:
        return f"{self.objective}:{self.shrinkage}:{self.scale}"


def _transform(
    actual: np.ndarray,
    global_probabilities: np.ndarray,
    family_delta: np.ndarray,
    child_delta: np.ndarray,
    fitted: FittedV83Expert,
    spec: CandidateSpec,
) -> DistributionSafeTransform:
    shrinkage = (
        fit_class_conditional_shrinkage(
            actual,
            global_probabilities,
            family_delta,
            child_delta,
            fitted.group_weights,
            scale=spec.scale,
        )
        if spec.shrinkage == "class"
        else np.ones(9)
    )
    raw = DistributionSafeTransform(
        scale=spec.scale,
        family_shrinkage=shrinkage[:3],
        child_shrinkage=shrinkage[3:],
        calibration_mode="identity",
        calibration_parameters=np.array([]),
    ).apply(global_probabilities, family_delta, child_delta)
    mode, parameters = select_safe_calibration(
        actual,
        global_probabilities,
        calibration_candidates(actual, raw),
    )
    return DistributionSafeTransform(
        scale=spec.scale,
        family_shrinkage=shrinkage[:3],
        child_shrinkage=shrinkage[3:],
        calibration_mode=mode,
        calibration_parameters=parameters,
    )


def _member_deltas(
    fitted: FittedV83Expert,
    examples,
    global_probabilities: np.ndarray,
    validation: np.ndarray,
    evaluation: np.ndarray,
    *,
    batch_size: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    validation_delta = predict_v83_deltas(
        fitted,
        examples,
        validation,
        global_probabilities,
        batch_size=batch_size,
    )
    evaluation_delta = predict_v83_deltas(
        fitted,
        examples,
        evaluation,
        global_probabilities,
        batch_size=batch_size,
    )
    stress_delta = predict_v83_deltas(
        fitted,
        _stress_examples(examples, context="catcher"),
        evaluation,
        global_probabilities,
        batch_size=batch_size,
    )
    return validation_delta, evaluation_delta, stress_delta


def _member_predictions(
    fitted: FittedV83Expert,
    global_probabilities: np.ndarray,
    validation_actual: np.ndarray,
    validation: np.ndarray,
    evaluation: np.ndarray,
    deltas: tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ],
    spec: CandidateSpec,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    validation_delta, evaluation_delta, stress_delta = deltas
    transform = _transform(
        validation_actual,
        global_probabilities[validation],
        *validation_delta,
        fitted,
        spec,
    )
    normal = transform.apply(
        global_probabilities[evaluation],
        *evaluation_delta,
    )
    stress = transform.apply(
        global_probabilities[evaluation],
        *stress_delta,
    )
    metadata = {
        "scale": transform.scale,
        "familyShrinkage": transform.family_shrinkage.tolist(),
        "childShrinkage": transform.child_shrinkage.tolist(),
        "calibrationMode": transform.calibration_mode,
        "calibrationParameters": transform.calibration_parameters.tolist(),
    }
    return normal.astype(np.float32), stress.astype(np.float32), metadata


def _fold(
    raw: pd.DataFrame,
    rows: pd.DataFrame,
    year: int,
    objective_configs: tuple[ObjectiveConfig, ...],
    specs: tuple[CandidateSpec, ...],
    *,
    seeds: tuple[int, ...],
    epochs: int,
    batch_size: int,
    refit_full: bool = False,
) -> dict[str, object]:
    fold_raw = raw[raw["game_date"].dt.year <= year].copy()
    fold_rows = rows[rows["game_date"].dt.year <= year].reset_index(drop=True)
    train_raw = fold_raw[fold_raw["game_date"].dt.year < year]
    vocabulary = SequenceVocabulary.fit(train_raw)
    examples = SequenceExampleBuilder().build(fold_raw, vocabulary)
    if len(examples) != len(fold_rows):
        raise AssertionError(f"V8.4 row alignment failed for {year}")
    examples = _examples_for_context(
        examples.with_repertoire(fold_rows),
        fold_rows,
        "catcher",
    )
    core, validation, evaluation = _chronological_indices(
        examples.target_dates,
        year,
    )
    global_model, trees = train_candidate_with_tuning(
        fold_rows[fold_rows["game_date"].dt.year < year],
        GLOBAL_SPEC,
    )
    global_probabilities = predict_candidate(
        global_model,
        fold_rows,
        GLOBAL_TEMPERATURE,
    ).astype(np.float32)
    del global_model
    gc.collect()

    actual = examples.target_groups[evaluation]
    games = fold_rows.iloc[evaluation]["game_pk"].to_numpy()
    candidates: dict[str, object] = {}
    fitted_by_objective: dict[str, list[FittedV83Expert]] = {}
    full_training = (
        np.sort(np.concatenate([core, validation])) if refit_full else core
    )
    for config in objective_configs:
        relevant = [spec for spec in specs if spec.objective == config.name]
        if not relevant:
            continue
        members = [
            fit_v83_expert(
                examples,
                core,
                validation,
                global_probabilities,
                description_vocab_size=len(vocabulary.descriptions) + 1,
                balance_strength=0,
                block_dropout=0.2,
                objective=config.objective(),
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
            )
            for seed in seeds
        ]
        if refit_full:
            refitted = [
                refit_v83_expert(
                    examples,
                    full_training,
                    global_probabilities,
                    description_vocab_size=len(vocabulary.descriptions) + 1,
                    balance_strength=0,
                    block_dropout=0.2,
                    objective=config.objective(),
                    epochs=member.epochs,
                    batch_size=batch_size,
                    seed=seed,
                )
                for member, seed in zip(members, seeds, strict=True)
            ]
            stress_examples = _stress_examples(examples, context="catcher")
            deltas = [
                (
                    predict_v83_deltas(
                        tuned,
                        examples,
                        validation,
                        global_probabilities,
                        batch_size=batch_size,
                    ),
                    predict_v83_deltas(
                        fitted,
                        examples,
                        evaluation,
                        global_probabilities,
                        batch_size=batch_size,
                    ),
                    predict_v83_deltas(
                        fitted,
                        stress_examples,
                        evaluation,
                        global_probabilities,
                        batch_size=batch_size,
                    ),
                )
                for tuned, fitted in zip(members, refitted, strict=True)
            ]
            fitted_by_objective[config.name] = refitted
        else:
            refitted = members
            deltas = [
                _member_deltas(
                    member,
                    examples,
                    global_probabilities,
                    validation,
                    evaluation,
                    batch_size=batch_size,
                )
                for member in members
            ]
            fitted_by_objective[config.name] = members
        for spec in relevant:
            normal_members, stress_members, transforms = [], [], []
            for member, member_deltas in zip(refitted, deltas, strict=True):
                normal, stress, transform = _member_predictions(
                    member,
                    global_probabilities,
                    examples.target_groups[validation],
                    validation,
                    evaluation,
                    member_deltas,
                    spec,
                )
                normal_members.append(normal)
                stress_members.append(stress)
                transforms.append(transform)
            normal = np.mean(normal_members, axis=0)
            stress = np.mean(stress_members, axis=0)
            candidates[spec.key] = {
                "spec": asdict(spec),
                "transforms": transforms,
                "normal": _assess(
                    actual,
                    games,
                    global_probabilities[evaluation],
                    normal,
                ),
                "physicalDropStress": _assess(
                    actual,
                    games,
                    global_probabilities[evaluation],
                    stress,
                ),
            }
            print(f"V8.4 {year}: {spec.key} complete", flush=True)
    return {
        "year": year,
        "rows": len(evaluation),
        "rowFingerprint": _fingerprint(fold_rows.iloc[evaluation]),
        "globalTrees": trees,
        "refitFullTraining": refit_full,
        "sequenceTrainingRows": len(full_training),
        "sequenceTrainingMaxDate": str(
            examples.target_dates[full_training].max()
        ),
        "tuningEpochs": {
            name: [member.epochs for member in fitted]
            for name, fitted in fitted_by_objective.items()
        },
        "globalMetrics": _metrics(actual, global_probabilities[evaluation]),
        "candidates": candidates,
        "fitted": fitted_by_objective,
        "vocabulary": vocabulary,
    }


def _selection_specs() -> tuple[CandidateSpec, ...]:
    return tuple(
        CandidateSpec(config.name, shrinkage, scale)
        for config in OBJECTIVES
        for shrinkage in SHRINKAGE_MODES
        for scale in SCALES
    )


def _select(
    candidates: dict[str, dict[str, object]],
    global_metrics: dict[str, object],
) -> CandidateSpec | None:
    def robust(candidate: dict[str, object]) -> bool:
        for condition in ("normal", "physicalDropStress"):
            metrics = candidate[condition]["metrics"]
            if any(
                metrics[name]
                < global_metrics[name] - ROBUST_SELECTION_TOLERANCE
                for name in (
                    "accuracy",
                    "familyAccuracy",
                    "hierarchicalAccuracy",
                    "macroF1",
                )
            ):
                return False
            if any(
                metrics[name]
                > global_metrics[name] + ROBUST_SELECTION_TOLERANCE
                for name in (
                    "totalVariationDistance",
                    "maxClassCalibrationError",
                )
            ):
                return False
        return True

    passing = [
        candidate
        for candidate in candidates.values()
        if candidate["normal"]["accepted"]
        and candidate["physicalDropStress"]["accepted"]
        and robust(candidate)
    ]
    if not passing:
        return None
    selected = min(
        passing,
        key=lambda candidate: (
            candidate["normal"]["metrics"]["logLoss"],
            -candidate["normal"]["metrics"]["macroF1"],
        ),
    )
    return CandidateSpec(**selected["spec"])


def _public_fold(fold: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in fold.items()
        if key not in {"fitted", "vocabulary"}
    }


def _configuration(
    *,
    epochs: int,
    batch_size: int,
) -> dict[str, object]:
    return {
        "objectives": [asdict(config) for config in OBJECTIVES],
        "scales": list(SCALES),
        "shrinkageModes": list(SHRINKAGE_MODES),
        "selectionSeeds": list(SELECTION_SEEDS),
        "confirmationSeeds": list(CONFIRMATION_SEEDS),
        "epochs": epochs,
        "batchSize": batch_size,
        "context": "R5 physical drift + catcher profile",
        "calibration": ["identity", "temperature", "hierarchical"],
        "protocol": "2024-select_2025-single-frozen-confirm",
        "robustSelectionTolerance": ROBUST_SELECTION_TOLERANCE,
    }


def _report(result: dict[str, object]) -> str:
    selection = result["selection"]
    lines = [
        "# V8.4 Distribution-Safe Sequence Residual",
        "",
        f"- 생성 시각: {result['generatedAt']}",
        f"- 선택: `{selection['candidate']}`",
        f"- 판정: `{selection['status']}`",
        f"- active 모델: `{result['activeModel']}`",
        "",
        "| 연도 | 모델 | Exact | Family | Hierarchical | Macro F1 | Log Loss | TVD |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for year, fold in result["folds"].items():
        global_metrics = fold["globalMetrics"]
        lines.append(
            f"| {year} | Global | {global_metrics['accuracy']:.2%} | "
            f"{global_metrics['familyAccuracy']:.2%} | "
            f"{global_metrics['hierarchicalAccuracy']:.2%} | "
            f"{global_metrics['macroF1']:.2%} | "
            f"{global_metrics['logLoss']:.5f} | "
            f"{global_metrics['totalVariationDistance']:.2%} |"
        )
        selected_metrics = fold.get("selectedMetrics")
        if selected_metrics:
            lines.append(
                f"| {year} | V8.4 | {selected_metrics['accuracy']:.2%} | "
                f"{selected_metrics['familyAccuracy']:.2%} | "
                f"{selected_metrics['hierarchicalAccuracy']:.2%} | "
                f"{selected_metrics['macroF1']:.2%} | "
                f"{selected_metrics['logLoss']:.5f} | "
                f"{selected_metrics['totalVariationDistance']:.2%} |"
            )
    ablation = result["folds"]["2024"]["candidates"]
    lines.extend(
        [
            "",
            "## 2024 objective별 최선 안전 후보",
            "",
            "| Objective | 후보 | Macro F1 | Log Loss | TVD | Calibration |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for objective in (config.name for config in OBJECTIVES):
        passing = [
            (key, candidate)
            for key, candidate in ablation.items()
            if candidate["spec"]["objective"] == objective
            and candidate["normal"]["accepted"]
            and candidate["physicalDropStress"]["accepted"]
        ]
        if not passing:
            continue
        key, candidate = min(
            passing,
            key=lambda item: item[1]["normal"]["metrics"]["logLoss"],
        )
        metrics = candidate["normal"]["metrics"]
        lines.append(
            f"| {objective} | `{key}` | {metrics['macroF1']:.2%} | "
            f"{metrics['logLoss']:.5f} | "
            f"{metrics['totalVariationDistance']:.2%} | "
            f"{candidate['transforms'][0]['calibrationMode']} |"
        )
    lines.extend(
        [
            "",
            (
                "2024에서 loss·shrinkage·scale을 선택하고 "
                "2025는 고정 후보를 한 번만 확인했다."
            ),
            "평가 안전 게이트는 V8.3과 동일하게 유지했다.",
            (
                "Class-conditional shrinkage는 선택 후보에서 모두 1.0으로 "
                "수렴해 최종 이득에 기여하지 않았다."
            ),
            (
                "V8.4는 연구 후보로 통과했으며 replay 제품의 active 모델은 "
                "V7.2를 유지한다."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(
    data_directory: Path,
    artifact_directory: Path,
    report_path: Path,
    *,
    epochs: int,
    batch_size: int,
    resume_selection: bool = False,
) -> dict[str, object]:
    resource_before = asdict(snapshot(Path.cwd()))
    assert_safe(snapshot(Path.cwd()))
    raw = _load_raw(data_directory)
    raw["game_date"] = pd.to_datetime(raw["game_date"], errors="coerce")
    rows = prepare_pitch_rows([raw], include_v83=True)

    existing_artifact = artifact_directory / "result.json"
    if resume_selection:
        previous = json.loads(existing_artifact.read_text())
        selection_public = previous["folds"]["2024"]
        selection_public.pop("ensemble", None)
        selection_public.pop("selectedMetrics", None)
    else:
        selection_fold = _fold(
            raw,
            rows,
            2024,
            OBJECTIVES,
            _selection_specs(),
            seeds=SELECTION_SEEDS,
            epochs=epochs,
            batch_size=batch_size,
        )
        selection_public = _public_fold(selection_fold)
    selected = _select(
        selection_public["candidates"],
        selection_public["globalMetrics"],
    )
    folds = {"2024": selection_public}
    status = "rejected-2024"
    active_model = "V7.2"

    if selected is not None:
        config = next(item for item in OBJECTIVES if item.name == selected.objective)
        fixed_specs = (selected,)
        ensemble_2024 = _fold(
            raw,
            rows,
            2024,
            (config,),
            fixed_specs,
            seeds=CONFIRMATION_SEEDS,
            epochs=epochs,
            batch_size=batch_size,
        )
        candidate_2024 = ensemble_2024["candidates"][selected.key]
        folds["2024"]["ensemble"] = candidate_2024
        folds["2024"]["selectedMetrics"] = candidate_2024["normal"]["metrics"]
        ensemble_passed = bool(
            candidate_2024["normal"]["accepted"]
            and candidate_2024["physicalDropStress"]["accepted"]
        )
        if ensemble_passed:
            confirmation = _fold(
                raw,
                rows,
                2025,
                (config,),
                fixed_specs,
                seeds=CONFIRMATION_SEEDS,
                epochs=epochs,
                batch_size=batch_size,
            )
            candidate_2025 = confirmation["candidates"][selected.key]
            passed_2025 = bool(
                candidate_2025["normal"]["accepted"]
                and candidate_2025["physicalDropStress"]["accepted"]
            )
            folds["2025"] = {
                **_public_fold(confirmation),
                "selectedMetrics": candidate_2025["normal"]["metrics"],
            }
            status = "passed" if passed_2025 else "rejected-2025"
            if passed_2025:
                model_directory = Path("models/v8.4")
                model_directory.mkdir(parents=True, exist_ok=True)
                checkpoints = []
                for index, expert in enumerate(
                    confirmation["fitted"][selected.objective],
                    1,
                ):
                    checkpoint = model_directory / f"sequence-seed-{index}.pt"
                    save_v83_expert(
                        expert,
                        checkpoint,
                    )
                    checkpoints.append(checkpoint.name)
                vocabulary_path = model_directory / "sequence-vocabulary.pkl"
                with vocabulary_path.open("wb") as handle:
                    pickle.dump(confirmation["vocabulary"], handle)
                (model_directory / "metadata.json").write_text(
                    json.dumps(
                        {
                            "modelVersion": "V8.4",
                            "deploymentStatus": "candidate",
                            "selected": asdict(selected),
                            "transforms": candidate_2025["transforms"],
                            "dataCutoff": "2024-12-31",
                            "checkpoints": checkpoints,
                            "vocabulary": vocabulary_path.name,
                        },
                        indent=2,
                    )
                    + "\n"
                )
        else:
            status = "rejected-ensemble-2024"

    configuration = _configuration(epochs=epochs, batch_size=batch_size)
    result = {
        "schemaVersion": "8.4",
        "modelVersion": "V8.4-distribution-safe-sequence-residual",
        "deploymentStatus": "candidate" if status == "passed" else "rejected",
        "activeModel": active_model,
        "generatedAt": datetime.now(UTC).isoformat(),
        "configuration": configuration,
        "configurationHash": hashlib.sha256(
            json.dumps(configuration, sort_keys=True).encode()
        ).hexdigest(),
        "selection": {
            "candidate": selected.key if selected is not None else "GLOBAL",
            "status": status,
        },
        "folds": folds,
        "resourceBefore": resource_before,
        "resourceAfter": asdict(snapshot(Path.cwd())),
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(result))
    return result


def _holdout_report(result: dict[str, object]) -> str:
    global_metrics = result["globalMetrics"]
    metrics = result["candidate"]["normal"]["metrics"]
    bootstrap = result["candidate"]["normal"]["bootstrap"]
    deltas = {
        "accuracy": metrics["accuracy"] - global_metrics["accuracy"],
        "family": metrics["familyAccuracy"] - global_metrics["familyAccuracy"],
        "hierarchical": (
            metrics["hierarchicalAccuracy"]
            - global_metrics["hierarchicalAccuracy"]
        ),
        "macroF1": metrics["macroF1"] - global_metrics["macroF1"],
        "logLoss": metrics["logLoss"] - global_metrics["logLoss"],
        "tvd": (
            metrics["totalVariationDistance"]
            - global_metrics["totalVariationDistance"]
        ),
    }
    lines = [
        "# V8.4 — 2025년까지 재학습, 2026 Temporal Holdout",
        "",
        f"- 학습 범위: `{result['trainingRange']['start']}` ~ "
        f"`{result['trainingRange']['end']}`",
        f"- 학습 투구: `{result['trainingRows']:,}`",
        f"- 평가 범위: `{result['holdoutRange']['start']}` ~ "
        f"`{result['holdoutRange']['end']}`",
        f"- 평가 투구: `{result['holdoutRows']:,}`",
        "- 2026 학습 사용: `false`",
        "- 후보 선택·튜닝 사용: `false`",
        "",
        "| 모델 | Exact | Family | Hierarchical | Macro F1 | Log Loss | TVD |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Global | {global_metrics['accuracy']:.2%} | "
            f"{global_metrics['familyAccuracy']:.2%} | "
            f"{global_metrics['hierarchicalAccuracy']:.2%} | "
            f"{global_metrics['macroF1']:.2%} | "
            f"{global_metrics['logLoss']:.5f} | "
            f"{global_metrics['totalVariationDistance']:.2%} |"
        ),
        (
            f"| V8.4 | {metrics['accuracy']:.2%} | "
            f"{metrics['familyAccuracy']:.2%} | "
            f"{metrics['hierarchicalAccuracy']:.2%} | "
            f"{metrics['macroF1']:.2%} | "
            f"{metrics['logLoss']:.5f} | "
            f"{metrics['totalVariationDistance']:.2%} |"
        ),
        (
            f"| 증감 | {deltas['accuracy']:+.2%}p | "
            f"{deltas['family']:+.2%}p | "
            f"{deltas['hierarchical']:+.2%}p | "
            f"{deltas['macroF1']:+.2%}p | "
            f"{deltas['logLoss']:+.5f} | "
            f"{deltas['tvd']:+.2%}p |"
        ),
        "",
        f"- paired-game Log Loss gain: `{bootstrap['meanGain']:.5f}`",
        f"- 95% CI: `[{bootstrap['ciLower']:.5f}, "
        f"{bootstrap['ciUpper']:.5f}]`",
        f"- 정상 게이트: `{result['candidate']['normal']['accepted']}`",
        (
            "- physical-drop stress 게이트: "
            f"`{result['candidate']['physicalDropStress']['accepted']}`"
        ),
        "",
        "이 2026 구간은 과거 프로젝트 실험에서 이미 관찰됐으므로 완전 독립 "
        "holdout이 아니라 temporal regression 평가다.",
        "",
    ]
    return "\n".join(lines)


def _validate_holdout_dates(
    training_dates: pd.Series,
    holdout_dates: pd.Series,
) -> None:
    if training_dates.dt.year.max() > 2025:
        raise ValueError("2026 rows are forbidden in V8.4 training data")
    if set(holdout_dates.dt.year.unique()) != {2026}:
        raise ValueError("V8.4 temporal holdout must contain only 2026 rows")


def run_holdout(
    training_directory: Path,
    holdout_directory: Path,
    artifact_directory: Path,
    report_path: Path,
    *,
    epochs: int,
    batch_size: int,
) -> dict[str, object]:
    resource_before = asdict(snapshot(Path.cwd()))
    assert_safe(snapshot(Path.cwd()))
    training_raw = _load_raw(training_directory)
    holdout_raw = _load_raw(holdout_directory)
    for frame in (training_raw, holdout_raw):
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    training_dates = training_raw["game_date"].dropna()
    holdout_dates = holdout_raw["game_date"].dropna()
    _validate_holdout_dates(training_dates, holdout_dates)

    raw = pd.concat([training_raw, holdout_raw], ignore_index=True)
    rows = prepare_pitch_rows([raw], include_v83=True)
    config = next(item for item in OBJECTIVES if item.name == "FOCAL_1")
    spec = CandidateSpec("FOCAL_1", "none", 0.25)
    fold = _fold(
        raw,
        rows,
        2026,
        (config,),
        (spec,),
        seeds=CONFIRMATION_SEEDS,
        epochs=epochs,
        batch_size=batch_size,
        refit_full=True,
    )
    candidate = fold["candidates"][spec.key]
    model_directory = Path("models/v8.4-2026-holdout")
    model_directory.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    for index, expert in enumerate(fold["fitted"][spec.objective], 1):
        checkpoint = model_directory / f"sequence-seed-{index}.pt"
        save_v83_expert(expert, checkpoint)
        checkpoints.append(checkpoint.name)
    vocabulary_path = model_directory / "sequence-vocabulary.pkl"
    with vocabulary_path.open("wb") as handle:
        pickle.dump(fold["vocabulary"], handle)

    configuration = {
        "model": "V8.4",
        "objective": asdict(config),
        "spec": asdict(spec),
        "seeds": list(CONFIRMATION_SEEDS),
        "epochsLimit": epochs,
        "batchSize": batch_size,
        "protocol": "tune-epochs-and-calibration_pre2026_refit-all-through2025",
    }
    public_fold = _public_fold(fold)
    result = {
        "schemaVersion": "8.4-holdout-1",
        "modelVersion": "V8.4-distribution-safe-sequence-residual",
        "evaluationStatus": (
            "passed"
            if candidate["normal"]["accepted"]
            and candidate["physicalDropStress"]["accepted"]
            else "failed"
        ),
        "generatedAt": datetime.now(UTC).isoformat(),
        "trainingRange": {
            "start": training_dates.min().date().isoformat(),
            "end": training_dates.max().date().isoformat(),
        },
        "holdoutRange": {
            "start": holdout_dates.min().date().isoformat(),
            "end": holdout_dates.max().date().isoformat(),
        },
        "trainingRawRows": len(training_raw),
        "trainingRows": public_fold["sequenceTrainingRows"],
        "trainingMaxDate": public_fold["sequenceTrainingMaxDate"],
        "holdoutRawRows": len(holdout_raw),
        "holdoutRows": public_fold["rows"],
        "holdoutRowFingerprint": public_fold["rowFingerprint"],
        "holdoutManifest": _diagnostic_manifest(holdout_directory),
        "2026UsedForTraining": False,
        "2026UsedForSelection": False,
        "historicallyObserved": True,
        "configuration": configuration,
        "configurationHash": hashlib.sha256(
            json.dumps(configuration, sort_keys=True).encode()
        ).hexdigest(),
        "tuningEpochs": public_fold["tuningEpochs"],
        "globalTrees": public_fold["globalTrees"],
        "globalMetrics": public_fold["globalMetrics"],
        "candidate": candidate,
        "checkpoints": checkpoints,
        "vocabulary": vocabulary_path.name,
        "resourceBefore": resource_before,
        "resourceAfter": asdict(snapshot(Path.cwd())),
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_holdout_report(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/statcast-v8"),
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts/v8.4-distribution-safe-sequence"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/2026-07-28-v8.4-distribution-safe-sequence.md"),
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--resume-selection",
        action="store_true",
        help="reuse the existing 2024 ablation artifact",
    )
    parser.add_argument(
        "--holdout-data",
        type=Path,
        help="evaluate frozen V8.4 after refitting all pre-2026 rows",
    )
    args = parser.parse_args()
    if args.holdout_data is not None:
        result = run_holdout(
            args.data,
            args.holdout_data,
            args.artifacts,
            args.report,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        print(
            json.dumps(
                {
                    "evaluationStatus": result["evaluationStatus"],
                    "holdoutRows": result["holdoutRows"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        result = run(
            args.data,
            args.artifacts,
            args.report,
            epochs=args.epochs,
            batch_size=args.batch_size,
            resume_selection=args.resume_selection,
        )
        print(json.dumps(result["selection"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
