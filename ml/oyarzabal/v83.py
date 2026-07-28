"""Train and evaluate the V8.3 robust Global-conditioned sequence residual."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .features import V83_CONTEXT_FEATURES, prepare_pitch_rows
from .metrics import bootstrap_log_loss_gain
from .modeling import predict_candidate, train_candidate_with_tuning
from .resources import assert_safe, snapshot
from .sequence import SequenceExampleBuilder, SequenceExamples, SequenceVocabulary
from .training import GLOBAL_SPEC, GLOBAL_TEMPERATURE
from .v8 import _chronological_indices, _fingerprint, _load_raw, _metrics
from .v83_sequence import (
    apply_hierarchical_calibration,
    fit_hierarchical_calibration,
    fit_v83_expert,
    hierarchical_residual_probabilities,
    predict_v83_deltas,
    save_v83_expert,
)

SCALES = (0.25, 0.5, 0.75, 1.0)
SEEDS = (8301, 8302, 8303)
PHYSICAL_CONTEXT = tuple(
    name
    for name in V83_CONTEXT_FEATURES
    if "catcher" not in name and "battery" not in name
)
CATCHER_CONTEXT = tuple(
    name for name in V83_CONTEXT_FEATURES if "catcher" in name or "battery" in name
)


@dataclass(frozen=True)
class V83Config:
    name: str
    block_dropout: float
    balance_strength: float
    context: str


CORE_CONFIGS = (
    V83Config("R0_residual_mask", 0.0, 0.0, "base"),
    V83Config("R1_block_dropout", 0.2, 0.0, "base"),
    V83Config("R2_balance_010", 0.2, 0.1, "base"),
    V83Config("R3_balance_020", 0.2, 0.2, "base"),
)


def _examples_for_context(
    base: SequenceExamples,
    rows: pd.DataFrame,
    context: str,
) -> SequenceExamples:
    if context == "base":
        return base
    physical = base.with_current_context(rows, PHYSICAL_CONTEXT)
    if context == "physical":
        return physical
    if context == "catcher":
        return physical.with_current_context(rows, CATCHER_CONTEXT)
    raise ValueError(f"unknown V8.3 context: {context}")


def _stress_examples(
    examples: SequenceExamples,
    *,
    context: str,
) -> SequenceExamples:
    source = examples.source_numeric.copy()
    source[:, 3:] = np.nan
    current = examples.current_numeric.copy()
    if context in {"physical", "catcher"}:
        base_width = current.shape[1] - len(PHYSICAL_CONTEXT)
        if context == "catcher":
            base_width -= len(CATCHER_CONTEXT)
        current[:, base_width : base_width + len(PHYSICAL_CONTEXT)] = np.nan
    return replace(examples, source_numeric=source, current_numeric=current)


def _assess(
    actual: np.ndarray,
    games: np.ndarray,
    global_probabilities: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, object]:
    global_metrics = _metrics(actual, global_probabilities)
    metrics = _metrics(actual, candidate)
    bootstrap = bootstrap_log_loss_gain(
        games,
        actual,
        global_probabilities,
        candidate,
        samples=500,
    )
    reasons = _failure_reasons(global_metrics, metrics, bootstrap)
    return {
        "metrics": metrics,
        "bootstrap": bootstrap,
        "failureReasons": reasons,
        "accepted": not reasons,
    }


def _failure_reasons(
    reference: dict[str, object],
    candidate: dict[str, object],
    bootstrap: dict[str, object],
) -> list[str]:
    reasons = []
    if candidate["logLoss"] >= reference["logLoss"]:
        reasons.append("log_loss_not_improved")
    if bootstrap["ciLower"] <= 0:
        reasons.append("log_loss_gain_ci_not_positive")
    for name, reason in (
        ("accuracy", "accuracy_drop_gt_0.5pp"),
        ("familyAccuracy", "family_accuracy_drop_gt_0.5pp"),
        ("hierarchicalAccuracy", "hierarchical_accuracy_drop_gt_0.5pp"),
        ("macroF1", "macro_f1_drop_gt_0.5pp"),
    ):
        if candidate[name] < reference[name] - 0.005:
            reasons.append(reason)
    for name, reason in (
        ("totalVariationDistance", "tvd_regression_gt_0.5pp"),
        (
            "maxClassCalibrationError",
            "calibration_regression_gt_0.5pp",
        ),
    ):
        if candidate[name] > reference[name] + 0.005:
            reasons.append(reason)
    major_zero = [
        name
        for name in candidate["zeroRecallClasses"]
        if candidate["actualDistribution"][name] >= 0.05
    ]
    if major_zero:
        reasons.append("major_zero_recall")
    return reasons


def _fit_candidate(
    config: V83Config,
    examples: SequenceExamples,
    global_probabilities: np.ndarray,
    core: np.ndarray,
    validation: np.ndarray,
    evaluation: np.ndarray,
    *,
    description_vocab_size: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    fitted = fit_v83_expert(
        examples,
        core,
        validation,
        global_probabilities,
        description_vocab_size=description_vocab_size,
        balance_strength=config.balance_strength,
        block_dropout=config.block_dropout,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
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
    stress = _stress_examples(examples, context=config.context)
    stress_delta = predict_v83_deltas(
        fitted,
        stress,
        evaluation,
        global_probabilities,
        batch_size=batch_size,
    )
    result = {
        "fitted": fitted,
        "validationLogLoss": fitted.validation_log_loss,
        "epochs": fitted.epochs,
        "scales": {},
    }
    for scale in SCALES:
        validation_probabilities = hierarchical_residual_probabilities(
            global_probabilities[validation],
            *validation_delta,
            scale,
        )
        calibration = fit_hierarchical_calibration(
            examples.target_groups[validation],
            validation_probabilities,
        )
        normal = apply_hierarchical_calibration(
            hierarchical_residual_probabilities(
                global_probabilities[evaluation],
                *evaluation_delta,
                scale,
            ),
            calibration,
        )
        stressed = apply_hierarchical_calibration(
            hierarchical_residual_probabilities(
                global_probabilities[evaluation],
                *stress_delta,
                scale,
            ),
            calibration,
        )
        result["scales"][str(scale)] = {
            "calibration": calibration.tolist(),
            "probabilities": normal.astype(np.float32),
            "stressProbabilities": stressed.astype(np.float32),
        }
    return result


def _public_candidate(
    candidate: dict[str, object],
    actual: np.ndarray,
    games: np.ndarray,
    global_probabilities: np.ndarray,
) -> dict[str, object]:
    return {
        "validationLogLoss": candidate["validationLogLoss"],
        "epochs": candidate["epochs"],
        "scales": {
            scale: {
                "calibration": values["calibration"],
                "normal": _assess(
                    actual,
                    games,
                    global_probabilities,
                    values["probabilities"],
                ),
                "physicalDropStress": _assess(
                    actual,
                    games,
                    global_probabilities,
                    values["stressProbabilities"],
                ),
            }
            for scale, values in candidate["scales"].items()
        },
    }


def _select(candidates: dict[str, dict[str, object]]) -> tuple[str, float]:
    passing = []
    for name, candidate in candidates.items():
        for scale, result in candidate["public"]["scales"].items():
            if (
                result["normal"]["accepted"]
                and result["physicalDropStress"]["accepted"]
            ):
                passing.append((name, float(scale), result["normal"]["metrics"]))
    if not passing:
        return "GLOBAL", 0.0
    return min(
        passing,
        key=lambda item: (item[2]["logLoss"], -item[2]["macroF1"]),
    )[:2]


def _fold(
    raw: pd.DataFrame,
    rows: pd.DataFrame,
    year: int,
    configs: tuple[V83Config, ...],
    *,
    epochs: int,
    batch_size: int,
    seeds: tuple[int, ...] | None = None,
) -> tuple[dict[str, object], dict[str, dict[str, object]], SequenceExamples]:
    fold_raw = raw[raw["game_date"].dt.year <= year].copy()
    fold_rows = rows[rows["game_date"].dt.year <= year].reset_index(drop=True)
    train_raw = fold_raw[fold_raw["game_date"].dt.year < year]
    vocabulary = SequenceVocabulary.fit(train_raw)
    examples = SequenceExampleBuilder().build(fold_raw, vocabulary)
    if len(examples) != len(fold_rows):
        raise AssertionError(f"V8.3 row alignment failed for {year}")
    base = examples.with_repertoire(fold_rows)
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
    candidates = {}
    for config in configs:
        selected_examples = _examples_for_context(base, fold_rows, config.context)
        members = [
            _fit_candidate(
                config,
                selected_examples,
                global_probabilities,
                core,
                validation,
                evaluation,
                description_vocab_size=len(vocabulary.descriptions) + 1,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
            )
            for seed in (seeds or (8300 + year,))
        ]
        fitted = {
            "fittedExperts": [member["fitted"] for member in members],
            "validationLogLoss": float(
                np.mean([member["validationLogLoss"] for member in members])
            ),
            "epochs": [member["epochs"] for member in members],
            "scales": {},
        }
        for scale in map(str, SCALES):
            fitted["scales"][scale] = {
                "calibration": [
                    member["scales"][scale]["calibration"] for member in members
                ],
                "probabilities": np.mean(
                    [member["scales"][scale]["probabilities"] for member in members],
                    axis=0,
                ).astype(np.float32),
                "stressProbabilities": np.mean(
                    [
                        member["scales"][scale]["stressProbabilities"]
                        for member in members
                    ],
                    axis=0,
                ).astype(np.float32),
            }
        fitted["public"] = _public_candidate(
            fitted,
            actual,
            games,
            global_probabilities[evaluation],
        )
        candidates[config.name] = fitted
        print(f"V8.3 {year}: {config.name} complete", flush=True)
    return (
        {
            "year": year,
            "rows": len(evaluation),
            "rowFingerprint": _fingerprint(fold_rows.iloc[evaluation]),
            "globalTrees": trees,
            "globalMetrics": _metrics(
                actual,
                global_probabilities[evaluation],
            ),
            "evaluation": evaluation,
            "globalProbabilities": global_probabilities,
            "actual": actual,
            "games": games,
            "vocabulary": vocabulary,
        },
        candidates,
        base,
    )


def _report(result: dict[str, object]) -> str:
    selection = result["selection"]
    lines = [
        "# V8.3 Robust Global-Conditioned Sequence Residual",
        "",
        f"- 생성 시각: {result['generatedAt']}",
        f"- 선택 후보: `{selection['config']}`",
        f"- sequence scale: `{selection['scale']}`",
        f"- 판정: `{selection['status']}`",
        "",
        "| 연도 | 모델 | Exact | Family | Hierarchical | Macro F1 | Log Loss |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for year, fold in result["folds"].items():
        metrics = fold["globalMetrics"]
        lines.append(
            f"| {year} | Global | {metrics['accuracy']:.2%} | "
            f"{metrics['familyAccuracy']:.2%} | "
            f"{metrics['hierarchicalAccuracy']:.2%} | "
            f"{metrics['macroF1']:.2%} | {metrics['logLoss']:.5f} |"
        )
        if fold.get("selectedMetrics"):
            metrics = fold["selectedMetrics"]
            lines.append(
                f"| {year} | V8.3 | {metrics['accuracy']:.2%} | "
                f"{metrics['familyAccuracy']:.2%} | "
                f"{metrics['hierarchicalAccuracy']:.2%} | "
                f"{metrics['macroF1']:.2%} | {metrics['logLoss']:.5f} |"
            )
    fold_2024 = result["folds"]["2024"]
    if "ablation" in fold_2024:
        lines.extend(
            [
                "",
                "## 2024 ablation 최선 결과",
                "",
                "| 후보 | Scale | Macro F1 | Log Loss | 정상 실패 | 결측 실패 |",
                "|---|---:|---:|---:|---|---|",
            ]
        )
        for name, candidate in fold_2024["ablation"].items():
            scale, values = min(
                candidate["scales"].items(),
                key=lambda item: item[1]["normal"]["metrics"]["logLoss"],
            )
            normal = values["normal"]
            stress = values["physicalDropStress"]
            lines.append(
                f"| {name} | {scale} | {normal['metrics']['macroF1']:.2%} | "
                f"{normal['metrics']['logLoss']:.5f} | "
                f"{', '.join(normal.get('failureReasons', [])) or '-'} | "
                f"{', '.join(stress.get('failureReasons', [])) or '-'} |"
            )
    lines.extend(
        [
            "",
            "2026 공개 구간은 회귀 진단 전용이며 후보 선택에 사용하지 않는다.",
            "모든 rolling·physical·catcher 값은 현재 투구를 제외한다.",
            (
                "2024의 모든 후보가 안전 게이트를 통과하지 못해 "
                "2025·Personalizer 단계는 실행하지 않았다."
                if selection["status"] == "rejected"
                else ""
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _diagnostic_manifest(directory: Path | None) -> dict[str, object] | None:
    if directory is None or not (directory / "manifest.json").exists():
        return None
    raw = json.loads((directory / "manifest.json").read_text())
    return {
        "start": raw["start"],
        "end": raw["end"],
        "schema": raw.get("schema"),
        "schemaFingerprint": raw.get("schemaFingerprint"),
        "rows": sum(int(shard.get("rows", 0)) for shard in raw["shards"]),
        "probe": raw.get("probe"),
        "shards": [
            {
                "file": Path(shard["path"]).name,
                "rows": shard.get("rows"),
                "sha256": shard.get("sha256"),
            }
            for shard in raw["shards"]
        ],
    }


def run(
    data_directory: Path,
    diagnostic_directory: Path | None,
    artifact_directory: Path,
    report_path: Path,
    *,
    epochs: int,
    batch_size: int,
) -> dict[str, object]:
    assert_safe(snapshot(Path.cwd()))
    raw = _load_raw(data_directory)
    if diagnostic_directory is not None:
        raw = pd.concat(
            [raw, _load_raw(diagnostic_directory)],
            ignore_index=True,
        )
    raw["game_date"] = pd.to_datetime(raw["game_date"], errors="coerce")
    rows = prepare_pitch_rows([raw], include_v83=True)

    fold_2024, core_candidates, base_2024 = _fold(
        raw,
        rows,
        2024,
        CORE_CONFIGS,
        epochs=epochs,
        batch_size=batch_size,
    )
    core_name, _ = _select(core_candidates)
    selected_core = next(
        (config for config in CORE_CONFIGS if config.name == core_name),
        CORE_CONFIGS[1],
    )
    feature_configs = (
        V83Config(
            "R4_physical_drift",
            0.2,
            selected_core.balance_strength,
            "physical",
        ),
        V83Config(
            "R5_catcher_profile",
            0.2,
            selected_core.balance_strength,
            "catcher",
        ),
    )
    fold_features, feature_candidates, _ = _fold(
        raw,
        rows,
        2024,
        feature_configs,
        epochs=epochs,
        batch_size=batch_size,
    )
    del base_2024
    all_candidates = {**core_candidates, **feature_candidates}
    selected_name, selected_scale = _select(all_candidates)
    selection_status = "selected" if selected_name != "GLOBAL" else "rejected"
    selected_config = next(
        (
            config
            for config in (*CORE_CONFIGS, *feature_configs)
            if config.name == selected_name
        ),
        None,
    )

    folds: dict[str, object] = {
        "2024": {
            **{
                key: value
                for key, value in fold_2024.items()
                if key
                not in {
                    "evaluation",
                    "globalProbabilities",
                    "actual",
                    "games",
                    "vocabulary",
                }
            },
            "ablation": {
                name: candidate["public"] for name, candidate in all_candidates.items()
            },
        }
    }
    if selected_config is not None:
        fold_2024_ensemble, ensemble_2024, _ = _fold(
            raw,
            rows,
            2024,
            (selected_config,),
            epochs=epochs,
            batch_size=batch_size,
            seeds=SEEDS,
        )
        chosen_2024 = ensemble_2024[selected_name]["public"]["scales"][
            str(selected_scale)
        ]
        folds["2024"]["selectedMetrics"] = chosen_2024["normal"]["metrics"]
        folds["2024"]["selectedStressMetrics"] = chosen_2024["physicalDropStress"][
            "metrics"
        ]
        folds["2024"]["ensembleAccepted"] = bool(
            chosen_2024["normal"]["accepted"]
            and chosen_2024["physicalDropStress"]["accepted"]
        )
        fold_2025, candidates_2025, _ = _fold(
            raw,
            rows,
            2025,
            (selected_config,),
            epochs=epochs,
            batch_size=batch_size,
            seeds=SEEDS,
        )
        candidate_2025 = candidates_2025[selected_name]
        chosen_2025 = candidate_2025["public"]["scales"][str(selected_scale)]
        passed_2025 = bool(
            folds["2024"]["ensembleAccepted"]
            and chosen_2025["normal"]["accepted"]
            and chosen_2025["physicalDropStress"]["accepted"]
        )
        selection_status = "active" if passed_2025 else "rejected-2025"
        folds["2025"] = {
            **{
                key: value
                for key, value in fold_2025.items()
                if key
                not in {
                    "evaluation",
                    "globalProbabilities",
                    "actual",
                    "games",
                    "vocabulary",
                }
            },
            "selectedMetrics": chosen_2025["normal"]["metrics"],
            "selectedStressMetrics": chosen_2025["physicalDropStress"]["metrics"],
            "accepted": passed_2025,
        }
        model_directory = Path("models/v8.3")
        checkpoint_fold = fold_2025
        checkpoint_candidate = candidate_2025
        if diagnostic_directory is not None:
            fold_2026, candidates_2026, _ = _fold(
                raw,
                rows,
                2026,
                (selected_config,),
                epochs=epochs,
                batch_size=batch_size,
                seeds=SEEDS,
            )
            checkpoint_fold = fold_2026
            checkpoint_candidate = candidates_2026[selected_name]
            chosen_2026 = checkpoint_candidate["public"]["scales"][str(selected_scale)]
            folds["2026"] = {
                **{
                    key: value
                    for key, value in fold_2026.items()
                    if key
                    not in {
                        "evaluation",
                        "globalProbabilities",
                        "actual",
                        "games",
                        "vocabulary",
                    }
                },
                "diagnosticOnly": True,
                "selectedMetrics": chosen_2026["normal"]["metrics"],
                "selectedStressMetrics": chosen_2026["physicalDropStress"]["metrics"],
            }
        checkpoints = []
        for index, expert in enumerate(checkpoint_candidate["fittedExperts"], 1):
            path = model_directory / f"sequence-seed-{index}.pt"
            save_v83_expert(expert, path)
            checkpoints.append(str(path))
        with (model_directory / "sequence-vocabulary.pkl").open("wb") as handle:
            pickle.dump(checkpoint_fold["vocabulary"], handle)
        (model_directory / "sequence-metadata.json").parent.mkdir(
            parents=True, exist_ok=True
        )
        (model_directory / "sequence-metadata.json").write_text(
            json.dumps(
                {
                    "modelVersion": "V8.3",
                    "dataCutoff": "2025-12-31",
                    "deploymentStatus": selection_status,
                    "config": asdict(selected_config),
                    "sequenceScale": selected_scale,
                    "checkpoints": checkpoints,
                    "calibration": checkpoint_candidate["scales"][str(selected_scale)][
                        "calibration"
                    ],
                    "schema": {
                        "physicalContext": list(PHYSICAL_CONTEXT),
                        "catcherContext": list(CATCHER_CONTEXT),
                    },
                },
                indent=2,
            )
            + "\n"
        )

    configuration = {
        "coreConfigs": [asdict(config) for config in CORE_CONFIGS],
        "featureConfigs": [asdict(config) for config in feature_configs],
        "scales": list(SCALES),
        "ensembleSeeds": list(SEEDS),
        "epochs": epochs,
        "batchSize": batch_size,
        "historyLength": 16,
        "bootstrapSamples": 500,
        "protocol": "2024-select_2025-frozen-confirm_2026-diagnostic-only",
    }
    result = {
        "schemaVersion": "8.3",
        "modelVersion": "V8.3-global-conditioned-sequence-residual",
        "deploymentStatus": (
            "active" if selection_status == "active" else "rejected"
        ),
        "activeModel": "V8.3" if selection_status == "active" else "V7.2",
        "generatedAt": datetime.now(UTC).isoformat(),
        "configuration": configuration,
        "configurationHash": hashlib.sha256(
            json.dumps(configuration, sort_keys=True).encode()
        ).hexdigest(),
        "selection": {
            "config": selected_name,
            "scale": selected_scale,
            "status": selection_status,
        },
        "folds": folds,
        "diagnosticData": _diagnostic_manifest(diagnostic_directory),
        "resourceAfter": asdict(snapshot(Path.cwd())),
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/statcast-v8"),
    )
    parser.add_argument("--diagnostic-data", type=Path)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts/v8.3-robust-sequence"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/2026-07-28-v8.3-robust-sequence.md"),
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    result = run(
        args.data,
        args.diagnostic_data,
        args.artifacts,
        args.report,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(json.dumps(result["selection"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
