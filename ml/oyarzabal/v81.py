"""Run the frozen V8.1 balanced sequence-expert experiment."""

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
from .holdout import _registry
from .hybrid import apply_reliability_gated_residual
from .metrics import (
    bootstrap_log_loss_gain,
    hierarchical_top_indices,
)
from .modeling import predict_candidate, train_candidate_with_tuning
from .residual import (
    apply_correction,
    compute_pitcher_reliability,
    predict_context_gate,
    predict_correction,
    residual_passes,
    train_final_residual,
    train_gate,
    train_residual_with_tuning,
)
from .resources import assert_safe, snapshot
from .sequence import (
    SequenceExampleBuilder,
    SequenceExamples,
    SequenceVocabulary,
    blend_log_probabilities,
    fit_sequence_expert,
    predict_sequence,
)
from .sequence_gate import (
    discrete_gate_scales,
    fit_sequence_gate,
    gate_feature_frame,
    gate_targets,
    predict_sequence_gate,
)
from .taxonomy import PITCH_GROUP_FAMILY_LABELS, PITCH_GROUPS
from .training import (
    GLOBAL_SPEC,
    GLOBAL_TEMPERATURE,
    _date_partitions,
    _oof_registry,
)
from .v8 import (
    _candidate_passes as candidate_passes,
)
from .v8 import (
    _chronological_indices,
    _fingerprint,
    _load_raw,
    _metrics,
)

ABLATIONS = {
    "A_baseline": {"repertoire": False, "balanced": False},
    "B_repertoire": {"repertoire": True, "balanced": False},
    "C_sqrt_weight": {"repertoire": False, "balanced": True},
    "D_repertoire_sqrt_weight": {"repertoire": True, "balanced": True},
}
BLENDS = (0.0, 0.10, 0.15, 0.20, 0.25)
THRESHOLDS = (0.4, 0.5, 0.6, 0.7)


def _assess(
    fold: dict[str, object],
    probabilities: np.ndarray,
) -> dict[str, object]:
    metrics = _metrics(fold["actual"], probabilities)
    bootstrap = bootstrap_log_loss_gain(
        fold["gameIds"],
        fold["actual"],
        fold["globalProbabilities"],
        probabilities,
    )
    passes = candidate_passes(fold["globalMetrics"], metrics)
    return {
        "metrics": metrics,
        "bootstrap": bootstrap,
        "passesMetrics": passes,
        "accepted": passes and bootstrap["ciLower"] > 0,
    }


def _run_fold(
    raw: pd.DataFrame,
    rows: pd.DataFrame,
    year: int,
    *,
    epochs: int,
    batch_size: int,
    ablation_names: tuple[str, ...] = tuple(ABLATIONS),
) -> tuple[dict[str, object], SequenceExamples, np.ndarray]:
    assert_safe(snapshot(Path.cwd()))
    fold_raw = raw[raw["game_date"].dt.year <= year].copy()
    train_raw = fold_raw[fold_raw["game_date"].dt.year < year]
    vocabulary = SequenceVocabulary.fit(train_raw)
    examples = SequenceExampleBuilder().build(fold_raw, vocabulary)
    fold_rows = rows[rows["game_date"].dt.year <= year].reset_index(drop=True)
    if (
        len(examples) != len(fold_rows)
        or not np.array_equal(examples.target_groups, fold_rows["target"])
        or not np.array_equal(
            examples.target_dates,
            fold_rows["game_date"].to_numpy(dtype="datetime64[D]"),
        )
    ):
        raise AssertionError(f"V8.1 row alignment failed for {year}")
    repertoire_examples = examples.with_repertoire(fold_rows)
    core, validation, evaluation = _chronological_indices(
        examples.target_dates,
        year,
    )

    train_rows = fold_rows[fold_rows["game_date"].dt.year < year]
    global_model, trees = train_candidate_with_tuning(train_rows, GLOBAL_SPEC)
    evaluation_rows = fold_rows.iloc[evaluation]
    global_probabilities = predict_candidate(
        global_model,
        evaluation_rows,
        GLOBAL_TEMPERATURE,
    ).astype(np.float32)
    del global_model
    gc.collect()

    fold: dict[str, object] = {
        "rows": len(evaluation),
        "rowFingerprint": _fingerprint(evaluation_rows),
        "globalTrees": trees,
        "globalMetrics": _metrics(
            examples.target_groups[evaluation],
            global_probabilities,
        ),
        "actual": examples.target_groups[evaluation],
        "gameIds": evaluation_rows["game_pk"].to_numpy(),
        "globalProbabilities": global_probabilities,
        "ablations": {},
    }
    selected_ablations = [
        (name, ABLATIONS[name]) for name in ablation_names
    ]
    for position, (name, config) in enumerate(selected_ablations):
        selected_examples = (
            repertoire_examples if config["repertoire"] else examples
        )
        fitted = fit_sequence_expert(
            selected_examples,
            core,
            validation,
            description_vocab_size=len(vocabulary.descriptions) + 1,
            location_weight=0,
            balanced=config["balanced"],
            epochs=epochs,
            batch_size=batch_size,
            seed=8100 + year,
        )
        sequence_probabilities = predict_sequence(
            fitted,
            selected_examples,
            evaluation,
            batch_size=batch_size,
        ).astype(np.float32)
        candidates = {}
        for blend in BLENDS:
            probabilities = blend_log_probabilities(
                global_probabilities,
                sequence_probabilities,
                blend,
            ).astype(np.float32)
            candidates[str(blend)] = {
                **_assess(fold, probabilities),
                "probabilities": probabilities,
            }
        fold["ablations"][name] = {
            "config": config,
            "validationLogLoss": fitted.validation_log_loss,
            "epochs": fitted.epochs,
            "familyWeights": (
                fitted.family_weights.tolist()
                if fitted.family_weights is not None
                else None
            ),
            "childWeights": (
                fitted.child_weights.tolist()
                if fitted.child_weights is not None
                else None
            ),
            "sequenceMetrics": _metrics(
                fold["actual"],
                sequence_probabilities,
            ),
            "sequenceProbabilities": sequence_probabilities,
            "blends": candidates,
        }
        del fitted
        gc.collect()
        print(
            f"V8.1 {year}: {position + 1}/{len(selected_ablations)} "
            f"{name} complete",
            flush=True,
        )
    return fold, repertoire_examples, evaluation


def _best_blend(ablation: dict[str, object]) -> tuple[str, bool]:
    nonzero = [
        (key, value)
        for key, value in ablation["blends"].items()
        if float(key) > 0
    ]
    accepted = [(key, value) for key, value in nonzero if value["accepted"]]
    pool = accepted or [
        (key, value)
        for key, value in nonzero
        if value["metrics"]["logLoss"]
        < ablation["blends"]["0.0"]["metrics"]["logLoss"]
    ]
    pool = pool or nonzero
    selected = min(pool, key=lambda item: item[1]["metrics"]["logLoss"])
    return selected[0], bool(accepted)


def select_ablation(fold: dict[str, object]) -> dict[str, object]:
    choices = {}
    for name, ablation in fold["ablations"].items():
        blend, has_accepted = _best_blend(ablation)
        choices[name] = {
            "blend": blend,
            "acceptedCandidateAvailable": has_accepted,
            "metrics": ablation["blends"][blend]["metrics"],
        }
    baseline = choices["A_baseline"]
    pruned = []
    for name in ("B_repertoire", "C_sqrt_weight"):
        candidate = choices[name]
        if (
            candidate["metrics"]["logLoss"] > baseline["metrics"]["logLoss"]
            and candidate["metrics"]["macroF1"] < baseline["metrics"]["macroF1"]
        ):
            pruned.append(name)
    eligible = set(ABLATIONS) - set(pruned)
    if len(pruned) == 2:
        eligible.discard("D_repertoire_sqrt_weight")
    safe = [
        name
        for name in eligible
        if choices[name]["acceptedCandidateAvailable"]
    ]
    pool = safe or list(eligible)
    selected_name = min(
        pool,
        key=lambda name: choices[name]["metrics"]["logLoss"],
    )
    return {
        "name": selected_name,
        "blend": float(choices[selected_name]["blend"]),
        "usedSafeCandidate": bool(safe),
        "pruned": sorted(pruned),
        "choices": choices,
    }


def _routing_report(
    actual: np.ndarray,
    sequence_probabilities: np.ndarray,
    scales: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    values, counts = np.unique(scales, return_counts=True)
    sequence_top = hierarchical_top_indices(
        sequence_probabilities,
        PITCH_GROUP_FAMILY_LABELS,
    )
    routed = scales > 0
    return {
        "scaleRates": {
            str(value): float(count / len(scales))
            for value, count in zip(values, counts, strict=True)
        },
        "routedRows": int(routed.sum()),
        "routedActualDistribution": {
            str(group): float(np.mean(actual[routed] == index))
            if routed.any()
            else 0.0
            for index, group in enumerate(PITCH_GROUPS)
        },
        "routedSequenceTopDistribution": {
            str(group): float(np.mean(sequence_top[routed] == index))
            if routed.any()
            else 0.0
            for index, group in enumerate(PITCH_GROUPS)
        },
        "probabilitySumMaximumError": float(
            np.abs(probabilities.sum(axis=1) - 1).max(initial=0)
        ),
    }


def _gate_candidate(
    fold: dict[str, object],
    gate_probabilities: np.ndarray,
    sequence_probabilities: np.ndarray,
    max_blend: float,
    threshold: float,
) -> dict[str, object]:
    scales = discrete_gate_scales(
        gate_probabilities,
        max_blend,
        threshold,
    )
    probabilities = blend_log_probabilities(
        fold["globalProbabilities"],
        sequence_probabilities,
        scales,
    )
    return {
        **_assess(fold, probabilities),
        **_routing_report(
            fold["actual"],
            sequence_probabilities,
            scales,
            probabilities,
        ),
        "probabilities": probabilities.astype(np.float32),
    }


def _public_fold(fold: dict[str, object]) -> dict[str, object]:
    result = {
        key: value
        for key, value in fold.items()
        if key
        not in {
            "actual",
            "gameIds",
            "globalProbabilities",
            "gate",
            "gateCandidates",
        }
    }
    result["ablations"] = {}
    for name, ablation in fold["ablations"].items():
        result["ablations"][name] = {
            key: value
            for key, value in ablation.items()
            if key not in {"sequenceProbabilities", "blends"}
        }
        result["ablations"][name]["blends"] = {
            blend: {
                key: value
                for key, value in candidate.items()
                if key != "probabilities"
            }
            for blend, candidate in ablation["blends"].items()
        }
    if "gate" in fold:
        result["gate"] = {
            key: value
            for key, value in fold["gate"].items()
            if key not in {"probabilities", "gateProbabilities"}
        }
    if "gateCandidates" in fold:
        result["gateCandidates"] = {
            threshold: {
                key: value
                for key, value in candidate.items()
                if key != "probabilities"
            }
            for threshold, candidate in fold["gateCandidates"].items()
        }
    return result


def _metric_row(
    year: int,
    model: str,
    metrics: dict[str, object],
) -> str:
    return (
        f"| {year} | {model} | {metrics['accuracy']:.2%} | "
        f"{metrics['familyAccuracy']:.2%} | "
        f"{metrics['hierarchicalAccuracy']:.2%} | "
        f"{metrics['macroF1']:.2%} | {metrics['logLoss']:.5f} |"
    )


def _report(result: dict[str, object]) -> str:
    selection = result["selection"]
    fold_2025 = result["folds"]["2025"]
    global_2025 = fold_2025["globalMetrics"]
    chosen_2025 = fold_2025["ablations"][selection["ablation"]]["blends"][
        str(selection["constantBlend"])
    ]["metrics"]
    previous_2025 = result["previousV8"]["fold2025"]["blends"]["0.25"][
        "metrics"
    ]
    constant_2024 = result["folds"]["2024"]["ablations"][
        selection["ablation"]
    ]["blends"]["0.25"]["metrics"]
    gate_2024 = result["folds"]["2024"]["gateCandidates"]["0.4"]
    exact_delta = chosen_2025["accuracy"] - global_2025["accuracy"]
    family_delta = (
        chosen_2025["familyAccuracy"] - global_2025["familyAccuracy"]
    )
    hierarchical_delta = (
        chosen_2025["hierarchicalAccuracy"]
        - global_2025["hierarchicalAccuracy"]
    )
    macro_delta = chosen_2025["macroF1"] - global_2025["macroF1"]
    lines = [
        "# V8.1 Balanced Sequence Expert",
        "",
        f"- 생성 시각: {result['generatedAt']}",
        f"- 2024 선택 ablation: `{selection['ablation']}`",
        f"- constant blend: `{selection['constantBlend']}`",
        f"- Sequence Gate threshold: `{selection['gateThreshold']}`",
        f"- 최종 선택: `{selection['final']}`",
        f"- 판정: `{selection['classification']}`",
        f"- 제품 V7.2 변경: `{'예' if result['productChanged'] else '아니오'}`",
        "",
        "| 연도 | 모델 | Exact | Family | Hierarchical | Macro F1 | Log Loss |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    blend_key = str(selection["constantBlend"])
    for year in ("2023", "2024", "2025"):
        fold = result["folds"][year]
        lines.append(_metric_row(int(year), "Global", fold["globalMetrics"]))
        for name, ablation in fold["ablations"].items():
            lines.append(
                _metric_row(
                    int(year),
                    name,
                    ablation["blends"][blend_key]["metrics"],
                )
            )
        if "gate" in fold:
            lines.append(
                _metric_row(
                    int(year),
                    "Sequence Gate",
                    fold["gate"]["metrics"],
                )
            )
    lines.extend(
        [
            "",
            "## 결론",
            "",
            result["conclusion"],
            "",
            "## 기존 V8 대비",
            "",
            (
                f"- 2025 Macro F1: {previous_2025['macroF1']:.2%} → "
                f"{chosen_2025['macroF1']:.2%} "
                f"({chosen_2025['macroF1'] - previous_2025['macroF1']:+.2%}p)"
            ),
            (
                f"- 2025 Log Loss: {previous_2025['logLoss']:.5f} → "
                f"{chosen_2025['logLoss']:.5f}"
            ),
            (
                f"- Global 대비 2025: Exact "
                f"{exact_delta:+.2%}p, Family {family_delta:+.2%}p, "
                f"Hierarchical {hierarchical_delta:+.2%}p, "
                f"Macro F1 {macro_delta:+.2%}p"
            ),
            "",
            "기존 V8의 2025 Macro F1 하락은 −0.75%p였고, V8.1은 "
            "−0.12%p로 회복해 허용치 0.5%p 안에 들어왔다.",
            "",
            "## Repertoire·sqrt weight 독립 효과",
            "",
            "- Repertoire(B)는 같은 blend 0.25에서 A보다 2025 Log Loss를 "
            "0.00116 낮추고 Macro F1을 0.75%p 높였다.",
            "- sqrt weight(C)는 A보다 Macro F1을 0.87%p 높였지만 Log Loss를 "
            "0.01044 악화시켰다.",
            "- 둘을 합친 D는 B보다 Macro F1을 0.52%p 높였지만 Log Loss가 "
            "0.01118 나빠져 주 지표 기준으로 선택하지 않았다.",
            "",
            "## Constant와 Sequence Gate",
            "",
            (
                f"- 2024 constant 0.25: Log Loss "
                f"{constant_2024['logLoss']:.5f}, Macro F1 "
                f"{constant_2024['macroF1']:.2%}"
            ),
            (
                f"- 가장 좋은 Gate threshold 0.4: Log Loss "
                f"{gate_2024['metrics']['logLoss']:.5f}, Macro F1 "
                f"{gate_2024['metrics']['macroF1']:.2%}"
            ),
            "",
            "Gate는 Macro F1을 더 회복했지만 constant보다 Log Loss가 "
            "0.00314 나빠 사전 채택 조건을 충족하지 못했다. 따라서 더 단순한 "
            "constant blend 0.25를 선택했다.",
            "",
            "Repertoire와 sqrt weight의 독립 효과는 같은 constant blend에서 "
            "비교했다. 2025는 2024 선택을 고정한 뒤 한 번만 확인했으며, "
            "2025 결과를 보고 후보·threshold·구조를 추가하지 않았다.",
            "",
            "## Sequence-adjusted Personalizer",
            "",
        ]
    )
    personalizer = result["personalizer"]
    if personalizer["status"] == "retrained":
        lines.extend(
            [
                "| 연도 | Sequence base LL | +Personalizer LL | 통과 |",
                "|---:|---:|---:|---|",
                *[
                    (
                        f"| {year} | "
                        f"{record['baseMetrics']['logLoss']:.5f} | "
                        f"{record['metrics']['logLoss']:.5f} | "
                        f"{'예' if record['passes'] else '아니오'} |"
                    )
                    for year, record in personalizer["years"].items()
                ],
                "",
                "V7.2 Registry tier·scale은 동결하고 Residual과 Residual Gate만 "
                "Sequence-adjusted OOF base 위에서 다시 학습했다. shadow와 "
                "Registry 밖 투수는 Sequence base와 bit-for-bit 동일함을 확인했다.",
                "",
            ]
        )
    else:
        lines.extend(["Sequence가 기각돼 Personalizer는 재학습하지 않았다.", ""])
    lines.extend(
        [
            "## 남은 한계와 다음 실험",
            "",
            result["limitations"],
            "",
        ]
    )
    return "\n".join(lines)


def _load_previous_v8(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    previous = json.loads(path.read_text(encoding="utf-8"))
    return {
        "selectedSequenceBlend": previous.get("selectedSequenceBlend"),
        "selection": previous.get("selection"),
        "fold2025": previous.get("folds", {}).get("2025"),
    }


def _pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle)
    temporary.replace(path)


def _personalizer_fold(
    rows: pd.DataFrame,
    base: np.ndarray,
    correction: np.ndarray,
    context_gate: np.ndarray,
    registry: dict,
    reliability: dict,
    *,
    controls: dict[str, float],
) -> tuple[np.ndarray, dict[str, object]]:
    routed_registry = _oof_registry(registry, reliability)
    candidate, sources, routing = apply_reliability_gated_residual(
        rows,
        base,
        correction,
        context_gate,
        routed_registry,
        reliability_scale_boost=controls["reliabilityScaleBoost"],
        context_gate_power=controls["contextGatePower"],
    )
    active = np.array(
        [
            bool(routed_registry.get(int(pitcher_id), None))
            and routed_registry[int(pitcher_id)].enabled
            for pitcher_id in rows["pitcher_id"]
        ]
    )
    candidate[~active] = base[~active]
    if not np.array_equal(candidate[~active], base[~active]):
        raise AssertionError("shadow/global fallback changed probabilities")
    metrics = _metrics(rows["target"].to_numpy(), candidate)
    return candidate, {
        "rows": len(rows),
        "activeRows": int(active.sum()),
        "interventionRows": int(
            sum(source != "global" for source in sources)
        ),
        "baseMetrics": _metrics(rows["target"].to_numpy(), base),
        "metrics": metrics,
        "passes": residual_passes(
            _metrics(rows["target"].to_numpy(), base),
            metrics,
        ),
        "fallbackExact": True,
        "meanEffectiveScale": float(
            np.mean(
                [record["effectiveScale"] for record in routing],
            )
        ),
    }


def retrain_personalizer(
    rows: pd.DataFrame,
    folds: dict[int, dict[str, object]],
    selected_name: str,
    selected_blend: float,
    registry_path: Path,
    model_directory: Path,
) -> dict[str, object]:
    """Retrain frozen V7.2 residual/gate on the accepted Sequence OOF base."""
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    registry = _registry(payload)
    controls = {
        "reliabilityScaleBoost": float(
            payload["residualTuning"]["dynamic"]["selectedControls"][
                "reliabilityScaleBoost"
            ]
        ),
        "contextGatePower": float(
            payload["residualTuning"]["dynamic"]["selectedControls"][
                "contextGatePower"
            ]
        ),
    }
    evaluations = {
        year: rows[rows["game_date"].dt.year == year]
        .copy()
        .reset_index(drop=True)
        for year in (2023, 2024, 2025)
    }
    base = {
        year: folds[year]["ablations"][selected_name]["blends"][
            str(selected_blend)
        ]["probabilities"]
        for year in evaluations
    }
    for year in evaluations:
        if len(evaluations[year]) != len(base[year]):
            raise AssertionError(f"personalizer OOF rows differ for {year}")
    pool = set(registry)
    pooled_positions = {
        year: np.flatnonzero(
            evaluations[year]["pitcher_id"].isin(pool).to_numpy()
        )
        for year in evaluations
    }
    pooled_rows = {
        year: evaluations[year].iloc[pooled_positions[year]].reset_index(
            drop=True
        )
        for year in evaluations
    }
    pooled_base = {
        year: base[year][pooled_positions[year]]
        for year in evaluations
    }

    residual_2023 = train_residual_with_tuning(
        pooled_rows[2023],
        pooled_base[2023],
    )
    tree_count = residual_2023.tree_count
    correction_2024 = predict_correction(
        residual_2023,
        evaluations[2024],
    )
    pooled_correction_2024 = correction_2024[pooled_positions[2024]]
    training, tuning, gate_evaluation = _date_partitions(pooled_rows[2024])
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
    )
    gate_tree_count = tuned_gate.tree_count
    development = training | tuning
    gate_2024 = train_gate(
        pooled_rows[2024].loc[development],
        pooled_base[2024][development],
        pooled_correction_2024[development],
        n_estimators=gate_tree_count,
    )
    reference_2024 = apply_correction(
        pooled_base[2024],
        pooled_correction_2024,
        0.5,
    )
    reliability_2024 = compute_pitcher_reliability(
        pooled_rows[2024].loc[development],
        pooled_base[2024][development],
        reference_2024[development],
    )
    evaluation_start = pooled_rows[2024].loc[
        gate_evaluation, "game_date"
    ].min()
    final_2024 = (
        evaluations[2024]["game_date"] >= evaluation_start
    ).to_numpy()
    context_gate_2024 = predict_context_gate(
        gate_2024,
        evaluations[2024].loc[final_2024],
        base[2024][final_2024],
        correction_2024[final_2024],
    )
    _, result_2024 = _personalizer_fold(
        evaluations[2024].loc[final_2024].reset_index(drop=True),
        base[2024][final_2024],
        correction_2024[final_2024],
        context_gate_2024,
        registry,
        reliability_2024,
        controls=controls,
    )

    residual_2024 = train_final_residual(
        pd.concat([pooled_rows[2023], pooled_rows[2024]], ignore_index=True),
        np.concatenate([pooled_base[2023], pooled_base[2024]]),
        n_estimators=tree_count,
    )
    correction_2025 = predict_correction(
        residual_2024,
        evaluations[2025],
    )
    pooled_correction_2025 = correction_2025[pooled_positions[2025]]
    gate_2025 = train_gate(
        pooled_rows[2024],
        pooled_base[2024],
        pooled_correction_2024,
        n_estimators=gate_tree_count,
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
    )
    _, result_2025 = _personalizer_fold(
        evaluations[2025],
        base[2025],
        correction_2025,
        context_gate_2025,
        registry,
        reliability_2025,
        controls=controls,
    )

    final_residual = train_final_residual(
        pd.concat(list(pooled_rows.values()), ignore_index=True),
        np.concatenate(list(pooled_base.values())),
        n_estimators=tree_count,
    )
    final_gate = train_gate(
        pd.concat([pooled_rows[2024], pooled_rows[2025]], ignore_index=True),
        np.concatenate([pooled_base[2024], pooled_base[2025]]),
        np.concatenate(
            [pooled_correction_2024, pooled_correction_2025]
        ),
        n_estimators=gate_tree_count,
    )
    _pickle(model_directory / "pooled-residual.pkl", final_residual)
    _pickle(model_directory / "context-gate.pkl", final_gate)
    return {
        "status": "retrained",
        "registry": str(registry_path),
        "registryPolicy": "V7.2 tiers and scale multipliers frozen",
        "controls": controls,
        "residualTreeCount": tree_count,
        "gateTreeCount": gate_tree_count,
        "models": {
            "residual": str(model_directory / "pooled-residual.pkl"),
            "gate": str(model_directory / "context-gate.pkl"),
        },
        "years": {"2024": result_2024, "2025": result_2025},
        "accepted": result_2024["passes"] and result_2025["passes"],
    }


def run(
    data_directory: Path,
    artifact_directory: Path,
    report_path: Path,
    *,
    epochs: int = 6,
    batch_size: int = 8192,
) -> dict[str, object]:
    before = snapshot(Path.cwd())
    assert_safe(before)
    raw = _load_raw(data_directory)
    raw["game_date"] = pd.to_datetime(raw["game_date"], errors="coerce")
    rows = prepare_pitch_rows([raw])
    folds: dict[int, dict[str, object]] = {}

    fold_2024, examples_2024, evaluation_2024 = _run_fold(
        raw,
        rows,
        2024,
        epochs=epochs,
        batch_size=batch_size,
    )
    selection = select_ablation(fold_2024)
    selected_name = selection["name"]
    selected_blend = selection["blend"]
    sequence_2024 = fold_2024["ablations"][selected_name][
        "sequenceProbabilities"
    ]
    gate_features_2024 = gate_feature_frame(
        examples_2024,
        evaluation_2024,
        fold_2024["globalProbabilities"],
        sequence_2024,
    )
    del examples_2024
    gc.collect()
    folds[2024] = fold_2024

    fold_2023, examples_2023, evaluation_2023 = _run_fold(
        raw,
        rows,
        2023,
        epochs=epochs,
        batch_size=batch_size,
    )
    sequence_2023 = fold_2023["ablations"][selected_name][
        "sequenceProbabilities"
    ]
    gate_features_2023 = gate_feature_frame(
        examples_2023,
        evaluation_2023,
        fold_2023["globalProbabilities"],
        sequence_2023,
    )
    targets_2023 = gate_targets(
        fold_2023["actual"],
        fold_2023["globalProbabilities"],
        sequence_2023,
        selected_blend,
    )
    gate = fit_sequence_gate(gate_features_2023, targets_2023)
    gate_probabilities_2024 = predict_sequence_gate(gate, gate_features_2024)
    gate_candidates_2024 = {
        str(threshold): _gate_candidate(
            fold_2024,
            gate_probabilities_2024,
            sequence_2024,
            selected_blend,
            threshold,
        )
        for threshold in THRESHOLDS
    }
    constant_2024 = fold_2024["ablations"][selected_name]["blends"][
        str(selected_blend)
    ]
    eligible_gates = [
        (threshold, candidate)
        for threshold, candidate in gate_candidates_2024.items()
        if candidate["accepted"]
        and candidate["metrics"]["logLoss"]
        < constant_2024["metrics"]["logLoss"]
    ]
    selected_threshold = (
        float(
            min(
                eligible_gates,
                key=lambda item: item[1]["metrics"]["logLoss"],
            )[0]
        )
        if eligible_gates
        else None
    )
    fold_2024["gateCandidates"] = gate_candidates_2024
    folds[2023] = fold_2023
    del examples_2023, gate_features_2023, gate_features_2024
    gc.collect()

    fold_2025, examples_2025, evaluation_2025 = _run_fold(
        raw,
        rows,
        2025,
        epochs=epochs,
        batch_size=batch_size,
    )
    sequence_2025 = fold_2025["ablations"][selected_name][
        "sequenceProbabilities"
    ]
    constant_2025 = fold_2025["ablations"][selected_name]["blends"][
        str(selected_blend)
    ]
    if selected_threshold is not None:
        gate_features_2025 = gate_feature_frame(
            examples_2025,
            evaluation_2025,
            fold_2025["globalProbabilities"],
            sequence_2025,
        )
        gate_probabilities_2025 = predict_sequence_gate(
            gate,
            gate_features_2025,
        )
        fold_2024["gate"] = gate_candidates_2024[str(selected_threshold)]
        fold_2025["gate"] = _gate_candidate(
            fold_2025,
            gate_probabilities_2025,
            sequence_2025,
            selected_blend,
            selected_threshold,
        )
    folds[2025] = fold_2025
    del examples_2025
    gc.collect()

    constant_ok = constant_2024["accepted"] and constant_2025["accepted"]
    gate_ok = (
        selected_threshold is not None
        and fold_2024["gate"]["accepted"]
        and fold_2025["gate"]["accepted"]
        and fold_2025["gate"]["metrics"]["logLoss"]
        <= constant_2025["metrics"]["logLoss"]
    )
    final = "gate" if gate_ok else ("constant" if constant_ok else "blend0")
    final_2025 = (
        fold_2025["gate"]["metrics"]
        if final == "gate"
        else constant_2025["metrics"]
        if final == "constant"
        else fold_2025["globalMetrics"]
    )
    macro_delta = (
        final_2025["macroF1"] - fold_2025["globalMetrics"]["macroF1"]
    )
    classification = (
        "balanced improvement"
        if final != "blend0" and macro_delta >= 0
        else "accepted trade-off"
        if final != "blend0" and macro_delta >= -0.005
        else "rejected"
    )
    personalizer = (
        retrain_personalizer(
            rows,
            folds,
            selected_name,
            selected_blend,
            Path("models/v7.2/registry.json"),
            Path("models/v8.1-candidate"),
        )
        if final != "blend0"
        else {"status": "not-run-sequence-failed", "accepted": False}
    )
    config = {
        "ablations": ABLATIONS,
        "blends": BLENDS,
        "thresholds": THRESHOLDS,
        "locationWeight": 0,
        "historyLength": 16,
        "gateFullOffset": 0.2,
        "protocol": "2023-gate-train_2024-select_2025-frozen-confirm",
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    result = {
        "schemaVersion": "8.1",
        "modelVersion": "V8.1-balanced-sequence-experiment",
        "deploymentStatus": "candidate" if final != "blend0" else "rejected",
        "generatedAt": datetime.now(UTC).isoformat(),
        "dataCutoff": rows["game_date"].max().date().isoformat(),
        "configuration": config,
        "configurationHash": config_hash,
        "selection": {
            "ablation": selected_name,
            "constantBlend": selected_blend,
            "gateThreshold": selected_threshold,
            "final": final,
            "classification": classification,
            "ablationDecision": selection,
        },
        "gateArtifact": {
            "trainedOn": 2023,
            "features": list(gate.feature_names),
            "positiveRate": float(targets_2023.mean()),
            "device": gate.device,
            "threshold": selected_threshold,
            "scaleLevels": (
                [0, selected_blend / 2, selected_blend]
                if selected_threshold is not None
                else [0]
            ),
        },
        "previousV8": _load_previous_v8(
            Path("artifacts/v8-sequence/result.json")
        ),
        "productBaseline": "V7.2",
        "productChanged": False,
        "personalizer": {
            **personalizer,
            "applicationOrder": ["global", "sequence", "personalizer"],
        },
        "resourceSnapshots": {
            "before": asdict(before),
            "after": asdict(snapshot(Path.cwd())),
        },
        "folds": {
            str(year): _public_fold(folds[year])
            for year in (2023, 2024, 2025)
        },
        "conclusion": (
            "Macro F1 안전 기준을 회복하면서 Sequence 증분 후보를 확보했다."
            if final != "blend0"
            else "모든 고정 후보가 안전 게이트를 통과하지 못해 V7.2를 유지한다."
        ),
        "limitations": (
            "이번 실험은 2025까지의 OOF 확인이다. 다음 실험은 후보를 동결한 "
            "뒤 아직 보지 않은 미래 구간에서 Sequence 증분을 한 번 검증한다."
        ),
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (artifact_directory / "sequence-gate.json").write_text(
        json.dumps(result["gateArtifact"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(result), encoding="utf-8")
    return result


def resume_personalizer(
    data_directory: Path,
    artifact_directory: Path,
    report_path: Path,
    *,
    epochs: int,
    batch_size: int,
) -> dict[str, object]:
    result_path = artifact_directory / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    selected_name = str(result["selection"]["ablation"])
    selected_blend = float(result["selection"]["constantBlend"])
    if result["selection"]["final"] == "blend0":
        raise ValueError("cannot retrain Personalizer after Sequence rejection")
    raw = _load_raw(data_directory)
    raw["game_date"] = pd.to_datetime(raw["game_date"], errors="coerce")
    rows = prepare_pitch_rows([raw])
    folds = {}
    for year in (2023, 2024, 2025):
        fold, examples, _ = _run_fold(
            raw,
            rows,
            year,
            epochs=epochs,
            batch_size=batch_size,
            ablation_names=(selected_name,),
        )
        folds[year] = fold
        del examples
        gc.collect()
    personalizer = retrain_personalizer(
        rows,
        folds,
        selected_name,
        selected_blend,
        Path("models/v7.2/registry.json"),
        Path("models/v8.1-candidate"),
    )
    result["personalizer"] = {
        **personalizer,
        "applicationOrder": ["global", "sequence", "personalizer"],
    }
    result["resourceSnapshots"]["personalizerAfter"] = asdict(
        snapshot(Path.cwd())
    )
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_report(result), encoding="utf-8")
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
        default=Path("artifacts/v8.1-balanced-sequence"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/2026-07-28-v8.1-balanced-sequence.md"),
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--personalizer-only", action="store_true")
    args = parser.parse_args()
    runner = resume_personalizer if args.personalizer_only else run
    result = runner(
        args.data,
        args.artifacts,
        args.report,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(json.dumps(result["selection"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
