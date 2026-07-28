"""Evaluate the V8 causal sequence expert on frozen chronological folds."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .features import prepare_pitch_rows
from .metrics import (
    bootstrap_log_loss_gain,
    evaluate_diagnostics,
)
from .modeling import predict_candidate, train_candidate_with_tuning
from .resources import assert_safe, snapshot
from .sequence import (
    SequenceExampleBuilder,
    SequenceVocabulary,
    blend_log_probabilities,
    fit_sequence_expert,
    predict_sequence,
)
from .taxonomy import PITCH_GROUP_FAMILY_LABELS, PITCH_GROUPS
from .training import GLOBAL_SPEC, GLOBAL_TEMPERATURE

V8_FOLDS = (2023, 2024, 2025)
SEQUENCE_BLEND_WEIGHTS = (0.0, 0.25, 0.5)
LOCATION_WEIGHTS = (0.0, 0.1, 0.25, 0.5)
V74_PERSONALIZATION = {
    "reliabilityScaleBoost": 1.5,
    "contextGatePower": 0.5,
    "fullTierBoost": 4.0,
    "limitedTierBoost": 4.0,
    "shadowTierBoost": 0.0,
    "scaleCap": 0.5,
    "jsDivergenceCap": 0.05,
    "classProbabilityShiftCap": 0.2,
}


def _load_raw(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Statcast shards in {directory}")
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)


def _metrics(actual: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    return evaluate_diagnostics(
        actual,
        probabilities,
        labels=range(len(PITCH_GROUPS)),
        names=[str(group) for group in PITCH_GROUPS],
        family_labels=PITCH_GROUP_FAMILY_LABELS,
    )


def _fingerprint(rows: pd.DataFrame) -> str:
    columns = [
        "game_date",
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "pitcher_id",
        "target",
    ]
    hashed = pd.util.hash_pandas_object(rows[columns], index=False)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()


def _chronological_indices(
    dates: np.ndarray,
    evaluation_year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    training = np.flatnonzero(years < evaluation_year)
    evaluation = np.flatnonzero(years == evaluation_year)
    unique_dates = np.unique(dates[training])
    if len(unique_dates) < 2 or not len(evaluation):
        raise ValueError(f"empty V8 fold for {evaluation_year}")
    cutoff = unique_dates[max(1, int(len(unique_dates) * 0.8))]
    core = training[dates[training] < cutoff]
    validation = training[dates[training] >= cutoff]
    if not len(core) or not len(validation):
        raise ValueError(f"empty V8 tuning split for {evaluation_year}")
    return core, validation, evaluation


def _candidate_passes(
    global_metrics: dict[str, object],
    candidate_metrics: dict[str, object],
) -> bool:
    major_zero = [
        name
        for name in candidate_metrics["zeroRecallClasses"]
        if candidate_metrics["actualDistribution"][name] >= 0.05
    ]
    return (
        candidate_metrics["logLoss"] < global_metrics["logLoss"]
        and candidate_metrics["accuracy"] >= global_metrics["accuracy"] - 0.005
        and candidate_metrics["familyAccuracy"]
        >= global_metrics["familyAccuracy"] - 0.005
        and candidate_metrics["hierarchicalAccuracy"]
        >= global_metrics["hierarchicalAccuracy"] - 0.005
        and candidate_metrics["macroF1"] >= global_metrics["macroF1"] - 0.005
        and candidate_metrics["totalVariationDistance"]
        <= global_metrics["totalVariationDistance"] + 0.005
        and candidate_metrics["maxClassCalibrationError"]
        <= global_metrics["maxClassCalibrationError"] + 0.005
        and not major_zero
    )


def select_blend(
    folds: dict[int, dict[str, object]],
) -> tuple[float, dict[str, object]]:
    """Select only from frozen weights; 2023 is diagnostic, 2024/25 decide."""
    candidates = []
    for weight in SEQUENCE_BLEND_WEIGHTS[1:]:
        key = str(weight)
        passes = all(
            _candidate_passes(
                folds[year]["globalMetrics"],
                folds[year]["blends"][key]["metrics"],
            )
            for year in (2024, 2025)
        )
        actual = np.concatenate(
            [folds[year]["actual"] for year in (2024, 2025)]
        )
        reference = np.concatenate(
            [folds[year]["globalProbabilities"] for year in (2024, 2025)]
        )
        candidate = np.concatenate(
            [folds[year]["blends"][key]["probabilities"] for year in (2024, 2025)]
        )
        games = np.concatenate(
            [folds[year]["gameIds"] for year in (2024, 2025)]
        )
        bootstrap = bootstrap_log_loss_gain(
            games,
            actual,
            reference,
            candidate,
        )
        candidates.append(
            {
                "weight": weight,
                "passesMetrics": passes,
                "bootstrap": bootstrap,
                "accepted": passes and bootstrap["ciLower"] > 0,
            }
        )
    accepted = [item for item in candidates if item["accepted"]]
    selected = max(
        accepted,
        key=lambda item: item["bootstrap"]["meanGain"],
        default=None,
    )
    return (
        float(selected["weight"]) if selected else 0.0,
        {"candidates": candidates, "accepted": selected is not None},
    )


def _public_fold(value: dict[str, object]) -> dict[str, object]:
    public = {
        key: item
        for key, item in value.items()
        if key
        not in {
            "actual",
            "gameIds",
            "globalProbabilities",
            "sequenceProbabilities",
        }
    }
    public["blends"] = {
        weight: {"metrics": candidate["metrics"]}
        for weight, candidate in value["blends"].items()
    }
    return public


def _report(result: dict[str, object]) -> str:
    lines = [
        "# V8 Transformer Sequence Expert",
        "",
        f"- 생성 시각: {result['generatedAt']}",
        f"- 선택 blend: `{result['selectedSequenceBlend']}`",
        f"- 선택 결과: `{'통과' if result['selection']['accepted'] else '미통과'}`",
        (
            "- 개인화: V7.4 scale과 safety cap은 동결. "
            "Sequence base gate 통과 후에만 재학습"
        ),
        "",
        "| 평가 연도 | 모델 | Exact | Family | Hierarchical | Macro F1 | Log Loss |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for year, fold in result["folds"].items():
        rows = [("Global", fold["globalMetrics"])]
        rows.append(("Transformer", fold["sequenceMetrics"]))
        rows.extend(
            (f"Blend {weight}", candidate["metrics"])
            for weight, candidate in fold["blends"].items()
            if float(weight) > 0
        )
        for name, metrics in rows:
            lines.append(
                f"| {year} | {name} | {metrics['accuracy']:.2%} | "
                f"{metrics['familyAccuracy']:.2%} | "
                f"{metrics['hierarchicalAccuracy']:.2%} | "
                f"{metrics['macroF1']:.2%} | {metrics['logLoss']:.5f} |"
            )
    lines.extend(
        [
            "",
            "이번 표는 Sequence의 Global 대비 증분 평가다. 후보가 탈락해 "
            "제품 V7.4는 변경하지 않았고 Personalizer 재학습도 실행하지 않았다.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    data_directory: Path,
    artifact_directory: Path,
    report_path: Path,
    *,
    epochs: int = 6,
    batch_size: int = 2048,
) -> dict[str, object]:
    raw = _load_raw(data_directory)
    raw["game_date"] = pd.to_datetime(raw["game_date"], errors="coerce")
    rows = prepare_pitch_rows([raw])
    folds: dict[int, dict[str, object]] = {}

    for year in V8_FOLDS:
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
            raise AssertionError(f"V8 row alignment failed for {year}")
        core, validation, evaluation = _chronological_indices(
            examples.target_dates,
            year,
        )

        train_rows = fold_rows[fold_rows["game_date"].dt.year < year]
        global_model, trees = train_candidate_with_tuning(
            train_rows,
            GLOBAL_SPEC,
        )
        evaluation_rows = fold_rows.iloc[evaluation]
        global_probabilities = predict_candidate(
            global_model,
            evaluation_rows,
            GLOBAL_TEMPERATURE,
        )
        del global_model
        gc.collect()

        fitted = [
            fit_sequence_expert(
                examples,
                core,
                validation,
                description_vocab_size=len(vocabulary.descriptions) + 1,
                location_weight=location_weight,
                epochs=epochs,
                batch_size=batch_size,
                seed=737 + year,
            )
            for location_weight in LOCATION_WEIGHTS
        ]
        selected_expert = min(
            fitted,
            key=lambda candidate: candidate.validation_log_loss,
        )
        sequence_probabilities = predict_sequence(
            selected_expert,
            examples,
            evaluation,
            batch_size=batch_size,
        )
        actual = examples.target_groups[evaluation]
        blends = {}
        for weight in SEQUENCE_BLEND_WEIGHTS:
            probabilities = blend_log_probabilities(
                global_probabilities,
                sequence_probabilities,
                weight,
            )
            blends[str(weight)] = {
                "metrics": _metrics(actual, probabilities),
                "probabilities": probabilities,
            }
        folds[year] = {
            "rows": len(actual),
            "rowFingerprint": _fingerprint(evaluation_rows),
            "globalTrees": trees,
            "locationCandidates": [
                {
                    "weight": expert.location_weight,
                    "validationLogLoss": expert.validation_log_loss,
                    "epochs": expert.epochs,
                }
                for expert in fitted
            ],
            "selectedLocationWeight": selected_expert.location_weight,
            "globalMetrics": _metrics(actual, global_probabilities),
            "sequenceMetrics": _metrics(actual, sequence_probabilities),
            "blends": blends,
            "actual": actual,
            "gameIds": evaluation_rows["game_pk"].to_numpy(),
            "globalProbabilities": global_probabilities,
            "sequenceProbabilities": sequence_probabilities,
        }
        del examples, fitted, selected_expert
        gc.collect()

    selected_weight, selection = select_blend(folds)
    result = {
        "schemaVersion": 8,
        "modelVersion": "V8-sequence-candidate",
        "deploymentStatus": "shadow",
        "generatedAt": datetime.now(UTC).isoformat(),
        "dataDirectory": str(data_directory),
        "dataCutoff": rows["game_date"].max().date().isoformat(),
        "architecture": {
            "historyLength": 16,
            "dModel": 64,
            "layers": 2,
            "heads": 4,
            "feedforward": 128,
            "dropout": 0.1,
            "locationWeights": list(LOCATION_WEIGHTS),
        },
        "personalization": {
            "baseline": "V7.4",
            "applicationOrder": ["global", "sequence", "personalizer"],
            "controls": V74_PERSONALIZATION,
            "status": (
                "frozen-awaiting-retrain"
                if selected_weight > 0
                else "not-run-sequence-gate-failed"
            ),
        },
        "selectedSequenceBlend": selected_weight,
        "selection": selection,
        "folds": {
            str(year): _public_fold(value)
            for year, value in folds.items()
        },
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    result_path = artifact_directory / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
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
        default=Path("artifacts/v8-sequence"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/2026-07-28-v8-sequence.md"),
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    result = run(
        args.data,
        args.artifacts,
        args.report,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "selectedSequenceBlend": result["selectedSequenceBlend"],
                "accepted": result["selection"]["accepted"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
