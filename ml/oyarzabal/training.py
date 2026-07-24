"""Train the MLB-wide global model and validated pitcher specialists."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import traceback
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .features import prepare_pitch_rows
from .hybrid import (
    RegistryEntry,
    select_blend_weight,
    serialize_registry_entry,
    specialist_eligibility,
)
from .metrics import evaluate_diagnostics
from .modeling import (
    CandidateSpec,
    apply_temperature,
    fit_temperature,
    predict_candidate,
    train_candidate_with_tuning,
    train_final_candidate,
)
from .resources import assert_safe, snapshot
from .taxonomy import PITCH_GROUPS

PILOT_PITCHERS = {
    543037: "Gerrit Cole",
    656427: "Jack Flaherty",
    595014: "Blake Treinen",
    592454: "Tommy Kahnle",
    641482: "Nestor Cortes",
}
GLOBAL_SPEC = CandidateSpec("global", "sqrt", 6, 8)


def specialist_specs() -> tuple[CandidateSpec, ...]:
    return tuple(
        CandidateSpec("global", weight_mode, depth, 8)
        for weight_mode in ("none", "sqrt")
        for depth in (3, 4)
    )


def validation_folds(
    rows: pd.DataFrame, evaluation_years: tuple[int, ...] = (2024, 2025)
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
    )


def train_hybrid(
    rows: pd.DataFrame,
    model_directory: Path,
    *,
    pilot_pitchers: dict[int, str] = PILOT_PITCHERS,
) -> dict[str, object]:
    folds = validation_folds(rows)
    global_predictions = []
    global_actual = []
    global_trees = []
    specialist_predictions: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    specialist_global: dict[int, list[np.ndarray]] = defaultdict(list)
    specialist_actual: dict[int, list[np.ndarray]] = defaultdict(list)
    specialist_trees: dict[tuple[int, str], list[int]] = defaultdict(list)
    eligibility = {
        pitcher_id: specialist_eligibility(rows[rows["pitcher_id"] == pitcher_id])
        for pitcher_id in pilot_pitchers
    }

    for _year, train_mask, evaluation_mask in folds:
        assert_safe(snapshot(Path.cwd()))
        train = rows[train_mask]
        evaluation = rows[evaluation_mask]
        global_model, trees = train_candidate_with_tuning(train, GLOBAL_SPEC)
        global_fold = predict_candidate(global_model, evaluation, temperature=1)
        global_predictions.append(global_fold)
        global_actual.append(evaluation["target"].to_numpy())
        global_trees.append(trees)

        for pitcher_id in pilot_pitchers:
            if not eligibility[pitcher_id]["eligible"]:
                continue
            pitcher_train = train[train["pitcher_id"] == pitcher_id]
            pitcher_evaluation = evaluation[evaluation["pitcher_id"] == pitcher_id]
            if pitcher_evaluation.empty or len(pitcher_train) < 2:
                continue
            specialist_global[pitcher_id].append(
                predict_candidate(global_model, pitcher_evaluation, temperature=1)
            )
            specialist_actual[pitcher_id].append(
                pitcher_evaluation["target"].to_numpy()
            )
            for spec in specialist_specs():
                specialist_model, tree_count = train_candidate_with_tuning(
                    pitcher_train, spec, n_estimators=500
                )
                specialist_predictions[(pitcher_id, spec.name)].append(
                    predict_candidate(
                        specialist_model, pitcher_evaluation, temperature=1
                    )
                )
                specialist_trees[(pitcher_id, spec.name)].append(tree_count)
                del specialist_model
                gc.collect()
        del global_model
        gc.collect()

    registry: dict[int, RegistryEntry] = {}
    validations = {}
    global_actual_values = np.concatenate(global_actual)
    global_raw = np.concatenate(global_predictions)
    global_temperature = fit_temperature(global_actual_values, global_raw)
    selected_specs: dict[int, CandidateSpec] = {}
    for pitcher_id, name in pilot_pitchers.items():
        if not eligibility[pitcher_id]["eligible"]:
            registry[pitcher_id] = RegistryEntry(
                pitcher_id, False, 0, "", reason="eligibility"
            )
            validations[str(pitcher_id)] = {
                "name": name,
                "eligibility": eligibility[pitcher_id],
            }
            continue
        if not specialist_actual[pitcher_id]:
            registry[pitcher_id] = RegistryEntry(
                pitcher_id, False, 0, "", reason="missing_validation_rows"
            )
            continue
        actual = np.concatenate(specialist_actual[pitcher_id])
        global_values = apply_temperature(
            np.concatenate(specialist_global[pitcher_id]), global_temperature
        )
        candidate_results = []
        for spec in specialist_specs():
            specialist_raw = np.concatenate(
                specialist_predictions[(pitcher_id, spec.name)]
            )
            temperature = fit_temperature(actual, specialist_raw)
            selection = select_blend_weight(
                actual,
                global_values,
                apply_temperature(specialist_raw, temperature),
            )
            candidate_results.append(
                {"spec": spec, "temperature": temperature, "selection": selection}
            )
        accepted = [
            candidate
            for candidate in candidate_results
            if candidate["selection"]["accepted"]
        ]
        selected = (
            min(
                accepted,
                key=lambda candidate: (
                    candidate["selection"]["metrics"]["logLoss"],
                    -candidate["selection"]["metrics"]["macroF1"],
                ),
            )
            if accepted
            else None
        )
        enabled = selected is not None
        selected_spec = selected["spec"] if selected else None
        specialist_temperature = selected["temperature"] if selected else 1
        selection = selected["selection"] if selected else None
        model_name = f"specialist-{pitcher_id}.pkl" if enabled else ""
        if selected_spec:
            selected_specs[pitcher_id] = selected_spec
        registry[pitcher_id] = RegistryEntry(
            pitcher_id=pitcher_id,
            enabled=enabled,
            specialist_weight=(
                float(selection["specialistWeight"]) if selection else 0
            ),
            model=model_name,
            data_cutoff=rows["game_date"].max().date().isoformat(),
            reason=None if enabled else "validation_failed",
            temperature=specialist_temperature,
            spec=selected_spec.name if selected_spec else None,
        )
        validations[str(pitcher_id)] = {
            "name": name,
            "eligibility": eligibility[pitcher_id],
            "selectedSpec": selected_spec.name if selected_spec else None,
            "candidates": [
                {
                    "spec": candidate["spec"].name,
                    "temperature": candidate["temperature"],
                    "selection": candidate["selection"],
                }
                for candidate in candidate_results
            ],
        }

    global_tree_count = int(np.median(global_trees))
    global_model = train_final_candidate(rows, GLOBAL_SPEC, global_tree_count)
    _atomic_pickle(model_directory / "global.pkl", global_model)
    del global_model
    gc.collect()

    for pitcher_id, entry in registry.items():
        if not entry.enabled:
            continue
        pitcher_rows = rows[rows["pitcher_id"] == pitcher_id]
        spec = selected_specs[pitcher_id]
        tree_count = int(np.median(specialist_trees[(pitcher_id, spec.name)]))
        model = train_final_candidate(pitcher_rows, spec, tree_count)
        _atomic_pickle(model_directory / entry.model, model)
        del model
        gc.collect()

    global_oof = apply_temperature(global_raw, global_temperature)
    result = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(UTC).isoformat(),
        "dataCutoff": rows["game_date"].max().date().isoformat(),
        "rows": len(rows),
        "global": {
            "model": "global.pkl",
            "treeCount": global_tree_count,
            "temperature": global_temperature,
            "features": "id-independent",
            "metrics": _metrics(global_actual_values, global_oof),
        },
        "specialists": {
            str(pitcher_id): serialize_registry_entry(
                entry, name=pilot_pitchers[pitcher_id]
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
    parser.add_argument("--models", type=Path, default=Path("models/hybrid"))
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
