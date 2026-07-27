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
    apply_logit_bias,
    fit_logit_bias,
    personalizer_passes,
    select_personalizer_strength,
    serialize_registry_entry,
    specialist_eligibility,
)
from .metrics import evaluate_diagnostics
from .modeling import (
    CandidateSpec,
    apply_temperature,
    fit_temperature,
    predict_candidate,
    select_candidate,
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
    specs = global_specs()
    global_predictions: dict[int, dict[str, np.ndarray]] = {}
    global_actual: dict[int, np.ndarray] = {}
    global_trees: dict[str, list[int]] = defaultdict(list)
    evaluations: dict[int, pd.DataFrame] = {}
    eligibility = {
        pitcher_id: specialist_eligibility(
            rows[
                (rows["pitcher_id"] == pitcher_id)
                & (rows["game_date"].dt.year < 2025)
            ]
        )
        for pitcher_id in pilot_pitchers
    }

    for year, train_mask, evaluation_mask in folds:
        assert_safe(snapshot(Path.cwd()))
        train = rows[train_mask]
        evaluation = rows[evaluation_mask]
        evaluations[year] = evaluation
        global_actual[year] = evaluation["target"].to_numpy()
        global_predictions[year] = {}
        for spec in specs:
            model, trees = train_candidate_with_tuning(train, spec)
            global_predictions[year][spec.name] = predict_candidate(
                model, evaluation, temperature=1
            )
            global_trees[spec.name].append(trees)
            del model
            gc.collect()

    registry: dict[int, RegistryEntry] = {}
    validations = {}
    tuning_candidates = []
    for spec in specs:
        raw = global_predictions[2024][spec.name]
        temperature = fit_temperature(global_actual[2024], raw)
        probabilities = apply_temperature(raw, temperature)
        tuning_candidates.append(
            {
                "name": spec.name,
                "spec": {
                    "featureSet": spec.feature_set,
                    "weightMode": spec.weight_mode,
                    "maxDepth": spec.max_depth,
                    "minChildWeight": spec.min_child_weight,
                },
                "temperature": temperature,
                "metrics": _metrics(global_actual[2024], probabilities),
                "bestIterations": global_trees[spec.name],
            }
        )

    reference = next(
        candidate
        for candidate in tuning_candidates
        if candidate["name"] == GLOBAL_SPEC.name
    )
    selected = select_candidate(
        reference["metrics"],
        [candidate for candidate in tuning_candidates if candidate is not reference],
    )
    accepted_global = selected is not None
    selected = selected or reference
    selected_spec = next(spec for spec in specs if spec.name == selected["name"])
    global_temperature = float(selected["temperature"])
    tuning_global = apply_temperature(
        global_predictions[2024][selected_spec.name], global_temperature
    )
    test_global = apply_temperature(
        global_predictions[2025][selected_spec.name], global_temperature
    )

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

        tuning_positions = np.flatnonzero(
            evaluations[2024]["pitcher_id"].to_numpy() == pitcher_id
        )
        test_positions = np.flatnonzero(
            evaluations[2025]["pitcher_id"].to_numpy() == pitcher_id
        )
        if not len(tuning_positions) or not len(test_positions):
            registry[pitcher_id] = RegistryEntry(
                pitcher_id, False, 0, "", reason="missing_validation_rows"
            )
            continue

        pitcher_tuning = evaluations[2024].iloc[tuning_positions]
        dates = np.array(
            sorted(pitcher_tuning["game_date"].dt.normalize().unique())
        )
        if len(dates) < 2:
            registry[pitcher_id] = RegistryEntry(
                pitcher_id, False, 0, "", reason="missing_personalizer_split"
            )
            continue
        cutoff = dates[max(1, min(len(dates) - 1, int(len(dates) * 0.6)))]
        calibration_mask = (
            pitcher_tuning["game_date"].dt.normalize().to_numpy() < cutoff
        )
        evaluation_mask = ~calibration_mask
        selection = select_personalizer_strength(
            global_actual[2024][tuning_positions][calibration_mask],
            tuning_global[tuning_positions][calibration_mask],
            global_actual[2024][tuning_positions][evaluation_mask],
            tuning_global[tuning_positions][evaluation_mask],
        )
        selected_on_validation = bool(selection["accepted"])
        strength = (
            float(selection["personalizerStrength"])
            if selected_on_validation
            else None
        )

        test_global_metrics = _metrics(
            global_actual[2025][test_positions],
            test_global[test_positions],
        )
        test_metrics = test_global_metrics
        deployment_bias = None
        effective_weight = 0.0
        if strength is not None:
            evaluation_bias = fit_logit_bias(
                global_actual[2024][tuning_positions],
                tuning_global[tuning_positions],
                prior_strength=strength,
            )
            test_metrics = _metrics(
                global_actual[2025][test_positions],
                apply_logit_bias(test_global[test_positions], evaluation_bias),
            )
        accepted_on_test = (
            strength is not None
            and personalizer_passes(test_global_metrics, test_metrics)
        )
        if accepted_on_test and strength is not None:
            deployment_actual = np.concatenate(
                [
                    global_actual[2024][tuning_positions],
                    global_actual[2025][test_positions],
                ]
            )
            deployment_probabilities = np.concatenate(
                [
                    tuning_global[tuning_positions],
                    test_global[test_positions],
                ]
            )
            deployment_bias = fit_logit_bias(
                deployment_actual,
                deployment_probabilities,
                prior_strength=strength,
            )
            effective_weight = len(deployment_actual) / (
                len(deployment_actual) + strength
            )

        registry[pitcher_id] = RegistryEntry(
            pitcher_id=pitcher_id,
            enabled=accepted_on_test,
            specialist_weight=float(effective_weight),
            model="logit-bias" if accepted_on_test else "",
            data_cutoff=rows["game_date"].max().date().isoformat(),
            reason=(
                None
                if accepted_on_test
                else (
                    "test_validation_failed"
                    if selected_on_validation
                    else "validation_failed"
                )
            ),
            spec="pitcher-logit-bias" if accepted_on_test else None,
            logit_bias=(
                tuple(float(value) for value in deployment_bias)
                if deployment_bias is not None
                else None
            ),
            personalizer_strength=strength if accepted_on_test else None,
        )
        validations[str(pitcher_id)] = {
            "name": name,
            "eligibility": eligibility[pitcher_id],
            "selectionYear": 2024,
            "testYear": 2025,
            "selection": selection,
            "acceptedOnTest": accepted_on_test,
            "testGlobalMetrics": test_global_metrics,
            "testMetrics": test_metrics,
        }

    global_tree_count = int(np.median(global_trees[selected_spec.name]))
    global_model = train_final_candidate(rows, selected_spec, global_tree_count)
    _atomic_pickle(model_directory / "global.pkl", global_model)
    del global_model
    gc.collect()

    result = {
        "schemaVersion": 3,
        "pitchGroups": [str(group) for group in PITCH_GROUPS],
        "generatedAt": datetime.now(UTC).isoformat(),
        "dataCutoff": rows["game_date"].max().date().isoformat(),
        "rows": len(rows),
        "global": {
            "model": "global.pkl",
            "selectedSpec": selected_spec.name,
            "treeCount": global_tree_count,
            "temperature": global_temperature,
            "features": "id-independent",
            "selectionYear": 2024,
            "testYear": 2025,
            "acceptedCandidate": accepted_global,
            "tuningCandidates": tuning_candidates,
            "metrics": _metrics(global_actual[2025], test_global),
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
