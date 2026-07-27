"""Train the frozen Global model and pooled contextual residual."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import traceback
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .features import prepare_pitch_rows
from .hybrid import (
    RegistryEntry,
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
    pitcher_residual_passes,
    predict_correction,
    residual_passes,
    select_residual_scale,
    train_final_residual,
    train_residual_with_tuning,
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
GLOBAL_TEMPERATURE = 1.0465
ACTIVE_LIMIT = 25
PROVISIONAL_LIMIT = 5
MIN_EVALUATION_PITCHES = 300


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
    )


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
        pitcher_id: pilot_pitchers.get(pitcher_id, f"선수 #{pitcher_id}")
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

    pooled_rows = {
        year: evaluation[evaluation["pitcher_id"].isin(pool_set)]
        for year, evaluation in evaluations.items()
    }
    pooled_positions = {
        year: np.flatnonzero(
            evaluation["pitcher_id"].isin(pool_set).to_numpy()
        )
        for year, evaluation in evaluations.items()
    }
    pooled_global = {
        year: global_predictions[year][pooled_positions[year]]
        for year in pooled_rows
    }

    residual_2023 = train_residual_with_tuning(
        pooled_rows[2023],
        pooled_global[2023],
    )
    correction_2024 = predict_correction(residual_2023, pooled_rows[2024])
    scale_selection = select_residual_scale(
        pooled_rows[2024]["target"].to_numpy(),
        pooled_global[2024],
        correction_2024,
    )
    residual_scale = float(scale_selection["scale"])
    residual_tree_count = residual_2023.tree_count
    del residual_2023
    gc.collect()

    residual_2024 = train_final_residual(
        pd.concat([pooled_rows[2023], pooled_rows[2024]], ignore_index=True),
        np.concatenate([pooled_global[2023], pooled_global[2024]]),
        n_estimators=residual_tree_count,
    )
    correction_2025 = predict_correction(residual_2024, pooled_rows[2025])
    pooled_candidate_2025 = apply_correction(
        pooled_global[2025],
        correction_2025,
        residual_scale,
    )
    aggregate_test_global = _metrics(
        pooled_rows[2025]["target"].to_numpy(),
        pooled_global[2025],
    )
    aggregate_test_candidate = _metrics(
        pooled_rows[2025]["target"].to_numpy(),
        pooled_candidate_2025,
    )
    aggregate_test_passed = (
        scale_selection["accepted"]
        and residual_passes(aggregate_test_global, aggregate_test_candidate)
    )

    candidate_2024 = apply_correction(
        pooled_global[2024],
        correction_2024,
        residual_scale,
    )
    validations: dict[str, object] = {}
    active_candidates = []
    provisional_candidates = []
    for pitcher_id in pool:
        positions = {
            year: np.flatnonzero(
                pooled_rows[year]["pitcher_id"].to_numpy() == pitcher_id
            )
            for year in (2023, 2024, 2025)
        }
        support = {str(year): int(len(value)) for year, value in positions.items()}
        if support["2024"]:
            validation_passed, validation_global, validation_candidate = (
                pitcher_residual_passes(
                    pooled_rows[2024]["target"].to_numpy()[positions[2024]],
                    pooled_global[2024][positions[2024]],
                    candidate_2024[positions[2024]],
                    min_support=MIN_EVALUATION_PITCHES,
                )
            )
        else:
            validation_passed = False
            validation_global = None
            validation_candidate = None
        if support["2025"]:
            test_passed, test_global, test_candidate = pitcher_residual_passes(
                pooled_rows[2025]["target"].to_numpy()[positions[2025]],
                pooled_global[2025][positions[2025]],
                pooled_candidate_2025[positions[2025]],
                min_support=MIN_EVALUATION_PITCHES,
            )
        else:
            test_passed = False
            test_global = None
            test_candidate = None

        score = 0.0
        if validation_global is not None and validation_candidate is not None:
            score += float(
                validation_global["logLoss"] - validation_candidate["logLoss"]
            )
        if test_global is not None and test_candidate is not None:
            score += float(test_global["logLoss"] - test_candidate["logLoss"])
        record = {
            "pitcherId": pitcher_id,
            "name": names[pitcher_id],
            "eligibility": eligibility[pitcher_id],
            "support": support,
            "validationPassed": validation_passed,
            "testPassed": test_passed,
            "validationGlobalMetrics": validation_global,
            "validationMetrics": validation_candidate,
            "testGlobalMetrics": test_global,
            "testMetrics": test_candidate,
            "score": score,
        }
        validations[str(pitcher_id)] = record
        if (
            aggregate_test_passed
            and support["2023"] >= MIN_EVALUATION_PITCHES
            and validation_passed
            and test_passed
        ):
            active_candidates.append(record)
        elif (
            aggregate_test_passed
            and support["2023"] >= MIN_EVALUATION_PITCHES
            and validation_passed
            and support["2025"] == 0
        ):
            provisional_candidates.append(record)

    active = sorted(
        active_candidates,
        key=lambda value: (
            int(value["pitcherId"] not in pilot_pitchers),
            -value["score"],
            -value["support"]["2025"],
        ),
    )[:ACTIVE_LIMIT]
    provisional = sorted(
        provisional_candidates,
        key=lambda value: (
            int(value["pitcherId"] not in pilot_pitchers),
            -value["score"],
            -value["support"]["2024"],
        ),
    )[:PROVISIONAL_LIMIT]
    active_ids = {int(value["pitcherId"]) for value in active}
    provisional_ids = {int(value["pitcherId"]) for value in provisional}
    ranks = {
        int(value["pitcherId"]): rank
        for rank, value in enumerate([*active, *provisional], start=1)
    }

    registry: dict[int, RegistryEntry] = {}
    for pitcher_id in pool:
        support = validations[str(pitcher_id)]["support"]
        if pitcher_id in active_ids:
            status = "active"
            reason = None
        elif pitcher_id in provisional_ids:
            status = "provisional"
            reason = "missing_2025_rows"
        else:
            status = "inactive"
            record = validations[str(pitcher_id)]
            if support["2023"] < MIN_EVALUATION_PITCHES:
                reason = "insufficient_2023_support"
            elif support["2024"] < MIN_EVALUATION_PITCHES:
                reason = "insufficient_2024_support"
            elif not record["validationPassed"]:
                reason = "validation_failed"
            elif support["2025"] and not record["testPassed"]:
                reason = "test_failed"
            elif not aggregate_test_passed:
                reason = "aggregate_test_failed"
            else:
                reason = "exposure_cap"
        pitcher_rows = rows[rows["pitcher_id"] == pitcher_id]
        enabled = status != "inactive"
        registry[pitcher_id] = RegistryEntry(
            pitcher_id=pitcher_id,
            enabled=enabled,
            specialist_weight=residual_scale if enabled else 0,
            model="pooled-residual.pkl" if enabled else "",
            data_cutoff=pitcher_rows["game_date"].max().date().isoformat(),
            reason=reason,
            spec="pooled-contextual-residual" if enabled else None,
            status=status,
            residual_scale=residual_scale if enabled else None,
            selection_rank=ranks.get(pitcher_id),
            support=support,
        )

    residual_training_rows = pd.concat(
        [pooled_rows[year] for year in (2023, 2024, 2025)],
        ignore_index=True,
    )
    residual_training_global = np.concatenate(
        [pooled_global[year] for year in (2023, 2024, 2025)]
    )
    final_residual = train_final_residual(
        residual_training_rows,
        residual_training_global,
        n_estimators=residual_tree_count,
    )
    _atomic_pickle(model_directory / "pooled-residual.pkl", final_residual)
    del final_residual, residual_2024
    gc.collect()

    global_tree_count = int(np.median(global_trees))
    global_model = train_final_candidate(rows, GLOBAL_SPEC, global_tree_count)
    _atomic_pickle(model_directory / "global.pkl", global_model)
    del global_model
    gc.collect()

    result = {
        "schemaVersion": 4,
        "pitchGroups": [str(group) for group in PITCH_GROUPS],
        "generatedAt": datetime.now(UTC).isoformat(),
        "dataCutoff": rows["game_date"].max().date().isoformat(),
        "rows": len(rows),
        "global": {
            "model": "global.pkl",
            "selectedSpec": GLOBAL_SPEC.name,
            "treeCount": global_tree_count,
            "temperature": GLOBAL_TEMPERATURE,
            "features": "id-independent",
            "selectionYear": "fixed-from-prior-experiment",
            "testYear": 2025,
            "acceptedCandidate": True,
            "bestIterations": global_trees,
            "metrics": _metrics(
                global_actual[2025],
                global_predictions[2025],
            ),
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
            "selectedScale": residual_scale,
            "selection": scale_selection,
            "testGlobalMetrics": aggregate_test_global,
            "testMetrics": aggregate_test_candidate,
            "testPassed": aggregate_test_passed,
            "activeCount": len(active),
            "provisionalCount": len(provisional),
            "activeLimit": ACTIVE_LIMIT,
            "provisionalLimit": PROVISIONAL_LIMIT,
        },
        "specialists": {
            str(pitcher_id): serialize_registry_entry(
                entry, name=names[pitcher_id]
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
