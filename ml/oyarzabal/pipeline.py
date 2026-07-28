"""Train the MVP candidate and build static replay artifacts."""

from __future__ import annotations

import argparse
import gc
import json
import traceback
import urllib.request
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .features import chronological_split, group_names, prepare_pitch_rows
from .hybrid import (
    RegistryEntry,
    apply_reliability_gated_residual,
    serialize_registry_entry,
    specialist_eligibility,
)
from .metrics import evaluate_diagnostics, hierarchical_top_indices
from .modeling import (
    apply_temperature,
    fit_temperature,
    predict_candidate,
    select_candidate,
    train_candidate_with_tuning,
    train_final_candidate,
)
from .residual import (
    apply_correction,
    compute_pitcher_reliability,
    maximum_safe_scale_multiplier,
    pitcher_residual_passes,
    predict_context_gate,
    predict_correction,
    train_final_residual,
    train_gate,
    train_residual_with_tuning,
)
from .resources import assert_safe, snapshot
from .taxonomy import (
    PITCH_GROUP_FAMILY_LABELS,
    PITCH_GROUP_NAMES,
    PITCH_GROUPS,
    UNSUPPORTED_CONTEXT,
    PitchGroup,
    context_pitch,
    family_names,
    group_families,
    group_pitch,
    normalize_group_probabilities,
    serialize_probabilities,
)
from .training import GLOBAL_SPEC, PILOT_PITCHERS, global_specs

GAME_PK = 775300
GAME_DATE = pd.Timestamp("2024-10-25")
RELIABILITY_SCALE_BOOST = 1.5
CONTEXT_GATE_POWER = 0.5
LIMITED_SCALE_BOOST = 1.0


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _load_history(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no history parquet files in {directory}")
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)


def _prior_probabilities(
    train: pd.DataFrame, target: pd.DataFrame, smoothing: float = 1.0
) -> np.ndarray:
    """Count-conditioned pitcher prior with deterministic backoff."""
    groups = len(PITCH_GROUPS)
    global_counts = np.bincount(train["target"], minlength=groups).astype(float)
    pitcher_counts = {
        key: np.bincount(part["target"], minlength=groups).astype(float)
        for key, part in train.groupby("pitcher", sort=False)
    }
    situation_counts = {
        key: np.bincount(part["target"], minlength=groups).astype(float)
        for key, part in train.groupby(["pitcher", "balls", "strikes"], sort=False)
    }
    rows = []
    for row in target.itertuples(index=False):
        key = (row.pitcher, row.balls, row.strikes)
        counts = situation_counts.get(key)
        if counts is None or counts.sum() < 20:
            counts = pitcher_counts.get(row.pitcher, global_counts)
        rows.append((counts + smoothing) / (counts.sum() + smoothing * groups))
    return np.asarray(rows)


def _parsed_probabilities(series: pd.Series) -> np.ndarray:
    rows = []
    group_labels = {str(group) for group in PITCH_GROUPS}
    for value in series:
        raw = json.loads(value)
        grouped = (
            serialize_probabilities(raw)
            if raw and set(raw) <= group_labels
            else serialize_probabilities(normalize_group_probabilities(raw))
        )
        rows.append([grouped[str(group)] for group in PITCH_GROUPS])
    return np.asarray(rows)


def _person_names(person_ids: set[int]) -> dict[int, str]:
    names = {person_id: f"선수 #{person_id}" for person_id in person_ids}
    if not person_ids:
        return names
    ids = ",".join(map(str, sorted(person_ids)))
    url = f"https://statsapi.mlb.com/api/v1/people?personIds={ids}"
    try:
        with urllib.request.urlopen(url, timeout=8) as response:
            payload = json.load(response)
        names.update(
            {
                int(person["id"]): person["fullName"]
                for person in payload.get("people", [])
            }
        )
    except Exception as error:
        print(f"MLB player-name lookup skipped: {error}")
    return names


def _prediction(probabilities: np.ndarray) -> dict[str, object]:
    values = {
        str(group): round(float(probability), 6)
        for group, probability in zip(PITCH_GROUPS, probabilities, strict=True)
    }
    top_index = int(
        hierarchical_top_indices(
            np.asarray(probabilities)[None, :],
            PITCH_GROUP_FAMILY_LABELS,
        )[0]
    )
    top = str(PITCH_GROUPS[top_index])
    return {
        "topPitch": top,
        "confidence": values[top],
        "probabilities": values,
    }


def _finite(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 2)


def _build_game(
    rows: pd.DataFrame,
    source_predictions: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    names: dict[int, str],
    model_sources: list[dict[str, object]],
) -> dict[str, object]:
    if len(rows) != len(source_predictions):
        raise ValueError(
            f"game/source prediction row mismatch: {len(rows)} != "
            f"{len(source_predictions)}"
        )
    expected = [group_pitch(code) for code in source_predictions["actual_pitch_type"]]
    if list(rows["target_group"]) != expected:
        raise ValueError("game pitch order does not match captured model predictions")

    recent: dict[tuple[int, int], list[str]] = {}
    pitches = []
    for sequence, row in enumerate(rows.itertuples(index=False)):
        plate_appearance = (int(row.game_pk), int(row.at_bat_number))
        history = recent.setdefault(plate_appearance, [])
        actual_group = str(row.target_group)
        count = f"{int(row.balls)}-{int(row.strikes)}"
        xgb_top = _prediction(probabilities["xgboost"][sequence])["topPitch"]
        explanation = [
            f"{count} 카운트와 {int(row.outs_when_up)}아웃 상황을 반영했습니다.",
            (
                "직전 구종은 "
                f"{PITCH_GROUP_NAMES.get(group_pitch(history[-1]), history[-1])}"
                if history
                else "이 타석의 첫 투구입니다."
            ),
            f"XGBoost 최우선 예측은 {PITCH_GROUP_NAMES[PitchGroup(xgb_top)]}입니다.",
        ]
        pitcher_id = int(row.pitcher)
        batter_id = int(row.batter)
        pitches.append(
            {
                "sequence": sequence + 1,
                "inning": int(row.inning),
                "half": str(row.inning_topbot),
                "atBatNumber": int(row.at_bat_number),
                "pitchNumber": int(row.pitch_number),
                "pitcher": {
                    "id": pitcher_id,
                    "name": names[pitcher_id],
                    "throws": str(row.p_throws),
                },
                "batter": {
                    "id": batter_id,
                    "name": names[batter_id],
                    "stand": str(row.stand),
                },
                "count": {
                    "balls": int(row.balls),
                    "strikes": int(row.strikes),
                },
                "outs": int(row.outs_when_up),
                "bases": [bool(row.base_1), bool(row.base_2), bool(row.base_3)],
                "score": {
                    "away": int(row.away_score),
                    "home": int(row.home_score),
                },
                "recentPitches": history[-3:],
                "modelSource": model_sources[sequence],
                "predictions": {
                    model: _prediction(values[sequence])
                    for model, values in probabilities.items()
                },
                "explanations": explanation,
                "actual": {
                    "rawCode": str(row.pitch_type),
                    "pitchGroup": actual_group,
                    "description": str(row.description),
                    "speed": _finite(row.release_speed),
                    "zone": None if pd.isna(row.zone) else int(row.zone),
                },
            }
        )
        history.append(str(row.pitch_type))

    first = rows.iloc[0]
    return {
        "id": str(int(first["game_pk"])),
        "date": first["game_date"].date().isoformat(),
        "title": "2024 월드 시리즈 1차전",
        "awayTeam": str(first["away_team"]),
        "homeTeam": str(first["home_team"]),
        "venue": str(first.get("venue_name", "Dodger Stadium")),
        "evaluationMode": "historical_showcase",
        "caveat": (
            "제품 검증용 역사적 쇼케이스이며, 동결된 2026 홀드아웃 결과가 아닙니다."
        ),
        "metrics": metrics,
        "pitchCount": len(pitches),
        "pitches": pitches,
    }


def _diagnostics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    return evaluate_diagnostics(
        labels,
        probabilities,
        labels=range(len(PITCH_GROUPS)),
        names=[str(group) for group in PITCH_GROUPS],
        family_labels=PITCH_GROUP_FAMILY_LABELS,
    )


def _pregame_hybrid(
    history: pd.DataFrame, game: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], dict[str, object]]:
    training, validation = chronological_split(history)
    global_candidates = []
    for spec in global_specs():
        model, trees = train_candidate_with_tuning(training, spec)
        raw = predict_candidate(model, validation, temperature=1)
        temperature = fit_temperature(validation["target"].to_numpy(), raw)
        probabilities = apply_temperature(raw, temperature)
        global_candidates.append(
            {
                "name": spec.name,
                "spec": spec,
                "treeCount": trees,
                "temperature": temperature,
                "probabilities": probabilities,
                "metrics": _diagnostics(
                    validation["target"].to_numpy(), probabilities
                ),
            }
        )
        del model
        gc.collect()

    reference = next(
        candidate
        for candidate in global_candidates
        if candidate["name"] == GLOBAL_SPEC.name
    )
    selected = select_candidate(
        reference["metrics"],
        [candidate for candidate in global_candidates if candidate is not reference],
    )
    accepted_global = selected is not None
    selected = selected or reference
    selected_spec = selected["spec"]
    global_trees = int(selected["treeCount"])
    global_temperature = float(selected["temperature"])
    global_validation_probabilities = selected["probabilities"]

    pitcher_ids = [int(value) for value in sorted(history["pitcher_id"].unique())]
    eligibility = {
        pitcher_id: specialist_eligibility(
            history[history["pitcher_id"] == pitcher_id]
        )
        for pitcher_id in pitcher_ids
    }
    pool = tuple(
        pitcher_id
        for pitcher_id, result in eligibility.items()
        if result["eligible"]
    )
    pooled_mask = validation["pitcher_id"].isin(pool).to_numpy()
    pooled_validation = validation[pooled_mask].copy().reset_index(drop=True)
    pooled_global = global_validation_probabilities[pooled_mask]
    dates = np.array(
        sorted(pooled_validation["game_date"].dt.normalize().unique())
    )
    if len(dates) < 10:
        raise ValueError("pregame V7 validation requires at least ten dates")
    residual_cutoff = dates[int(len(dates) * 0.5)]
    gate_cutoff = dates[int(len(dates) * 0.75)]
    tune_cutoff = dates[int(len(dates) * 0.85)]
    normalized = pooled_validation["game_date"].dt.normalize().to_numpy()
    residual_mask = normalized < residual_cutoff
    gate_mask = (normalized >= residual_cutoff) & (normalized < gate_cutoff)
    tune_mask = (normalized >= gate_cutoff) & (normalized < tune_cutoff)
    evaluation_mask = normalized >= tune_cutoff
    residual = train_residual_with_tuning(
        pooled_validation[residual_mask],
        pooled_global[residual_mask],
    )
    development_mask = gate_mask | tune_mask | evaluation_mask
    development_correction = predict_correction(
        residual, pooled_validation[development_mask]
    )
    development_rows = (
        pooled_validation[development_mask].copy().reset_index(drop=True)
    )
    development_global = pooled_global[development_mask]
    development_dates = development_rows["game_date"].dt.normalize().to_numpy()
    gate_train = development_dates < gate_cutoff
    gate_tune = (development_dates >= gate_cutoff) & (
        development_dates < tune_cutoff
    )
    gate_evaluation = development_dates >= tune_cutoff
    fitted_gate = train_gate(
        development_rows[gate_train],
        development_global[gate_train],
        development_correction[gate_train],
        n_estimators=1_000,
        tuning=(
            development_rows[gate_tune],
            development_global[gate_tune],
            development_correction[gate_tune],
        ),
    )
    gate_tree_count = fitted_gate.tree_count
    del fitted_gate
    gate_development = gate_train | gate_tune
    fitted_gate = train_gate(
        development_rows[gate_development],
        development_global[gate_development],
        development_correction[gate_development],
        n_estimators=gate_tree_count,
    )
    reference_development = apply_correction(
        development_global[gate_development],
        development_correction[gate_development],
        0.5,
    )
    reliability_development = compute_pitcher_reliability(
        development_rows[gate_development],
        development_global[gate_development],
        reference_development,
        samples=200,
    )
    temporary_registry = {
        pitcher_id: RegistryEntry(
            pitcher_id=pitcher_id,
            enabled=True,
            specialist_weight=0,
            model="pooled-contextual-residual",
            status="full",
            reliability=float(values["reliability"]),
            reliability_components=values,
        )
        for pitcher_id, values in reliability_development.items()
    }
    context_gate = predict_context_gate(
        fitted_gate,
        development_rows[gate_evaluation],
        development_global[gate_evaluation],
        development_correction[gate_evaluation],
    )
    evaluation_candidate, _, evaluation_routing = (
        apply_reliability_gated_residual(
            development_rows[gate_evaluation],
            development_global[gate_evaluation],
            development_correction[gate_evaluation],
            context_gate,
            temporary_registry,
            reliability_scale_boost=RELIABILITY_SCALE_BOOST,
            context_gate_power=CONTEXT_GATE_POWER,
        )
    )
    evaluation_rows = (
        development_rows[gate_evaluation].copy().reset_index(drop=True)
    )
    evaluation_global = development_global[gate_evaluation]
    evaluation_correction = development_correction[gate_evaluation]
    validation_results: dict[str, object] = {}
    tiers: dict[int, str] = {}
    scale_multipliers: dict[int, float] = {}
    for pitcher_id in pool:
        positions = np.flatnonzero(
            evaluation_rows["pitcher_id"].to_numpy() == pitcher_id
        )
        if not len(positions):
            validation_results[str(pitcher_id)] = {
                "name": PILOT_PITCHERS.get(pitcher_id, f"선수 #{pitcher_id}"),
                "eligibility": eligibility[pitcher_id],
                "support": 0,
                "accepted": False,
                "tier": "shadow",
                "scaleMultiplier": 0.0,
            }
            tiers[pitcher_id] = "shadow"
            scale_multipliers[pitcher_id] = 0.0
            continue
        passed, global_metrics, candidate_metrics = pitcher_residual_passes(
            evaluation_rows["target"].to_numpy()[positions],
            evaluation_global[positions],
            evaluation_candidate[positions],
            min_support=50,
        )
        safe_scale = (
            maximum_safe_scale_multiplier(
                evaluation_rows["target"].to_numpy()[positions],
                evaluation_global[positions],
                evaluation_correction[positions],
                np.asarray(
                    [
                        float(evaluation_routing[position]["effectiveScale"])
                        for position in positions
                    ]
                ),
            )
            if len(positions) >= 50
            else {
                "maxSafeMultiplier": 0.0,
                "failureReasons": ["insufficient_support"],
            }
        )
        if passed:
            tier = "full"
            scale_multiplier = 1.0
        elif float(safe_scale["maxSafeMultiplier"]) > 0:
            tier = "limited"
            scale_multiplier = float(safe_scale["maxSafeMultiplier"])
        else:
            tier = "shadow"
            scale_multiplier = 0.0
        score = float(global_metrics["logLoss"] - candidate_metrics["logLoss"])
        record = {
            "name": PILOT_PITCHERS.get(pitcher_id, f"선수 #{pitcher_id}"),
            "eligibility": eligibility[pitcher_id],
            "support": int(len(positions)),
            "accepted": bool(passed),
            "globalMetrics": global_metrics,
            "metrics": candidate_metrics,
            "score": score,
            "tier": tier,
            "scaleMultiplier": scale_multiplier,
            "incrementalFailureReasons": safe_scale["failureReasons"],
        }
        validation_results[str(pitcher_id)] = record
        tiers[pitcher_id] = tier
        scale_multipliers[pitcher_id] = scale_multiplier

    reference_all_development = apply_correction(
        development_global,
        development_correction,
        0.5,
    )
    reliability_final = compute_pitcher_reliability(
        development_rows,
        development_global,
        reference_all_development,
        samples=200,
    )
    registry = {}
    for pitcher_id in pool:
        status = tiers[pitcher_id]
        enabled = status in {"full", "limited"}
        pitcher_rows = history[history["pitcher_id"] == pitcher_id]
        reliability = reliability_final.get(
            pitcher_id,
            {
                "n": 0,
                "supportCoefficient": 0,
                "pAll": 0,
                "pRecent": 0,
                "reliability": 0,
                "recentSupport": 0,
            },
        )
        registry[pitcher_id] = RegistryEntry(
            pitcher_id=pitcher_id,
            enabled=enabled,
            specialist_weight=0,
            model="pooled-contextual-residual" if enabled else "",
            data_cutoff=pitcher_rows["game_date"].max().date().isoformat(),
            reason=(
                None
                if status == "full"
                else "incremental_gate_only"
                if status == "limited"
                else "validation_failed"
            ),
            spec="v7-incremental-residual" if enabled else None,
            status=status,
            reliability=float(reliability["reliability"]),
            reliability_components=reliability,
            support={
                "pregameValidation": int(
                    validation_results[str(pitcher_id)]["support"]
                )
            },
            scale_multiplier=scale_multipliers[pitcher_id],
            incremental_validation={
                "pregame": {
                    "support": validation_results[str(pitcher_id)]["support"],
                    "failureReasons": validation_results[str(pitcher_id)].get(
                        "incrementalFailureReasons",
                        ["insufficient_support"],
                    ),
                }
            },
        )
    residual_tree_count = residual.tree_count
    del residual
    gc.collect()
    residual = train_final_residual(
        pooled_validation,
        pooled_global,
        n_estimators=residual_tree_count,
    )
    final_gate = train_gate(
        development_rows,
        development_global,
        development_correction,
        n_estimators=gate_tree_count,
    )

    global_model = train_final_candidate(history, selected_spec, global_trees)
    global_game = predict_candidate(global_model, game, temperature=global_temperature)
    game_correction = predict_correction(residual, game)
    game_context_gate = predict_context_gate(
        final_gate,
        game,
        global_game,
        game_correction,
    )
    final_game, source_types, routing = apply_reliability_gated_residual(
        game,
        global_game,
        game_correction,
        game_context_gate,
        registry,
        prediction_dates=game["game_date"].dt.date.tolist(),
        limited_scale_boost=LIMITED_SCALE_BOOST,
        reliability_scale_boost=RELIABILITY_SCALE_BOOST,
        context_gate_power=CONTEXT_GATE_POWER,
    )
    model_sources = []
    for _pitcher_id, source_type, route in zip(
        game["pitcher_id"], source_types, routing, strict=True
    ):
        weight = float(route["effectiveScale"])
        model_sources.append(
            {
                "type": source_type,
                "label": (
                    "V7.2 Reliability-tuned Residual"
                    if source_type != "global"
                    else "MLB Global XGBoost"
                ),
                "globalWeight": 1 - weight,
                "specialistWeight": weight,
                **route,
            }
        )
    selection_artifact = {
        "globalSpec": selected_spec.name,
        "globalCandidateAccepted": accepted_global,
        "globalTemperature": global_temperature,
        "globalCandidates": [
            {
                "name": candidate["name"],
                "treeCount": candidate["treeCount"],
                "temperature": candidate["temperature"],
                "metrics": candidate["metrics"],
            }
            for candidate in global_candidates
        ],
        "residualSelection": {
            "formula": (
                "min(0.5, 1.5 * reliability) * sqrt(contextGate) "
                "* registryScaleMultiplier"
            ),
            "reliabilityScaleBoost": RELIABILITY_SCALE_BOOST,
            "contextGatePower": CONTEXT_GATE_POWER,
            "limitedScaleBoost": LIMITED_SCALE_BOOST,
            "tierCounts": dict(
                sorted(
                    Counter(tiers.values()).items()
                )
            ),
            "jsCap": 0.05,
            "probabilityShiftCap": 0.20,
        },
        "residualTreeCount": residual_tree_count,
        "gateTreeCount": gate_tree_count,
        "trainingPool": len(pool),
        "validation": validation_results,
        "specialists": {
            str(pitcher_id): serialize_registry_entry(
                entry,
                name=PILOT_PITCHERS.get(
                    pitcher_id,
                    f"선수 #{pitcher_id}",
                ),
            )
            for pitcher_id, entry in registry.items()
        },
    }
    return final_game, global_game, model_sources, selection_artifact


def build_demo(args: argparse.Namespace) -> None:
    generated_at = datetime.now(UTC).isoformat()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_directory = args.runs / run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    before = snapshot(Path.cwd())
    assert_safe(before)
    _atomic_json(run_directory / "resources-before.json", asdict(before))

    history_raw = _load_history(args.history)
    history_raw["game_date"] = pd.to_datetime(history_raw["game_date"], errors="coerce")
    history_raw = history_raw[history_raw["game_date"] < GAME_DATE]
    game_raw = pd.read_parquet(args.game)
    unsupported = game_raw["pitch_type"].map(context_pitch).eq(UNSUPPORTED_CONTEXT)
    if unsupported.any():
        codes = sorted(game_raw.loc[unsupported, "pitch_type"].dropna().unique())
        raise ValueError(f"showcase contains unsupported target pitches: {codes}")

    combined = prepare_pitch_rows([history_raw, game_raw])
    history = combined[combined["game_date"] < GAME_DATE].copy()
    game = combined[combined["game_pk"] == GAME_PK].copy().reset_index(drop=True)
    (
        selected_probabilities,
        global_probabilities,
        model_sources,
        experiment,
    ) = _pregame_hybrid(history, game)

    captured = pd.read_csv(args.predictions)
    game_probabilities = {
        "final": selected_probabilities,
        "xgboost": global_probabilities,
        "similarity": _parsed_probabilities(captured["similarity_probabilities"]),
        "baseline": _parsed_probabilities(captured["baseline_probabilities"]),
    }
    game_metrics = {
        name: _diagnostics(game["target"].to_numpy(), values)
        for name, values in game_probabilities.items()
    }
    people = set(game["pitcher"].astype(int)) | set(game["batter"].astype(int))
    names = _person_names(people)
    game_artifact = _build_game(
        game,
        captured,
        game_probabilities,
        game_metrics,
        names,
        model_sources,
    )
    pitcher_count = int(history["pitcher_id"].nunique())
    data_scope = {
        "startDate": history["game_date"].min().date().isoformat(),
        "endDate": history["game_date"].max().date().isoformat(),
        "pitcherCount": pitcher_count,
        "mlbWide": pitcher_count >= 100,
    }
    if not data_scope["mlbWide"]:
        game_artifact["caveat"] += (
            f" 현재 산출물은 {pitcher_count}명 캐시로 만든 중간 검증본입니다."
        )

    game_path = args.output / "games" / f"{GAME_PK}.json"
    _atomic_json(game_path, game_artifact)
    validation_metrics = {}
    manifest = {
        "schemaVersion": 8,
        "decisionRule": "family-sum-then-child",
        "generatedAt": generated_at,
        "evaluationMode": "historical_showcase",
        "deploymentStatus": "active",
        "deploymentBasis": "taxonomy-product-generation",
        "performanceCertification": "prospective-pending",
        "caveat": game_artifact["caveat"],
        "dataScope": data_scope,
        "finalModel": "v7.2-reliability-tuned-residual",
        "pitchGroups": group_names(),
        "pitchFamilies": family_names(),
        "pitchGroupFamilies": group_families(),
        "models": {
            "final": "V7.2 Reliability-tuned Residual (Current)",
            "xgboost": "MLB Global XGBoost",
            "similarity": "PitchPredict Similarity",
            "baseline": "상황별 빈도 기준선",
        },
        "validationMetrics": validation_metrics,
        "selection": {
            "accepted": any(
                specialist["enabled"]
                for specialist in experiment["specialists"].values()
            ),
            "referenceName": GLOBAL_SPEC.name,
            "selectedName": "v7.2-reliability-tuned-residual",
            "folds": ["pregame-chronological-validation"],
            "candidates": experiment["validation"],
            "specialists": experiment["specialists"],
        },
        "games": [
            {
                "id": str(GAME_PK),
                "title": game_artifact["title"],
                "date": game_artifact["date"],
                "matchup": f"{game_artifact['awayTeam']} @ {game_artifact['homeTeam']}",
                "pitchCount": game_artifact["pitchCount"],
                "path": f"/data/games/{GAME_PK}.json",
            }
        ],
    }
    _atomic_json(args.output / "manifest.json", manifest)

    after = snapshot(Path.cwd())
    _atomic_json(run_directory / "resources-after.json", asdict(after))
    _atomic_json(
        run_directory / "result.json",
        {
            "runId": run_id,
            "generatedAt": generated_at,
            "historyRows": len(history),
            "targetGroups": len(PITCH_GROUPS),
            "validationMetrics": validation_metrics,
            "gameMetrics": game_metrics,
            "selection": experiment,
            "output": str(args.output),
        },
    )
    print(f"Built {len(game)} replay pitches at {game_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--game",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("web/public/data"),
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("artifacts/runs"),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    existing_runs = set(args.runs.glob("*")) if args.runs.exists() else set()
    try:
        build_demo(args)
    except Exception as error:
        new_runs = set(args.runs.glob("*")) - existing_runs
        if new_runs:
            run_directory = max(new_runs, key=lambda path: path.stat().st_mtime)
            _atomic_json(
                run_directory / "error.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise


if __name__ == "__main__":
    main()
