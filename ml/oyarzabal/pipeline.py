"""Train the MVP candidate and build static replay artifacts."""

from __future__ import annotations

import argparse
import gc
import json
import traceback
import urllib.request
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .features import chronological_split, group_names, prepare_pitch_rows
from .hybrid import (
    RegistryEntry,
    fit_logit_bias,
    personalize_by_pitcher,
    select_personalizer_strength,
    serialize_registry_entry,
    specialist_eligibility,
)
from .metrics import evaluate_diagnostics
from .modeling import (
    apply_temperature,
    fit_temperature,
    predict_candidate,
    select_candidate,
    train_candidate_with_tuning,
    train_final_candidate,
)
from .resources import assert_safe, snapshot
from .taxonomy import (
    PITCH_GROUP_NAMES,
    PITCH_GROUPS,
    UNSUPPORTED_CONTEXT,
    PitchGroup,
    context_pitch,
    group_pitch,
    normalize_group_probabilities,
    serialize_probabilities,
)
from .training import GLOBAL_SPEC, PILOT_PITCHERS, global_specs

DEFAULT_SOURCE = Path("/root/workspace/pitchpredict-smoke-test")
GAME_PK = 775300
GAME_DATE = pd.Timestamp("2024-10-25")


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
    for value in series:
        raw = json.loads(value)
        grouped = serialize_probabilities(normalize_group_probabilities(raw))
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
    top = str(PITCH_GROUPS[int(np.argmax(probabilities))])
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
        xgb_top = str(PITCH_GROUPS[int(np.argmax(probabilities["xgboost"][sequence]))])
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

    registry: dict[int, RegistryEntry] = {}
    validation_results: dict[str, object] = {}

    for pitcher_id, name in PILOT_PITCHERS.items():
        pitcher_history = history[history["pitcher_id"] == pitcher_id]
        eligibility = specialist_eligibility(pitcher_history)
        pitcher_validation = validation[validation["pitcher_id"] == pitcher_id]
        if (
            not eligibility["eligible"]
            or pitcher_validation.empty
            or pitcher_validation["game_date"].dt.normalize().nunique() < 2
        ):
            registry[pitcher_id] = RegistryEntry(
                pitcher_id, False, 0, "", reason="eligibility_or_validation"
            )
            validation_results[str(pitcher_id)] = {
                "name": name,
                "eligibility": eligibility,
            }
            continue
        positions = validation.index.get_indexer(pitcher_validation.index)
        dates = np.array(
            sorted(pitcher_validation["game_date"].dt.normalize().unique())
        )
        cutoff = dates[max(1, min(len(dates) - 1, int(len(dates) * 0.6)))]
        calibration_mask = (
            pitcher_validation["game_date"].dt.normalize().to_numpy() < cutoff
        )
        evaluation_mask = ~calibration_mask
        actual = pitcher_validation["target"].to_numpy()
        probabilities = global_validation_probabilities[positions]
        selection = select_personalizer_strength(
            actual[calibration_mask],
            probabilities[calibration_mask],
            actual[evaluation_mask],
            probabilities[evaluation_mask],
        )
        enabled = bool(selection["accepted"])
        strength = (
            float(selection["personalizerStrength"]) if enabled else None
        )
        bias = (
            fit_logit_bias(actual, probabilities, prior_strength=strength)
            if strength is not None
            else None
        )
        effective_weight = (
            len(actual) / (len(actual) + strength) if strength is not None else 0
        )
        registry[pitcher_id] = RegistryEntry(
            pitcher_id,
            enabled,
            float(effective_weight),
            "logit-bias" if enabled else "",
            data_cutoff=history["game_date"].max().date().isoformat(),
            reason=None if enabled else "validation_failed",
            spec="pitcher-logit-bias" if enabled else None,
            logit_bias=(
                tuple(float(value) for value in bias) if bias is not None else None
            ),
            personalizer_strength=strength,
        )
        validation_results[str(pitcher_id)] = {
            "name": name,
            "eligibility": eligibility,
            "selection": selection,
        }

    global_model = train_final_candidate(history, selected_spec, global_trees)
    global_game = predict_candidate(global_model, game, temperature=global_temperature)
    final_game, source_types = personalize_by_pitcher(
        game["pitcher_id"].to_numpy(), global_game, registry
    )
    model_sources = []
    for pitcher_id, source_type in zip(
        game["pitcher_id"], source_types, strict=True
    ):
        entry = registry.get(int(pitcher_id))
        weight = entry.specialist_weight if source_type == "hybrid" and entry else 0
        model_sources.append(
            {
                "type": source_type,
                "label": (
                    "Pitcher Personalizer + Global"
                    if source_type == "hybrid"
                    else "MLB Global XGBoost"
                ),
                "globalWeight": 1 - weight,
                "specialistWeight": weight,
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
        "validation": validation_results,
        "specialists": {
            str(pitcher_id): serialize_registry_entry(
                entry, name=PILOT_PITCHERS[pitcher_id]
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
        "schemaVersion": 4,
        "generatedAt": generated_at,
        "evaluationMode": "historical_showcase",
        "caveat": game_artifact["caveat"],
        "dataScope": data_scope,
        "finalModel": "global-pitcher-personalizer",
        "pitchGroups": group_names(),
        "models": {
            "final": "검증형 Global + Pitcher Personalizer",
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
            "selectedName": "global-pitcher-personalizer",
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_SOURCE / "data/cache/pitchpredict/pitcher",
    )
    parser.add_argument(
        "--game",
        type=Path,
        default=DEFAULT_SOURCE / f"data/cache/games/{GAME_PK}.parquet",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_SOURCE / "outputs/predictions.csv",
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
    return parser


def main() -> None:
    args = _parser().parse_args()
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
