"""Evaluate frozen models on post-cutoff Statcast data."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .features import prepare_pitch_rows
from .hybrid import (
    RegistryEntry,
    apply_logit_bias,
    apply_pooled_residual_by_pitcher,
    apply_reliability_gated_residual,
)
from .metrics import bootstrap_log_loss_gain, evaluate_diagnostics
from .modeling import predict_candidate
from .residual import (
    predict_context_gate,
    predict_correction,
    relative_pitcher_failure_reasons,
)
from .taxonomy import PITCH_GROUP_FAMILY_LABELS, PITCH_GROUPS

DEFAULT_CUTOFF = pd.Timestamp("2025-12-31")
PROSPECTIVE_START = pd.Timestamp("2026-07-26")
COMPARISON_COHORT_ID = "v5-enabled-pitchers-v1"
FROZEN_BENCHMARK = {
    "start": "2026-03-25",
    "end": "2026-07-25",
    "rows": 28_734,
    "sampleSha256": "a2d7de0347b98e9cac05fe1ee22eedc1eab33aa5430e770089755f8945198232",
}
# Frozen from models/hybrid/registry.json. Model registries may change; this benchmark
# cohort must not, otherwise cross-version personalizer comparisons become invalid.
V5_EVALUATION_PITCHER_IDS = frozenset(
    {
        458681,
        543037,
        543135,
        548389,
        554430,
        592836,
        592866,
        601713,
        605135,
        607259,
        607536,
        607625,
        608344,
        622491,
        622663,
        641154,
        641540,
        642547,
        650644,
        656288,
        656427,
        663559,
        663623,
        665152,
        669194,
        669373,
        669923,
        676440,
        676710,
        686613,
    }
)


def _frames(directory: Path) -> list[pd.DataFrame]:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Statcast shards in {directory}")
    return [pd.read_parquet(path) for path in files]


def _dates(frames: list[pd.DataFrame]) -> pd.Series:
    return pd.concat(
        [pd.to_datetime(frame["game_date"], errors="coerce") for frame in frames],
        ignore_index=True,
    ).dropna()


def frozen_rows(
    history_directory: Path,
    holdout_directory: Path,
    *,
    cutoff: pd.Timestamp = DEFAULT_CUTOFF,
) -> pd.DataFrame:
    history = _frames(history_directory)
    holdout = _frames(holdout_directory)
    history_dates = _dates(history)
    holdout_dates = _dates(holdout)
    if history_dates.empty or history_dates.max().normalize() > cutoff:
        raise ValueError("training history contains rows after the frozen cutoff")
    if holdout_dates.empty or holdout_dates.min().normalize() <= cutoff:
        raise ValueError("holdout contains rows on or before the frozen cutoff")

    rows = prepare_pitch_rows([*history, *holdout])
    evaluation = rows[rows["game_date"].dt.normalize() > cutoff].copy()
    if evaluation.empty:
        raise ValueError("holdout has no supported target pitches")
    return evaluation


def _registry(payload: dict[str, object]) -> dict[int, RegistryEntry]:
    cutoff = pd.Timestamp(str(payload["dataCutoff"]))
    if cutoff.normalize() > DEFAULT_CUTOFF:
        raise ValueError("registry was fitted with post-2025 data")

    entries = {}
    for raw in payload["specialists"].values():
        pitcher_id = int(raw["pitcherId"])
        data_cutoff = raw.get("dataCutoff")
        if data_cutoff and pd.Timestamp(data_cutoff).normalize() > DEFAULT_CUTOFF:
            raise ValueError(f"pitcher {pitcher_id} uses post-2025 data")
        entries[pitcher_id] = RegistryEntry(
            pitcher_id=pitcher_id,
            enabled=bool(raw["enabled"]),
            specialist_weight=float(raw["specialistWeight"]),
            model=str(raw["model"]),
            data_cutoff=data_cutoff,
            reason=raw.get("reason"),
            spec=raw.get("spec"),
            status=str(raw["status"]),
            residual_scale=raw.get("residualScale"),
            reliability=raw.get("reliability"),
            reliability_components=raw.get("reliabilityComponents"),
            selection_rank=raw.get("selectionRank"),
            support=raw.get("support"),
            scale_multiplier=float(raw.get("scaleMultiplier", 1.0)),
            stale=bool(raw.get("stale", raw.get("status") == "provisional")),
            incremental_validation=raw.get("incrementalValidation"),
        )
    return entries


def _metrics(rows: pd.DataFrame, probabilities: np.ndarray) -> dict[str, object]:
    return evaluate_diagnostics(
        rows["target"].to_numpy(),
        probabilities,
        labels=range(len(PITCH_GROUPS)),
        names=[str(group) for group in PITCH_GROUPS],
        family_labels=PITCH_GROUP_FAMILY_LABELS,
    )


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_sample_fingerprint(rows: pd.DataFrame) -> str:
    """Identify the exact labeled rows used by a benchmark."""
    columns = [
        "game_date",
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "pitcher_id",
        "batter_id",
        "target",
    ]
    canonical = rows[columns].copy()
    canonical["game_date"] = canonical["game_date"].dt.strftime("%Y-%m-%d")
    canonical = canonical.sort_values(columns, kind="stable")
    payload = canonical.to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def promotion_sample_is_prospective(rows: pd.DataFrame) -> bool:
    return bool(
        not rows.empty
        and rows["game_date"].min().normalize() >= PROSPECTIVE_START
    )


def _predict_models(
    rows: pd.DataFrame,
    model_directory: Path,
) -> dict[str, object]:
    registry_path = model_directory / "registry.json"
    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    registry = _registry(registry_payload)

    global_path = model_directory / str(registry_payload["global"]["model"])
    residual_path = model_directory / str(registry_payload["residual"]["model"])
    with global_path.open("rb") as handle:
        global_model = pickle.load(handle)
    with residual_path.open("rb") as handle:
        residual_model = pickle.load(handle)

    global_probabilities = predict_candidate(
        global_model,
        rows,
        temperature=float(registry_payload["global"]["temperature"]),
    )
    calibration = registry_payload["global"].get("calibration", {})
    calibration_bias = calibration.get("bias")
    if calibration_bias is not None:
        global_probabilities = apply_logit_bias(
            global_probabilities,
            calibration_bias,
        )
    correction = predict_correction(residual_model, rows)
    model_paths = {
        "global": {"path": str(global_path), "sha256": _hash(global_path)},
        "residual": {
            "path": str(residual_path),
            "sha256": _hash(residual_path),
        },
        "registry": {
            "path": str(registry_path),
            "sha256": _hash(registry_path),
        },
    }
    if int(registry_payload["schemaVersion"]) >= 5:
        gate_path = model_directory / str(registry_payload["gate"]["model"])
        with gate_path.open("rb") as handle:
            gate_model = pickle.load(handle)
        context_gate = predict_context_gate(
            gate_model,
            rows,
            global_probabilities,
            correction,
        )
        final_probabilities, sources, routing = apply_reliability_gated_residual(
            rows,
            global_probabilities,
            correction,
            context_gate,
            registry,
            prediction_dates=[value.date() for value in rows["game_date"]],
        )
        model_paths["gate"] = {
            "path": str(gate_path),
            "sha256": _hash(gate_path),
        }
    else:
        final_probabilities, sources = apply_pooled_residual_by_pitcher(
            rows["pitcher_id"].to_numpy(),
            global_probabilities,
            correction,
            registry,
            prediction_dates=[value.date() for value in rows["game_date"]],
        )
        routing = [
            {
                "pitcherReliability": entry.residual_scale if entry else 0,
                "contextGate": 1 if source != "global" else 0,
                "effectiveScale": entry.residual_scale if entry else 0,
                "capReason": None,
                "hardGateReason": (
                    None if source != "global" else "registry_inactive"
                ),
            }
            for source, pitcher_id in zip(
                sources,
                rows["pitcher_id"],
                strict=True,
            )
            for entry in [registry.get(int(pitcher_id))]
        ]
    return {
        "registryPayload": registry_payload,
        "registry": registry,
        "global": global_probabilities,
        "final": final_probabilities,
        "sources": sources,
        "routing": routing,
        "models": model_paths,
    }


def _paired(
    rows: pd.DataFrame,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    samples: int = 1_000,
) -> dict[str, float | int]:
    return bootstrap_log_loss_gain(
        rows["game_pk"].to_numpy(),
        rows["target"].to_numpy(),
        reference,
        candidate,
        samples=samples,
    )


def _breakdowns(
    rows: pd.DataFrame,
    global_probabilities: np.ndarray,
    final_probabilities: np.ndarray,
) -> tuple[dict[str, object], dict[str, object]]:
    pitchers = {}
    for pitcher_id, positions in rows.groupby("pitcher_id", sort=True).indices.items():
        indices = np.asarray(positions)
        selected = rows.iloc[indices]
        pitchers[str(int(pitcher_id))] = {
            "n": int(len(indices)),
            "global": _metrics(selected, global_probabilities[indices]),
            "final": _metrics(selected, final_probabilities[indices]),
            "pairedBootstrap": _paired(
                selected,
                global_probabilities[indices],
                final_probabilities[indices],
                samples=500,
            ),
        }
    months = {}
    month_keys = rows["game_date"].dt.to_period("M").astype(str)
    for month, positions in rows.groupby(month_keys, sort=True).indices.items():
        indices = np.asarray(positions)
        selected = rows.iloc[indices]
        months[str(month)] = {
            "n": int(len(indices)),
            "global": _metrics(selected, global_probabilities[indices]),
            "final": _metrics(selected, final_probabilities[indices]),
            "pairedBootstrap": _paired(
                selected,
                global_probabilities[indices],
                final_probabilities[indices],
                samples=500,
            ),
        }
    return pitchers, months


def evaluate_frozen_holdout(
    rows: pd.DataFrame,
    model_directory: Path,
    *,
    reference_model_directory: Path | None = None,
    evaluation_start: pd.Timestamp | None = None,
) -> dict[str, object]:
    if evaluation_start is not None:
        rows = rows[
            rows["game_date"].dt.normalize() >= evaluation_start.normalize()
        ].copy()
        if rows.empty:
            raise ValueError("no holdout rows at or after evaluation start")
    predicted = _predict_models(rows, model_directory)
    registry_payload = predicted["registryPayload"]
    registry = predicted["registry"]
    global_probabilities = predicted["global"]
    final_probabilities = predicted["final"]
    sources = predicted["sources"]
    routing = predicted["routing"]
    reference_predictions = (
        _predict_models(rows, reference_model_directory)
        if reference_model_directory is not None
        else None
    )
    reference_probabilities = (
        reference_predictions["final"] if reference_predictions else None
    )

    pool_ids = set(registry)
    enabled_ids = {
        pitcher_id for pitcher_id, entry in registry.items() if entry.enabled
    }
    masks = {
        "allMlb": np.ones(len(rows), dtype=bool),
        "comparisonCohort": rows["pitcher_id"]
        .isin(V5_EVALUATION_PITCHER_IDS)
        .to_numpy(),
        "registryPool": rows["pitcher_id"].isin(pool_ids).to_numpy(),
        "enabledPitchers": rows["pitcher_id"].isin(enabled_ids).to_numpy(),
    }
    comparison_rows = rows.loc[masks["comparisonCohort"]]
    comparison_fingerprint = evaluation_sample_fingerprint(comparison_rows)
    benchmark_start = rows["game_date"].min().date().isoformat()
    benchmark_end = rows["game_date"].max().date().isoformat()
    is_frozen_benchmark = (
        benchmark_start == FROZEN_BENCHMARK["start"]
        and benchmark_end == FROZEN_BENCHMARK["end"]
    )
    if is_frozen_benchmark and (
        len(comparison_rows) != FROZEN_BENCHMARK["rows"]
        or comparison_fingerprint != FROZEN_BENCHMARK["sampleSha256"]
    ):
        raise ValueError("frozen V5 evaluation sample changed")
    scope_roles = {
        "allMlb": "overall-comparison",
        "comparisonCohort": "personalizer-comparison",
        "registryPool": "model-diagnostic",
        "enabledPitchers": "model-diagnostic",
    }
    scopes = {}
    for name, mask in masks.items():
        selected = rows.loc[mask]
        scope = {
            "role": scope_roles[name],
            "global": _metrics(selected, global_probabilities[mask]),
            "final": _metrics(selected, final_probabilities[mask]),
            "pairedBootstrap": _paired(
                selected,
                global_probabilities[mask],
                final_probabilities[mask],
            ),
        }
        if reference_probabilities is not None:
            scope["reference"] = _metrics(
                selected,
                reference_probabilities[mask],
            )
            scope["candidateVsReference"] = _paired(
                selected,
                reference_probabilities[mask],
                final_probabilities[mask],
            )
        scopes[name] = scope
    pitchers, months = _breakdowns(
        rows,
        global_probabilities,
        final_probabilities,
    )
    scale_values = np.array(
        [float(value["effectiveScale"]) for value in routing],
        dtype=float,
    )
    cap_counts = Counter(
        str(value["capReason"])
        for value in routing
        if value["capReason"] is not None
    )
    hard_counts = Counter(
        str(value["hardGateReason"])
        for value in routing
        if value["hardGateReason"] is not None
    )
    registry_tiers = {}
    pitcher_ids = rows["pitcher_id"].to_numpy(dtype=int)
    for tier in ("full", "limited", "shadow"):
        tier_ids = {
            pitcher_id
            for pitcher_id, entry in registry.items()
            if entry.status == tier
        }
        mask = np.isin(pitcher_ids, list(tier_ids))
        selected = rows.loc[mask]
        registry_tiers[tier] = {
            "pitchers": len(tier_ids),
            "rows": int(np.count_nonzero(mask)),
            "intervenedRows": int(np.count_nonzero((scale_values > 0) & mask)),
            "global": _metrics(selected, global_probabilities[mask]),
            "final": _metrics(selected, final_probabilities[mask]),
            "pairedBootstrap": (
                _paired(
                    selected,
                    global_probabilities[mask],
                    final_probabilities[mask],
                )
                if mask.any()
                else None
            ),
        }
    reference = None
    promotion = None
    if reference_predictions is not None and reference_probabilities is not None:
        promotion_probabilities = final_probabilities.copy()
        player_failures = []
        for pitcher_id, entry in registry.items():
            if not entry.enabled:
                continue
            positions = np.flatnonzero(
                rows["pitcher_id"].to_numpy(dtype=int) == pitcher_id
            )
            selected = rows.iloc[positions]
            if len(positions) < 300:
                reasons = ["insufficient_support"]
                player_global = _metrics(
                    selected,
                    global_probabilities[positions],
                )
                player_candidate = _metrics(
                    selected,
                    final_probabilities[positions],
                )
            else:
                player_global = _metrics(
                    selected,
                    global_probabilities[positions],
                )
                player_candidate = _metrics(
                    selected,
                    final_probabilities[positions],
                )
                reasons = relative_pitcher_failure_reasons(
                    player_global,
                    player_candidate,
                )
            if reasons:
                promotion_probabilities[positions] = global_probabilities[positions]
                player_failures.append(
                    {
                        "pitcherId": pitcher_id,
                        "support": int(len(positions)),
                        "failureReasons": reasons,
                        "global": player_global,
                        "candidate": player_candidate,
                    }
                )
        paired = _paired(
            rows,
            reference_probabilities,
            promotion_probabilities,
        )
        reference_metrics = _metrics(rows, reference_probabilities)
        candidate_metrics = _metrics(rows, promotion_probabilities)
        comparison_mask = masks["comparisonCohort"]
        comparison_reference = _metrics(
            comparison_rows,
            reference_probabilities[comparison_mask],
        )
        comparison_candidate = _metrics(
            comparison_rows,
            promotion_probabilities[comparison_mask],
        )
        comparison_paired = _paired(
            comparison_rows,
            reference_probabilities[comparison_mask],
            promotion_probabilities[comparison_mask],
        )
        days = (
            rows["game_date"].max().normalize()
            - rows["game_date"].min().normalize()
        ).days + 1
        is_prospective = promotion_sample_is_prospective(rows)
        failed_ids = {value["pitcherId"] for value in player_failures}
        intervention_mask = (scale_values > 0) & ~rows["pitcher_id"].isin(
            failed_ids
        ).to_numpy()
        intervened = int(np.count_nonzero(intervention_mask))
        enough_data = (
            is_prospective
            and days >= 30
            and len(rows) >= 100_000
            and intervened >= 15_000
        )
        overall_metric_passed = (
            paired["ciLower"] > 0
            and candidate_metrics["accuracy"]
            >= reference_metrics["accuracy"] - 0.005
            and candidate_metrics["macroF1"]
            >= reference_metrics["macroF1"] - 0.005
            and candidate_metrics["hierarchicalAccuracy"]
            >= reference_metrics["hierarchicalAccuracy"] - 0.005
            and not candidate_metrics["zeroRecallClasses"]
            and candidate_metrics["maxClassShareError"] <= 0.20
            and candidate_metrics["totalVariationDistance"] <= 0.20
            and candidate_metrics["maxClassCalibrationError"] <= 0.10
        )
        comparison_metric_passed = (
            comparison_paired["ciLower"] > 0
            and comparison_candidate["accuracy"]
            >= comparison_reference["accuracy"] - 0.005
            and comparison_candidate["macroF1"]
            >= comparison_reference["macroF1"] - 0.005
            and comparison_candidate["hierarchicalAccuracy"]
            >= comparison_reference["hierarchicalAccuracy"] - 0.005
            and not comparison_candidate["zeroRecallClasses"]
            and comparison_candidate["maxClassShareError"] <= 0.20
            and comparison_candidate["totalVariationDistance"] <= 0.20
            and comparison_candidate["maxClassCalibrationError"] <= 0.10
        )
        metric_passed = overall_metric_passed and comparison_metric_passed
        reference = {
            "models": reference_predictions["models"],
            "metrics": reference_metrics,
        }
        promotion = {
            "prospectiveStart": PROSPECTIVE_START.date().isoformat(),
            "prospectiveEvaluation": bool(is_prospective),
            "days": int(days),
            "rows": int(len(rows)),
            "intervenedRows": intervened,
            "enoughData": enough_data,
            "pairedBootstrap": paired,
            "deactivatedPitchers": player_failures,
            "overallMetricPassed": overall_metric_passed,
            "comparisonCohort": {
                "id": COMPARISON_COHORT_ID,
                "reference": comparison_reference,
                "candidate": comparison_candidate,
                "pairedBootstrap": comparison_paired,
                "metricPassed": comparison_metric_passed,
            },
            "metricPassed": metric_passed,
            "promoted": bool(enough_data and metric_passed),
        }

    return {
        "schemaVersion": 5,
        "decisionRule": "family-sum-then-child",
        "generatedAt": datetime.now(UTC).isoformat(),
        "trainingCutoff": str(registry_payload["dataCutoff"]),
        "holdoutStart": rows["game_date"].min().date().isoformat(),
        "holdoutEnd": rows["game_date"].max().date().isoformat(),
        "evaluationCohort": {
            "id": COMPARISON_COHORT_ID,
            "sourceModel": "V5",
            "definition": "pitcher_id in frozen V5 enabled registry",
            "pitcherIds": sorted(V5_EVALUATION_PITCHER_IDS),
            "rows": int(len(comparison_rows)),
            "sampleSha256": comparison_fingerprint,
            "frozenBenchmarkMatch": is_frozen_benchmark,
        },
        "models": predicted["models"],
        "reference": reference,
        "promotion": promotion,
        "routing": dict(sorted(Counter(sources).items())),
        "routingDiagnostics": {
            "effectiveScale": {
                "p10": float(np.quantile(scale_values, 0.10)),
                "p50": float(np.quantile(scale_values, 0.50)),
                "p90": float(np.quantile(scale_values, 0.90)),
                "nonZero": int(np.count_nonzero(scale_values > 0)),
            },
            "capReasons": dict(sorted(cap_counts.items())),
            "hardGateReasons": dict(sorted(hard_counts.items())),
        },
        "registryTiers": registry_tiers,
        "scopes": scopes,
        "pitchers": pitchers,
        "months": months,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("data/raw/statcast"))
    parser.add_argument(
        "--holdout", type=Path, default=Path("data/holdout/statcast-2026")
    )
    parser.add_argument("--models", type=Path, default=Path("models/v7"))
    parser.add_argument("--reference-models", type=Path)
    parser.add_argument("--start", type=pd.Timestamp)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = evaluate_frozen_holdout(
        frozen_rows(args.history, args.holdout),
        args.models,
        reference_model_directory=args.reference_models,
        evaluation_start=args.start,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote holdout evaluation to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
