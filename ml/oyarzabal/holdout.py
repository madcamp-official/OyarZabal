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
from .hybrid import RegistryEntry, apply_pooled_residual_by_pitcher
from .metrics import evaluate_diagnostics
from .modeling import predict_candidate
from .residual import predict_correction
from .taxonomy import PITCH_GROUPS

DEFAULT_CUTOFF = pd.Timestamp("2025-12-31")


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
            selection_rank=raw.get("selectionRank"),
            support=raw.get("support"),
        )
    return entries


def _metrics(rows: pd.DataFrame, probabilities: np.ndarray) -> dict[str, object]:
    return evaluate_diagnostics(
        rows["target"].to_numpy(),
        probabilities,
        labels=range(len(PITCH_GROUPS)),
        names=[str(group) for group in PITCH_GROUPS],
    )


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_frozen_holdout(
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
    correction = predict_correction(residual_model, rows)
    final_probabilities, sources = apply_pooled_residual_by_pitcher(
        rows["pitcher_id"].to_numpy(),
        global_probabilities,
        correction,
        registry,
        prediction_dates=[value.date() for value in rows["game_date"]],
    )

    pool_ids = set(registry)
    enabled_ids = {
        pitcher_id for pitcher_id, entry in registry.items() if entry.enabled
    }
    masks = {
        "allMlb": np.ones(len(rows), dtype=bool),
        "registryPool": rows["pitcher_id"].isin(pool_ids).to_numpy(),
        "enabledPitchers": rows["pitcher_id"].isin(enabled_ids).to_numpy(),
    }
    scopes = {}
    for name, mask in masks.items():
        selected = rows.loc[mask]
        scopes[name] = {
            "global": _metrics(selected, global_probabilities[mask]),
            "final": _metrics(selected, final_probabilities[mask]),
        }

    return {
        "schemaVersion": 1,
        "generatedAt": datetime.now(UTC).isoformat(),
        "trainingCutoff": str(registry_payload["dataCutoff"]),
        "holdoutStart": rows["game_date"].min().date().isoformat(),
        "holdoutEnd": rows["game_date"].max().date().isoformat(),
        "models": {
            "global": {"path": str(global_path), "sha256": _hash(global_path)},
            "residual": {
                "path": str(residual_path),
                "sha256": _hash(residual_path),
            },
            "registry": {
                "path": str(registry_path),
                "sha256": _hash(registry_path),
            },
        },
        "routing": dict(sorted(Counter(sources).items())),
        "scopes": scopes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("data/raw/statcast"))
    parser.add_argument(
        "--holdout", type=Path, default=Path("data/holdout/statcast-2026")
    )
    parser.add_argument("--models", type=Path, default=Path("models/hybrid"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = evaluate_frozen_holdout(
        frozen_rows(args.history, args.holdout),
        args.models,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
