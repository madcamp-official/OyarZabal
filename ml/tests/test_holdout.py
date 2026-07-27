import json
from pathlib import Path

import pandas as pd
import pytest
from oyarzabal.holdout import _registry, frozen_rows


def _pitch(path: Path, game_date: str, game_pk: int) -> None:
    pd.DataFrame(
        {
            "pitch_type": ["FF"],
            "game_date": [game_date],
            "game_pk": [game_pk],
            "pitcher": [10],
            "batter": [20],
        }
    ).to_parquet(path)


def test_frozen_rows_keeps_2026_out_of_history(tmp_path: Path) -> None:
    history = tmp_path / "history"
    holdout = tmp_path / "holdout"
    history.mkdir()
    holdout.mkdir()
    _pitch(history / "2025.parquet", "2025-09-01", 1)
    _pitch(holdout / "2026.parquet", "2026-04-01", 2)

    rows = frozen_rows(history, holdout)
    assert rows["game_date"].dt.year.tolist() == [2026]

    _pitch(history / "leak.parquet", "2026-01-01", 3)
    with pytest.raises(ValueError, match="training history"):
        frozen_rows(history, holdout)


def test_registry_rejects_any_post_2025_training_data() -> None:
    payload = {
        "dataCutoff": "2025-12-31",
        "specialists": {
            "10": {
                "pitcherId": 10,
                "enabled": True,
                "specialistWeight": 0.5,
                "model": "pooled-residual.pkl",
                "dataCutoff": "2026-01-01",
                "status": "active",
                "residualScale": 0.5,
            }
        },
    }
    with pytest.raises(ValueError, match="post-2025"):
        _registry(json.loads(json.dumps(payload)))


def test_registry_reads_v6_reliability_metadata() -> None:
    payload = {
        "schemaVersion": 5,
        "dataCutoff": "2025-11-01",
        "specialists": {
            "10": {
                "pitcherId": 10,
                "enabled": True,
                "specialistWeight": 0,
                "model": "pooled-residual.pkl",
                "dataCutoff": "2025-09-01",
                "status": "active",
                "residualScale": None,
                "reliability": 0.2,
                "reliabilityComponents": {
                    "n": 1_000,
                    "supportCoefficient": 0.5,
                    "pAll": 0.8,
                    "pRecent": 0.5,
                },
                "selectionRank": None,
                "support": {"2025": 500},
            }
        },
    }

    entry = _registry(payload)[10]

    assert entry.reliability == 0.2
    assert entry.reliability_components["pRecent"] == 0.5

    payload["dataCutoff"] = "2026-01-01"
    payload["specialists"]["10"]["dataCutoff"] = "2025-12-31"
    with pytest.raises(ValueError, match="post-2025"):
        _registry(json.loads(json.dumps(payload)))
