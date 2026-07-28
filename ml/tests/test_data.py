from datetime import date

import pandas as pd
import pytest
from oyarzabal.data import (
    V8_EXTRA_COLUMNS,
    collect_statcast_shards,
    month_windows,
    probe_statcast_schema,
)


def test_month_windows_clip_to_requested_dates():
    assert month_windows(date(2024, 1, 30), date(2024, 3, 2)) == [
        (date(2024, 1, 30), date(2024, 1, 31)),
        (date(2024, 2, 1), date(2024, 2, 29)),
        (date(2024, 3, 1), date(2024, 3, 2)),
    ]


def test_collection_is_atomic_and_resumes_completed_months(tmp_path):
    calls: list[tuple[str, str]] = []

    def fetch(start: str, end: str) -> pd.DataFrame:
        calls.append((start, end))
        return pd.DataFrame(
            {
                "game_date": [end],
                "game_pk": [1],
                "pitch_type": ["FF"],
                "pitcher": [10],
                "batter": [20],
                "unused": ["drop me"],
            }
        )

    first = collect_statcast_shards(
        date(2024, 1, 1),
        date(2024, 2, 2),
        tmp_path,
        fetcher=fetch,
        columns=("game_date", "game_pk", "pitch_type", "pitcher", "batter"),
        resource_check=lambda: None,
    )
    second = collect_statcast_shards(
        date(2024, 1, 1),
        date(2024, 2, 2),
        tmp_path,
        fetcher=fetch,
        columns=("game_date", "game_pk", "pitch_type", "pitcher", "batter"),
        resource_check=lambda: None,
    )

    assert calls == [("2024-01-01", "2024-01-31"), ("2024-02-01", "2024-02-02")]
    assert first["downloadedRows"] == 2
    assert second["downloadedRows"] == 0
    assert second["skippedShards"] == 2
    assert len(first["schemaFingerprint"]) == 64
    assert all(len(shard["sha256"]) == 64 for shard in first["shards"])
    assert all(len(shard["sha256"]) == 64 for shard in second["shards"])
    assert list(pd.read_parquet(tmp_path / "2024-01.parquet")) == [
        "game_date",
        "game_pk",
        "pitch_type",
        "pitcher",
        "batter",
    ]
    assert not list(tmp_path.glob("*.tmp"))


def test_collection_extends_an_existing_partial_month(tmp_path):
    columns = (
        "game_date",
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "pitch_type",
        "pitcher",
        "batter",
    )
    pd.DataFrame(
        {
            "game_date": ["2026-07-25"],
            "game_pk": [1],
            "at_bat_number": [1],
            "pitch_number": [1],
            "pitch_type": ["FF"],
            "pitcher": [10],
            "batter": [20],
        }
    ).to_parquet(tmp_path / "2026-07.parquet", index=False)
    calls: list[tuple[str, str]] = []

    def fetch(start: str, end: str) -> pd.DataFrame:
        calls.append((start, end))
        return pd.DataFrame(
            {
                "game_date": ["2026-07-26"],
                "game_pk": [2],
                "at_bat_number": [1],
                "pitch_number": [1],
                "pitch_type": ["SL"],
                "pitcher": [30],
                "batter": [40],
            }
        )

    result = collect_statcast_shards(
        date(2026, 7, 26),
        date(2026, 7, 28),
        tmp_path,
        fetcher=fetch,
        columns=columns,
        resource_check=lambda: None,
    )

    merged = pd.read_parquet(tmp_path / "2026-07.parquet")
    assert calls == [("2026-07-26", "2026-07-28")]
    assert result["downloadedRows"] == 1
    assert list(merged["game_pk"]) == [1, 2]


def test_v8_probe_requires_extended_physical_schema():
    frame = pd.DataFrame({name: [1] for name in V8_EXTRA_COLUMNS})
    result = probe_statcast_schema(frame, V8_EXTRA_COLUMNS)

    assert result["rows"] == 1
    assert result["missingRates"]["fielder_2"] == 0
    with pytest.raises(ValueError, match="release_spin_rate"):
        probe_statcast_schema(frame.drop(columns="release_spin_rate"), V8_EXTRA_COLUMNS)
