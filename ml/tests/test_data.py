from datetime import date

import pandas as pd
from oyarzabal.data import collect_statcast_shards, month_windows


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
                "game_date": [start],
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
    assert list(pd.read_parquet(tmp_path / "2024-01.parquet")) == [
        "game_date",
        "game_pk",
        "pitch_type",
        "pitcher",
        "batter",
    ]
    assert not list(tmp_path.glob("*.tmp"))
