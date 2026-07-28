"""Resumable monthly Statcast collection."""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import signal
import traceback
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .resources import assert_safe, snapshot

STATCAST_COLUMNS = (
    "pitch_type",
    "game_date",
    "release_speed",
    "batter",
    "pitcher",
    "description",
    "zone",
    "stand",
    "p_throws",
    "home_team",
    "away_team",
    "balls",
    "strikes",
    "plate_x",
    "plate_z",
    "on_3b",
    "on_2b",
    "on_1b",
    "outs_when_up",
    "inning",
    "inning_topbot",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "home_score",
    "away_score",
    "bat_score",
    "fld_score",
    "n_thruorder_pitcher",
    "pitcher_days_since_prev_game",
)
V8_EXTRA_COLUMNS = (
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "sz_top",
    "sz_bot",
    "fielder_2",
)
V8_STATCAST_COLUMNS = (*STATCAST_COLUMNS, *V8_EXTRA_COLUMNS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_statcast_schema(
    frame: pd.DataFrame,
    columns: Sequence[str] = V8_STATCAST_COLUMNS,
) -> dict[str, object]:
    """Fail before collection if live Statcast lacks the frozen V8 schema."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Statcast probe is missing columns: {missing}")
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "requiredColumns": list(columns),
        "missingRates": {
            name: float(frame[name].isna().mean()) if len(frame) else None
            for name in V8_EXTRA_COLUMNS
        },
    }


def month_windows(start: date, end: date) -> list[tuple[date, date]]:
    if start > end:
        raise ValueError("start date must not be after end date")
    windows = []
    cursor = start
    while cursor <= end:
        month_end = date(
            cursor.year,
            cursor.month,
            calendar.monthrange(cursor.year, cursor.month)[1],
        )
        clipped_end = min(month_end, end)
        windows.append((cursor, clipped_end))
        cursor = clipped_end + timedelta(days=1)
    return windows


def _valid_shard(path: Path, columns: Sequence[str]) -> bool:
    if not path.exists():
        return False
    try:
        return set(columns) <= set(pq.read_schema(path).names)
    except Exception:
        return False


def collect_statcast_shards(
    start: date,
    end: date,
    output: Path,
    *,
    fetcher: Callable[[str, str], pd.DataFrame],
    columns: Sequence[str] = STATCAST_COLUMNS,
    resource_check: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Download monthly shards, safely resuming files already written."""
    output.mkdir(parents=True, exist_ok=True)
    check = resource_check or (lambda: assert_safe(snapshot(output)))
    downloaded_rows = 0
    skipped = 0
    shards = []

    for window_start, window_end in month_windows(start, end):
        path = output / f"{window_start:%Y-%m}.parquet"
        existing = None
        if _valid_shard(path, columns):
            existing = pd.read_parquet(path, columns=list(columns))
            dates = pd.to_datetime(existing["game_date"], errors="coerce")
            latest = dates.max()
            if pd.notna(latest) and latest.date() >= window_end:
                skipped += 1
                shards.append(
                    {
                        "path": str(path),
                        "status": "skipped",
                        "rows": len(existing),
                        "sha256": _sha256(path),
                    }
                )
                continue
            if pd.notna(latest):
                window_start = max(
                    window_start,
                    latest.date(),
                )

        check()
        for attempt in range(1, 4):
            try:
                frame = fetcher(window_start.isoformat(), window_end.isoformat())
                break
            except Exception:
                if attempt == 3:
                    raise
                check()
                print(f"retrying {window_start:%Y-%m} ({attempt}/3)")
        missing = set(columns) - set(frame.columns)
        if missing and not frame.empty:
            raise ValueError(f"Statcast response is missing columns: {sorted(missing)}")
        selected = frame.reindex(columns=columns)
        if existing is not None:
            selected = pd.concat([existing, selected], ignore_index=True)
            identity = ("game_pk", "at_bat_number", "pitch_number")
            selected = selected.drop_duplicates(
                subset=(
                    list(identity)
                    if set(identity) <= set(selected.columns)
                    else list(selected.columns)
                ),
                keep="last",
            )
        temporary = path.with_suffix(".parquet.tmp")
        selected.to_parquet(temporary, index=False)
        temporary.replace(path)
        downloaded_rows += len(frame)
        shards.append(
            {
                "path": str(path),
                "status": "extended" if existing is not None else "downloaded",
                "rows": len(selected),
                "sha256": _sha256(path),
            }
        )
        print(f"saved {path} ({len(selected):,} rows)")

    result = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "downloadedRows": downloaded_rows,
        "skippedShards": skipped,
        "shards": shards,
        "schemaFingerprint": hashlib.sha256(
            json.dumps(list(columns), separators=(",", ":")).encode()
        ).hexdigest(),
    }
    manifest = output / "manifest.json"
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2022, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--v8",
        action="store_true",
        help="collect the isolated V8 physical/catcher schema",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="disable pybaseball's within-month parallel requests",
    )
    args = parser.parse_args()
    output = args.output or Path(
        "data/raw/statcast-v8" if args.v8 else "data/raw/statcast"
    )
    columns = V8_STATCAST_COLUMNS if args.v8 else STATCAST_COLUMNS

    from pybaseball import statcast

    def fetch(start: str, end: str) -> pd.DataFrame:
        def timed_out(_signum: int, _frame: object) -> None:
            raise TimeoutError(f"Statcast request timed out: {start} to {end}")

        previous = signal.signal(signal.SIGALRM, timed_out)
        signal.alarm(180)
        try:
            return statcast(
                start_dt=start,
                end_dt=end,
                verbose=False,
                parallel=not args.serial,
            )
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    try:
        probe_date = min(
            args.end,
            max(args.start, date(args.start.year, 4, 15)),
        )
        probe = (
            probe_statcast_schema(
                fetch(probe_date.isoformat(), probe_date.isoformat())
            )
            if args.v8
            else None
        )
        result = collect_statcast_shards(
            args.start,
            args.end,
            output,
            fetcher=fetch,
            columns=columns,
        )
        if probe is not None:
            result["schema"] = "v8"
            result["probe"] = probe
            manifest = output / "manifest.json"
            temporary = manifest.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(result, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(manifest)
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        (output / "error.json").write_text(
            json.dumps(
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
