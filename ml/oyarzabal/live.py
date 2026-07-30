"""Live Gameday state and point-in-time prediction helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    V83_CONTEXT_FEATURES,
    prepare_pitch_rows,
)
from .holdout import _registry
from .hybrid import apply_logit_bias, apply_reliability_gated_residual
from .metrics import hierarchical_top_indices
from .modeling import predict_candidate
from .residual import predict_context_gate, predict_correction
from .sequence import (
    REPERTOIRE_COLUMNS,
    SequenceExampleBuilder,
    SequenceExamples,
    SequenceVocabulary,
)
from .taxonomy import (
    PITCH_GROUP_FAMILY_LABELS,
    PITCH_GROUPS,
    context_pitch,
    group_pitch,
)

UNKNOWN = "UNKNOWN"
HISTORY_COLUMNS = (
    "pitch_type",
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    "balls",
    "strikes",
    "stand",
    "p_throws",
    "description",
    "fielder_2",
    "release_speed",
    "plate_x",
    "plate_z",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "sz_top",
    "sz_bot",
    "outs_when_up",
    "inning",
    "on_1b",
    "on_2b",
    "on_3b",
    "n_thruorder_pitcher",
)
V84_DESCRIPTIONS = (
    "automatic_ball",
    "automatic_strike",
    "ball",
    "blocked_ball",
    "bunt_foul_tip",
    "called_strike",
    "foul",
    "foul_bunt",
    "foul_tip",
    "hit_by_pitch",
    "hit_into_play",
    "missed_bunt",
    "pitchout",
    "swinging_strike",
    "swinging_strike_blocked",
)
V84_VOCABULARY = SequenceVocabulary(
    descriptions={
        value: index + 1 for index, value in enumerate(V84_DESCRIPTIONS)
    },
    pitchers={},
    batters={},
    catchers={},
)


def _counts() -> Counter[str]:
    return Counter({str(group): 0 for group in PITCH_GROUPS})


def _copy_nested(
    values: dict[object, Counter[str]],
) -> dict[object, Counter[str]]:
    return {key: Counter(counts) for key, counts in values.items()}


@dataclass(frozen=True)
class HistorySnapshot:
    global_counts: Counter[str]
    career_counts: Counter[str]
    season_counts: Counter[str]
    recent_contexts: tuple[str, ...]
    count_counts: dict[tuple[int, int], Counter[str]]
    stand_counts: dict[str, Counter[str]]
    transition_counts: dict[str, Counter[str]]
    previous_game_date: date | None


@dataclass(frozen=True)
class PitchRecord:
    sequence: int
    at_bat_index: int
    event_index: int
    play_id: str
    pitcher_id: int
    batter_id: int
    stand: str
    throws: str
    balls: int
    strikes: int
    raw_code: str
    context_group: str
    target_group: str | None
    description: str
    release_speed: float | None
    plate_x: float | None
    plate_z: float | None
    release_spin_rate: float | None


@dataclass(frozen=True)
class LiveContext:
    game_pk: int
    game_date: date
    status: str
    at_bat_index: int
    target_sequence: int
    pitcher_id: int
    pitcher_name: str
    batter_id: int
    batter_name: str
    catcher_id: int
    throws: str
    stand: str
    balls: int
    strikes: int
    outs: int
    inning: int
    half: str
    bases: tuple[bool, bool, bool]
    away_score: int
    home_score: int
    next_pitch_number: int
    state_key: str


@dataclass(frozen=True)
class LiveView:
    context: LiveContext | None
    pitches: tuple[PitchRecord, ...]
    game: dict[str, Any]
    repertoire: tuple[str, ...]
    feature_row: pd.DataFrame | None
    sequence_examples: SequenceExamples | None


class HistoryStore:
    """Load historical Statcast once and expose pitcher snapshots by date."""

    def __init__(self, directories: tuple[Path, ...]):
        self.directories = directories
        self._rows: pd.DataFrame | None = None
        self._snapshots: dict[tuple[int, date], HistorySnapshot] = {}
        self._sequences: dict[str, SequenceExamples] = {}

    @classmethod
    def from_frame(cls, rows: pd.DataFrame) -> HistoryStore:
        store = cls(())
        store._rows = store._prepare(rows)
        return store

    @staticmethod
    def _prepare(rows: pd.DataFrame) -> pd.DataFrame:
        frame = rows.copy()
        for name in HISTORY_COLUMNS:
            if name not in frame:
                frame[name] = np.nan
        frame = frame[list(HISTORY_COLUMNS)]
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
        frame["context_group"] = frame["pitch_type"].map(context_pitch)
        frame["target_group"] = frame["pitch_type"].map(group_pitch)
        frame = frame[
            frame["game_date"].notna() & frame["context_group"].notna()
        ].copy()
        frame["pitcher"] = pd.to_numeric(frame["pitcher"], errors="coerce")
        frame["batter"] = pd.to_numeric(frame["batter"], errors="coerce")
        frame = frame[frame["pitcher"].notna() & frame["batter"].notna()]
        frame["pitcher"] = frame["pitcher"].astype("int64")
        frame["batter"] = frame["batter"].astype("int64")
        frame["balls"] = pd.to_numeric(frame["balls"], errors="coerce").fillna(0)
        frame["strikes"] = pd.to_numeric(
            frame["strikes"], errors="coerce"
        ).fillna(0)
        frame["stand"] = frame["stand"].fillna(UNKNOWN).astype(str)
        frame["p_throws"] = frame["p_throws"].fillna(UNKNOWN).astype(str)
        return frame.sort_values(
            ["game_date", "game_pk", "at_bat_number", "pitch_number"],
            kind="stable",
        ).reset_index(drop=True)

    def _load(self) -> pd.DataFrame:
        if self._rows is not None:
            return self._rows
        files = [
            path
            for directory in self.directories
            for path in sorted(directory.glob("*.parquet"))
        ]
        if not files:
            raise FileNotFoundError(
                "no historical Statcast parquet files; configure "
                "OYARZABAL_HISTORY_DIRS"
            )
        frames = []
        for path in files:
            frames.append(pd.read_parquet(path, columns=list(HISTORY_COLUMNS)))
        self._rows = self._prepare(pd.concat(frames, ignore_index=True))
        return self._rows

    def snapshot(self, pitcher_id: int, game_date: date) -> HistorySnapshot:
        cache_key = (pitcher_id, game_date)
        cached = self._snapshots.get(cache_key)
        if cached is not None:
            return cached
        rows = self._load()
        before = rows[rows["game_date"].dt.date < game_date]
        supported = before["target_group"].notna()
        global_counts = _counts()
        global_counts.update(
            str(group) for group in before.loc[supported, "target_group"]
        )

        pitcher = before[before["pitcher"].eq(pitcher_id)].copy()
        supported_pitcher = pitcher["target_group"].notna()
        career = _counts()
        career.update(
            str(group)
            for group in pitcher.loc[supported_pitcher, "target_group"]
        )
        season = _counts()
        in_season = pitcher["game_date"].dt.year.eq(game_date.year)
        season.update(
            str(group)
            for group in pitcher.loc[
                supported_pitcher & in_season, "target_group"
            ]
        )

        recent_contexts = tuple(
            str(value) for value in pitcher["context_group"].tail(100)
        )
        count_counts: dict[tuple[int, int], Counter[str]] = defaultdict(_counts)
        stand_counts: dict[str, Counter[str]] = defaultdict(_counts)
        for row in pitcher.loc[supported_pitcher].itertuples(index=False):
            group = str(row.target_group)
            count_counts[(int(row.balls), int(row.strikes))][group] += 1
            stand_counts[str(row.stand)][group] += 1

        previous = (
            pitcher.groupby(["game_pk", "pitcher"], sort=False)["context_group"]
            .shift(1)
            .fillna(UNKNOWN)
        )
        transition_counts: dict[str, Counter[str]] = defaultdict(_counts)
        for prior, group in zip(
            previous,
            pitcher["target_group"],
            strict=True,
        ):
            if pd.notna(group):
                transition_counts[str(prior)][str(group)] += 1

        previous_game_date = (
            pitcher["game_date"].max().date() if not pitcher.empty else None
        )
        snapshot = HistorySnapshot(
            global_counts=global_counts,
            career_counts=career,
            season_counts=season,
            recent_contexts=recent_contexts,
            count_counts=_copy_nested(count_counts),
            stand_counts=_copy_nested(stand_counts),
            transition_counts=_copy_nested(transition_counts),
            previous_game_date=previous_game_date,
        )
        if len(self._snapshots) >= 256:
            self._snapshots.pop(next(iter(self._snapshots)))
        self._snapshots[cache_key] = snapshot
        return snapshot

    def sequence_examples(
        self,
        context: LiveContext,
        records: tuple[PitchRecord, ...],
    ) -> SequenceExamples:
        """Build the current V8.4 example from prior Statcast and Gameday rows."""
        cached = self._sequences.get(context.state_key)
        if cached is not None:
            return cached
        rows = self._load()
        before = rows["game_date"].dt.date < context.game_date
        relevant = rows["pitcher"].eq(context.pitcher_id)
        if context.catcher_id:
            relevant |= pd.to_numeric(
                rows["fielder_2"],
                errors="coerce",
            ).eq(context.catcher_id)
        historical = rows.loc[before & relevant, list(HISTORY_COLUMNS)]

        live_rows = []
        for record in records:
            live_rows.append(
                {
                    "pitch_type": record.raw_code,
                    "game_date": pd.Timestamp(context.game_date),
                    "game_pk": context.game_pk,
                    "at_bat_number": record.at_bat_index,
                    "pitch_number": record.event_index + 1,
                    "pitcher": record.pitcher_id,
                    "batter": record.batter_id,
                    "balls": record.balls,
                    "strikes": record.strikes,
                    "stand": record.stand,
                    "p_throws": record.throws,
                    "description": record.description,
                    "fielder_2": context.catcher_id or np.nan,
                    "release_speed": record.release_speed,
                    "plate_x": record.plate_x,
                    "plate_z": record.plate_z,
                    "release_spin_rate": record.release_spin_rate,
                }
            )
        live_rows.append(
            {
                # The target is a placeholder; no target field enters inference.
                "pitch_type": "FF",
                "game_date": pd.Timestamp(context.game_date),
                "game_pk": context.game_pk,
                "at_bat_number": context.at_bat_index,
                "pitch_number": context.next_pitch_number,
                "pitcher": context.pitcher_id,
                "batter": context.batter_id,
                "balls": context.balls,
                "strikes": context.strikes,
                "stand": context.stand,
                "p_throws": context.throws,
                "description": "ball",
                "fielder_2": context.catcher_id or np.nan,
                "outs_when_up": context.outs,
                "inning": context.inning,
                "on_1b": 1 if context.bases[0] else np.nan,
                "on_2b": 1 if context.bases[1] else np.nan,
                "on_3b": 1 if context.bases[2] else np.nan,
                "n_thruorder_pitcher": sum(
                    record.pitcher_id == context.pitcher_id
                    and record.batter_id == context.batter_id
                    and record.at_bat_index < context.at_bat_index
                    for record in records
                )
                + 1,
            }
        )
        raw = pd.concat(
            [historical, pd.DataFrame(live_rows)],
            ignore_index=True,
            sort=False,
        )
        prepared = prepare_pitch_rows([raw], include_v83=True)
        sequence_raw = raw[
            pd.to_numeric(raw["pitcher"], errors="coerce").eq(
                context.pitcher_id
            )
        ].tail(17)
        examples = SequenceExampleBuilder().build(
            sequence_raw,
            V84_VOCABULARY,
        )
        current = examples.current_numeric.copy()
        extra = np.full(
            (
                len(examples),
                18 + len(V83_CONTEXT_FEATURES),
            ),
            np.nan,
            dtype=np.float32,
        )
        current_row = prepared.iloc[-1]
        season = current_row[list(REPERTOIRE_COLUMNS[:6])].to_numpy(
            dtype=np.float32
        )
        recent = current_row[list(REPERTOIRE_COLUMNS[6:])].to_numpy(
            dtype=np.float32
        )
        extra[-1] = np.concatenate(
            [
                season,
                recent,
                recent - season,
                current_row[list(V83_CONTEXT_FEATURES)].to_numpy(
                    dtype=np.float32
                ),
            ]
        )
        result = replace(
            examples,
            current_numeric=np.column_stack([current, extra]).astype(
                np.float32
            ),
            repertoire_context=extra[:, :18],
        )
        # ponytail: keep only the polled state; cache more if concurrent games matter.
        self._sequences.clear()
        self._sequences[context.state_key] = result
        return result


def _number(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _description(details: dict[str, Any]) -> str:
    code = str(details.get("code", ""))
    mapped = {
        "B": "ball",
        "C": "called_strike",
        "S": "swinging_strike",
        "F": "foul",
        "L": "foul_bunt",
        "T": "foul_tip",
        "X": "hit_into_play",
        "D": "hit_into_play_no_out",
        "E": "hit_into_play_score",
        "W": "blocked_ball",
        "H": "hit_by_pitch",
        "P": "pitchout",
    }
    if code in mapped:
        return mapped[code]
    return str(details.get("description", "NONE")).strip().lower().replace(" ", "_")


def pitch_records(payload: dict[str, Any]) -> tuple[PitchRecord, ...]:
    records = []
    for play in payload.get("liveData", {}).get("plays", {}).get("allPlays", []):
        matchup = play.get("matchup", {})
        pitcher = matchup.get("pitcher", {})
        batter = matchup.get("batter", {})
        if not pitcher.get("id") or not batter.get("id"):
            continue
        balls = 0
        strikes = 0
        for event in play.get("playEvents", []):
            before_balls, before_strikes = balls, strikes
            count = event.get("count") or {}
            if "balls" in count:
                balls = int(count["balls"])
            if "strikes" in count:
                strikes = int(count["strikes"])
            if not event.get("isPitch"):
                continue
            details = event.get("details") or {}
            raw_code = str((details.get("type") or {}).get("code", ""))
            context = context_pitch(raw_code)
            if context is None:
                continue
            pitch_data = event.get("pitchData") or {}
            coordinates = pitch_data.get("coordinates") or {}
            breaks = pitch_data.get("breaks") or {}
            records.append(
                PitchRecord(
                    sequence=len(records) + 1,
                    at_bat_index=int(play.get("atBatIndex", 0)),
                    event_index=int(event.get("index", 0)),
                    play_id=str(event.get("playId", "")),
                    pitcher_id=int(pitcher["id"]),
                    batter_id=int(batter["id"]),
                    stand=str((matchup.get("batSide") or {}).get("code", UNKNOWN)),
                    throws=str(
                        (matchup.get("pitchHand") or {}).get("code", UNKNOWN)
                    ),
                    balls=before_balls,
                    strikes=before_strikes,
                    raw_code=raw_code,
                    context_group=str(context),
                    target_group=(
                        str(group_pitch(raw_code))
                        if group_pitch(raw_code) is not None
                        else None
                    ),
                    description=_description(details),
                    release_speed=_number(pitch_data.get("startSpeed")),
                    plate_x=_number(coordinates.get("pX")),
                    plate_z=_number(coordinates.get("pZ")),
                    release_spin_rate=_number(breaks.get("spinRate")),
                )
            )
    return tuple(records)


def _game(payload: dict[str, Any], game_pk: int) -> dict[str, Any]:
    game_data = payload.get("gameData", {})
    teams = game_data.get("teams", {})
    status = game_data.get("status", {})
    return {
        "gamePk": game_pk,
        "date": str((game_data.get("datetime") or {}).get("officialDate", "")),
        "awayTeam": str((teams.get("away") or {}).get("name", "Away")),
        "homeTeam": str((teams.get("home") or {}).get("name", "Home")),
        "status": str(status.get("abstractGameState", "Unknown")),
        "detailedStatus": str(status.get("detailedState", "Unknown")),
    }


def current_context(
    payload: dict[str, Any],
    game_pk: int,
    records: tuple[PitchRecord, ...],
) -> LiveContext | None:
    game = _game(payload, game_pk)
    if game["status"] not in {"Live", "Preview"}:
        return None
    current = payload.get("liveData", {}).get("plays", {}).get("currentPlay")
    if not current or (current.get("about") or {}).get("isComplete"):
        return None
    matchup = current.get("matchup") or {}
    pitcher = matchup.get("pitcher") or {}
    batter = matchup.get("batter") or {}
    if not pitcher.get("id") or not batter.get("id"):
        return None
    count = current.get("count") or {}
    about = current.get("about") or {}
    linescore = payload.get("liveData", {}).get("linescore") or {}
    offense = linescore.get("offense") or {}
    defense = linescore.get("defense") or {}
    catcher = defense.get("catcher") or {}
    team_scores = linescore.get("teams") or {}
    away_score = int((team_scores.get("away") or {}).get("runs") or 0)
    home_score = int((team_scores.get("home") or {}).get("runs") or 0)
    at_bat_index = int(current.get("atBatIndex", about.get("atBatIndex", 0)))
    pitch_count = sum(
        bool(event.get("isPitch")) for event in current.get("playEvents", [])
    )
    game_date = date.fromisoformat(game["date"])
    raw_state = {
        "gamePk": game_pk,
        "atBatIndex": at_bat_index,
        "targetSequence": len(records) + 1,
        "pitcherId": int(pitcher["id"]),
        "batterId": int(batter["id"]),
        "balls": int(count.get("balls") or 0),
        "strikes": int(count.get("strikes") or 0),
        "outs": int(count.get("outs") or linescore.get("outs") or 0),
        "inning": int(about.get("inning") or linescore.get("currentInning") or 1),
        "half": "Top" if bool(about.get("isTopInning")) else "Bot",
        "bases": [
            bool(offense.get("first")),
            bool(offense.get("second")),
            bool(offense.get("third")),
        ],
        "awayScore": away_score,
        "homeScore": home_score,
    }
    state_key = hashlib.sha256(
        json.dumps(raw_state, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    return LiveContext(
        game_pk=game_pk,
        game_date=game_date,
        status=game["detailedStatus"],
        at_bat_index=at_bat_index,
        target_sequence=len(records) + 1,
        pitcher_id=int(pitcher["id"]),
        pitcher_name=str(pitcher.get("fullName", f"#{pitcher['id']}")),
        batter_id=int(batter["id"]),
        batter_name=str(batter.get("fullName", f"#{batter['id']}")),
        catcher_id=int(catcher.get("id") or 0),
        throws=str((matchup.get("pitchHand") or {}).get("code", UNKNOWN)),
        stand=str((matchup.get("batSide") or {}).get("code", UNKNOWN)),
        balls=raw_state["balls"],
        strikes=raw_state["strikes"],
        outs=raw_state["outs"],
        inning=raw_state["inning"],
        half=raw_state["half"],
        bases=tuple(raw_state["bases"]),
        away_score=away_score,
        home_score=home_score,
        next_pitch_number=pitch_count + 1,
        state_key=state_key,
    )


def _rate(
    counts: Counter[str],
    backing: dict[str, float],
    strength: float,
) -> dict[str, float]:
    total = sum(counts.values())
    return {
        str(group): (
            counts[str(group)] + strength * backing[str(group)]
        )
        / (total + strength)
        for group in PITCH_GROUPS
    }


def _global_rate(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    return {
        str(group): (counts[str(group)] + 1) / (total + len(PITCH_GROUPS))
        for group in PITCH_GROUPS
    }


def _streak(contexts: list[str]) -> int:
    if not contexts:
        return 0
    last = contexts[-1]
    count = 0
    for value in reversed(contexts):
        if value != last:
            break
        count += 1
    return count


def build_feature_row(
    payload: dict[str, Any],
    context: LiveContext,
    records: tuple[PitchRecord, ...],
    history: HistoryStore,
) -> pd.DataFrame:
    snapshot = history.snapshot(context.pitcher_id, context.game_date)
    global_counts = Counter(snapshot.global_counts)
    career_counts = Counter(snapshot.career_counts)
    season_counts = Counter(snapshot.season_counts)
    recent = deque(snapshot.recent_contexts, maxlen=100)
    count_counts = _copy_nested(snapshot.count_counts)
    stand_counts = _copy_nested(snapshot.stand_counts)
    transition_counts = _copy_nested(snapshot.transition_counts)

    game_contexts: list[str] = []
    plate_contexts: list[str] = []
    previous_by_pitcher: dict[int, str] = defaultdict(lambda: UNKNOWN)
    previous_record: PitchRecord | None = None
    for record in records:
        target = record.target_group
        if target is not None:
            global_counts[target] += 1
        if record.at_bat_index == context.at_bat_index:
            plate_contexts.append(record.context_group)
        if record.pitcher_id != context.pitcher_id:
            continue
        prior = previous_by_pitcher[record.pitcher_id]
        if target is not None:
            career_counts[target] += 1
            season_counts[target] += 1
            count_counts.setdefault(
                (record.balls, record.strikes), _counts()
            )[target] += 1
            stand_counts.setdefault(record.stand, _counts())[target] += 1
            transition_counts.setdefault(prior, _counts())[target] += 1
        previous_by_pitcher[record.pitcher_id] = record.context_group
        recent.append(record.context_group)
        game_contexts.append(record.context_group)
        previous_record = record

    global_rates = _global_rate(global_counts)
    career_rates = _rate(career_counts, global_rates, 50)
    season_rates = _rate(season_counts, career_rates, 30)
    game_counts = _counts()
    game_counts.update(value for value in game_contexts if value in game_counts)
    game_rates = _rate(game_counts, season_rates, 20)
    recent_counts = _counts()
    recent_counts.update(value for value in recent if value in recent_counts)
    recent_rates = _rate(recent_counts, career_rates, 20)
    situation_counts = count_counts.get(
        (context.balls, context.strikes), _counts()
    )
    situation_rates = _rate(situation_counts, career_rates, 20)
    handed_counts = stand_counts.get(context.stand, _counts())
    handed_rates = _rate(handed_counts, career_rates, 30)
    previous_group = game_contexts[-1] if game_contexts else UNKNOWN
    transition = transition_counts.get(previous_group, _counts())
    transition_rates = _rate(transition, career_rates, 15)

    pa_counts = _counts()
    pa_counts.update(value for value in plate_contexts if value in pa_counts)
    pa_total = sum(pa_counts.values())
    fastball = pa_counts["FOUR_SEAM"] + pa_counts["MOVING_FASTBALL"]
    breaking = pa_counts["SLIDER"] + pa_counts["CURVE"]
    offspeed = pa_counts["CHANGEUP"] + pa_counts["SPLITTER_FORK"]

    plays = payload.get("liveData", {}).get("plays", {}).get("allPlays", [])
    times_faced = sum(
        int(play.get("atBatIndex", -1)) < context.at_bat_index
        and int((play.get("matchup", {}).get("pitcher") or {}).get("id") or -1)
        == context.pitcher_id
        and int((play.get("matchup", {}).get("batter") or {}).get("id") or -1)
        == context.batter_id
        for play in plays
    )
    days_since = (
        (context.game_date - snapshot.previous_game_date).days
        if snapshot.previous_game_date is not None
        else np.nan
    )
    batting_score = (
        context.away_score if context.half == "Top" else context.home_score
    )
    fielding_score = (
        context.home_score if context.half == "Top" else context.away_score
    )
    row: dict[str, object] = {
        "game_date": pd.Timestamp(context.game_date),
        "game_pk": context.game_pk,
        "pitcher": str(context.pitcher_id),
        "batter": str(context.batter_id),
        "pitcher_id": context.pitcher_id,
        "batter_id": context.batter_id,
        "balls": context.balls,
        "strikes": context.strikes,
        "outs_when_up": context.outs,
        "inning": context.inning,
        "base_1": int(context.bases[0]),
        "base_2": int(context.bases[1]),
        "base_3": int(context.bases[2]),
        "score_diff": batting_score - fielding_score,
        "game_pitch_count": len(game_contexts),
        "n_thruorder_pitcher": times_faced + 1,
        "pitcher_days_since_prev_game": days_since,
        "prev_release_speed": (
            previous_record.release_speed if previous_record else np.nan
        ),
        "prev_plate_x": previous_record.plate_x if previous_record else np.nan,
        "prev_plate_z": previous_record.plate_z if previous_record else np.nan,
        "pitch_number": context.next_pitch_number,
        "prev_pitch_streak": _streak(game_contexts),
        "pa_prev_pitch_streak": _streak(plate_contexts),
        "pa_distinct_pitch_groups": sum(value > 0 for value in pa_counts.values()),
        "pa_fastball_share": (fastball + 1) / (pa_total + 3),
        "pa_breaking_share": (breaking + 1) / (pa_total + 3),
        "pa_offspeed_share": (offspeed + 1) / (pa_total + 3),
        "game_distinct_pitch_groups": sum(
            value > 0 for value in game_counts.values()
        ),
        "stand": context.stand,
        "p_throws": context.throws,
        "inning_topbot": context.half,
        "prev_pitch_group": previous_group,
        "prev_description": (
            previous_record.description if previous_record else "NONE"
        ),
        "pa_prev_pitch_1": (
            plate_contexts[-1] if len(plate_contexts) >= 1 else UNKNOWN
        ),
        "pa_prev_pitch_2": (
            plate_contexts[-2] if len(plate_contexts) >= 2 else UNKNOWN
        ),
        "pa_prev_pitch_3": (
            plate_contexts[-3] if len(plate_contexts) >= 3 else UNKNOWN
        ),
        "pa_first_pitch_group": plate_contexts[0] if plate_contexts else UNKNOWN,
        "career_support": sum(career_counts.values()),
        "count_support": sum(situation_counts.values()),
        "stand_support": sum(handed_counts.values()),
        "transition_support": sum(transition.values()),
    }
    for group in PITCH_GROUPS:
        key = str(group)
        row[f"pa_count_{key}"] = pa_counts[key]
        for prefix, values in (
            ("career_rate_", career_rates),
            ("season_rate_", season_rates),
            ("recent_100_rate_", recent_rates),
            ("game_rate_", game_rates),
            ("count_rate_", situation_rates),
            ("stand_rate_", handed_rates),
            ("transition_rate_", transition_rates),
        ):
            row[f"{prefix}{key}"] = values[key]
        row[f"game_delta_{key}"] = game_rates[key] - season_rates[key]
    for name in NUMERIC_FEATURES:
        row.setdefault(name, np.nan)
    for name in CATEGORICAL_FEATURES:
        row.setdefault(name, UNKNOWN)
    return pd.DataFrame([row])


class LivePredictor:
    """Load the complete frozen V7.2 Global + residual pipeline once."""

    requires_sequence = False

    def __init__(self, model_directory: Path = Path("models/v7.2")):
        self.model_directory = model_directory
        registry_path = model_directory / "registry.json"
        self.payload = json.loads(registry_path.read_text(encoding="utf-8"))
        self.registry = _registry(self.payload)
        global_config = self.payload["global"]
        residual_config = self.payload["residual"]
        with (model_directory / global_config["model"]).open("rb") as handle:
            self.global_model = pickle.load(handle)
        with (model_directory / residual_config["model"]).open("rb") as handle:
            self.residual_model = pickle.load(handle)
        with (model_directory / self.payload["gate"]["model"]).open("rb") as handle:
            self.gate_model = pickle.load(handle)
        self.global_temperature = float(global_config["temperature"])
        self.calibration_bias = (
            global_config.get("calibration", {}).get("bias")
        )
        self.residual_controls = {
            "full_tier_boost": float(
                residual_config.get("fullTierBoost", 1.0)
            ),
            "limited_scale_boost": float(
                residual_config.get(
                    "limitedTierBoost",
                    residual_config.get("limitedScaleBoost", 1.0),
                )
            ),
            "reliability_scale_boost": float(
                residual_config.get("reliabilityScaleBoost", 1.0)
            ),
            "context_gate_power": float(
                residual_config.get("contextGatePower", 1.0)
            ),
            "scale_cap": float(residual_config.get("scaleCap", 0.5)),
            "js_divergence_cap": float(
                residual_config.get(
                    "jsDivergenceCap",
                    residual_config.get("jsCap", 0.05),
                )
            ),
            "class_probability_shift_cap": float(
                residual_config.get(
                    "classProbabilityShiftCap",
                    residual_config.get("probabilityShiftCap", 0.20),
                )
            ),
        }

    def predict(
        self,
        rows: pd.DataFrame,
        _sequence_examples: SequenceExamples | None,
    ) -> dict[str, Any]:
        global_probabilities = predict_candidate(
            self.global_model,
            rows,
            temperature=self.global_temperature,
        )
        if self.calibration_bias is not None:
            global_probabilities = apply_logit_bias(
                global_probabilities,
                self.calibration_bias,
            )
        correction = predict_correction(self.residual_model, rows)
        context_gate = predict_context_gate(
            self.gate_model,
            rows,
            global_probabilities,
            correction,
        )
        probabilities, sources, routing = apply_reliability_gated_residual(
            rows,
            global_probabilities,
            correction,
            context_gate,
            self.registry,
            prediction_dates=[
                pd.Timestamp(value).date() for value in rows["game_date"]
            ],
            **self.residual_controls,
        )
        values = probabilities[0]
        top_index = int(
            hierarchical_top_indices(
                values[None, :],
                PITCH_GROUP_FAMILY_LABELS,
            )[0]
        )
        top = str(PITCH_GROUPS[top_index])
        route = dict(routing[0])
        route["pipeline"] = "v7.2-residual"
        return {
            "modelVersion": str(self.payload.get("modelVersion", "V7.2")),
            "topPitch": top,
            "confidence": round(float(values[top_index]), 6),
            "probabilities": {
                str(group): round(float(value), 6)
                for group, value in zip(PITCH_GROUPS, values, strict=True)
            },
            "source": sources[0],
            "routing": route,
        }


def build_live_view(
    payload: dict[str, Any],
    game_pk: int,
    history: HistoryStore,
    *,
    include_sequence: bool = True,
) -> LiveView:
    copied = copy.deepcopy(payload)
    records = pitch_records(copied)
    context = current_context(copied, game_pk, records)
    row = (
        build_feature_row(copied, context, records, history)
        if context is not None
        else None
    )
    sequence_examples = (
        history.sequence_examples(context, records)
        if context is not None and include_sequence
        else None
    )
    repertoire: tuple[str, ...] = ()
    if context is not None:
        snapshot = history.snapshot(context.pitcher_id, context.game_date)
        counts = Counter(snapshot.season_counts)
        counts.update(
            record.target_group
            for record in records
            if record.pitcher_id == context.pitcher_id
            and record.target_group is not None
        )
        if not sum(counts.values()):
            counts = Counter(snapshot.career_counts)
        repertoire = tuple(
            str(group) for group in PITCH_GROUPS if counts[str(group)] > 0
        )
    return LiveView(
        context=context,
        pitches=records,
        game=_game(copied, game_pk),
        repertoire=repertoire,
        feature_row=row,
        sequence_examples=sequence_examples,
    )


def serialize_context(context: LiveContext) -> dict[str, Any]:
    return {
        "stateKey": context.state_key,
        "targetSequence": context.target_sequence,
        "atBatIndex": context.at_bat_index,
        "pitchNumber": context.next_pitch_number,
        "pitcher": {
            "id": context.pitcher_id,
            "name": context.pitcher_name,
            "throws": context.throws,
        },
        "batter": {
            "id": context.batter_id,
            "name": context.batter_name,
            "stand": context.stand,
        },
        "count": {
            "balls": context.balls,
            "strikes": context.strikes,
        },
        "outs": context.outs,
        "inning": context.inning,
        "half": context.half,
        "bases": list(context.bases),
        "score": {
            "away": context.away_score,
            "home": context.home_score,
        },
    }


def serialize_pitch(record: PitchRecord) -> dict[str, Any]:
    return {
        "sequence": record.sequence,
        "pitchGroup": record.target_group,
        "rawCode": record.raw_code,
        "description": record.description,
    }
