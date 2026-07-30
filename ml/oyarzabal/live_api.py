"""Local API for the OyarZabal Gameday Chrome side panel."""

from __future__ import annotations

import copy
import json
import os
import sqlite3
import threading
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request

from .live import (
    HistoryStore,
    LivePredictor,
    LiveView,
    build_live_view,
    current_context,
    pitch_records,
    serialize_context,
)
from .taxonomy import (
    PITCH_GROUP_FAMILIES,
    PITCH_GROUP_NAMES,
    PITCH_GROUPS,
    group_pitch,
)

DEFAULT_API_PORT = 8010
DEFAULT_WINDOW_SECONDS = 8


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _serialize_view_context(view: LiveView) -> dict[str, Any]:
    if view.context is None or view.feature_row is None:
        raise ValueError("live view has no prediction context")
    row = view.feature_row.iloc[0]

    def rates(prefix: str) -> dict[str, float]:
        return {
            str(group): float(row[f"{prefix}{group}"])
            for group in PITCH_GROUPS
        }

    context = serialize_context(view.context)
    context["insights"] = {
        "repertoire": list(view.repertoire),
        "rates": {
            "season": rates("season_rate_"),
            "recent100": rates("recent_100_rate_"),
            "game": rates("game_rate_"),
            "count": rates("count_rate_"),
            "stand": rates("stand_rate_"),
        },
        "support": {
            "count": int(row["count_support"]),
            "stand": int(row["stand_support"]),
        },
        "workload": {
            "gamePitches": int(row["game_pitch_count"]),
            "timesFaced": int(row["n_thruorder_pitcher"]),
            "plateAppearancePitches": sum(
                int(row[f"pa_count_{group}"]) for group in PITCH_GROUPS
            ),
        },
    }
    return context


def fetch_gameday(game_pk: int) -> dict[str, Any]:
    url = (
        f"https://ws.statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        "?language=en"
    )
    request_value = urllib.request.Request(
        url,
        headers={
            "User-Agent": "OyarZabal-live-prototype/0.1",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    with urllib.request.urlopen(request_value, timeout=8) as response:
        return json.load(response)


def _connect(path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    return connection


def _init_database(path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as connection:
        connection.executescript(
            """
            PRAGMA journal_mode = WAL;

            CREATE TABLE IF NOT EXISTS live_rounds (
                id TEXT PRIMARY KEY,
                game_pk INTEGER NOT NULL,
                state_key TEXT NOT NULL,
                target_sequence INTEGER NOT NULL,
                context_json TEXT NOT NULL,
                prediction_json TEXT NOT NULL,
                opened_at TEXT NOT NULL,
                deadline TEXT NOT NULL,
                status TEXT NOT NULL,
                actual_pitch TEXT,
                UNIQUE (game_pk, state_key)
            );

            CREATE TABLE IF NOT EXISTS live_answers (
                round_id TEXT NOT NULL REFERENCES live_rounds(id),
                session_id TEXT NOT NULL,
                selected_pitch TEXT NOT NULL,
                received_at TEXT NOT NULL,
                PRIMARY KEY (round_id, session_id)
            );

            CREATE TABLE IF NOT EXISTS live_round_sessions (
                round_id TEXT NOT NULL REFERENCES live_rounds(id),
                session_id TEXT NOT NULL,
                seen_at TEXT NOT NULL,
                PRIMARY KEY (round_id, session_id)
            );

            CREATE INDEX IF NOT EXISTS live_rounds_game_sequence
                ON live_rounds (game_pk, target_sequence);
            """
        )


class LiveService:
    def __init__(
        self,
        *,
        database_path: str,
        history: HistoryStore,
        predictor: LivePredictor,
        fetcher: Callable[[int], dict[str, Any]] = fetch_gameday,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ):
        self.database_path = database_path
        self.history = history
        self.predictor = predictor
        self.include_sequence = bool(
            getattr(predictor, "requires_sequence", False)
        )
        self.fetcher = fetcher
        self.window_seconds = window_seconds
        self._predictions: dict[str, dict[str, Any]] = {}
        self._views: dict[str, LiveView] = {}
        self._lock = threading.Lock()
        _init_database(database_path)

    def _prediction(self, view: LiveView) -> dict[str, Any]:
        if (
            view.context is None
            or view.feature_row is None
        ):
            raise ValueError("live view has no prediction context")
        key = view.context.state_key
        if key not in self._predictions:
            self._predictions[key] = self.predictor.predict(
                view.feature_row,
                view.sequence_examples,
            )
        return self._predictions[key]

    def _reconcile(
        self,
        connection: sqlite3.Connection,
        game_pk: int,
        view: LiveView,
    ) -> None:
        active = connection.execute(
            """
            SELECT id, state_key, target_sequence
            FROM live_rounds
            WHERE game_pk = ? AND status IN ('open', 'locked')
            """,
            (game_pk,),
        ).fetchall()
        current_key = view.context.state_key if view.context else None
        current_sequence = (
            view.context.target_sequence if view.context else len(view.pitches) + 1
        )
        for round_value in active:
            target_sequence = int(round_value["target_sequence"])
            if target_sequence <= len(view.pitches):
                pitch = view.pitches[target_sequence - 1]
                status = "result" if pitch.target_group is not None else "cancelled"
                connection.execute(
                    """
                    UPDATE live_rounds
                    SET status = ?, actual_pitch = ?
                    WHERE id = ?
                    """,
                    (status, pitch.target_group, round_value["id"]),
                )
            elif (
                target_sequence == current_sequence
                and current_key is not None
                and round_value["state_key"] != current_key
            ):
                connection.execute(
                    "UPDATE live_rounds SET status = 'cancelled' WHERE id = ?",
                    (round_value["id"],),
                )

    def _ensure_round(
        self,
        connection: sqlite3.Connection,
        view: LiveView,
    ) -> sqlite3.Row | None:
        if view.context is None:
            return None
        prediction = self._prediction(view)
        round_id = (
            f"{view.context.game_pk}-"
            f"{view.context.target_sequence}-"
            f"{view.context.state_key[:12]}"
        )
        opened = _utcnow()
        deadline = opened + timedelta(seconds=self.window_seconds)
        connection.execute(
            """
            INSERT OR IGNORE INTO live_rounds (
                id, game_pk, state_key, target_sequence,
                context_json, prediction_json,
                opened_at, deadline, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')
            """,
            (
                round_id,
                view.context.game_pk,
                view.context.state_key,
                view.context.target_sequence,
                json.dumps(_serialize_view_context(view), ensure_ascii=False),
                json.dumps(prediction, ensure_ascii=False),
                opened.isoformat(),
                deadline.isoformat(),
            ),
        )
        connection.execute(
            """
            UPDATE live_rounds
            SET status = 'locked'
            WHERE id = ? AND status = 'open' AND deadline <= ?
            """,
            (round_id, _utcnow().isoformat()),
        )
        return connection.execute(
            "SELECT * FROM live_rounds WHERE id = ?",
            (round_id,),
        ).fetchone()

    def refresh(self, game_pk: int) -> tuple[LiveView, sqlite3.Row | None]:
        payload = self.fetcher(game_pk)
        for _ in range(3):
            records = pitch_records(payload)
            context = current_context(payload, game_pk, records)
            state_key = context.state_key if context else None
            view = self._views.get(state_key) if state_key else None
            built = view is None
            if built:
                view = build_live_view(
                    payload,
                    game_pk,
                    self.history,
                    include_sequence=self.include_sequence,
                )
            with self._lock, _connect(self.database_path) as connection:
                self._reconcile(connection, game_pk, view)
                round_value = self._ensure_round(connection, view)
            if not built or state_key is None:
                return view, round_value

            latest = self.fetcher(game_pk)
            latest_records = pitch_records(latest)
            latest_context = current_context(latest, game_pk, latest_records)
            latest_key = latest_context.state_key if latest_context else None
            if latest_key == state_key:
                self._views[state_key] = view
                if len(self._views) > 32:
                    self._views.pop(next(iter(self._views)))
                return view, round_value
            payload = latest
        return view, round_value

    def snapshot(self, game_pk: int, session_id: str) -> dict[str, Any]:
        view, round_value = self.refresh(game_pk)
        response: dict[str, Any] = {
            "game": view.game,
            "pitchGroups": {
                str(group): PITCH_GROUP_NAMES[group] for group in PITCH_GROUPS
            },
            "pitchGroupFamilies": {
                str(group): str(PITCH_GROUP_FAMILIES[group])
                for group in PITCH_GROUPS
            },
            "serverTime": _utcnow().isoformat(),
        }
        with _connect(self.database_path) as connection:
            if round_value is not None:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO live_round_sessions (
                        round_id, session_id, seen_at
                    ) VALUES (?, ?, ?)
                    """,
                    (
                        round_value["id"],
                        session_id,
                        _utcnow().isoformat(),
                    ),
                )
            last_result = connection.execute(
                """
                SELECT
                    r.id, r.context_json, r.prediction_json, r.actual_pitch,
                    a.selected_pitch
                FROM live_rounds r
                JOIN live_round_sessions s
                    ON s.round_id = r.id AND s.session_id = ?
                LEFT JOIN live_answers a
                    ON a.round_id = r.id AND a.session_id = s.session_id
                WHERE
                    r.game_pk = ?
                    AND r.status = 'result'
                ORDER BY r.target_sequence DESC
                LIMIT 1
                """,
                (session_id, game_pk),
            ).fetchone()
        response["lastResult"] = (
            {
                "roundId": last_result["id"],
                "context": json.loads(last_result["context_json"]),
                "prediction": json.loads(last_result["prediction_json"]),
                "selectedPitch": last_result["selected_pitch"],
                "actualPitch": last_result["actual_pitch"],
            }
            if last_result
            else None
        )
        notice = None
        if round_value is not None:
            previous_sequence = int(round_value["target_sequence"]) - 1
            if 0 < previous_sequence <= len(view.pitches):
                previous_pitch = view.pitches[previous_sequence - 1]
                if previous_pitch.target_group is None:
                    notice = {
                        "type": "unsupportedPitch",
                        "rawCode": previous_pitch.raw_code,
                    }
            if notice is None:
                with _connect(self.database_path) as connection:
                    cancelled = connection.execute(
                        """
                        SELECT 1
                        FROM live_rounds
                        WHERE
                            game_pk = ?
                            AND target_sequence = ?
                            AND status = 'cancelled'
                            AND id != ?
                        LIMIT 1
                        """,
                        (
                            game_pk,
                            round_value["target_sequence"],
                            round_value["id"],
                        ),
                    ).fetchone()
                if cancelled is not None:
                    notice = {"type": "stateChanged"}
        response["roundNotice"] = notice
        if round_value is None:
            response["round"] = {
                "status": (
                    "final" if view.game["status"] == "Final" else "syncing"
                )
            }
            return response

        with _connect(self.database_path) as connection:
            answer = connection.execute(
                """
                SELECT selected_pitch, received_at
                FROM live_answers
                WHERE round_id = ? AND session_id = ?
                """,
                (round_value["id"], session_id),
            ).fetchone()
        round_status = str(round_value["status"])
        visible = round_status in {"locked", "result"}
        round_payload: dict[str, Any] = {
            "id": round_value["id"],
            "status": round_status,
            "openedAt": round_value["opened_at"],
            "deadline": round_value["deadline"],
            "context": json.loads(round_value["context_json"]),
            "prediction": (
                json.loads(round_value["prediction_json"]) if visible else None
            ),
            "actualPitch": round_value["actual_pitch"],
            "answer": (
                {
                    "selectedPitch": answer["selected_pitch"],
                    "receivedAt": answer["received_at"],
                }
                if answer
                else None
            ),
        }
        response["round"] = round_payload
        return response

    def answer(
        self,
        round_id: str,
        session_id: str,
        selected_pitch: str,
    ) -> dict[str, Any]:
        if selected_pitch not in {str(group) for group in PITCH_GROUPS}:
            raise ValueError("지원하지 않는 구종입니다.")
        with _connect(self.database_path) as connection:
            round_value = connection.execute(
                "SELECT * FROM live_rounds WHERE id = ?",
                (round_id,),
            ).fetchone()
        if round_value is None:
            raise LookupError("알 수 없는 대결입니다.")

        self.refresh(int(round_value["game_pk"]))
        received_at = _utcnow()
        with self._lock, _connect(self.database_path) as connection:
            round_value = connection.execute(
                "SELECT * FROM live_rounds WHERE id = ?",
                (round_id,),
            ).fetchone()
            if (
                round_value["status"] != "open"
                or received_at >= _timestamp(round_value["deadline"])
                or round_value["actual_pitch"] is not None
            ):
                raise RuntimeError("이미 마감된 대결입니다.")
            connection.execute(
                """
                INSERT OR IGNORE INTO live_round_sessions (
                    round_id, session_id, seen_at
                ) VALUES (?, ?, ?)
                """,
                (round_id, session_id, received_at.isoformat()),
            )
            connection.execute(
                """
                INSERT INTO live_answers (
                    round_id, session_id, selected_pitch, received_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT (round_id, session_id) DO UPDATE SET
                    selected_pitch = excluded.selected_pitch,
                    received_at = excluded.received_at
                """,
                (
                    round_id,
                    session_id,
                    selected_pitch,
                    received_at.isoformat(),
                ),
            )
        return {
            "roundId": round_id,
            "selectedPitch": selected_pitch,
            "receivedAt": received_at.isoformat(),
            "status": "draft",
        }


@dataclass
class ReplayState:
    """PROTOTYPE: one in-memory completed-game replay."""

    game_pk: int
    generation: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    pitch_index: int = 0
    answer: dict[str, str] | None = None
    last_result: dict[str, Any] | None = None
    game: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    prediction: dict[str, Any] | None = None
    actual_pitch: str | None = None
    round_id: str | None = None


def _replay_targets(payload: dict[str, Any]) -> list[tuple[int, int, str]]:
    targets = []
    plays = payload.get("liveData", {}).get("plays", {}).get("allPlays", [])
    for play_index, play in enumerate(plays):
        for event_index, event in enumerate(play.get("playEvents", [])):
            if not event.get("isPitch"):
                continue
            raw_code = str(
                (((event.get("details") or {}).get("type") or {}).get("code"))
                or ""
            )
            group = group_pitch(raw_code)
            if group is not None:
                targets.append((play_index, event_index, str(group)))
    return targets


def _count_before_pitch(
    plays: list[dict[str, Any]],
    play_index: int,
    event_index: int,
) -> dict[str, int]:
    current_events = plays[play_index].get("playEvents", [])
    for event in reversed(current_events[:event_index]):
        count = event.get("count")
        if count:
            return {
                "balls": int(count.get("balls") or 0),
                "strikes": int(count.get("strikes") or 0),
                "outs": int(count.get("outs") or 0),
            }
    previous_outs = 0
    if play_index:
        previous_events = plays[play_index - 1].get("playEvents", [])
        if previous_events:
            previous_outs = int(
                (previous_events[-1].get("count") or {}).get("outs") or 0
            )
    return {"balls": 0, "strikes": 0, "outs": previous_outs % 3}


_BASE_NAMES = {"1B": "first", "2B": "second", "3B": "third"}


def _replay_offense(
    plays: list[dict[str, Any]],
    play_index: int,
    event_index: int,
) -> dict[str, dict[str, Any]]:
    target_about = plays[play_index].get("about") or {}
    target_half = (
        int(target_about.get("inning") or 1),
        bool(target_about.get("isTopInning")),
    )
    occupied: dict[str, dict[str, Any]] = {}
    for index, play in enumerate(plays[: play_index + 1]):
        about = play.get("about") or {}
        if (
            int(about.get("inning") or 1),
            bool(about.get("isTopInning")),
        ) != target_half:
            continue
        movements: dict[int, list[dict[str, Any]]] = {}
        for fallback_index, runner in enumerate(play.get("runners") or []):
            details = runner.get("details") or {}
            raw_play_index = details.get("playIndex")
            movement_index = (
                int(raw_play_index)
                if isinstance(raw_play_index, (int, float))
                else fallback_index
            )
            if index == play_index and (
                raw_play_index is None or movement_index >= event_index
            ):
                continue
            movements.setdefault(movement_index, []).append(runner)
        for movement_index in sorted(movements):
            runners = movements[movement_index]
            for runner in runners:
                movement = runner.get("movement") or {}
                start = _BASE_NAMES.get(
                    movement.get("start") or movement.get("originBase")
                )
                if start:
                    occupied.pop(start, None)
                runner_id = (
                    ((runner.get("details") or {}).get("runner") or {}).get("id")
                )
                if runner_id:
                    occupied = {
                        base: value
                        for base, value in occupied.items()
                        if value.get("id") != runner_id
                    }
            for runner in runners:
                movement = runner.get("movement") or {}
                end = _BASE_NAMES.get(movement.get("end"))
                if not end or movement.get("isOut"):
                    continue
                runner_value = (
                    (runner.get("details") or {}).get("runner") or {"id": True}
                )
                occupied[end] = runner_value
    return {
        name: occupied[name]
        for name in ("first", "second", "third")
        if name in occupied
    }


def _replay_payload(
    payload: dict[str, Any],
    target: tuple[int, int, str],
) -> dict[str, Any]:
    """Return the completed feed as it looked immediately before one pitch."""
    play_index, event_index, _ = target
    copied = copy.deepcopy(payload)
    source_plays = payload["liveData"]["plays"]["allPlays"]
    current = copy.deepcopy(source_plays[play_index])
    current["playEvents"] = current.get("playEvents", [])[:event_index]
    current["count"] = _count_before_pitch(
        source_plays,
        play_index,
        event_index,
    )
    current.setdefault("about", {})["isComplete"] = False
    current.pop("result", None)

    plays = copied["liveData"]["plays"]
    plays["allPlays"] = [
        *copy.deepcopy(source_plays[:play_index]),
        current,
    ]
    plays["currentPlay"] = current

    about = current.get("about") or {}
    previous_result = (
        source_plays[play_index - 1].get("result") or {}
        if play_index
        else {}
    )
    linescore = copied["liveData"].setdefault("linescore", {})
    linescore.update(
        {
            "currentInning": int(about.get("inning") or 1),
            "inningState": "Top"
            if bool(about.get("isTopInning"))
            else "Bottom",
            "isTopInning": bool(about.get("isTopInning")),
            "outs": current["count"]["outs"],
            "offense": _replay_offense(
                source_plays,
                play_index,
                event_index,
            ),
            "teams": {
                "away": {
                    "runs": int(previous_result.get("awayScore") or 0)
                },
                "home": {
                    "runs": int(previous_result.get("homeScore") or 0)
                },
            },
        }
    )
    copied["gameData"]["status"] = {
        "abstractGameState": "Live",
        "detailedState": "Replay Test",
    }
    return copied


class ReplayService:
    """PROTOTYPE: drive completed feeds through the production live path."""

    def __init__(self, live: LiveService):
        self.live = live
        self._feeds: dict[int, dict[str, Any]] = {}
        self._states: dict[tuple[str, int], ReplayState] = {}
        self._lock = threading.Lock()

    def _feed(self, game_pk: int) -> dict[str, Any]:
        if game_pk not in self._feeds:
            payload = self.live.fetcher(game_pk)
            if not _replay_targets(payload):
                raise ValueError("재생할 투구가 없는 경기입니다.")
            self._feeds[game_pk] = payload
        return self._feeds[game_pk]

    def start(self, game_pk: int, session_id: str) -> dict[str, Any]:
        self._feed(game_pk)
        with self._lock:
            # ponytail: one process-local map; add shared storage for multi-worker use.
            if len(self._states) >= 128:
                self._states.pop(next(iter(self._states)))
            self._states[(session_id, game_pk)] = ReplayState(game_pk=game_pk)
        return self.snapshot(game_pk, session_id)

    def _state(self, game_pk: int, session_id: str) -> ReplayState:
        state = self._states.get((session_id, game_pk))
        if state is None:
            raise LookupError("먼저 테스트 모드를 시작하세요.")
        return state

    def _prepare(self, state: ReplayState) -> None:
        if state.context is not None or state.round_id is not None:
            return
        feed = self._feed(state.game_pk)
        targets = _replay_targets(feed)
        if state.pitch_index >= len(targets):
            state.game = build_live_view(
                feed,
                state.game_pk,
                self.live.history,
                include_sequence=self.live.include_sequence,
            ).game
            return
        target = targets[state.pitch_index]
        view = build_live_view(
            _replay_payload(feed, target),
            state.game_pk,
            self.live.history,
            include_sequence=self.live.include_sequence,
        )
        if view.context is None:
            raise ValueError("투구 직전 상태를 만들지 못했습니다.")
        state.game = view.game
        state.context = _serialize_view_context(view)
        state.prediction = self.live._prediction(view)
        state.actual_pitch = target[2]
        state.round_id = (
            f"replay-{state.generation}-"
            f"{state.pitch_index + 1}-{view.context.state_key[:8]}"
        )

    def snapshot(self, game_pk: int, session_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._state(game_pk, session_id)
            self._prepare(state)
            response: dict[str, Any] = {
                "mode": "replay",
                "game": state.game,
                "pitchGroups": {
                    str(group): PITCH_GROUP_NAMES[group]
                    for group in PITCH_GROUPS
                },
                "pitchGroupFamilies": {
                    str(group): str(PITCH_GROUP_FAMILIES[group])
                    for group in PITCH_GROUPS
                },
                "serverTime": _utcnow().isoformat(),
                "lastResult": state.last_result,
            }
            if state.context is None:
                response["round"] = {"status": "final"}
                return response
            response["round"] = {
                "id": state.round_id,
                "mode": "replay",
                "status": "locked" if state.answer else "open",
                "openedAt": None,
                "deadline": None,
                "context": state.context,
                "prediction": state.prediction if state.answer else None,
                "actualPitch": None,
                "answer": state.answer,
                "pitchNumber": state.pitch_index + 1,
            }
            return response

    def answer(
        self,
        round_id: str,
        session_id: str,
        selected_pitch: str,
    ) -> dict[str, Any]:
        if selected_pitch not in {str(group) for group in PITCH_GROUPS}:
            raise ValueError("지원하지 않는 구종입니다.")
        with self._lock:
            state = next(
                (
                    state
                    for (stored_session_id, _), state in self._states.items()
                    if stored_session_id == session_id
                    and state.round_id == round_id
                ),
                None,
            )
            if state is None:
                raise LookupError("먼저 테스트 모드를 시작하세요.")
            self._prepare(state)
            if state.round_id != round_id:
                raise LookupError("현재 테스트 투구가 아닙니다.")
            if state.answer is not None:
                raise RuntimeError("이미 예측을 확정했습니다.")
            received_at = _utcnow().isoformat()
            state.answer = {
                "selectedPitch": selected_pitch,
                "receivedAt": received_at,
            }
            return {
                "roundId": round_id,
                "selectedPitch": selected_pitch,
                "receivedAt": received_at,
                "status": "locked",
            }

    def advance(self, game_pk: int, session_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._state(game_pk, session_id)
            self._prepare(state)
            if state.context is None:
                raise RuntimeError("이미 마지막 투구입니다.")
            if state.answer is None:
                raise RuntimeError("먼저 예측을 확정하세요.")
            state.last_result = {
                "roundId": state.round_id,
                "context": state.context,
                "prediction": state.prediction,
                "selectedPitch": state.answer["selectedPitch"],
                "actualPitch": state.actual_pitch,
            }
            state.pitch_index += 1
            state.answer = None
            state.context = None
            state.prediction = None
            state.actual_pitch = None
            state.round_id = None
            self._prepare(state)
        return self.snapshot(game_pk, session_id)


def _history_directories() -> tuple[Path, ...]:
    configured = os.environ.get(
        "OYARZABAL_HISTORY_DIRS",
        os.pathsep.join(
            ("data/raw/statcast-v8", "data/holdout/statcast-v8-2026")
        ),
    )
    return tuple(Path(value) for value in configured.split(os.pathsep) if value)


def create_app(
    *,
    database_path: str | None = None,
    history: HistoryStore | None = None,
    predictor: LivePredictor | None = None,
    fetcher: Callable[[int], dict[str, Any]] = fetch_gameday,
    window_seconds: int | None = None,
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    service = LiveService(
        database_path=database_path
        or os.environ.get(
            "OYARZABAL_LIVE_DB_PATH",
            "data/live/live.sqlite3",
        ),
        history=history or HistoryStore(_history_directories()),
        predictor=predictor
        or LivePredictor(
            Path(
                os.environ.get(
                    "OYARZABAL_LIVE_MODEL_DIR",
                    "models/v7.2",
                )
            ),
        ),
        fetcher=fetcher,
        window_seconds=window_seconds
        if window_seconds is not None
        else DEFAULT_WINDOW_SECONDS,
    )
    replay = ReplayService(service)

    @app.after_request
    def headers(response: Response) -> Response:
        origin = request.headers.get("Origin", "")
        if origin.startswith("chrome-extension://") or origin in {
            "http://127.0.0.1:8010",
            "http://localhost:8010",
        }:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.errorhandler(413)
    def payload_too_large(_: Exception) -> tuple[Response, int]:
        return jsonify({"error": "요청이 너무 큽니다."}), 413

    @app.get("/api/health")
    def health() -> Response:
        return jsonify({"ok": True})

    @app.post("/api/replay/sessions")
    def start_replay() -> tuple[Response, int] | Response:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "요청 형식이 올바르지 않습니다."}), 400
        session_id = str(payload.get("sessionId") or "")
        try:
            uuid.UUID(session_id)
            game_pk = int(payload.get("gamePk"))
        except (TypeError, ValueError):
            return jsonify({"error": "올바른 세션과 gamePk가 필요합니다."}), 400
        try:
            return jsonify(replay.start(game_pk, session_id))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            return jsonify({"error": f"리플레이 시작 실패: {error}"}), 502

    @app.get("/api/replay/games/<int:game_pk>")
    def replay_snapshot(game_pk: int) -> tuple[Response, int] | Response:
        session_id = request.args.get("sessionId", "")
        try:
            uuid.UUID(session_id)
            return jsonify(replay.snapshot(game_pk, session_id))
        except ValueError:
            return jsonify({"error": "올바른 세션 ID가 필요합니다."}), 400
        except LookupError as error:
            return jsonify({"error": str(error)}), 404
        except (OSError, KeyError, json.JSONDecodeError) as error:
            return jsonify({"error": f"리플레이 동기화 실패: {error}"}), 502

    @app.post("/api/replay/rounds/<round_id>/answers")
    def submit_replay_answer(round_id: str) -> tuple[Response, int]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "요청 형식이 올바르지 않습니다."}), 400
        session_id = str(payload.get("sessionId") or "")
        try:
            uuid.UUID(session_id)
            result = replay.answer(
                round_id,
                session_id,
                str(payload.get("selectedPitch")),
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except LookupError as error:
            return jsonify({"error": str(error)}), 404
        except RuntimeError as error:
            return jsonify({"error": str(error)}), 409
        return jsonify(result), 201

    @app.post("/api/replay/sessions/<session_id>/advance")
    def advance_replay(session_id: str) -> tuple[Response, int] | Response:
        payload = request.get_json(silent=True)
        try:
            uuid.UUID(session_id)
            game_pk = int((payload or {}).get("gamePk"))
            return jsonify(replay.advance(game_pk, session_id))
        except (TypeError, ValueError):
            return jsonify({"error": "올바른 세션과 gamePk가 필요합니다."}), 400
        except LookupError as error:
            return jsonify({"error": str(error)}), 404
        except RuntimeError as error:
            return jsonify({"error": str(error)}), 409
        except (OSError, KeyError, json.JSONDecodeError) as error:
            return jsonify({"error": f"다음 투구 준비 실패: {error}"}), 502

    @app.get("/api/live/games/<int:game_pk>")
    def game_snapshot(game_pk: int) -> tuple[Response, int] | Response:
        session_id = request.args.get("sessionId", "")
        try:
            uuid.UUID(session_id)
        except ValueError:
            return jsonify({"error": "올바른 세션 ID가 필요합니다."}), 400
        try:
            return jsonify(service.snapshot(game_pk, session_id))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            return jsonify({"error": f"경기 동기화 실패: {error}"}), 502

    @app.post("/api/live/rounds/<round_id>/answers")
    def submit_answer(round_id: str) -> tuple[Response, int]:
        if not request.is_json:
            return jsonify({"error": "JSON 요청만 허용됩니다."}), 415
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "요청 형식이 올바르지 않습니다."}), 400
        session_id = payload.get("sessionId")
        selected_pitch = payload.get("selectedPitch")
        try:
            uuid.UUID(str(session_id))
        except ValueError:
            return jsonify({"error": "올바른 세션 ID가 필요합니다."}), 400
        try:
            result = service.answer(
                round_id,
                str(session_id),
                str(selected_pitch),
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except LookupError as error:
            return jsonify({"error": str(error)}), 404
        except RuntimeError as error:
            return jsonify({"error": str(error)}), 409
        except (OSError, KeyError, json.JSONDecodeError) as error:
            return jsonify({"error": f"경기 동기화 실패: {error}"}), 502
        return jsonify(result), 200

    return app


def main() -> None:
    app = create_app()
    app.run(
        host=os.environ.get("OYARZABAL_LIVE_HOST", "127.0.0.1"),
        port=int(os.environ.get("OYARZABAL_LIVE_PORT", str(DEFAULT_API_PORT))),
        debug=False,
    )


if __name__ == "__main__":
    main()
