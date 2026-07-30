from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from flask import Flask, Response, jsonify, request

from oyarzabal.daily_challenge import GROUP_FAMILY, fetch_daily_scenarios

SEOUL = ZoneInfo("Asia/Seoul")
MAX_RESPONSE_MS = 10 * 60 * 1000


class DailyValidationError(ValueError):
    pass


def seoul_today() -> date:
    return datetime.now(SEOUL).date()


def init_daily_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS daily_challenges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            public_id TEXT NOT NULL UNIQUE,
            challenge_date TEXT NOT NULL UNIQUE,
            source_date TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('published')),
            created_at TEXT NOT NULL,
            published_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS daily_scenarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            challenge_id INTEGER NOT NULL REFERENCES daily_challenges(id)
                ON DELETE CASCADE,
            scenario_key TEXT NOT NULL,
            position INTEGER NOT NULL CHECK (position BETWEEN 1 AND 3),
            pitcher TEXT NOT NULL,
            batter TEXT NOT NULL,
            game_date TEXT NOT NULL,
            inning TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_score INTEGER NOT NULL,
            home_score INTEGER NOT NULL,
            outs INTEGER NOT NULL CHECK (outs BETWEEN 0 AND 2),
            bases_json TEXT NOT NULL,
            outcome TEXT NOT NULL,
            game_url TEXT NOT NULL,
            repertoire_json TEXT NOT NULL,
            available_groups_json TEXT NOT NULL,
            UNIQUE (challenge_id, position),
            UNIQUE (challenge_id, scenario_key)
        );
        CREATE TABLE IF NOT EXISTS daily_pitches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario_id INTEGER NOT NULL REFERENCES daily_scenarios(id)
                ON DELETE CASCADE,
            pitch_number INTEGER NOT NULL,
            count TEXT NOT NULL,
            actual_group TEXT NOT NULL,
            raw_label TEXT NOT NULL,
            result TEXT NOT NULL,
            play_id TEXT NOT NULL,
            model_group TEXT,
            model_confidence REAL,
            model_probabilities_json TEXT,
            model_version TEXT,
            UNIQUE (scenario_id, pitch_number)
        );
        CREATE TABLE IF NOT EXISTS daily_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            public_id TEXT NOT NULL UNIQUE,
            challenge_id INTEGER NOT NULL REFERENCES daily_challenges(id),
            player_hash TEXT NOT NULL,
            nickname TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            total_points INTEGER NOT NULL DEFAULT 0,
            detail_hits INTEGER NOT NULL DEFAULT 0,
            family_hits INTEGER NOT NULL DEFAULT 0,
            UNIQUE (challenge_id, player_hash)
        );
        CREATE TABLE IF NOT EXISTS daily_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_id INTEGER NOT NULL REFERENCES daily_attempts(id)
                ON DELETE CASCADE,
            scenario_id INTEGER NOT NULL REFERENCES daily_scenarios(id),
            pitch_id INTEGER NOT NULL REFERENCES daily_pitches(id),
            selected_family TEXT NOT NULL,
            selected_group TEXT NOT NULL,
            actual_family TEXT NOT NULL,
            actual_group TEXT NOT NULL,
            family_hit INTEGER NOT NULL CHECK (family_hit IN (0, 1)),
            detail_hit INTEGER NOT NULL CHECK (detail_hit IN (0, 1)),
            points INTEGER NOT NULL CHECK (points IN (0, 1, 3)),
            response_ms INTEGER NOT NULL CHECK (response_ms >= 0),
            answered_at TEXT NOT NULL,
            UNIQUE (attempt_id, pitch_id)
        );
        CREATE INDEX IF NOT EXISTS idx_daily_attempts_challenge_score
            ON daily_attempts (
                challenge_id, total_points DESC, detail_hits DESC
            );
        CREATE INDEX IF NOT EXISTS idx_daily_answers_attempt
            ON daily_answers (attempt_id, id);
        """
    )
    scenario_table_sql = connection.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE type = 'table' AND name = 'daily_scenarios'"
    ).fetchone()["sql"]
    if "position IN (1, 2)" in scenario_table_sql:
        connection.execute("PRAGMA foreign_keys = OFF")
        try:
            connection.executescript(
                """
                BEGIN;
                CREATE TABLE daily_scenarios_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    challenge_id INTEGER NOT NULL REFERENCES daily_challenges(id)
                        ON DELETE CASCADE,
                    scenario_key TEXT NOT NULL,
                    position INTEGER NOT NULL CHECK (position BETWEEN 1 AND 3),
                    pitcher TEXT NOT NULL,
                    batter TEXT NOT NULL,
                    game_date TEXT NOT NULL,
                    inning TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_score INTEGER NOT NULL,
                    home_score INTEGER NOT NULL,
                    outs INTEGER NOT NULL CHECK (outs BETWEEN 0 AND 2),
                    bases_json TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    game_url TEXT NOT NULL,
                    repertoire_json TEXT NOT NULL,
                    available_groups_json TEXT NOT NULL,
                    UNIQUE (challenge_id, position),
                    UNIQUE (challenge_id, scenario_key)
                );
                INSERT INTO daily_scenarios_new SELECT * FROM daily_scenarios;
                DROP TABLE daily_scenarios;
                ALTER TABLE daily_scenarios_new RENAME TO daily_scenarios;
                COMMIT;
                """
            )
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.execute("PRAGMA foreign_keys = ON")
    pitch_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(daily_pitches)")
    }
    for name, definition in (
        ("model_group", "TEXT"),
        ("model_confidence", "REAL"),
        ("model_probabilities_json", "TEXT"),
        ("model_version", "TEXT"),
    ):
        if name not in pitch_columns:
            connection.execute(
                f"ALTER TABLE daily_pitches ADD COLUMN {name} {definition}"
            )


def _validate_iso_date(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise DailyValidationError(f"{label}가 올바르지 않습니다.")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise DailyValidationError(f"{label}가 올바르지 않습니다.") from error


def _validate_participant_id(value: Any) -> str:
    if not isinstance(value, str):
        raise DailyValidationError("브라우저 식별자가 올바르지 않습니다.")
    try:
        parsed = uuid.UUID(value)
    except ValueError as error:
        raise DailyValidationError("브라우저 식별자가 올바르지 않습니다.") from error
    if str(parsed) != value.lower():
        raise DailyValidationError("브라우저 식별자가 올바르지 않습니다.")
    return str(parsed)


def _validate_generated_scenarios(scenarios: Any) -> list[dict[str, Any]]:
    if not isinstance(scenarios, list) or len(scenarios) != 3:
        raise DailyValidationError("Daily 문제에는 정확히 세 타석이 필요합니다.")
    scenario_ids: set[str] = set()
    pitcher_names: set[str] = set()
    game_dates: set[str] = set()
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise DailyValidationError("Daily 타석 형식이 올바르지 않습니다.")
        scenario_id = scenario.get("id")
        pitcher = scenario.get("pitcher")
        if (
            not isinstance(scenario_id, str)
            or not scenario_id
            or scenario_id in scenario_ids
            or not isinstance(pitcher, str)
            or not pitcher
            or pitcher in pitcher_names
        ):
            raise DailyValidationError("서로 다른 투수의 고유한 세 타석이 필요합니다.")
        try:
            game_dates.add(date.fromisoformat(scenario.get("date", "")).isoformat())
        except (TypeError, ValueError) as error:
            raise DailyValidationError(
                "Daily 경기 날짜가 올바르지 않습니다."
            ) from error
        pitches = scenario.get("pitches")
        if not isinstance(pitches, list) or not 3 <= len(pitches) <= 6:
            raise DailyValidationError("각 Daily 타석은 3~6구여야 합니다.")
        available_groups = scenario.get("availableGroups")
        if (
            not isinstance(available_groups, list)
            or not available_groups
            or any(group not in GROUP_FAMILY for group in available_groups)
        ):
            raise DailyValidationError("투수의 구종 목록이 올바르지 않습니다.")
        for pitch_number, pitch in enumerate(pitches, start=1):
            model_prediction = pitch.get("modelPrediction")
            probabilities = (
                model_prediction.get("probabilities")
                if isinstance(model_prediction, dict)
                else None
            )
            if (
                not isinstance(pitch, dict)
                or pitch.get("number") != pitch_number
                or pitch.get("actual") not in available_groups
                or not isinstance(pitch.get("playId"), str)
                or not pitch["playId"]
                or not isinstance(model_prediction, dict)
                or not isinstance(probabilities, dict)
                or model_prediction.get("selected") not in probabilities
                or model_prediction.get("selected") not in available_groups
                or not isinstance(model_prediction.get("modelVersion"), str)
                or not probabilities
                or set(probabilities) - set(available_groups)
                or any(
                    type(value) not in {int, float}
                    or not math.isfinite(value)
                    or value < 0
                    for value in probabilities.values()
                )
                or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-4)
                or type(model_prediction.get("confidence")) not in {int, float}
                or not math.isclose(
                    model_prediction["confidence"],
                    probabilities[model_prediction["selected"]],
                    abs_tol=1e-4,
                )
            ):
                raise DailyValidationError("Daily 투구 기록이 올바르지 않습니다.")
        scenario_ids.add(scenario_id)
        pitcher_names.add(pitcher)
    if len(game_dates) != 1:
        raise DailyValidationError("세 Daily 타석의 경기 날짜가 같아야 합니다.")
    return scenarios


def _update_daily_model_predictions(
    connection: sqlite3.Connection,
    challenge_id: int,
    scenarios: list[dict[str, Any]],
) -> None:
    for scenario in scenarios:
        for pitch in scenario["pitches"]:
            prediction = pitch["modelPrediction"]
            connection.execute(
                """
                UPDATE daily_pitches
                SET
                    model_group = ?,
                    model_confidence = ?,
                    model_probabilities_json = ?,
                    model_version = ?
                WHERE scenario_id = (
                    SELECT id
                    FROM daily_scenarios
                    WHERE challenge_id = ? AND scenario_key = ?
                ) AND pitch_number = ?
                """,
                (
                    prediction["selected"],
                    prediction["confidence"],
                    json.dumps(
                        prediction["probabilities"],
                        ensure_ascii=False,
                    ),
                    prediction["modelVersion"],
                    challenge_id,
                    scenario["id"],
                    pitch["number"],
                ),
            )


def publish_daily_challenge(
    connection: sqlite3.Connection,
    *,
    challenge_date: date,
    source_date: date,
    scenarios: list[dict[str, Any]],
) -> sqlite3.Row:
    scenarios = _validate_generated_scenarios(scenarios)
    existing = connection.execute(
        "SELECT * FROM daily_challenges WHERE challenge_date = ?",
        (challenge_date.isoformat(),),
    ).fetchone()
    if existing:
        existing_scenarios = connection.execute(
            """
            SELECT scenario_key
            FROM daily_scenarios
            WHERE challenge_id = ?
            ORDER BY position
            """,
            (existing["id"],),
        ).fetchall()
        existing_keys = [row["scenario_key"] for row in existing_scenarios]
        generated_keys = [scenario["id"] for scenario in scenarios]
        if existing_keys == generated_keys:
            _update_daily_model_predictions(
                connection,
                existing["id"],
                scenarios,
            )
            return existing
        if existing_keys != generated_keys[:2]:
            raise DailyValidationError(
                "기존 Daily 문제와 새 세 타석의 구성이 일치하지 않습니다."
            )
        scenarios_to_insert = scenarios[2:]
        challenge_id = existing["id"]
        start_position = 3
    else:
        now = datetime.now(UTC).isoformat()
        cursor = connection.execute(
            """
            INSERT INTO daily_challenges (
                public_id, challenge_date, source_date, status, created_at, published_at
            ) VALUES (?, ?, ?, 'published', ?, ?)
            """,
            (
                str(uuid.uuid4()),
                challenge_date.isoformat(),
                source_date.isoformat(),
                now,
                now,
            ),
        )
        challenge_id = cursor.lastrowid
        scenarios_to_insert = scenarios
        start_position = 1

    for position, scenario in enumerate(scenarios_to_insert, start=start_position):
        scenario_cursor = connection.execute(
            """
            INSERT INTO daily_scenarios (
                challenge_id, scenario_key, position, pitcher, batter, game_date,
                inning, away_team, home_team, away_score, home_score, outs,
                bases_json, outcome, game_url, repertoire_json,
                available_groups_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                challenge_id,
                scenario["id"],
                position,
                scenario["pitcher"],
                scenario["batter"],
                scenario["date"],
                scenario["inning"],
                scenario["awayTeam"],
                scenario["homeTeam"],
                scenario["awayScore"],
                scenario["homeScore"],
                scenario["outs"],
                json.dumps(scenario["bases"], ensure_ascii=False),
                scenario["outcome"],
                scenario["gameUrl"],
                json.dumps(scenario["repertoire"], ensure_ascii=False),
                json.dumps(scenario["availableGroups"], ensure_ascii=False),
            ),
        )
        scenario_id = scenario_cursor.lastrowid
        connection.executemany(
            """
            INSERT INTO daily_pitches (
                scenario_id, pitch_number, count, actual_group,
                raw_label, result, play_id, model_group, model_confidence,
                model_probabilities_json, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    scenario_id,
                    pitch["number"],
                    pitch["count"],
                    pitch["actual"],
                    pitch["rawLabel"],
                    pitch["result"],
                    pitch["playId"],
                    pitch["modelPrediction"]["selected"],
                    pitch["modelPrediction"]["confidence"],
                    json.dumps(
                        pitch["modelPrediction"]["probabilities"],
                        ensure_ascii=False,
                    ),
                    pitch["modelPrediction"]["modelVersion"],
                )
                for pitch in scenario["pitches"]
            ],
        )
    if existing:
        connection.execute(
            """
            UPDATE daily_attempts
            SET completed_at = NULL
            WHERE challenge_id = ?
            """,
            (challenge_id,),
        )
        _update_daily_model_predictions(connection, challenge_id, scenarios)
    return connection.execute(
        "SELECT * FROM daily_challenges WHERE id = ?",
        (challenge_id,),
    ).fetchone()


def _challenge_for_date(
    connection: sqlite3.Connection,
    challenge_date: date,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT *
        FROM daily_challenges
        WHERE challenge_date = ? AND status = 'published'
        """,
        (challenge_date.isoformat(),),
    ).fetchone()


def _challenge_summary(
    connection: sqlite3.Connection,
    challenge: sqlite3.Row,
) -> dict[str, Any]:
    participants = connection.execute(
        """
        SELECT COUNT(*) AS participants
        FROM daily_attempts
        WHERE challenge_id = ? AND completed_at IS NOT NULL
        """,
        (challenge["id"],),
    ).fetchone()["participants"]
    scenario_count = connection.execute(
        "SELECT COUNT(*) AS count FROM daily_scenarios WHERE challenge_id = ?",
        (challenge["id"],),
    ).fetchone()["count"]
    return {
        "id": challenge["public_id"],
        "number": challenge["id"],
        "date": challenge["challenge_date"],
        "sourceDate": challenge["source_date"],
        "scenarioCount": scenario_count,
        "participantCount": participants,
    }


def _scenario_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["scenario_key"],
        "pitcher": row["pitcher"],
        "batter": row["batter"],
        "date": row["game_date"],
        "inning": row["inning"],
        "awayTeam": row["away_team"],
        "homeTeam": row["home_team"],
        "awayScore": row["away_score"],
        "homeScore": row["home_score"],
        "outs": row["outs"],
        "bases": json.loads(row["bases_json"]),
        "outcome": row["outcome"],
        "gameUrl": row["game_url"],
        "repertoire": json.loads(row["repertoire_json"]),
        "availableGroups": json.loads(row["available_groups_json"]),
    }


def _ordered_pitches(
    connection: sqlite3.Connection,
    challenge_id: int,
) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT
            p.*,
            s.position AS scenario_position,
            s.scenario_key,
            s.challenge_id
        FROM daily_pitches p
        JOIN daily_scenarios s ON s.id = p.scenario_id
        WHERE s.challenge_id = ?
        ORDER BY s.position, p.pitch_number
        """,
        (challenge_id,),
    ).fetchall()


def _history_payload(
    connection: sqlite3.Connection,
    attempt_id: int,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            s.scenario_key,
            s.position AS scenario_position,
            p.pitch_number,
            p.count,
            p.raw_label,
            p.result,
            p.play_id,
            p.model_group,
            p.model_confidence,
            p.model_probabilities_json,
            p.model_version,
            a.selected_family,
            a.selected_group,
            a.actual_family,
            a.actual_group,
            a.family_hit,
            a.detail_hit,
            a.points,
            a.response_ms
        FROM daily_answers a
        JOIN daily_pitches p ON p.id = a.pitch_id
        JOIN daily_scenarios s ON s.id = a.scenario_id
        WHERE a.attempt_id = ?
        ORDER BY s.position, p.pitch_number
        """,
        (attempt_id,),
    ).fetchall()
    return [
        {
            "scenarioId": row["scenario_key"],
            "scenarioNumber": row["scenario_position"],
            "pitchNumber": row["pitch_number"],
            "count": row["count"],
            "rawLabel": row["raw_label"],
            "result": row["result"],
            "playId": row["play_id"],
            "modelPrediction": {
                "selected": row["model_group"],
                "confidence": row["model_confidence"],
                "probabilities": json.loads(row["model_probabilities_json"]),
                "modelVersion": row["model_version"],
            }
            if row["model_group"]
            else None,
            "selectedFamily": row["selected_family"],
            "selected": row["selected_group"],
            "actualFamily": row["actual_family"],
            "actual": row["actual_group"],
            "familyHit": bool(row["family_hit"]),
            "detailHit": bool(row["detail_hit"]),
            "points": row["points"],
            "responseMs": row["response_ms"],
        }
        for row in rows
    ]


def _streak(
    connection: sqlite3.Connection,
    player_hash: str,
    through_date: date,
) -> int:
    rows = connection.execute(
        """
        SELECT c.challenge_date
        FROM daily_attempts a
        JOIN daily_challenges c ON c.id = a.challenge_id
        WHERE a.player_hash = ? AND a.completed_at IS NOT NULL
          AND c.challenge_date <= ?
        GROUP BY c.challenge_date
        ORDER BY c.challenge_date DESC
        """,
        (player_hash, through_date.isoformat()),
    ).fetchall()
    streak = 0
    expected = through_date
    for row in rows:
        completed_date = date.fromisoformat(row["challenge_date"])
        if completed_date != expected:
            break
        streak += 1
        expected -= timedelta(days=1)
    return streak


def _leaderboard(
    connection: sqlite3.Connection,
    challenge_id: int,
    *,
    current_attempt_id: int,
) -> dict[str, Any]:
    rows = connection.execute(
        """
        SELECT
            id,
            nickname,
            total_points,
            detail_hits,
            family_hits,
            completed_at
        FROM daily_attempts
        WHERE challenge_id = ? AND completed_at IS NOT NULL
        """,
        (challenge_id,),
    ).fetchall()
    payload_rows = [
        {
            "rank": 0,
            "nickname": row["nickname"],
            "points": row["total_points"],
            "detailHits": row["detail_hits"],
            "familyHits": row["family_hits"],
            "isCurrent": row["id"] == current_attempt_id,
            "isModel": False,
            "_completedAt": row["completed_at"],
            "_id": row["id"],
        }
        for row in rows
    ]
    model_pitches = connection.execute(
        """
        SELECT p.actual_group, p.model_group, p.model_version
        FROM daily_pitches p
        JOIN daily_scenarios s ON s.id = p.scenario_id
        WHERE s.challenge_id = ? AND p.model_group IS NOT NULL
        ORDER BY s.position, p.pitch_number
        """,
        (challenge_id,),
    ).fetchall()
    if model_pitches:
        detail_hits = sum(
            row["actual_group"] == row["model_group"] for row in model_pitches
        )
        family_hits = sum(
            GROUP_FAMILY[row["actual_group"]]
            == GROUP_FAMILY[row["model_group"]]
            for row in model_pitches
        )
        model_version = model_pitches[0]["model_version"] or "Model"
        payload_rows.append(
            {
                "rank": 0,
                "nickname": f"OyarZabal {model_version}",
                "points": detail_hits * 3 + (family_hits - detail_hits),
                "detailHits": detail_hits,
                "familyHits": family_hits,
                "isCurrent": False,
                "isModel": True,
                "_completedAt": "",
                "_id": 0,
            }
        )
    payload_rows.sort(
        key=lambda row: (
            -row["points"],
            -row["detailHits"],
            row["isModel"],
            row["_completedAt"],
            row["_id"],
        )
    )
    previous_score: tuple[int, int] | None = None
    rank = 0
    for row in payload_rows:
        score = (row["points"], row["detailHits"])
        if score != previous_score:
            rank += 1
            previous_score = score
        row["rank"] = rank
    current = next(
        (row for row in payload_rows if row["isCurrent"]),
        None,
    )
    model = next((row for row in payload_rows if row["isModel"]), None)
    visible = payload_rows[:20]
    if current and current not in visible:
        visible.append(current)
    if model and model not in visible:
        visible.append(model)
    for row in visible:
        row.pop("_completedAt", None)
        row.pop("_id", None)
    return {
        "participantCount": len(rows),
        "currentRank": current["rank"] if current else None,
        "rows": visible,
    }


def _attempt_result(
    connection: sqlite3.Connection,
    attempt: sqlite3.Row,
) -> dict[str, Any]:
    challenge = connection.execute(
        "SELECT * FROM daily_challenges WHERE id = ?",
        (attempt["challenge_id"],),
    ).fetchone()
    pitch_count = connection.execute(
        """
        SELECT COUNT(*) AS pitch_count
        FROM daily_pitches p
        JOIN daily_scenarios s ON s.id = p.scenario_id
        WHERE s.challenge_id = ?
        """,
        (attempt["challenge_id"],),
    ).fetchone()["pitch_count"]
    leaderboard = _leaderboard(
        connection,
        attempt["challenge_id"],
        current_attempt_id=attempt["id"],
    )
    return {
        "points": attempt["total_points"],
        "maxPoints": pitch_count * 3,
        "pitchCount": pitch_count,
        "detailHits": attempt["detail_hits"],
        "familyHits": attempt["family_hits"],
        "rank": leaderboard["currentRank"],
        "streak": _streak(
            connection,
            attempt["player_hash"],
            date.fromisoformat(challenge["challenge_date"]),
        ),
        "leaderboard": leaderboard,
    }


def _attempt_state(
    connection: sqlite3.Connection,
    attempt: sqlite3.Row,
) -> dict[str, Any]:
    challenge = connection.execute(
        "SELECT * FROM daily_challenges WHERE id = ?",
        (attempt["challenge_id"],),
    ).fetchone()
    history = _history_payload(connection, attempt["id"])
    payload: dict[str, Any] = {
        "attemptId": attempt["public_id"],
        "nickname": attempt["nickname"],
        "challenge": _challenge_summary(connection, challenge),
        "history": history,
        "status": "completed" if attempt["completed_at"] else "playing",
    }
    if attempt["completed_at"]:
        payload["result"] = _attempt_result(connection, attempt)
        return payload

    pitches = _ordered_pitches(connection, attempt["challenge_id"])
    if len(history) >= len(pitches):
        raise RuntimeError("완료되지 않은 Daily 기록에 다음 투구가 없습니다.")
    current_pitch = pitches[len(history)]
    scenario = connection.execute(
        "SELECT * FROM daily_scenarios WHERE id = ?",
        (current_pitch["scenario_id"],),
    ).fetchone()
    payload["current"] = {
        "questionNumber": len(history) + 1,
        "scenarioNumber": current_pitch["scenario_position"],
        "scenarioCount": payload["challenge"]["scenarioCount"],
        "scenario": _scenario_payload(scenario),
        "pitch": {
            "number": current_pitch["pitch_number"],
            "count": current_pitch["count"],
        },
    }
    return payload


def daily_admin_summary(connection: sqlite3.Connection) -> dict[str, Any]:
    challenge_rows = connection.execute(
        """
        SELECT
            c.id,
            c.challenge_date,
            c.source_date,
            COUNT(DISTINCT CASE
                WHEN a.completed_at IS NOT NULL THEN a.id
            END) AS participants,
            COUNT(DISTINCT p.id) AS pitch_count
        FROM daily_challenges c
        LEFT JOIN daily_scenarios s ON s.challenge_id = c.id
        LEFT JOIN daily_pitches p ON p.scenario_id = s.id
        LEFT JOIN daily_attempts a ON a.challenge_id = c.id
        GROUP BY c.id
        ORDER BY c.challenge_date DESC
        LIMIT 30
        """
    ).fetchall()
    answer_rows = connection.execute(
        """
        SELECT
            s.challenge_id,
            a.family_hit,
            a.detail_hit,
            a.points
        FROM daily_answers a
        JOIN daily_scenarios s ON s.id = a.scenario_id
        """
    ).fetchall()
    model_rows = connection.execute(
        """
        SELECT
            s.challenge_id,
            p.actual_group,
            p.model_group
        FROM daily_pitches p
        JOIN daily_scenarios s ON s.id = p.scenario_id
        WHERE p.model_group IS NOT NULL
        """
    ).fetchall()

    def score_model(actual: str, predicted: str) -> tuple[int, int, int]:
        detail = int(actual == predicted)
        family = int(GROUP_FAMILY[actual] == GROUP_FAMILY[predicted])
        return detail, family, 3 if detail else 1 if family else 0

    answers_by_challenge: dict[int, list[sqlite3.Row]] = {}
    for row in answer_rows:
        answers_by_challenge.setdefault(row["challenge_id"], []).append(row)
    models_by_challenge: dict[int, list[tuple[int, int, int]]] = {}
    for row in model_rows:
        models_by_challenge.setdefault(row["challenge_id"], []).append(
            score_model(row["actual_group"], row["model_group"])
        )

    def rate(hits: int, count: int) -> float:
        return round(hits / count, 4) if count else 0

    challenges = []
    for row in challenge_rows:
        answers = answers_by_challenge.get(row["id"], [])
        models = models_by_challenge.get(row["id"], [])
        challenges.append(
            {
                "date": row["challenge_date"],
                "sourceDate": row["source_date"],
                "participants": row["participants"],
                "pitchCount": row["pitch_count"],
                "humanDetailRate": rate(
                    sum(answer["detail_hit"] for answer in answers),
                    len(answers),
                ),
                "humanFamilyRate": rate(
                    sum(answer["family_hit"] for answer in answers),
                    len(answers),
                ),
                "modelDetailRate": rate(
                    sum(detail for detail, _, _ in models),
                    len(models),
                ),
                "modelFamilyRate": rate(
                    sum(family for _, family, _ in models),
                    len(models),
                ),
                "modelPoints": sum(points for _, _, points in models),
                "modelMaxPoints": len(models) * 3,
            }
        )

    attempt_rows = connection.execute(
        """
        SELECT
            a.public_id,
            a.nickname,
            a.started_at,
            a.completed_at,
            a.total_points,
            a.detail_hits,
            a.family_hits,
            c.challenge_date,
            COUNT(answers.id) AS answered_pitches,
            (
                SELECT COUNT(*)
                FROM daily_pitches p
                JOIN daily_scenarios s ON s.id = p.scenario_id
                WHERE s.challenge_id = c.id
            ) AS pitch_count
        FROM daily_attempts a
        JOIN daily_challenges c ON c.id = a.challenge_id
        LEFT JOIN daily_answers answers ON answers.attempt_id = a.id
        GROUP BY a.id
        ORDER BY a.id DESC
        LIMIT 500
        """
    ).fetchall()
    all_models = [score for values in models_by_challenge.values() for score in values]
    return {
        "totals": {
            "challenges": len(challenge_rows),
            "completedAttempts": sum(row["participants"] for row in challenge_rows),
            "answers": len(answer_rows),
            "humanDetailRate": rate(
                sum(row["detail_hit"] for row in answer_rows),
                len(answer_rows),
            ),
            "humanFamilyRate": rate(
                sum(row["family_hit"] for row in answer_rows),
                len(answer_rows),
            ),
            "modelDetailRate": rate(
                sum(detail for detail, _, _ in all_models),
                len(all_models),
            ),
            "modelFamilyRate": rate(
                sum(family for _, family, _ in all_models),
                len(all_models),
            ),
        },
        "challenges": challenges,
        "attempts": [
            {
                "id": row["public_id"],
                "nickname": row["nickname"],
                "date": row["challenge_date"],
                "status": "completed" if row["completed_at"] else "playing",
                "points": row["total_points"],
                "maxPoints": row["pitch_count"] * 3,
                "detailHits": row["detail_hits"],
                "familyHits": row["family_hits"],
                "answeredPitches": row["answered_pitches"],
                "pitchCount": row["pitch_count"],
                "startedAt": row["started_at"],
                "completedAt": row["completed_at"],
            }
            for row in attempt_rows
        ],
    }


def register_daily_routes(
    app: Flask,
    *,
    database: Callable[[], sqlite3.Connection],
    require_admin: Callable[[], Response | None],
    validate_nickname: Callable[[Any], str],
    limiter_allow: Callable[[str], bool],
    client_ip: Callable[[], str],
    today_provider: Callable[[], date] = seoul_today,
    generator: Callable[
        [date, date],
        list[dict[str, Any]],
    ] = fetch_daily_scenarios,
) -> None:
    def reject_cross_origin() -> Response | None:
        allowed_origin = app.config["ALLOWED_ORIGIN"]
        origin = request.headers.get("Origin")
        if allowed_origin and origin and origin != allowed_origin:
            return jsonify({"error": "허용되지 않은 출처입니다."}), 403
        return None

    @app.get("/api/daily/challenge")
    def daily_challenge() -> tuple[Response, int] | Response:
        challenge = _challenge_for_date(database(), today_provider())
        if not challenge:
            return (
                jsonify({"error": "오늘의 Daily 문제가 아직 준비되지 않았습니다."}),
                404,
            )
        return jsonify(_challenge_summary(database(), challenge))

    @app.post("/api/admin/daily/generate")
    def generate_daily_challenge() -> tuple[Response, int] | Response:
        unauthorized = require_admin()
        if unauthorized is not None:
            return unauthorized
        payload = request.get_json(silent=True) if request.is_json else {}
        payload = payload if isinstance(payload, dict) else {}
        try:
            challenge_date = (
                _validate_iso_date(payload["challengeDate"], "문제 날짜")
                if "challengeDate" in payload
                else today_provider()
            )
            source_date = (
                _validate_iso_date(payload["sourceDate"], "경기 날짜")
                if "sourceDate" in payload
                else challenge_date - timedelta(days=1)
            )
            scenarios = generator(challenge_date, source_date)
            published_source_date = date.fromisoformat(scenarios[0]["date"])
            connection = database()
            with connection:
                challenge = publish_daily_challenge(
                    connection,
                    challenge_date=challenge_date,
                    source_date=published_source_date,
                    scenarios=scenarios,
                )
        except DailyValidationError as error:
            return jsonify({"error": str(error)}), 400
        except Exception:
            app.logger.exception("Daily challenge generation failed")
            return jsonify({"error": "Daily 문제 생성에 실패했습니다."}), 502
        return (
            jsonify(
                {
                    **_challenge_summary(database(), challenge),
                    "scenarios": [
                        {
                            "pitcher": scenario["pitcher"],
                            "batter": scenario["batter"],
                            "pitchCount": len(scenario["pitches"]),
                        }
                        for scenario in scenarios
                    ],
                }
            ),
            201,
        )

    @app.post("/api/daily/attempts")
    def start_daily_attempt() -> tuple[Response, int] | Response:
        rejected = reject_cross_origin()
        if rejected is not None:
            return rejected
        if not limiter_allow(client_ip()):
            return (
                jsonify({"error": "요청이 너무 많습니다. 잠시 후 다시 시도해 주세요."}),
                429,
            )
        if not request.is_json:
            return jsonify({"error": "JSON 요청만 허용됩니다."}), 415
        payload = request.get_json(silent=True)
        try:
            if not isinstance(payload, dict):
                raise DailyValidationError("요청 형식이 올바르지 않습니다.")
            nickname = validate_nickname(payload.get("nickname"))
            participant_id = _validate_participant_id(payload.get("participantId"))
        except (DailyValidationError, ValueError) as error:
            return jsonify({"error": str(error)}), 400

        connection = database()
        challenge = _challenge_for_date(connection, today_provider())
        if not challenge:
            return (
                jsonify({"error": "오늘의 Daily 문제가 아직 준비되지 않았습니다."}),
                404,
            )
        player_hash = hashlib.sha256(participant_id.encode()).hexdigest()
        attempt = connection.execute(
            """
            SELECT *
            FROM daily_attempts
            WHERE challenge_id = ? AND player_hash = ?
            """,
            (challenge["id"], player_hash),
        ).fetchone()
        created = attempt is None
        if created:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO daily_attempts (
                        public_id, challenge_id, player_hash, nickname, started_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        challenge["id"],
                        player_hash,
                        nickname,
                        datetime.now(UTC).isoformat(),
                    ),
                )
            attempt = connection.execute(
                "SELECT * FROM daily_attempts WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        return jsonify(_attempt_state(connection, attempt)), 201 if created else 200

    @app.post("/api/daily/attempts/<attempt_id>/answers")
    def answer_daily_pitch(attempt_id: str) -> tuple[Response, int] | Response:
        rejected = reject_cross_origin()
        if rejected is not None:
            return rejected
        if not limiter_allow(client_ip()):
            return (
                jsonify({"error": "요청이 너무 많습니다. 잠시 후 다시 시도해 주세요."}),
                429,
            )
        if not request.is_json:
            return jsonify({"error": "JSON 요청만 허용됩니다."}), 415
        try:
            uuid.UUID(attempt_id)
        except ValueError:
            return jsonify({"error": "Daily 기록을 찾을 수 없습니다."}), 404
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "요청 형식이 올바르지 않습니다."}), 400
        selected_family = payload.get("selectedFamily")
        selected_group = payload.get("selected")
        response_ms = payload.get("responseMs")
        if (
            selected_group not in GROUP_FAMILY
            or selected_family != GROUP_FAMILY.get(selected_group)
            or type(response_ms) is not int
            or not 0 <= response_ms <= MAX_RESPONSE_MS
        ):
            return jsonify({"error": "구종 답안이 올바르지 않습니다."}), 400

        connection = database()
        attempt = connection.execute(
            "SELECT * FROM daily_attempts WHERE public_id = ?",
            (attempt_id,),
        ).fetchone()
        if not attempt:
            return jsonify({"error": "Daily 기록을 찾을 수 없습니다."}), 404
        if attempt["completed_at"]:
            return jsonify({"error": "이미 완료한 Daily 기록입니다."}), 409
        history = _history_payload(connection, attempt["id"])
        pitches = _ordered_pitches(connection, attempt["challenge_id"])
        if len(history) >= len(pitches):
            return jsonify({"error": "이미 모든 투구에 답했습니다."}), 409
        pitch = pitches[len(history)]
        scenario = connection.execute(
            "SELECT * FROM daily_scenarios WHERE id = ?",
            (pitch["scenario_id"],),
        ).fetchone()
        available_groups = json.loads(scenario["available_groups_json"])
        if selected_group not in available_groups:
            return jsonify({"error": "해당 투수가 던지지 않는 구종입니다."}), 400

        actual_group = pitch["actual_group"]
        actual_family = GROUP_FAMILY[actual_group]
        detail_hit = selected_group == actual_group
        family_hit = selected_family == actual_family
        points = 3 if detail_hit else 1 if family_hit else 0
        now = datetime.now(UTC).isoformat()
        completed = len(history) + 1 == len(pitches)
        try:
            with connection:
                connection.execute(
                    """
                    INSERT INTO daily_answers (
                        attempt_id, scenario_id, pitch_id,
                        selected_family, selected_group,
                        actual_family, actual_group,
                        family_hit, detail_hit, points, response_ms, answered_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        attempt["id"],
                        pitch["scenario_id"],
                        pitch["id"],
                        selected_family,
                        selected_group,
                        actual_family,
                        actual_group,
                        int(family_hit),
                        int(detail_hit),
                        points,
                        response_ms,
                        now,
                    ),
                )
                connection.execute(
                    """
                    UPDATE daily_attempts
                    SET
                        total_points = total_points + ?,
                        detail_hits = detail_hits + ?,
                        family_hits = family_hits + ?,
                        completed_at = CASE WHEN ? THEN ? ELSE completed_at END
                    WHERE id = ?
                    """,
                    (
                        points,
                        int(detail_hit),
                        int(family_hit),
                        int(completed),
                        now,
                        attempt["id"],
                    ),
                )
        except sqlite3.IntegrityError:
            return jsonify({"error": "이미 처리된 투구입니다."}), 409

        refreshed = connection.execute(
            "SELECT * FROM daily_attempts WHERE id = ?",
            (attempt["id"],),
        ).fetchone()
        state = _attempt_state(connection, refreshed)
        state["reveal"] = {
            "scenarioId": scenario["scenario_key"],
            "scenarioNumber": scenario["position"],
            "pitchNumber": pitch["pitch_number"],
            "count": pitch["count"],
            "rawLabel": pitch["raw_label"],
            "result": pitch["result"],
            "playId": pitch["play_id"],
            "modelPrediction": {
                "selected": pitch["model_group"],
                "confidence": pitch["model_confidence"],
                "probabilities": json.loads(pitch["model_probabilities_json"]),
                "modelVersion": pitch["model_version"],
            }
            if pitch["model_group"]
            else None,
            "selectedFamily": selected_family,
            "selected": selected_group,
            "actualFamily": actual_family,
            "actual": actual_group,
            "familyHit": family_hit,
            "detailHit": detail_hit,
            "points": points,
            "responseMs": response_ms,
        }
        return jsonify(state), 201

    @app.get("/api/daily/leaderboard")
    def daily_leaderboard() -> tuple[Response, int] | Response:
        attempt_id = request.args.get("attemptId", "")
        try:
            uuid.UUID(attempt_id)
        except ValueError:
            return jsonify({"error": "완료한 Daily 기록이 필요합니다."}), 403
        connection = database()
        attempt = connection.execute(
            "SELECT * FROM daily_attempts WHERE public_id = ?",
            (attempt_id,),
        ).fetchone()
        if not attempt or not attempt["completed_at"]:
            return jsonify({"error": "완료한 Daily 기록이 필요합니다."}), 403
        return jsonify(
            _leaderboard(
                connection,
                attempt["challenge_id"],
                current_attempt_id=attempt["id"],
            )
        )
