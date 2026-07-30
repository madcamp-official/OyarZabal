from __future__ import annotations

import csv
import hmac
import io
import os
import sqlite3
import threading
import time
import unicodedata
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from flask import Flask, Response, g, jsonify, request

from oyarzabal.daily_api import (
    daily_admin_summary,
    init_daily_database,
    register_daily_routes,
)
from oyarzabal.memorable_sets import (
    MEMORABLE_SCENARIOS,
    MEMORABLE_VERSION_LABELS,
)

PITCH_GROUP_FAMILY = {
    "FOUR_SEAM": "FASTBALL",
    "MOVING_FASTBALL": "FASTBALL",
    "SLIDER": "BREAKING",
    "CURVE": "BREAKING",
    "CHANGEUP": "OFFSPEED",
    "SPLITTER_FORK": "OFFSPEED",
}

VERSION1_SCENARIOS = [
    {
        "id": "brown-rodriguez",
        "pitcher": "Hunter Brown",
        "batter": "Julio Rodríguez",
        "actual": ["CURVE", "SLIDER", "MOVING_FASTBALL"],
    },
    {
        "id": "flexen-judge",
        "pitcher": "Chris Flexen",
        "batter": "Aaron Judge",
        "actual": [
            "MOVING_FASTBALL",
            "MOVING_FASTBALL",
            "FOUR_SEAM",
            "FOUR_SEAM",
            "FOUR_SEAM",
        ],
    },
    {
        "id": "snell-tatis",
        "pitcher": "Blake Snell",
        "batter": "Fernando Tatis Jr.",
        "actual": ["CHANGEUP", "FOUR_SEAM", "CHANGEUP", "CURVE"],
    },
    {
        "id": "snell-carroll",
        "pitcher": "Blake Snell",
        "batter": "Corbin Carroll",
        "actual": ["CURVE", "SLIDER", "SLIDER"],
    },
    {
        "id": "kelly-cronenworth",
        "pitcher": "Merrill Kelly",
        "batter": "Jake Cronenworth",
        "actual": [
            "MOVING_FASTBALL",
            "MOVING_FASTBALL",
            "FOUR_SEAM",
            "CHANGEUP",
            "FOUR_SEAM",
            "CURVE",
            "MOVING_FASTBALL",
            "MOVING_FASTBALL",
            "FOUR_SEAM",
            "CHANGEUP",
        ],
    },
]

VERSION2_SCENARIOS = [
    {
        "id": "v2-brown-benintendi",
        "pitcher": "Hunter Brown",
        "batter": "Andrew Benintendi",
        "actual": [
            "CHANGEUP",
            "MOVING_FASTBALL",
            "MOVING_FASTBALL",
            "FOUR_SEAM",
            "CURVE",
        ],
    },
    {
        "id": "v2-peterson-alonso",
        "pitcher": "David Peterson",
        "batter": "Pete Alonso",
        "actual": [
            "CURVE",
            "CHANGEUP",
            "FOUR_SEAM",
            "SLIDER",
            "FOUR_SEAM",
            "FOUR_SEAM",
        ],
    },
    {
        "id": "v2-canning-albies",
        "pitcher": "Griffin Canning",
        "batter": "Ozzie Albies",
        "actual": ["SLIDER", "CHANGEUP", "SLIDER", "MOVING_FASTBALL"],
    },
    {
        "id": "v2-verlander-arenado",
        "pitcher": "Justin Verlander",
        "batter": "Nolan Arenado",
        "actual": [
            "SLIDER",
            "SLIDER",
            "SLIDER",
            "CURVE",
            "CHANGEUP",
            "FOUR_SEAM",
        ],
    },
    {
        "id": "v2-kelly-wilson",
        "pitcher": "Merrill Kelly",
        "batter": "Jacob Wilson",
        "actual": [
            "SLIDER",
            "CHANGEUP",
            "MOVING_FASTBALL",
            "MOVING_FASTBALL",
        ],
    },
]

BENCHMARK_SCENARIOS = {
    "v1": VERSION1_SCENARIOS,
    "v2": VERSION2_SCENARIOS,
    **MEMORABLE_SCENARIOS,
}
SCENARIOS = VERSION1_SCENARIOS
ALL_SCENARIOS = [
    scenario
    for version_scenarios in BENCHMARK_SCENARIOS.values()
    for scenario in version_scenarios
]
SCENARIO_BY_ID = {scenario["id"]: scenario for scenario in ALL_SCENARIOS}
SCENARIO_VERSION = {
    scenario["id"]: version
    for version, version_scenarios in BENCHMARK_SCENARIOS.items()
    for scenario in version_scenarios
}
EXPECTED_ANSWERS_BY_VERSION = {
    version: {
        (scenario["id"], pitch_number): actual
        for scenario in version_scenarios
        for pitch_number, actual in enumerate(scenario["actual"], start=1)
    }
    for version, version_scenarios in BENCHMARK_SCENARIOS.items()
}
EXPECTED_ANSWERS = EXPECTED_ANSWERS_BY_VERSION["v1"]
ALL_EXPECTED_ANSWERS = {
    answer_key: actual
    for version_answers in EXPECTED_ANSWERS_BY_VERSION.values()
    for answer_key, actual in version_answers.items()
}
AVAILABLE_GROUPS = {
    scenario["id"]: set(
        scenario.get(
            "availableGroups",
            (
                PITCH_GROUP_FAMILY
                if not (
                    scenario["id"].startswith("snell-")
                    or scenario["id"] == "v2-verlander-arenado"
                )
                else {"FOUR_SEAM", "SLIDER", "CURVE", "CHANGEUP"}
            ),
        )
    )
    for scenario in ALL_SCENARIOS
}
EXPERIENCE_LEVELS = {"new", "casual", "fan"}
MAX_RESPONSE_MS = 10 * 60 * 1000


class ValidationError(ValueError):
    pass


def _validate_nickname(value: Any) -> str:
    if not isinstance(value, str):
        raise ValidationError("닉네임을 입력해 주세요.")
    nickname = " ".join(unicodedata.normalize("NFKC", value).split())
    if not 1 <= len(nickname) <= 20 or any(
        not (character.isalnum() or character in " _-") for character in nickname
    ):
        raise ValidationError(
            "닉네임은 1~20자의 한글, 영문, 숫자, 공백, _, -만 가능합니다."
        )
    return nickname


class RateLimiter:
    def __init__(self, limit: int, window_seconds: int = 3600):
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            attempts = self._requests[key]
            while attempts and attempts[0] < cutoff:
                attempts.popleft()
            if len(attempts) >= self.limit:
                return False
            attempts.append(now)
            return True


def _validate_submission(
    payload: Any,
) -> tuple[str, str, str, list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise ValidationError("요청 형식이 올바르지 않습니다.")

    nickname = _validate_nickname(payload.get("nickname"))
    experience = payload.get("experience")
    if experience not in EXPERIENCE_LEVELS:
        raise ValidationError("MLB 시청 경험 값이 올바르지 않습니다.")
    version = payload.get("version", "v1")
    if version not in BENCHMARK_SCENARIOS:
        raise ValidationError("Pitch Test 버전이 올바르지 않습니다.")
    expected_answers = EXPECTED_ANSWERS_BY_VERSION[version]
    scenario_order = {
        scenario["id"]: index
        for index, scenario in enumerate(BENCHMARK_SCENARIOS[version])
    }

    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, list) or len(raw_answers) != len(expected_answers):
        raise ValidationError(
            f"{len(expected_answers)}개 투구 답안이 모두 필요합니다."
        )

    seen: set[tuple[str, int]] = set()
    answers: list[dict[str, Any]] = []
    for raw_answer in raw_answers:
        if not isinstance(raw_answer, dict):
            raise ValidationError("답안 형식이 올바르지 않습니다.")

        scenario_id = raw_answer.get("scenarioId")
        pitch_number = raw_answer.get("pitchNumber")
        selected_family = raw_answer.get("selectedFamily")
        selected_group = raw_answer.get("selected")
        response_ms = raw_answer.get("responseMs")

        if (
            not isinstance(scenario_id, str)
            or scenario_id not in SCENARIO_BY_ID
            or SCENARIO_VERSION[scenario_id] != version
        ):
            raise ValidationError("알 수 없는 시나리오입니다.")
        if type(pitch_number) is not int:
            raise ValidationError("투구 번호가 올바르지 않습니다.")
        answer_key = (scenario_id, pitch_number)
        if answer_key not in expected_answers or answer_key in seen:
            raise ValidationError("투구 답안이 누락되었거나 중복되었습니다.")
        if selected_group not in AVAILABLE_GROUPS[scenario_id]:
            raise ValidationError("해당 투수가 던지지 않는 구종입니다.")
        if selected_family != PITCH_GROUP_FAMILY.get(selected_group):
            raise ValidationError("구종 계열과 상세 구종이 일치하지 않습니다.")
        if type(response_ms) is not int or not 0 <= response_ms <= MAX_RESPONSE_MS:
            raise ValidationError("응답 시간이 올바르지 않습니다.")

        actual_group = expected_answers[answer_key]
        actual_family = PITCH_GROUP_FAMILY[actual_group]
        detail_hit = selected_group == actual_group
        family_hit = selected_family == actual_family
        seen.add(answer_key)
        answers.append(
            {
                "scenario_id": scenario_id,
                "pitch_number": pitch_number,
                "selected_family": selected_family,
                "selected_group": selected_group,
                "actual_family": actual_family,
                "actual_group": actual_group,
                "family_hit": int(family_hit),
                "detail_hit": int(detail_hit),
                "points": 3 if detail_hit else 1 if family_hit else 0,
                "response_ms": response_ms,
            }
        )

    if seen != set(expected_answers):
        raise ValidationError(
            f"{len(expected_answers)}개 투구 답안이 모두 필요합니다."
        )
    answers.sort(
        key=lambda answer: (
            scenario_order[answer["scenario_id"]],
            answer["pitch_number"],
        )
    )
    return version, nickname, experience, answers


def _connect(path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    return connection


def _init_database(path: str) -> None:
    database = Path(path)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = _connect(path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                public_id TEXT NOT NULL UNIQUE,
                nickname TEXT,
                max_points INTEGER NOT NULL DEFAULT 105,
                benchmark_version TEXT NOT NULL DEFAULT 'v1',
                experience TEXT NOT NULL,
                submitted_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                submission_id INTEGER NOT NULL REFERENCES submissions(id)
                    ON DELETE CASCADE,
                scenario_id TEXT NOT NULL,
                pitch_number INTEGER NOT NULL,
                selected_family TEXT NOT NULL,
                selected_group TEXT NOT NULL,
                actual_family TEXT NOT NULL,
                actual_group TEXT NOT NULL,
                family_hit INTEGER NOT NULL CHECK (family_hit IN (0, 1)),
                detail_hit INTEGER NOT NULL CHECK (detail_hit IN (0, 1)),
                points INTEGER NOT NULL CHECK (points IN (0, 1, 3)),
                response_ms INTEGER NOT NULL CHECK (response_ms >= 0),
                UNIQUE (submission_id, scenario_id, pitch_number)
            );
            CREATE INDEX IF NOT EXISTS idx_answers_scenario_pitch
                ON answers (scenario_id, pitch_number);
            CREATE INDEX IF NOT EXISTS idx_submissions_submitted_at
                ON submissions (submitted_at);
            """
        )
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(submissions)")
        }
        if "nickname" not in columns:
            connection.execute("ALTER TABLE submissions ADD COLUMN nickname TEXT")
        if "max_points" not in columns:
            connection.execute(
                "ALTER TABLE submissions "
                "ADD COLUMN max_points INTEGER NOT NULL DEFAULT 105"
            )
        if "benchmark_version" not in columns:
            connection.execute(
                "ALTER TABLE submissions "
                "ADD COLUMN benchmark_version TEXT NOT NULL DEFAULT 'v1'"
            )
        init_daily_database(connection)
        connection.commit()
    finally:
        connection.close()


def _score(answers: list[dict[str, Any]]) -> dict[str, int | float]:
    detail_hits = sum(answer["detail_hit"] for answer in answers)
    family_hits = sum(answer["family_hit"] for answer in answers)
    total_points = sum(answer["points"] for answer in answers)
    average_response_ms = round(
        sum(answer["response_ms"] for answer in answers) / len(answers)
    )
    return {
        "totalPoints": total_points,
        "maxPoints": len(answers) * 3,
        "detailHits": detail_hits,
        "familyHits": family_hits,
        "averageResponseMs": average_response_ms,
    }


def _rate(hits: int | float, attempts: int | float) -> float:
    return round(hits / attempts, 4) if attempts else 0.0


def create_app(
    *,
    database_path: str | None = None,
    admin_token: str | None = None,
    rate_limit: int | None = None,
    daily_generator: Callable[[date, date], list[dict[str, Any]]] | None = None,
    daily_today: Callable[[], date] | None = None,
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
    app.config["DATABASE_PATH"] = database_path or os.environ.get(
        "BENCHMARK_DB_PATH", "data/benchmark/benchmark.sqlite3"
    )
    if admin_token is None:
        token_file = os.environ.get("BENCHMARK_ADMIN_TOKEN_FILE")
        admin_token = (
            Path(token_file).read_text(encoding="utf-8").strip()
            if token_file
            else os.environ.get("BENCHMARK_ADMIN_TOKEN", "")
        )
    app.config["ADMIN_TOKEN"] = admin_token
    if app.config["ADMIN_TOKEN"] and len(app.config["ADMIN_TOKEN"]) < 32:
        raise ValueError("BENCHMARK_ADMIN_TOKEN must contain at least 32 characters")
    app.config["ALLOWED_ORIGIN"] = os.environ.get("BENCHMARK_ALLOWED_ORIGIN", "")
    app.config["TRUST_PROXY"] = os.environ.get("BENCHMARK_TRUST_PROXY") == "1"
    limiter = RateLimiter(
        rate_limit
        if rate_limit is not None
        else int(os.environ.get("BENCHMARK_RATE_LIMIT", "30"))
    )
    daily_limiter = RateLimiter(
        int(os.environ.get("BENCHMARK_DAILY_RATE_LIMIT", "120"))
    )
    leaderboard_limiter = RateLimiter(
        int(os.environ.get("BENCHMARK_LEADERBOARD_RATE_LIMIT", "120"))
    )
    _init_database(app.config["DATABASE_PATH"])

    def database() -> sqlite3.Connection:
        if "database" not in g:
            g.database = _connect(app.config["DATABASE_PATH"])
        return g.database

    def client_ip() -> str:
        if app.config["TRUST_PROXY"]:
            forwarded = request.headers.get("X-Forwarded-For", "")
            if forwarded:
                return forwarded.split(",", maxsplit=1)[0].strip()
        return request.remote_addr or "unknown"

    def require_admin() -> Response | None:
        configured = app.config["ADMIN_TOKEN"]
        supplied = request.headers.get("Authorization", "")
        expected = f"Bearer {configured}"
        if not configured or not hmac.compare_digest(supplied, expected):
            return jsonify({"error": "관리자 인증이 필요합니다."}), 401
        return None

    @app.teardown_appcontext
    def close_database(_: BaseException | None) -> None:
        connection = g.pop("database", None)
        if connection is not None:
            connection.close()

    @app.after_request
    def security_headers(response: Response) -> Response:
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; frame-ancestors 'none'"
        )
        return response

    @app.errorhandler(413)
    def payload_too_large(_: Exception) -> tuple[Response, int]:
        return jsonify({"error": "요청이 너무 큽니다."}), 413

    @app.get("/api/health")
    def health() -> Response:
        return jsonify({"ok": True})

    daily_route_options: dict[str, Any] = {}
    if daily_generator is not None:
        daily_route_options["generator"] = daily_generator
    if daily_today is not None:
        daily_route_options["today_provider"] = daily_today
    register_daily_routes(
        app,
        database=database,
        require_admin=require_admin,
        validate_nickname=_validate_nickname,
        limiter_allow=daily_limiter.allow,
        client_ip=client_ip,
        **daily_route_options,
    )

    @app.post("/api/benchmark/submissions")
    def create_submission() -> tuple[Response, int]:
        allowed_origin = app.config["ALLOWED_ORIGIN"]
        origin = request.headers.get("Origin")
        if allowed_origin and origin and origin != allowed_origin:
            return jsonify({"error": "허용되지 않은 출처입니다."}), 403
        if not limiter.allow(client_ip()):
            return (
                jsonify(
                    {"error": "제출 횟수가 너무 많습니다. 잠시 후 다시 시도해 주세요."}
                ),
                429,
            )
        if not request.is_json:
            return jsonify({"error": "JSON 요청만 허용됩니다."}), 415

        try:
            version, nickname, experience, answers = _validate_submission(
                request.get_json(silent=True)
            )
        except ValidationError as error:
            return jsonify({"error": str(error)}), 400

        public_id = str(uuid.uuid4())
        submitted_at = datetime.now(UTC).isoformat()
        connection = database()
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO submissions (
                    public_id, nickname, max_points, benchmark_version,
                    experience, submitted_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    public_id,
                    nickname,
                    len(answers) * 3,
                    version,
                    experience,
                    submitted_at,
                ),
            )
            submission_id = cursor.lastrowid
            connection.executemany(
                """
                INSERT INTO answers (
                    submission_id, scenario_id, pitch_number,
                    selected_family, selected_group,
                    actual_family, actual_group,
                    family_hit, detail_hit, points, response_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        submission_id,
                        answer["scenario_id"],
                        answer["pitch_number"],
                        answer["selected_family"],
                        answer["selected_group"],
                        answer["actual_family"],
                        answer["actual_group"],
                        answer["family_hit"],
                        answer["detail_hit"],
                        answer["points"],
                        answer["response_ms"],
                    )
                    for answer in answers
                ],
            )

        return (
            jsonify(
                {
                    "submissionId": public_id,
                    **_score(answers),
                }
            ),
            201,
        )

    @app.get("/api/benchmark/memorable-leaderboard")
    def memorable_leaderboard() -> tuple[Response, int] | Response:
        version = request.args.get("version", "")
        if version not in MEMORABLE_SCENARIOS:
            return jsonify({"error": "알 수 없는 명승부 Set입니다."}), 400
        if not leaderboard_limiter.allow(f"{client_ip()}:{version}"):
            return jsonify({"error": "조회 횟수가 너무 많습니다."}), 429

        connection = database()
        participant_count = connection.execute(
            """
            SELECT COUNT(*) AS participants
            FROM submissions
            WHERE benchmark_version = ?
            """,
            (version,),
        ).fetchone()["participants"]
        rows = connection.execute(
            """
            SELECT
                s.nickname,
                SUM(a.points) AS points,
                SUM(a.detail_hit) AS detail_hits,
                SUM(a.family_hit) AS family_hits,
                AVG(a.response_ms) AS average_response_ms
            FROM submissions s
            JOIN answers a ON a.submission_id = s.id
            WHERE s.benchmark_version = ?
            GROUP BY s.id
            ORDER BY
                points DESC,
                detail_hits DESC,
                family_hits DESC,
                average_response_ms ASC,
                s.id ASC
            LIMIT 50
            """,
            (version,),
        ).fetchall()
        leaderboard_rows = []
        previous_score: tuple[int, int] | None = None
        rank = 0
        for row in rows:
            score = (row["points"], row["detail_hits"])
            if score != previous_score:
                rank += 1
                previous_score = score
            leaderboard_rows.append(
                {
                    "rank": rank,
                    "nickname": row["nickname"],
                    "points": row["points"],
                    "detailHits": row["detail_hits"],
                    "familyHits": row["family_hits"],
                    "averageResponseMs": round(row["average_response_ms"]),
                }
            )
        pitch_count = sum(
            len(scenario["actual"])
            for scenario in MEMORABLE_SCENARIOS[version]
        )
        return jsonify(
            {
                "version": version,
                "label": MEMORABLE_VERSION_LABELS[version],
                "participantCount": participant_count,
                "pitchCount": pitch_count,
                "maxPoints": pitch_count * 3,
                "rows": leaderboard_rows,
            }
        )

    @app.get("/api/admin/benchmark/summary")
    def admin_summary() -> tuple[Response, int] | Response:
        unauthorized = require_admin()
        if unauthorized is not None:
            return unauthorized

        connection = database()
        total = connection.execute(
            """
            SELECT
                COUNT(DISTINCT s.id) AS participants,
                COUNT(a.id) AS attempts,
                COALESCE(SUM(a.detail_hit), 0) AS detail_hits,
                COALESCE(SUM(a.family_hit), 0) AS family_hits,
                COALESCE(SUM(a.points), 0) AS points,
                COALESCE(AVG(a.response_ms), 0) AS average_response_ms
            FROM submissions s
            LEFT JOIN answers a ON a.submission_id = s.id
            WHERE s.benchmark_version IN ('v1', 'v2')
            """
        ).fetchone()
        average_score_rate = connection.execute(
            """
            SELECT COALESCE(AVG(score * 1.0 / max_points), 0) AS rate
            FROM (
                SELECT
                    s.id,
                    s.max_points,
                    COALESCE(SUM(a.points), 0) AS score
                FROM submissions s
                LEFT JOIN answers a ON a.submission_id = s.id
                WHERE s.benchmark_version IN ('v1', 'v2')
                GROUP BY s.id
            )
            """
        ).fetchone()["rate"]

        scenario_rows = connection.execute(
            """
            SELECT
                scenario_id,
                COUNT(*) AS attempts,
                COUNT(DISTINCT submission_id) AS participants,
                SUM(detail_hit) AS detail_hits,
                SUM(family_hit) AS family_hits,
                SUM(points) AS points
            FROM answers
            GROUP BY scenario_id
            """
        ).fetchall()
        scenario_stats = {row["scenario_id"]: row for row in scenario_rows}

        pitch_rows = connection.execute(
            """
            SELECT
                scenario_id,
                pitch_number,
                COUNT(*) AS attempts,
                SUM(detail_hit) AS detail_hits,
                SUM(family_hit) AS family_hits
            FROM answers
            GROUP BY scenario_id, pitch_number
            ORDER BY scenario_id, pitch_number
            """
        ).fetchall()

        submission_rows = connection.execute(
            """
            SELECT
                s.id,
                s.public_id,
                s.nickname,
                s.max_points,
                s.benchmark_version,
                s.experience,
                s.submitted_at,
                SUM(a.points) AS points,
                SUM(a.detail_hit) AS detail_hits,
                SUM(a.family_hit) AS family_hits,
                AVG(a.response_ms) AS average_response_ms
            FROM submissions s
            JOIN answers a ON a.submission_id = s.id
            GROUP BY s.id
            ORDER BY s.id DESC
            LIMIT 500
            """
        ).fetchall()

        distribution_rows = connection.execute(
            """
            SELECT score_percent AS score, COUNT(*) AS participants
            FROM (
                SELECT
                    s.id,
                    CAST(
                        ROUND(SUM(a.points) * 100.0 / s.max_points)
                        AS INTEGER
                    ) AS score_percent
                FROM submissions s
                JOIN answers a ON a.submission_id = s.id
                WHERE s.benchmark_version IN ('v1', 'v2')
                GROUP BY s.id
            )
            GROUP BY score_percent
            ORDER BY score_percent
            """
        ).fetchall()

        attempts = total["attempts"]
        participants = total["participants"]
        scenario_payload = []
        for scenario in ALL_SCENARIOS:
            if SCENARIO_VERSION[scenario["id"]] not in {"v1", "v2"}:
                continue
            stats = scenario_stats.get(scenario["id"])
            if not stats:
                continue
            scenario_attempts = stats["attempts"] if stats else 0
            scenario_payload.append(
                {
                    "id": scenario["id"],
                    "version": SCENARIO_VERSION[scenario["id"]],
                    "pitcher": scenario["pitcher"],
                    "batter": scenario["batter"],
                    "attempts": scenario_attempts,
                    "detailRate": _rate(
                        stats["detail_hits"], scenario_attempts
                    )
                    if stats
                    else 0,
                    "familyRate": _rate(
                        stats["family_hits"], scenario_attempts
                    )
                    if stats
                    else 0,
                    "averagePoints": round(
                        stats["points"] / max(1, stats["participants"]),
                        2,
                    )
                    if stats
                    else 0,
                }
            )
        submission_payload = [
            {
                "id": row["public_id"],
                "participantLabel": (
                    row["nickname"] or f"익명 #{row['id']:04d}"
                ),
                "experience": row["experience"],
                "version": row["benchmark_version"],
                "versionLabel": MEMORABLE_VERSION_LABELS.get(
                    row["benchmark_version"]
                ),
                "submittedAt": row["submitted_at"],
                "points": row["points"],
                "maxPoints": row["max_points"],
                "detailHits": row["detail_hits"],
                "familyHits": row["family_hits"],
                "averageResponseMs": round(row["average_response_ms"]),
            }
            for row in submission_rows
        ]
        memorable_chapters = []
        memorable_detail_hits = 0
        memorable_family_hits = 0
        for version, chapter_scenarios in MEMORABLE_SCENARIOS.items():
            chapter_stats = [
                scenario_stats.get(scenario["id"])
                for scenario in chapter_scenarios
            ]
            pitch_count = sum(
                len(scenario["actual"]) for scenario in chapter_scenarios
            )
            answer_count = sum(
                stats["attempts"] for stats in chapter_stats if stats
            )
            participants_count = (
                answer_count // pitch_count if pitch_count else 0
            )
            detail_count = sum(
                stats["detail_hits"] for stats in chapter_stats if stats
            )
            family_count = sum(
                stats["family_hits"] for stats in chapter_stats if stats
            )
            points = sum(
                stats["points"] for stats in chapter_stats if stats
            )
            memorable_detail_hits += detail_count
            memorable_family_hits += family_count
            memorable_chapters.append(
                {
                    "version": version,
                    "label": MEMORABLE_VERSION_LABELS[version],
                    "participants": participants_count,
                    "pitchCount": pitch_count,
                    "answers": answer_count,
                    "detailRate": _rate(detail_count, answer_count),
                    "familyRate": _rate(family_count, answer_count),
                    "averageScoreRate": _rate(
                        points,
                        participants_count * pitch_count * 3,
                    ),
                }
            )
        memorable_answers = sum(
            chapter["answers"] for chapter in memorable_chapters
        )
        memorable_participants = sum(
            chapter["participants"] for chapter in memorable_chapters
        )
        return jsonify(
            {
                "totals": {
                    "participants": participants,
                    "attempts": attempts,
                    "detailRate": _rate(total["detail_hits"], attempts),
                    "familyRate": _rate(total["family_hits"], attempts),
                    "averageScoreRate": round(average_score_rate, 4),
                    "averageResponseMs": round(total["average_response_ms"]),
                },
                "scenarios": scenario_payload,
                "pitches": [
                    {
                        "scenarioId": row["scenario_id"],
                        "pitchNumber": row["pitch_number"],
                        "attempts": row["attempts"],
                        "detailRate": _rate(
                            row["detail_hits"], row["attempts"]
                        ),
                        "familyRate": _rate(
                            row["family_hits"], row["attempts"]
                        ),
                    }
                    for row in pitch_rows
                    if (row["scenario_id"], row["pitch_number"])
                    in ALL_EXPECTED_ANSWERS
                    and SCENARIO_VERSION[row["scenario_id"]] in {"v1", "v2"}
                ],
                "scoreDistribution": [
                    {
                        "score": row["score"],
                        "participants": row["participants"],
                    }
                    for row in distribution_rows
                ],
                "submissions": [
                    row
                    for row in submission_payload
                    if row["version"] in {"v1", "v2"}
                ],
                "memorable": {
                    "totals": {
                        "completedAttempts": memorable_participants,
                        "answers": memorable_answers,
                        "detailRate": _rate(
                            memorable_detail_hits, memorable_answers
                        ),
                        "familyRate": _rate(
                            memorable_family_hits, memorable_answers
                        ),
                    },
                    "chapters": memorable_chapters,
                    "submissions": [
                        row
                        for row in submission_payload
                        if row["version"].startswith("memorable:")
                    ],
                },
                "daily": daily_admin_summary(connection),
            }
        )

    @app.get("/api/admin/benchmark/submissions.csv")
    def admin_csv() -> tuple[Response, int] | Response:
        unauthorized = require_admin()
        if unauthorized is not None:
            return unauthorized

        rows = database().execute(
            """
            SELECT
                s.id,
                s.public_id,
                s.nickname,
                s.max_points,
                s.benchmark_version,
                s.experience,
                s.submitted_at,
                SUM(a.points) AS points,
                SUM(a.detail_hit) AS detail_hits,
                SUM(a.family_hit) AS family_hits,
                ROUND(AVG(a.response_ms)) AS average_response_ms
            FROM submissions s
            JOIN answers a ON a.submission_id = s.id
            GROUP BY s.id
            ORDER BY s.id DESC
            """
        ).fetchall()
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(
            [
                "submission_id",
                "participant_label",
                "nickname",
                "benchmark_version",
                "experience",
                "submitted_at",
                "points",
                "max_points",
                "detail_hits",
                "family_hits",
                "average_response_ms",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["public_id"],
                    row["nickname"] or f"익명 #{row['id']:04d}",
                    row["nickname"] or "",
                    row["benchmark_version"],
                    row["experience"],
                    row["submitted_at"],
                    row["points"],
                    row["max_points"],
                    row["detail_hits"],
                    row["family_hits"],
                    row["average_response_ms"],
                ]
            )
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={
                "Content-Disposition": (
                    'attachment; filename="human-benchmark-submissions.csv"'
                )
            },
        )

    return app


def main() -> None:
    app = create_app()
    if len(app.config["ADMIN_TOKEN"]) < 32:
        raise SystemExit("BENCHMARK_ADMIN_TOKEN must contain at least 32 characters")
    app.run(
        host=os.environ.get("BENCHMARK_HOST", "127.0.0.1"),
        port=int(os.environ.get("BENCHMARK_PORT", "8000")),
        debug=False,
    )
