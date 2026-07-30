from __future__ import annotations

import sqlite3
from copy import deepcopy
from datetime import date

import pytest
from oyarzabal.benchmark_api import create_app
from oyarzabal.daily_api import init_daily_database

ADMIN_TOKEN = "test-admin-token-with-at-least-32-characters"
TODAY = date(2026, 7, 29)


def _scenarios() -> list[dict[str, object]]:
    repertoire = {
        "FASTBALL": ["포심", "싱커"],
        "BREAKING": ["슬라이더", "커브"],
        "OFFSPEED": ["체인지업"],
    }
    available_groups = [
        "FOUR_SEAM",
        "MOVING_FASTBALL",
        "SLIDER",
        "CURVE",
        "CHANGEUP",
    ]
    scenarios = [
        {
            "id": "daily-1-10",
            "pitcher": "Pitcher One",
            "batter": "Batter One",
            "date": "2026-07-28",
            "inning": "3회 초",
            "awayTeam": "NYY",
            "homeTeam": "CWS",
            "awayScore": 1,
            "homeScore": 0,
            "outs": 1,
            "bases": [True, False, False],
            "outcome": "삼진",
            "gameUrl": "https://www.mlb.com/gameday/1",
            "repertoire": repertoire,
            "availableGroups": available_groups,
            "pitches": [
                {
                    "number": 1,
                    "count": "0–0",
                    "actual": "FOUR_SEAM",
                    "rawLabel": "포심",
                    "result": "볼",
                    "playId": "play-1",
                },
                {
                    "number": 2,
                    "count": "1–0",
                    "actual": "SLIDER",
                    "rawLabel": "슬라이더",
                    "result": "파울",
                    "playId": "play-2",
                },
                {
                    "number": 3,
                    "count": "1–1",
                    "actual": "CHANGEUP",
                    "rawLabel": "체인지업",
                    "result": "헛스윙",
                    "playId": "play-3",
                },
            ],
        },
        {
            "id": "daily-2-20",
            "pitcher": "Pitcher Two",
            "batter": "Batter Two",
            "date": "2026-07-28",
            "inning": "7회 말",
            "awayTeam": "HOU",
            "homeTeam": "LAA",
            "awayScore": 2,
            "homeScore": 2,
            "outs": 0,
            "bases": [False, True, False],
            "outcome": "안타",
            "gameUrl": "https://www.mlb.com/gameday/2",
            "repertoire": repertoire,
            "availableGroups": available_groups,
            "pitches": [
                {
                    "number": 1,
                    "count": "0–0",
                    "actual": "MOVING_FASTBALL",
                    "rawLabel": "싱커",
                    "result": "볼",
                    "playId": "play-4",
                },
                {
                    "number": 2,
                    "count": "1–0",
                    "actual": "CURVE",
                    "rawLabel": "커브",
                    "result": "루킹 스트라이크",
                    "playId": "play-5",
                },
                {
                    "number": 3,
                    "count": "1–1",
                    "actual": "FOUR_SEAM",
                    "rawLabel": "포심",
                    "result": "인플레이",
                    "playId": "play-6",
                },
            ],
        },
        {
            "id": "daily-3-30",
            "pitcher": "Pitcher Three",
            "batter": "Batter Three",
            "date": "2026-07-28",
            "inning": "9회 초",
            "awayTeam": "SEA",
            "homeTeam": "BOS",
            "awayScore": 3,
            "homeScore": 2,
            "outs": 2,
            "bases": [False, False, True],
            "outcome": "볼넷",
            "gameUrl": "https://www.mlb.com/gameday/3",
            "repertoire": repertoire,
            "availableGroups": available_groups,
            "pitches": [
                {
                    "number": 1,
                    "count": "0–0",
                    "actual": "CURVE",
                    "rawLabel": "커브",
                    "result": "볼",
                    "playId": "play-7",
                },
                {
                    "number": 2,
                    "count": "1–0",
                    "actual": "FOUR_SEAM",
                    "rawLabel": "포심",
                    "result": "파울",
                    "playId": "play-8",
                },
                {
                    "number": 3,
                    "count": "1–1",
                    "actual": "CHANGEUP",
                    "rawLabel": "체인지업",
                    "result": "볼",
                    "playId": "play-9",
                },
            ],
        },
    ]
    for scenario in scenarios:
        for pitch in scenario["pitches"]:
            pitch["modelPrediction"] = {
                "selected": pitch["actual"],
                "confidence": 1.0,
                "probabilities": {pitch["actual"]: 1.0},
                "modelVersion": "V8.4-test",
            }
    return scenarios


def _client(tmp_path):
    scenarios = _scenarios()
    app = create_app(
        database_path=str(tmp_path / "benchmark.sqlite3"),
        admin_token=ADMIN_TOKEN,
        daily_generator=lambda _challenge_date, _source_date: deepcopy(scenarios),
        daily_today=lambda: TODAY,
    )
    app.config["TESTING"] = True
    return app.test_client()


def _generate(client) -> None:
    response = client.post(
        "/api/admin/daily/generate",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert response.status_code == 201


def _start(client, participant_id: str, nickname: str = "야구팬"):
    return client.post(
        "/api/daily/attempts",
        json={"participantId": participant_id, "nickname": nickname},
    )


def test_daily_database_migrates_two_scenario_constraint(tmp_path) -> None:
    connection = sqlite3.connect(tmp_path / "old.sqlite3")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        """
        CREATE TABLE daily_scenarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            challenge_id INTEGER NOT NULL REFERENCES daily_challenges(id)
                ON DELETE CASCADE,
            scenario_key TEXT NOT NULL,
            position INTEGER NOT NULL CHECK (position IN (1, 2)),
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
        )
        """
    )

    init_daily_database(connection)

    table_sql = connection.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE type = 'table' AND name = 'daily_scenarios'"
    ).fetchone()["sql"]
    assert "position BETWEEN 1 AND 3" in table_sql
    pitch_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(daily_pitches)")
    }
    assert {
        "model_group",
        "model_confidence",
        "model_probabilities_json",
        "model_version",
    } <= pitch_columns
    connection.close()


def test_daily_game_locks_answers_and_shows_leaderboard_after_completion(
    tmp_path,
) -> None:
    client = _client(tmp_path)
    assert client.post("/api/admin/daily/generate").status_code == 401
    assert client.get("/api/daily/challenge").status_code == 404
    _generate(client)

    challenge = client.get("/api/daily/challenge")
    assert challenge.status_code == 200
    assert challenge.json == {
        "id": challenge.json["id"],
        "number": 1,
        "date": "2026-07-29",
        "sourceDate": "2026-07-28",
        "scenarioCount": 3,
        "participantCount": 0,
    }

    participant_id = "11111111-1111-4111-8111-111111111111"
    started = _start(client, participant_id)
    assert started.status_code == 201
    assert started.json["status"] == "playing"
    assert started.json["current"]["scenarioNumber"] == 1
    assert started.json["current"]["pitch"] == {"number": 1, "count": "0–0"}
    assert "actual" not in started.json["current"]["pitch"]
    assert "playId" not in started.json["current"]["pitch"]
    assert "modelPrediction" not in started.json["current"]["pitch"]
    attempt_id = started.json["attemptId"]
    assert (
        client.get(f"/api/daily/leaderboard?attemptId={attempt_id}").status_code
        == 403
    )

    actual_groups = [
        "FOUR_SEAM",
        "SLIDER",
        "CHANGEUP",
        "MOVING_FASTBALL",
        "CURVE",
        "FOUR_SEAM",
        "CURVE",
        "FOUR_SEAM",
        "CHANGEUP",
    ]
    families = {
        "FOUR_SEAM": "FASTBALL",
        "MOVING_FASTBALL": "FASTBALL",
        "SLIDER": "BREAKING",
        "CURVE": "BREAKING",
        "CHANGEUP": "OFFSPEED",
    }
    response = None
    for index, actual in enumerate(actual_groups):
        response = client.post(
            f"/api/daily/attempts/{attempt_id}/answers",
            json={
                "selectedFamily": families[actual],
                "selected": actual,
                "responseMs": 1000 + index,
            },
        )
        assert response.status_code == 201
        assert response.json["reveal"]["actual"] == actual
        assert response.json["reveal"]["playId"] == f"play-{index + 1}"
        assert response.json["reveal"]["modelPrediction"] == {
            "selected": actual,
            "confidence": 1.0,
            "probabilities": {actual: 1.0},
            "modelVersion": "V8.4-test",
        }

    assert response is not None
    assert response.json["status"] == "completed"
    assert response.json["result"]["points"] == 27
    assert response.json["result"]["maxPoints"] == 27
    assert response.json["result"]["rank"] == 1
    assert response.json["result"]["streak"] == 1
    assert response.json["result"]["leaderboard"]["rows"][0]["nickname"] == "야구팬"
    assert response.json["result"]["leaderboard"]["rows"][1] == {
        "rank": 1,
        "nickname": "OyarZabal V8.4-test",
        "points": 27,
        "detailHits": 9,
        "familyHits": 9,
        "isCurrent": False,
        "isModel": True,
    }

    resumed = _start(client, participant_id, nickname="다른이름")
    assert resumed.status_code == 200
    assert resumed.json["status"] == "completed"
    assert resumed.json["nickname"] == "야구팬"
    assert resumed.json["result"]["rank"] == 1

    admin = client.get(
        "/api/admin/benchmark/summary",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert admin.json["daily"]["totals"] == {
        "challenges": 1,
        "completedAttempts": 1,
        "answers": 9,
        "humanDetailRate": 1.0,
        "humanFamilyRate": 1.0,
        "modelDetailRate": 1.0,
        "modelFamilyRate": 1.0,
    }
    assert admin.json["daily"]["attempts"][0]["nickname"] == "야구팬"


def test_daily_upgrade_keeps_two_scenario_answers_and_reopens_attempt(
    tmp_path,
) -> None:
    client = _client(tmp_path)
    _generate(client)
    database_path = tmp_path / "benchmark.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    with connection:
        connection.execute(
            """
            DELETE FROM daily_scenarios
            WHERE challenge_id = 1 AND position = 3
            """
        )
    connection.close()

    participant_id = "55555555-5555-4555-8555-555555555555"
    started = _start(client, participant_id)
    attempt_id = started.json["attemptId"]
    for selected_family, selected in [
        ("FASTBALL", "FOUR_SEAM"),
        ("BREAKING", "SLIDER"),
        ("OFFSPEED", "CHANGEUP"),
        ("FASTBALL", "MOVING_FASTBALL"),
        ("BREAKING", "CURVE"),
        ("FASTBALL", "FOUR_SEAM"),
    ]:
        response = client.post(
            f"/api/daily/attempts/{attempt_id}/answers",
            json={
                "selectedFamily": selected_family,
                "selected": selected,
                "responseMs": 1000,
            },
        )
        assert response.status_code == 201
    assert response.json["status"] == "completed"

    _generate(client)

    challenge = client.get("/api/daily/challenge")
    assert challenge.json["scenarioCount"] == 3
    assert challenge.json["participantCount"] == 0
    resumed = _start(client, participant_id)
    assert resumed.json["status"] == "playing"
    assert resumed.json["current"]["scenarioNumber"] == 3
    assert len(resumed.json["history"]) == 6
    connection = sqlite3.connect(database_path)
    assert connection.execute(
        "SELECT COUNT(*) FROM daily_pitches WHERE model_group IS NOT NULL"
    ).fetchone()[0] == 9
    connection.close()


def test_daily_rejects_invalid_input_and_ranks_official_attempts(tmp_path) -> None:
    client = _client(tmp_path)
    _generate(client)
    assert (
        _start(
            client,
            "22222222-2222-4222-8222-222222222222",
            nickname="<script>",
        ).status_code
        == 400
    )
    assert _start(client, "not-a-uuid").status_code == 400

    first = _start(client, "33333333-3333-4333-8333-333333333333", "첫번째")
    attempt_id = first.json["attemptId"]
    invalid = client.post(
        f"/api/daily/attempts/{attempt_id}/answers",
        json={
            "selectedFamily": "BREAKING",
            "selected": "FOUR_SEAM",
            "responseMs": 1,
        },
    )
    assert invalid.status_code == 400

    for _ in range(9):
        response = client.post(
            f"/api/daily/attempts/{attempt_id}/answers",
            json={
                "selectedFamily": "FASTBALL",
                "selected": "FOUR_SEAM",
                "responseMs": 1000,
            },
        )
        assert response.status_code == 201

    leaderboard = client.get(
        f"/api/daily/leaderboard?attemptId={attempt_id}"
    )
    assert leaderboard.status_code == 200
    assert leaderboard.json["participantCount"] == 1
    assert any(row["isCurrent"] for row in leaderboard.json["rows"])
    assert any(row["isModel"] for row in leaderboard.json["rows"])


def test_daily_rejects_cross_origin_attempts(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BENCHMARK_ALLOWED_ORIGIN", "https://pitchtest.example")
    client = _client(tmp_path)
    _generate(client)

    response = client.post(
        "/api/daily/attempts",
        json={
            "participantId": "44444444-4444-4444-8444-444444444444",
            "nickname": "야구팬",
        },
        headers={"Origin": "https://attacker.example"},
    )

    assert response.status_code == 403
