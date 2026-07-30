from __future__ import annotations

import sqlite3
from copy import deepcopy

import pytest
from oyarzabal.benchmark_api import (
    BENCHMARK_SCENARIOS,
    PITCH_GROUP_FAMILY,
    create_app,
)

ADMIN_TOKEN = "test-admin-token-with-at-least-32-characters"


def _payload(version: str = "v1") -> dict[str, object]:
    scenarios = BENCHMARK_SCENARIOS[version]
    return {
        "version": version,
        "nickname": "야구팬",
        "experience": "casual",
        "answers": [
            {
                "scenarioId": scenario["id"],
                "pitchNumber": pitch_number,
                "selectedFamily": "FASTBALL",
                "selected": "FOUR_SEAM",
                "responseMs": 1000,
            }
            for scenario in scenarios
            for pitch_number, _ in enumerate(scenario["actual"], start=1)
        ],
    }


def _memorable_payload(version: str = "memorable:alcs7-bottom-7") -> dict[str, object]:
    scenarios = BENCHMARK_SCENARIOS[version]
    return {
        "version": version,
        "nickname": "명승부팬",
        "experience": "fan",
        "answers": [
            {
                "scenarioId": scenario["id"],
                "pitchNumber": pitch_number,
                "selectedFamily": PITCH_GROUP_FAMILY[
                    scenario["availableGroups"][0]
                ],
                "selected": scenario["availableGroups"][0],
                "responseMs": 1500,
            }
            for scenario in scenarios
            for pitch_number, _ in enumerate(scenario["actual"], start=1)
        ],
    }


def _client(tmp_path, *, rate_limit: int = 30):
    app = create_app(
        database_path=str(tmp_path / "benchmark.sqlite3"),
        admin_token=ADMIN_TOKEN,
        rate_limit=rate_limit,
    )
    app.config["TESTING"] = True
    return app.test_client()


def test_server_scores_submission_and_protects_admin(tmp_path) -> None:
    client = _client(tmp_path)

    response = client.post("/api/benchmark/submissions", json=_payload())

    assert response.status_code == 201
    assert response.json["totalPoints"] == 28
    assert response.json["maxPoints"] == 75
    assert response.json["detailHits"] == 7
    assert response.json["familyHits"] == 14
    assert client.get("/api/admin/benchmark/summary").status_code == 401
    assert (
        client.get(
            "/api/admin/benchmark/summary",
            headers={"Authorization": "Bearer wrong-token"},
        ).status_code
        == 401
    )

    summary = client.get(
        "/api/admin/benchmark/summary",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )

    assert summary.status_code == 200
    assert summary.json["submissions"][0]["participantLabel"] == "야구팬"
    assert summary.json["totals"] == {
        "participants": 1,
        "attempts": 25,
        "detailRate": round(7 / 25, 4),
        "familyRate": round(14 / 25, 4),
        "averageScoreRate": round(28 / 75, 4),
        "averageResponseMs": 1000,
    }
    assert summary.json["submissions"][0]["maxPoints"] == 75
    assert len(summary.json["scenarios"]) == 5
    assert len(summary.json["pitches"]) == 25
    assert summary.json["submissions"][0]["version"] == "v1"


def test_accepts_version_2_and_keeps_versions_separate(tmp_path) -> None:
    client = _client(tmp_path)

    response = client.post("/api/benchmark/submissions", json=_payload("v2"))

    assert response.status_code == 201
    assert response.json["totalPoints"] == 20
    assert response.json["maxPoints"] == 75
    assert response.json["detailHits"] == 5
    assert response.json["familyHits"] == 10

    summary = client.get(
        "/api/admin/benchmark/summary",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert {scenario["version"] for scenario in summary.json["scenarios"]} == {"v2"}
    assert len(summary.json["scenarios"]) == 5
    assert summary.json["submissions"][0]["version"] == "v2"

    mismatched = _payload("v2")
    mismatched["answers"][0]["scenarioId"] = "brown-rodriguez"
    assert (
        client.post("/api/benchmark/submissions", json=mismatched).status_code
        == 400
    )


def test_saves_memorable_set_and_reports_it_separately(tmp_path) -> None:
    client = _client(tmp_path)
    version = "memorable:alcs7-bottom-7"

    response = client.post(
        "/api/benchmark/submissions",
        json=_memorable_payload(version),
    )

    assert response.status_code == 201
    assert response.json["maxPoints"] == 36

    summary = client.get(
        "/api/admin/benchmark/summary",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    ).json
    chapter = next(
        item for item in summary["memorable"]["chapters"]
        if item["version"] == version
    )
    assert summary["totals"]["participants"] == 0
    assert summary["memorable"]["totals"]["completedAttempts"] == 1
    assert summary["memorable"]["totals"]["answers"] == 12
    assert chapter["participants"] == 1
    assert chapter["pitchCount"] == 12
    assert summary["memorable"]["submissions"][0]["participantLabel"] == "명승부팬"
    assert summary["memorable"]["submissions"][0]["version"] == version

    leaderboard = client.get(
        "/api/benchmark/memorable-leaderboard",
        query_string={"version": version},
    )
    assert leaderboard.status_code == 200
    assert leaderboard.json["participantCount"] == 1
    assert leaderboard.json["pitchCount"] == 12
    assert leaderboard.json["maxPoints"] == 36
    assert leaderboard.json["rows"][0]["nickname"] == "명승부팬"
    assert leaderboard.json["rows"][0]["rank"] == 1
    assert "id" not in leaderboard.json["rows"][0]
    assert (
        client.get(
            "/api/benchmark/memorable-leaderboard",
            query_string={"version": "memorable:unknown"},
        ).status_code
        == 400
    )

    invalid = _memorable_payload(version)
    invalid["answers"][0]["scenarioId"] = "brown-rodriguez"
    assert (
        client.post("/api/benchmark/submissions", json=invalid).status_code
        == 400
    )


def test_rejects_invalid_answers_and_rate_limits(tmp_path) -> None:
    client = _client(tmp_path, rate_limit=1)
    invalid = deepcopy(_payload())
    snell_answer = next(
        answer
        for answer in invalid["answers"]
        if answer["scenarioId"] == "snell-tatis"
    )
    snell_answer["selected"] = "MOVING_FASTBALL"

    assert (
        client.post("/api/benchmark/submissions", json=invalid).status_code == 400
    )
    assert client.post("/api/benchmark/submissions", json=_payload()).status_code == 429


def test_csv_uses_nickname_and_current_max_points(tmp_path) -> None:
    client = _client(tmp_path)
    response = client.post("/api/benchmark/submissions", json=_payload())
    assert response.status_code == 201

    csv_response = client.get(
        "/api/admin/benchmark/submissions.csv",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )

    assert csv_response.status_code == 200
    assert "야구팬" in csv_response.text
    assert "nickname" in csv_response.text
    assert "max_points" in csv_response.text
    assert "benchmark_version" in csv_response.text


def test_rejects_cross_origin_submission_when_origin_is_configured(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCHMARK_ALLOWED_ORIGIN", "https://pitchtest.example")
    client = _client(tmp_path)

    response = client.post(
        "/api/benchmark/submissions",
        json=_payload(),
        headers={"Origin": "https://attacker.example"},
    )

    assert response.status_code == 403


def test_reads_admin_token_from_protected_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token_file = tmp_path / "admin-token"
    token_file.write_text(f"{ADMIN_TOKEN}\n", encoding="utf-8")
    monkeypatch.setenv("BENCHMARK_ADMIN_TOKEN_FILE", str(token_file))
    app = create_app(database_path=str(tmp_path / "benchmark.sqlite3"))
    app.config["TESTING"] = True

    response = app.test_client().get(
        "/api/admin/benchmark/summary",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )

    assert response.status_code == 200


def test_rejects_unsafe_nickname(tmp_path) -> None:
    client = _client(tmp_path)
    payload = _payload()
    payload["nickname"] = "<script>"

    response = client.post("/api/benchmark/submissions", json=payload)

    assert response.status_code == 400
    assert "닉네임" in response.json["error"]


def test_migrates_existing_anonymous_database(tmp_path) -> None:
    database_path = tmp_path / "benchmark.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                public_id TEXT NOT NULL UNIQUE,
                experience TEXT NOT NULL,
                submitted_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO submissions (public_id, experience, submitted_at)
            VALUES ('legacy-id', 'casual', '2026-07-27T00:00:00+00:00')
            """
        )

    create_app(
        database_path=str(database_path),
        admin_token=ADMIN_TOKEN,
    )

    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(submissions)")
        }
        max_points = connection.execute(
            "SELECT max_points FROM submissions WHERE public_id = 'legacy-id'"
        ).fetchone()[0]
    assert "nickname" in columns
    assert "max_points" in columns
    assert "benchmark_version" in columns
    assert max_points == 105
