from __future__ import annotations

import copy
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import oyarzabal.live_api as live_api
import pandas as pd
import pytest
from oyarzabal.live import HistoryStore, LivePredictor, build_live_view
from oyarzabal.live_api import _replay_payload, _replay_targets, create_app


class FakePredictor:
    def predict(
        self,
        _rows: pd.DataFrame,
        _sequence: object,
    ) -> dict[str, object]:
        return {
            "modelVersion": "test",
            "topPitch": "FOUR_SEAM",
            "confidence": 0.4,
            "probabilities": {
                "FOUR_SEAM": 0.4,
                "MOVING_FASTBALL": 0.2,
                "SLIDER": 0.15,
                "CURVE": 0.1,
                "CHANGEUP": 0.1,
                "SPLITTER_FORK": 0.05,
            },
            "source": "global",
            "routing": {},
        }


class AdvancingPredictor(FakePredictor):
    def __init__(self, current: dict[str, object]):
        self.current = current
        self.calls = 0

    def predict(
        self,
        rows: pd.DataFrame,
        sequence: object,
    ) -> dict[str, object]:
        self.calls += 1
        if self.calls == 1:
            updated = live_feed()
            play = updated["liveData"]["plays"]["currentPlay"]
            play["playEvents"].append(
                pitch_event(
                    index=1,
                    pitch_number=2,
                    code="SL",
                    balls=1,
                    strikes=1,
                )
            )
            play["count"] = {"balls": 1, "strikes": 1, "outs": 0}
            self.current["value"] = updated
        return super().predict(rows, sequence)


def history() -> HistoryStore:
    rows = pd.DataFrame(
        [
            {
                "pitch_type": code,
                "game_date": "2024-10-24",
                "game_pk": 1,
                "at_bat_number": index + 1,
                "pitch_number": 1,
                "pitcher": 10,
                "batter": 100 + index,
                "balls": index % 4,
                "strikes": index % 3,
                "stand": "R",
                "p_throws": "R",
            }
            for index, code in enumerate(
                ["FF", "SI", "SL", "CU", "CH", "FS"] * 5
            )
        ]
    )
    return HistoryStore.from_frame(rows)


def pitch_event(
    *,
    index: int,
    pitch_number: int,
    code: str,
    balls: int,
    strikes: int,
) -> dict[str, object]:
    return {
        "index": index,
        "isPitch": True,
        "pitchNumber": pitch_number,
        "playId": f"play-{index}",
        "details": {
            "code": "B" if balls else "C",
            "description": "Ball" if balls else "Called Strike",
            "type": {"code": code, "description": code},
        },
        "count": {"balls": balls, "strikes": strikes, "outs": 0},
        "pitchData": {
            "startSpeed": 95 + index,
            "coordinates": {"pX": 0.1 * index, "pZ": 2.5},
        },
    }


def live_feed() -> dict[str, object]:
    current_play = {
        "atBatIndex": 0,
        "about": {
            "atBatIndex": 0,
            "inning": 1,
            "isTopInning": True,
            "isComplete": False,
        },
        "matchup": {
            "pitcher": {"id": 10, "fullName": "Test Pitcher"},
            "batter": {"id": 20, "fullName": "Test Batter"},
            "pitchHand": {"code": "R"},
            "batSide": {"code": "L"},
        },
        "count": {"balls": 1, "strikes": 0, "outs": 0},
        "playEvents": [
            pitch_event(
                index=0,
                pitch_number=1,
                code="FF",
                balls=1,
                strikes=0,
            )
        ],
    }
    return {
        "gameData": {
            "datetime": {"officialDate": "2024-10-25"},
            "status": {
                "abstractGameState": "Live",
                "detailedState": "In Progress",
            },
            "teams": {
                "away": {"name": "Away"},
                "home": {"name": "Home"},
            },
        },
        "liveData": {
            "plays": {
                "allPlays": [current_play],
                "currentPlay": current_play,
            },
            "linescore": {
                "currentInning": 1,
                "outs": 0,
                "teams": {
                    "away": {"runs": 0},
                    "home": {"runs": 0},
                },
                "offense": {"first": {"id": 99}},
                "defense": {"catcher": {"id": 30}},
            },
        },
    }


def test_live_view_builds_next_pitch_features() -> None:
    feed = live_feed()
    store = history()
    view = build_live_view(feed, 777777, store)

    assert view.context is not None
    assert view.context.target_sequence == 2
    assert view.context.next_pitch_number == 2
    assert view.context.balls == 1
    assert view.context.bases == (True, False, False)
    assert len(view.pitches) == 1
    assert view.feature_row is not None
    assert view.sequence_examples is not None
    assert view.sequence_examples.current_numeric.shape[1] == 127
    row = view.feature_row.iloc[0]
    assert row["game_pitch_count"] == 1
    assert row["prev_release_speed"] == 95
    assert row["pa_prev_pitch_1"] == "FOUR_SEAM"
    assert row["n_thruorder_pitcher"] == 1
    repeated = build_live_view(feed, 777777, store)
    assert repeated.sequence_examples is view.sequence_examples


def test_live_predictor_uses_full_v72_residual_pipeline() -> None:
    model_directory = Path("models/v7.2")
    if not (model_directory / "registry.json").exists():
        pytest.skip("requires local V7.2 model artifacts")

    view = build_live_view(live_feed(), 777777, history())
    rows = view.feature_row.copy()
    rows["pitcher"] = "571945"
    rows["pitcher_id"] = 571945
    rows["count_support"] = 1_000
    rows["stand_support"] = 1_000
    rows["transition_support"] = 1_000

    prediction = LivePredictor(model_directory).predict(rows, None)

    assert prediction["modelVersion"] == "V7.2"
    assert prediction["source"] == "reliability-gated-residual"
    assert prediction["routing"]["pipeline"] == "v7.2-residual"
    assert prediction["routing"]["effectiveScale"] > 0


def test_live_api_uses_a_fixed_eight_second_server_window(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        live_api,
        "_utcnow",
        lambda: datetime.fromisoformat("2026-07-30T00:00:00+00:00"),
    )
    monkeypatch.setenv("OYARZABAL_LIVE_WINDOW_SECONDS", "99")
    app = create_app(
        database_path=str(tmp_path / "live.sqlite3"),
        history=history(),
        predictor=FakePredictor(),
        fetcher=lambda _game_pk: copy.deepcopy(live_feed()),
    )

    round_value = app.test_client().get(
        f"/api/live/games/777777?sessionId={uuid.uuid4()}"
    ).get_json()["round"]

    assert (
        datetime.fromisoformat(round_value["deadline"])
        - datetime.fromisoformat(round_value["openedAt"])
    ).total_seconds() == 8


def test_live_api_accepts_answer_and_scores_next_pitch(tmp_path) -> None:
    current = {"value": live_feed()}

    def fetcher(_game_pk: int) -> dict[str, object]:
        return copy.deepcopy(current["value"])

    app = create_app(
        database_path=str(tmp_path / "live.sqlite3"),
        history=history(),
        predictor=FakePredictor(),
        fetcher=fetcher,
        window_seconds=60,
    )
    client = app.test_client()
    session_id = str(uuid.uuid4())

    snapshot = client.get(
        f"/api/live/games/777777?sessionId={session_id}"
    ).get_json()
    assert snapshot["round"]["status"] == "open"
    assert snapshot["round"]["prediction"] is None
    insights = snapshot["round"]["context"]["insights"]
    assert set(insights["rates"]) == {
        "season",
        "recent100",
        "game",
        "count",
        "stand",
    }
    assert sum(insights["rates"]["season"].values()) == pytest.approx(1)
    assert set(insights["repertoire"]) == {
        "FOUR_SEAM",
        "MOVING_FASTBALL",
        "SLIDER",
        "CURVE",
        "CHANGEUP",
        "SPLITTER_FORK",
    }
    assert insights["workload"]["gamePitches"] == 1
    round_id = snapshot["round"]["id"]

    answer = client.post(
        f"/api/live/rounds/{round_id}/answers",
        json={
            "sessionId": session_id,
            "selectedPitch": "SLIDER",
        },
    )
    assert answer.status_code == 200
    assert answer.get_json()["status"] == "draft"
    assert "prediction" not in answer.get_json()

    updated = live_feed()
    play = updated["liveData"]["plays"]["currentPlay"]
    play["playEvents"].append(
        pitch_event(
            index=1,
            pitch_number=2,
            code="SL",
            balls=1,
            strikes=1,
        )
    )
    play["count"] = {"balls": 1, "strikes": 1, "outs": 0}
    current["value"] = updated

    result = client.get(
        f"/api/live/games/777777?sessionId={session_id}"
    ).get_json()
    assert result["lastResult"]["roundId"] == round_id
    assert result["lastResult"]["selectedPitch"] == "SLIDER"
    assert result["lastResult"]["actualPitch"] == "SLIDER"
    assert result["lastResult"]["prediction"]["topPitch"] == "FOUR_SEAM"
    assert result["round"]["context"]["inning"] == 1
    assert result["round"]["context"]["count"] == {
        "balls": 1,
        "strikes": 1,
    }


def test_live_api_updates_draft_until_the_server_deadline(
    tmp_path,
    monkeypatch,
) -> None:
    feed = live_feed()
    clock = {
        "now": datetime.fromisoformat("2026-07-30T00:00:00+00:00"),
    }
    monkeypatch.setattr(live_api, "_utcnow", lambda: clock["now"])
    app = create_app(
        database_path=str(tmp_path / "live.sqlite3"),
        history=history(),
        predictor=FakePredictor(),
        fetcher=lambda _game_pk: copy.deepcopy(feed),
        window_seconds=6,
    )
    client = app.test_client()
    session_id = str(uuid.uuid4())
    round_id = client.get(
        f"/api/live/games/777777?sessionId={session_id}"
    ).get_json()["round"]["id"]
    payload = {
        "sessionId": session_id,
        "selectedPitch": "FOUR_SEAM",
    }

    first = client.post(
        f"/api/live/rounds/{round_id}/answers",
        json=payload,
    )
    assert first.status_code in {200, 201}
    assert "prediction" not in first.get_json()
    open_round = client.get(
        f"/api/live/games/777777?sessionId={session_id}"
    ).get_json()["round"]
    assert open_round["status"] == "open"
    assert open_round["answer"]["selectedPitch"] == "FOUR_SEAM"
    assert open_round["prediction"] is None

    clock["now"] += timedelta(seconds=2)
    updated = client.post(
        f"/api/live/rounds/{round_id}/answers",
        json={
            "sessionId": session_id,
            "selectedPitch": "SLIDER",
        },
    )
    assert updated.status_code in {200, 201}
    open_round = client.get(
        f"/api/live/games/777777?sessionId={session_id}"
    ).get_json()["round"]
    assert open_round["answer"]["selectedPitch"] == "SLIDER"
    assert open_round["prediction"] is None

    clock["now"] += timedelta(seconds=4)
    locked_round = client.get(
        f"/api/live/games/777777?sessionId={session_id}"
    ).get_json()["round"]
    assert locked_round["status"] == "locked"
    assert locked_round["answer"]["selectedPitch"] == "SLIDER"
    assert locked_round["prediction"]["topPitch"] == "FOUR_SEAM"
    assert client.post(
        f"/api/live/rounds/{round_id}/answers",
        json=payload,
    ).status_code == 409


def test_live_api_locks_a_draft_when_the_pitch_arrives_early(tmp_path) -> None:
    current = {"value": live_feed()}
    app = create_app(
        database_path=str(tmp_path / "live.sqlite3"),
        history=history(),
        predictor=FakePredictor(),
        fetcher=lambda _game_pk: copy.deepcopy(current["value"]),
        window_seconds=60,
    )
    client = app.test_client()
    session_id = str(uuid.uuid4())
    round_id = client.get(
        f"/api/live/games/777777?sessionId={session_id}"
    ).get_json()["round"]["id"]
    assert client.post(
        f"/api/live/rounds/{round_id}/answers",
        json={
            "sessionId": session_id,
            "selectedPitch": "FOUR_SEAM",
        },
    ).status_code == 200

    updated = live_feed()
    play = updated["liveData"]["plays"]["currentPlay"]
    play["playEvents"].append(
        pitch_event(
            index=1,
            pitch_number=2,
            code="SL",
            balls=1,
            strikes=1,
        )
    )
    play["count"] = {"balls": 1, "strikes": 1, "outs": 0}
    current["value"] = updated

    rejected = client.post(
        f"/api/live/rounds/{round_id}/answers",
        json={
            "sessionId": session_id,
            "selectedPitch": "SLIDER",
        },
    )
    assert rejected.status_code == 409
    snapshot = client.get(
        f"/api/live/games/777777?sessionId={session_id}"
    ).get_json()
    assert snapshot["lastResult"]["roundId"] == round_id
    assert snapshot["lastResult"]["selectedPitch"] == "FOUR_SEAM"
    assert snapshot["lastResult"]["actualPitch"] == "SLIDER"


def test_live_api_scores_an_unselected_result_as_an_attempt(tmp_path) -> None:
    current = {"value": live_feed()}
    app = create_app(
        database_path=str(tmp_path / "live.sqlite3"),
        history=history(),
        predictor=FakePredictor(),
        fetcher=lambda _game_pk: copy.deepcopy(current["value"]),
        window_seconds=60,
    )
    client = app.test_client()
    session_id = str(uuid.uuid4())
    round_id = client.get(
        f"/api/live/games/777777?sessionId={session_id}"
    ).get_json()["round"]["id"]

    updated = live_feed()
    play = updated["liveData"]["plays"]["currentPlay"]
    play["playEvents"].append(
        pitch_event(
            index=1,
            pitch_number=2,
            code="SL",
            balls=1,
            strikes=1,
        )
    )
    play["count"] = {"balls": 1, "strikes": 1, "outs": 0}
    current["value"] = updated

    last_result = client.get(
        f"/api/live/games/777777?sessionId={session_id}"
    ).get_json()["lastResult"]
    assert last_result["roundId"] == round_id
    assert last_result["selectedPitch"] is None
    assert last_result["actualPitch"] == "SLIDER"


def test_live_api_does_not_score_rounds_a_session_never_saw(tmp_path) -> None:
    current = {"value": live_feed()}
    app = create_app(
        database_path=str(tmp_path / "live.sqlite3"),
        history=history(),
        predictor=FakePredictor(),
        fetcher=lambda _game_pk: copy.deepcopy(current["value"]),
        window_seconds=60,
    )
    client = app.test_client()
    client.get(f"/api/live/games/777777?sessionId={uuid.uuid4()}")

    updated = live_feed()
    play = updated["liveData"]["plays"]["currentPlay"]
    play["playEvents"].append(
        pitch_event(
            index=1,
            pitch_number=2,
            code="SL",
            balls=1,
            strikes=1,
        )
    )
    play["count"] = {"balls": 1, "strikes": 1, "outs": 0}
    current["value"] = updated

    late_session = str(uuid.uuid4())
    snapshot = client.get(
        f"/api/live/games/777777?sessionId={late_session}"
    ).get_json()
    assert snapshot["lastResult"] is None


def test_live_api_reports_cancelled_round_reason(tmp_path) -> None:
    current = {"value": live_feed()}
    app = create_app(
        database_path=str(tmp_path / "live.sqlite3"),
        history=history(),
        predictor=FakePredictor(),
        fetcher=lambda _game_pk: copy.deepcopy(current["value"]),
        window_seconds=60,
    )
    client = app.test_client()
    session_id = str(uuid.uuid4())
    first = client.get(
        f"/api/live/games/777777?sessionId={session_id}"
    ).get_json()

    moved = live_feed()
    moved["liveData"]["linescore"]["offense"] = {"second": {"id": 99}}
    current["value"] = moved
    state_changed = client.get(
        f"/api/live/games/777777?sessionId={session_id}"
    ).get_json()
    assert state_changed["round"]["id"] != first["round"]["id"]
    assert state_changed["roundNotice"] == {"type": "stateChanged"}

    play = moved["liveData"]["plays"]["currentPlay"]
    play["playEvents"].append(
        pitch_event(
            index=1,
            pitch_number=2,
            code="KN",
            balls=1,
            strikes=1,
        )
    )
    play["count"] = {"balls": 1, "strikes": 1, "outs": 0}
    current["value"] = moved
    unsupported = client.get(
        f"/api/live/games/777777?sessionId={session_id}"
    ).get_json()
    assert unsupported["roundNotice"] == {
        "type": "unsupportedPitch",
        "rawCode": "KN",
    }


def test_live_snapshot_does_not_return_state_that_advanced_during_prediction(
    tmp_path,
) -> None:
    current = {"value": live_feed()}

    def fetcher(_game_pk: int) -> dict[str, object]:
        return copy.deepcopy(current["value"])

    app = create_app(
        database_path=str(tmp_path / "live.sqlite3"),
        history=history(),
        predictor=AdvancingPredictor(current),
        fetcher=fetcher,
        window_seconds=60,
    )

    snapshot = app.test_client().get(
        f"/api/live/games/777777?sessionId={uuid.uuid4()}"
    ).get_json()

    assert snapshot["round"]["context"]["pitchNumber"] == 3
    assert snapshot["round"]["context"]["count"] == {
        "balls": 1,
        "strikes": 1,
    }


def test_replay_progress_is_kept_per_session_and_game(tmp_path) -> None:
    feed = live_feed()
    play = feed["liveData"]["plays"]["currentPlay"]
    play["playEvents"].append(
        pitch_event(
            index=1,
            pitch_number=2,
            code="SL",
            balls=1,
            strikes=1,
        )
    )
    play["about"]["isComplete"] = True
    feed["gameData"]["status"] = {
        "abstractGameState": "Final",
        "detailedState": "Final",
    }
    app = create_app(
        database_path=str(tmp_path / "live.sqlite3"),
        history=history(),
        predictor=FakePredictor(),
        fetcher=lambda _game_pk: copy.deepcopy(feed),
        window_seconds=60,
    )
    client = app.test_client()
    session_id = str(uuid.uuid4())

    first = client.post(
        "/api/replay/sessions",
        json={"sessionId": session_id, "gamePk": 1001},
    ).get_json()
    assert first["round"]["pitchNumber"] == 1
    assert "insights" in first["round"]["context"]
    round_id = first["round"]["id"]
    answer = client.post(
        f"/api/replay/rounds/{round_id}/answers",
        json={"sessionId": session_id, "selectedPitch": "FOUR_SEAM"},
    )
    assert answer.status_code == 201
    assert "prediction" not in answer.get_json()
    locked = client.get(
        f"/api/replay/games/1001?sessionId={session_id}"
    ).get_json()["round"]
    assert locked["status"] == "locked"
    assert locked["prediction"]["topPitch"] == "FOUR_SEAM"
    advanced = client.post(
        f"/api/replay/sessions/{session_id}/advance",
        json={"gamePk": 1001},
    ).get_json()
    assert advanced["round"]["pitchNumber"] == 2

    assert client.post(
        "/api/replay/sessions",
        json={"sessionId": session_id, "gamePk": 1002},
    ).status_code == 200
    resumed = client.get(
        f"/api/replay/games/1001?sessionId={session_id}"
    )
    assert resumed.status_code == 200
    assert resumed.get_json()["round"]["pitchNumber"] == 2


def test_replay_payload_updates_the_inning() -> None:
    feed = live_feed()
    first_play = feed["liveData"]["plays"]["currentPlay"]
    first_play["about"]["isComplete"] = True
    second_play = copy.deepcopy(first_play)
    second_play["atBatIndex"] = 1
    second_play["about"].update({"atBatIndex": 1, "inning": 2})
    feed["liveData"]["plays"] = {
        "allPlays": [first_play, second_play],
        "currentPlay": second_play,
    }

    targets = _replay_targets(feed)
    first_payload = _replay_payload(feed, targets[0])
    second_payload = _replay_payload(feed, targets[1])

    assert first_payload["liveData"]["linescore"]["currentInning"] == 1
    assert second_payload["liveData"]["linescore"]["currentInning"] == 2
    assert second_payload["liveData"]["plays"]["currentPlay"]["about"][
        "inning"
    ] == 2


def test_replay_payload_reconstructs_runners_before_pitch() -> None:
    feed = live_feed()
    first_play = feed["liveData"]["plays"]["currentPlay"]
    first_play["about"]["isComplete"] = True
    first_play["runners"] = [
        {
            "movement": {
                "start": None,
                "end": "1B",
                "isOut": False,
            },
            "details": {
                "runner": {"id": 99},
                "playIndex": 0,
            },
        }
    ]
    second_play = copy.deepcopy(first_play)
    second_play["atBatIndex"] = 1
    second_play["about"]["atBatIndex"] = 1
    second_play["runners"] = []
    feed["liveData"]["plays"] = {
        "allPlays": [first_play, second_play],
        "currentPlay": second_play,
    }

    target = _replay_targets(feed)[1]
    replay = _replay_payload(feed, target)

    assert replay["liveData"]["linescore"]["offense"] == {
        "first": {"id": 99}
    }
