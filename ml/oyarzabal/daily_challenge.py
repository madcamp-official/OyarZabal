from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

PITCH_TYPES: dict[str, tuple[str, str]] = {
    "FF": ("FOUR_SEAM", "포심"),
    "FC": ("MOVING_FASTBALL", "커터"),
    "SI": ("MOVING_FASTBALL", "싱커"),
    "SL": ("SLIDER", "슬라이더"),
    "ST": ("SLIDER", "스위퍼"),
    "SV": ("SLIDER", "스위퍼"),
    "CU": ("CURVE", "커브"),
    "KC": ("CURVE", "너클 커브"),
    "CH": ("CHANGEUP", "체인지업"),
    "FS": ("SPLITTER_FORK", "스플리터"),
    "FO": ("SPLITTER_FORK", "포크볼"),
}

GROUP_FAMILY = {
    "FOUR_SEAM": "FASTBALL",
    "MOVING_FASTBALL": "FASTBALL",
    "SLIDER": "BREAKING",
    "CURVE": "BREAKING",
    "CHANGEUP": "OFFSPEED",
    "SPLITTER_FORK": "OFFSPEED",
}

GROUP_ORDER = {
    "FOUR_SEAM": 0,
    "MOVING_FASTBALL": 1,
    "SLIDER": 2,
    "CURVE": 3,
    "CHANGEUP": 4,
    "SPLITTER_FORK": 5,
}

SOURCE_LOOKBACK_DAYS = 7

OUTCOME_LABELS = {
    "strikeout": "삼진",
    "strikeout_double_play": "삼진 병살",
    "walk": "볼넷",
    "intent_walk": "고의4구",
    "hit_by_pitch": "몸에 맞는 공",
    "single": "안타",
    "double": "2루타",
    "triple": "3루타",
    "home_run": "홈런",
    "field_out": "범타",
    "force_out": "포스 아웃",
    "grounded_into_double_play": "병살타",
    "fielders_choice": "야수 선택",
    "fielders_choice_out": "야수 선택 아웃",
    "sac_fly": "희생 플라이",
    "sac_bunt": "희생 번트",
    "double_play": "병살타",
    "field_error": "실책 출루",
}

CALL_LABELS = {
    "B": "볼",
    "*B": "볼",
    "C": "루킹 스트라이크",
    "S": "헛스윙",
    "F": "파울",
    "T": "파울 팁",
    "X": "인플레이",
    "D": "인플레이",
    "E": "인플레이",
    "L": "파울 번트",
    "W": "피치아웃",
}


class DailySourceError(RuntimeError):
    pass


def _fetch_json(url: str, timeout: int = 20) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "OyarZabal-Daily-Pitch-Test/1.0",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _fetch_text(url: str, timeout: int = 20) -> str:
    request = Request(
        url,
        headers={
            "Accept": "text/html",
            "User-Agent": "OyarZabal-Daily-Pitch-Test/1.0",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def sporty_video_available(play_id: str) -> bool:
    html = _fetch_text(
        f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
    )
    return (
        'id="sporty"' in html
        and f"var playId = '{play_id}';" in html
    )


def _all_videos_available(scenarios: list[dict[str, Any]]) -> bool:
    play_ids = [
        pitch["playId"]
        for scenario in scenarios
        for pitch in scenario["pitches"]
    ]
    with ThreadPoolExecutor(max_workers=min(8, len(play_ids))) as executor:
        return all(executor.map(sporty_video_available, play_ids))


def _before_bases(
    play: dict[str, Any],
    previous_play: dict[str, Any] | None,
) -> list[bool]:
    if not previous_play:
        return [False, False, False]
    about = play.get("about", {})
    previous_about = previous_play.get("about", {})
    if (
        about.get("inning") != previous_about.get("inning")
        or about.get("halfInning") != previous_about.get("halfInning")
    ):
        return [False, False, False]
    previous_matchup = previous_play.get("matchup", {})
    return [
        bool(previous_matchup.get("postOnFirst")),
        bool(previous_matchup.get("postOnSecond")),
        bool(previous_matchup.get("postOnThird")),
    ]


def _before_score(previous_play: dict[str, Any] | None) -> tuple[int, int]:
    if not previous_play:
        return 0, 0
    result = previous_play.get("result", {})
    return int(result.get("awayScore", 0)), int(result.get("homeScore", 0))


def _pitcher_repertoires(
    plays: list[dict[str, Any]],
) -> dict[int, dict[str, list[str]]]:
    repertoires: dict[int, dict[str, list[str]]] = {}
    for play in plays:
        pitcher_id = play.get("matchup", {}).get("pitcher", {}).get("id")
        if type(pitcher_id) is not int:
            continue
        repertoire = repertoires.setdefault(
            pitcher_id,
            {"FASTBALL": [], "BREAKING": [], "OFFSPEED": []},
        )
        for event in play.get("playEvents", []):
            if not event.get("isPitch"):
                continue
            pitch_code = event.get("details", {}).get("type", {}).get("code")
            if pitch_code not in PITCH_TYPES:
                continue
            group, raw_label = PITCH_TYPES[pitch_code]
            family = GROUP_FAMILY[group]
            if raw_label not in repertoire[family]:
                repertoire[family].append(raw_label)
    return repertoires


def extract_game_candidates(
    feed: dict[str, Any],
    *,
    game_pk: int,
    source_date: date,
) -> list[dict[str, Any]]:
    game_data = feed.get("gameData", {})
    teams = game_data.get("teams", {})
    away_team = teams.get("away", {}).get("abbreviation") or "AWAY"
    home_team = teams.get("home", {}).get("abbreviation") or "HOME"
    plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])
    if not isinstance(plays, list):
        return []
    repertoires = _pitcher_repertoires(plays)
    candidates: list[dict[str, Any]] = []
    previous_play: dict[str, Any] | None = None

    for play_index, play in enumerate(plays):
        matchup = play.get("matchup", {})
        pitcher = matchup.get("pitcher", {})
        batter = matchup.get("batter", {})
        pitcher_id = pitcher.get("id")
        play_events = [
            event for event in play.get("playEvents", []) if event.get("isPitch")
        ]
        supported_events = [
            (event_index, event)
            for event_index, event in enumerate(play.get("playEvents", []))
            if event.get("isPitch")
            if event.get("details", {}).get("type", {}).get("code") in PITCH_TYPES
            and isinstance(event.get("playId"), str)
        ]
        if (
            type(pitcher_id) is not int
            or len(play_events) != len(supported_events)
            or not 3 <= len(supported_events) <= 6
            or not play.get("about", {}).get("isComplete")
        ):
            previous_play = play
            continue

        pitch_rows: list[dict[str, Any]] = []
        balls = 0
        strikes = 0
        for pitch_number, (event_index, event) in enumerate(
            supported_events,
            start=1,
        ):
            details = event["details"]
            pitch_code = details["type"]["code"]
            group, raw_label = PITCH_TYPES[pitch_code]
            call_code = details.get("call", {}).get("code", details.get("code", ""))
            pitch_rows.append(
                {
                    "number": pitch_number,
                    "count": f"{balls}–{strikes}",
                    "actual": group,
                    "rawLabel": raw_label,
                    "result": CALL_LABELS.get(
                        call_code,
                        details.get("description") or "투구",
                    ),
                    "playId": event["playId"],
                    "_eventIndex": event_index,
                }
            )
            count = event.get("count", {})
            balls = min(int(count.get("balls", balls)), 3)
            strikes = min(int(count.get("strikes", strikes)), 2)

        repertoire = repertoires.get(
            pitcher_id,
            {"FASTBALL": [], "BREAKING": [], "OFFSPEED": []},
        )
        available_groups = sorted(
            {
                PITCH_TYPES[event["details"]["type"]["code"]][0]
                for candidate_play in plays
                if candidate_play.get("matchup", {}).get("pitcher", {}).get("id")
                == pitcher_id
                for event in candidate_play.get("playEvents", [])
                if event.get("isPitch")
                and event.get("details", {}).get("type", {}).get("code")
                in PITCH_TYPES
            },
            key=GROUP_ORDER.__getitem__,
        )
        about = play.get("about", {})
        away_score, home_score = _before_score(previous_play)
        inning_half = "초" if about.get("isTopInning") else "말"
        result = play.get("result", {})
        first_pitch = supported_events[0][1]
        candidates.append(
            {
                "id": f"daily-{game_pk}-{about.get('atBatIndex', len(candidates))}",
                "pitcher": pitcher.get("fullName", "Unknown Pitcher"),
                "pitcherId": pitcher_id,
                "batter": batter.get("fullName", "Unknown Batter"),
                "date": source_date.isoformat(),
                "inning": f"{about.get('inning', 1)}회 {inning_half}",
                "awayTeam": away_team,
                "homeTeam": home_team,
                "awayScore": away_score,
                "homeScore": home_score,
                "outs": int(first_pitch.get("count", {}).get("outs", 0)),
                "bases": _before_bases(play, previous_play),
                "outcome": OUTCOME_LABELS.get(
                    result.get("eventType"),
                    result.get("event") or "타석 완료",
                ),
                "outcomeType": result.get("eventType", ""),
                "gameUrl": f"https://www.mlb.com/gameday/{game_pk}",
                "repertoire": repertoire,
                "availableGroups": available_groups,
                "pitches": pitch_rows,
                "captivatingIndex": int(about.get("captivatingIndex", 0) or 0),
                "_gamePk": game_pk,
                "_playIndex": play_index,
            }
        )
        previous_play = play
    return candidates


def _selection_key(
    candidate: dict[str, Any],
    challenge_date: date,
) -> tuple[int, int, str]:
    digest = hashlib.sha256(
        f"{challenge_date.isoformat()}:{candidate['id']}".encode()
    ).hexdigest()
    return (
        abs(len(candidate["pitches"]) - 5),
        -int(candidate.get("captivatingIndex", 0)),
        digest,
    )


def select_daily_scenarios(
    candidates: list[dict[str, Any]],
    *,
    challenge_date: date,
) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda item: _selection_key(item, challenge_date))
    if not ordered:
        raise DailySourceError("조건에 맞는 전날 타석을 찾지 못했습니다.")
    first = ordered[0]
    first_bucket = (
        "strikeout"
        if "strikeout" in first.get("outcomeType", "")
        else "ball_in_play"
    )
    second = next(
        (
            candidate
            for candidate in ordered[1:]
            if candidate["pitcherId"] != first["pitcherId"]
            and (
                "strikeout"
                if "strikeout" in candidate.get("outcomeType", "")
                else "ball_in_play"
            )
            != first_bucket
        ),
        None,
    )
    if second is None:
        second = next(
            (
                candidate
                for candidate in ordered[1:]
                if candidate["pitcherId"] != first["pitcherId"]
            ),
            None,
        )
    if second is None:
        raise DailySourceError("서로 다른 투수의 타석 세 개를 찾지 못했습니다.")

    third = next(
        (
            candidate
            for candidate in ordered[1:]
            if candidate["pitcherId"]
            not in {first["pitcherId"], second["pitcherId"]}
        ),
        None,
    )
    if third is None:
        raise DailySourceError("서로 다른 투수의 타석 세 개를 찾지 못했습니다.")

    selected = [first, second, third]
    for scenario in selected:
        scenario.pop("captivatingIndex", None)
        scenario.pop("outcomeType", None)
        scenario.pop("pitcherId", None)
    return selected


def _attach_model_predictions(scenarios: list[dict[str, Any]]) -> None:
    from oyarzabal.live import HistoryStore, LivePredictor, build_live_view
    from oyarzabal.live_api import _replay_payload

    history_directories = tuple(
        Path(value)
        for value in os.environ.get(
            "OYARZABAL_HISTORY_DIRS",
            os.pathsep.join(
                ("data/raw/statcast-v8", "data/holdout/statcast-v8-2026")
            ),
        ).split(os.pathsep)
        if value
    )
    history = HistoryStore(history_directories)
    predictor = LivePredictor(
        Path(os.environ.get("OYARZABAL_MODEL_DIR", "models/v8.4")),
        Path(os.environ.get("OYARZABAL_GLOBAL_MODEL_DIR", "models/v7.2")),
    )
    feeds = {
        game_pk: _fetch_json(
            f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        )
        for game_pk in {int(scenario["_gamePk"]) for scenario in scenarios}
    }
    for scenario in scenarios:
        game_pk = int(scenario.pop("_gamePk"))
        play_index = int(scenario.pop("_playIndex"))
        feed = feeds[game_pk]
        for pitch in scenario["pitches"]:
            event_index = int(pitch.pop("_eventIndex"))
            view = build_live_view(
                _replay_payload(
                    feed,
                    (play_index, event_index, pitch["actual"]),
                ),
                game_pk,
                history,
            )
            if view.feature_row is None or view.sequence_examples is None:
                raise DailySourceError(
                    "모델 예측용 투구 직전 상태를 만들지 못했습니다."
                )
            prediction = predictor.predict(
                view.feature_row,
                view.sequence_examples,
            )
            available_probabilities = {
                group: float(probability)
                for group, probability in prediction["probabilities"].items()
                if group in scenario["availableGroups"]
            }
            total = sum(available_probabilities.values())
            probabilities = {
                group: round(probability / total, 6)
                for group, probability in available_probabilities.items()
            }
            selected = max(probabilities, key=probabilities.__getitem__)
            pitch["modelPrediction"] = {
                "selected": selected,
                "confidence": probabilities[selected],
                "probabilities": probabilities,
                "modelVersion": prediction["modelVersion"],
            }


def _fetch_scenarios_for_date(
    challenge_date: date,
    source_date: date,
) -> list[dict[str, Any]]:
    schedule = _fetch_json(
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={source_date.isoformat()}"
    )
    games = [
        game
        for date_row in schedule.get("dates", [])
        for game in date_row.get("games", [])
        if game.get("status", {}).get("detailedState") == "Final"
        and game.get("officialDate") == source_date.isoformat()
    ]
    games.sort(
        key=lambda game: hashlib.sha256(
            f"{challenge_date.isoformat()}:{game['gamePk']}".encode()
        ).hexdigest()
    )
    candidates: list[dict[str, Any]] = []
    for game in games:
        game_pk = int(game["gamePk"])
        feed = _fetch_json(
            f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        )
        candidates.extend(
            extract_game_candidates(
                feed,
                game_pk=game_pk,
                source_date=source_date,
            )
        )
        if len({candidate["pitcherId"] for candidate in candidates}) >= 4:
            break
    return select_daily_scenarios(candidates, challenge_date=challenge_date)


def fetch_daily_scenarios(
    challenge_date: date,
    source_date: date,
) -> list[dict[str, Any]]:
    errors: list[str] = []
    for days_ago in range(SOURCE_LOOKBACK_DAYS):
        candidate_date = source_date - timedelta(days=days_ago)
        try:
            scenarios = _fetch_scenarios_for_date(
                challenge_date,
                candidate_date,
            )
            if _all_videos_available(scenarios):
                _attach_model_predictions(scenarios)
                return scenarios
            errors.append(f"{candidate_date}: Savant 영상 준비 중")
        except Exception as error:
            errors.append(f"{candidate_date}: {error}")
    raise DailySourceError(
        f"최근 {SOURCE_LOOKBACK_DAYS}일 안에 영상이 모두 준비된 "
        "세 타석을 찾지 못했습니다. "
        + "; ".join(errors)
    )
