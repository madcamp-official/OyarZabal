from __future__ import annotations

from datetime import date, timedelta

from oyarzabal import daily_challenge


def test_uses_only_a_video_verified_date_within_seven_days(
    monkeypatch,
) -> None:
    source_date = date(2026, 7, 29)
    fetched_dates: list[date] = []
    checked_dates: list[str] = []
    attached_dates: list[str] = []

    def fetch_for_date(_challenge_date, candidate_date):
        fetched_dates.append(candidate_date)
        return [{"date": candidate_date.isoformat(), "pitches": []}]

    def videos_available(scenarios):
        checked_dates.append(scenarios[0]["date"])
        return scenarios[0]["date"] == (source_date - timedelta(days=6)).isoformat()

    def attach_predictions(scenarios):
        attached_dates.append(scenarios[0]["date"])

    monkeypatch.setattr(
        daily_challenge,
        "_fetch_scenarios_for_date",
        fetch_for_date,
    )
    monkeypatch.setattr(
        daily_challenge,
        "_all_videos_available",
        videos_available,
    )
    monkeypatch.setattr(
        daily_challenge,
        "_attach_model_predictions",
        attach_predictions,
    )

    scenarios = daily_challenge.fetch_daily_scenarios(
        date(2026, 7, 30),
        source_date,
    )

    expected_dates = [
        source_date - timedelta(days=days_ago)
        for days_ago in range(7)
    ]
    assert fetched_dates == expected_dates
    assert checked_dates == [item.isoformat() for item in expected_dates]
    assert scenarios[0]["date"] == expected_dates[-1].isoformat()
    assert attached_dates == [expected_dates[-1].isoformat()]


def test_checks_every_savant_play_id_and_rejects_a_missing_video(
    monkeypatch,
) -> None:
    checked: list[str] = []

    def video_available(play_id: str) -> bool:
        checked.append(play_id)
        return play_id != "missing"

    monkeypatch.setattr(
        daily_challenge,
        "sporty_video_available",
        video_available,
    )

    assert not daily_challenge._all_videos_available(
        [
            {
                "pitches": [
                    {"playId": "video-1"},
                    {"playId": "missing"},
                    {"playId": "video-2"},
                ]
            }
        ]
    )
    assert set(checked) == {"video-1", "missing", "video-2"}
