from __future__ import annotations

from datetime import date

from oyarzabal import daily_generate


def test_builds_tomorrows_challenge_for_the_pre_midnight_timer(
    monkeypatch,
) -> None:
    monkeypatch.delenv("DAILY_CHALLENGE_DATE", raising=False)
    monkeypatch.delenv("DAILY_SOURCE_DATE", raising=False)
    monkeypatch.setenv("DAILY_CHALLENGE_OFFSET_DAYS", "1")
    monkeypatch.setattr(
        daily_generate,
        "seoul_today",
        lambda: date(2026, 7, 30),
    )

    assert daily_generate.generation_payload() == {
        "challengeDate": "2026-07-31"
    }
