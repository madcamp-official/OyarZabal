from __future__ import annotations

import json
import os
from datetime import timedelta

from oyarzabal.benchmark_api import create_app
from oyarzabal.daily_api import seoul_today


def generation_payload() -> dict[str, str]:
    challenge_date = os.environ.get("DAILY_CHALLENGE_DATE")
    offset_days = int(os.environ.get("DAILY_CHALLENGE_OFFSET_DAYS", "0"))
    if not challenge_date and offset_days:
        challenge_date = (
            seoul_today() + timedelta(days=offset_days)
        ).isoformat()
    return {
        key: value
        for key, value in {
            "challengeDate": challenge_date,
            "sourceDate": os.environ.get("DAILY_SOURCE_DATE"),
        }.items()
        if value
    }


def main() -> None:
    app = create_app()
    token = app.config["ADMIN_TOKEN"]
    if len(token) < 32:
        raise SystemExit("BENCHMARK_ADMIN_TOKEN must contain at least 32 characters")
    payload = generation_payload()
    with app.test_client() as client:
        response = client.post(
            "/api/admin/daily/generate",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
    print(json.dumps(response.get_json(), ensure_ascii=False))
    if response.status_code not in {200, 201}:
        raise SystemExit(1)
