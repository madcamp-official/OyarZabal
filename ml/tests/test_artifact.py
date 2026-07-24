import json
from pathlib import Path

import pytest


def test_static_replay_uses_schema_v3_and_preserves_pitch_order() -> None:
    root = Path(__file__).parents[2] / "web/public/data"
    manifest = json.loads((root / "manifest.json").read_text())
    game = json.loads((root / "games/775300.json").read_text())

    assert manifest["schemaVersion"] == 3
    assert len(manifest["pitchGroups"]) == 6
    assert game["pitchCount"] == 307
    assert [pitch["sequence"] for pitch in game["pitches"]] == list(range(1, 308))
    for pitch in game["pitches"]:
        assert pitch["modelSource"]["type"] in {"global", "hybrid"}
        for prediction in pitch["predictions"].values():
            assert sum(prediction["probabilities"].values()) == pytest.approx(
                1, abs=1e-5
            )
