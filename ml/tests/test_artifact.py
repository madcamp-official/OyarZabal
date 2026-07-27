import json
from pathlib import Path

import pytest
from oyarzabal.taxonomy import PITCH_FAMILIES, PITCH_GROUPS


def test_static_replay_uses_schema_v7_and_preserves_pitch_order() -> None:
    root = Path(__file__).parents[2] / "web/public/data"
    manifest = json.loads((root / "manifest.json").read_text())
    game = json.loads((root / "games/775300.json").read_text())

    assert manifest["schemaVersion"] == 7
    assert manifest["deploymentStatus"] == "shadow"
    assert tuple(manifest["pitchGroups"]) == tuple(str(group) for group in PITCH_GROUPS)
    assert tuple(manifest["pitchFamilies"]) == tuple(
        str(family) for family in PITCH_FAMILIES
    )
    assert game["pitchCount"] == 307
    assert [pitch["sequence"] for pitch in game["pitches"]] == list(range(1, 308))
    for pitch in game["pitches"]:
        assert pitch["modelSource"]["type"] in {
            "global",
            "pooled-residual",
            "provisional-residual",
            "reliability-gated-residual",
        }
        assert "pitcherReliability" in pitch["modelSource"]
        assert "contextGate" in pitch["modelSource"]
        assert "effectiveScale" in pitch["modelSource"]
        assert pitch["modelSource"]["registryTier"] in {
            "full",
            "limited",
            "shadow",
        }
        assert "scaleMultiplier" in pitch["modelSource"]
        for prediction in pitch["predictions"].values():
            assert sum(prediction["probabilities"].values()) == pytest.approx(
                1, abs=1e-5
            )
    for metrics in game["metrics"].values():
        assert metrics["hierarchicalAccuracy"] == pytest.approx(
            (metrics["accuracy"] + metrics["familyAccuracy"]) / 2
        )
