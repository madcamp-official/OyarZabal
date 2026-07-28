import json
from pathlib import Path


def test_v81_artifact_freezes_selection_and_preserves_row_cohort():
    root = Path(__file__).parents[2] / "artifacts"
    current = json.loads(
        (root / "v8.1-balanced-sequence/result.json").read_text()
    )
    previous = json.loads((root / "v8-sequence/result.json").read_text())

    assert current["selection"]["ablation"] == "B_repertoire"
    assert current["selection"]["constantBlend"] == 0.25
    assert current["selection"]["gateThreshold"] is None
    assert current["selection"]["final"] == "constant"
    assert current["selection"]["classification"] == "accepted trade-off"
    assert current["protocolAudit"] == {
        "selectionYear": 2024,
        "confirmationYear": 2025,
        "confirmationInputs": [
            "global",
            "B_repertoire-blend-0.25",
            "sequence-gate-not-selected",
        ],
        "other2025Ablations": "reporting-only-never-read-by-selection",
        "retunedAfter2025": False,
        "opened2026SelectionUse": False,
    }
    assert current["personalizer"]["status"] == "retrained"
    assert current["personalizer"]["accepted"]
    for year in ("2023", "2024", "2025"):
        assert (
            current["folds"][year]["rowFingerprint"]
            == previous["folds"][year]["rowFingerprint"]
        )
    for year in ("2024", "2025"):
        assert current["personalizer"]["years"][year]["fallbackExact"]
