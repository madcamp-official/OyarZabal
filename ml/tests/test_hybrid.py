import numpy as np
import pandas as pd
import pytest
from oyarzabal.hybrid import (
    RegistryEntry,
    blend_by_pitcher,
    blend_probabilities,
    select_blend_weight,
    serialize_registry_entry,
    specialist_eligibility,
)


def test_blend_weight_selection_requires_improvement() -> None:
    actual = np.array([0, 1, 0, 1])
    global_probabilities = np.array(
        [
            [0.55, 0.45, 0, 0, 0, 0],
            [0.55, 0.45, 0, 0, 0, 0],
            [0.55, 0.45, 0, 0, 0, 0],
            [0.55, 0.45, 0, 0, 0, 0],
        ]
    )
    specialist_probabilities = np.array(
        [
            [0.9, 0.1, 0, 0, 0, 0],
            [0.1, 0.9, 0, 0, 0, 0],
            [0.9, 0.1, 0, 0, 0, 0],
            [0.1, 0.9, 0, 0, 0, 0],
        ]
    )

    selected = select_blend_weight(
        actual,
        global_probabilities,
        specialist_probabilities,
        weights=(0.0, 0.5, 1.0),
    )

    assert selected["accepted"] is True
    assert selected["specialistWeight"] == 1.0
    assert selected["metrics"]["logLoss"] < selected["globalMetrics"]["logLoss"]


def test_router_blends_enabled_specialist_and_falls_back_to_global() -> None:
    global_probabilities = np.array([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]])
    specialist = {10: np.array([[0.2, 0.8], [0.4, 0.6]])}
    registry = {
        10: RegistryEntry(
            pitcher_id=10,
            enabled=True,
            specialist_weight=0.75,
            model="specialist-10",
        ),
        20: RegistryEntry(
            pitcher_id=20,
            enabled=False,
            specialist_weight=1.0,
            model="specialist-20",
        ),
    }

    probabilities, sources = blend_by_pitcher(
        np.array([10, 20, 10]),
        global_probabilities,
        specialist,
        registry,
    )

    assert probabilities[0] == pytest.approx([0.3, 0.7])
    assert probabilities[1] == pytest.approx(global_probabilities[1])
    assert probabilities[2] == pytest.approx([0.5, 0.5])
    assert sources == ["hybrid", "global", "hybrid"]
    assert probabilities.sum(axis=1) == pytest.approx([1, 1, 1])


def test_specialist_eligibility_checks_volume_recency_and_repertoire() -> None:
    rows = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                ["2023-01-01", "2024-01-01", "2024-02-01", "2024-03-01"]
            ),
            "target": [0, 0, 1, 2],
        }
    )
    result = specialist_eligibility(
        rows,
        min_pitches=4,
        min_recent_pitches=3,
        min_pitch_groups=3,
        min_group_share=0.2,
    )
    assert result["eligible"] is True

    too_small = specialist_eligibility(rows.iloc[:2], min_pitches=4)
    assert too_small["eligible"] is False
    assert "pitch_count" in too_small["reasons"]


def test_blend_rejects_invalid_weight() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        blend_probabilities(np.array([[1, 0]]), np.array([[0, 1]]), 1.1)


def test_registry_serialization_uses_public_schema_names() -> None:
    entry = RegistryEntry(10, True, 0.75, "specialist-10.pkl", "2025-12-31")
    payload = serialize_registry_entry(entry, name="Pitcher")
    assert payload["pitcherId"] == 10
    assert payload["globalWeight"] == 0.25
    assert payload["specialistWeight"] == 0.75
    assert "pitcher_id" not in payload
