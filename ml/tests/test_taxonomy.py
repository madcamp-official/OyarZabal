import pytest
from oyarzabal.taxonomy import (
    PITCH_GROUPS,
    UNSUPPORTED_CONTEXT,
    PitchGroup,
    context_pitch,
    group_pitch,
    normalize_group_probabilities,
)


def test_pitch_codes_are_grouped_and_exclusions_are_dropped() -> None:
    assert group_pitch("FF") is PitchGroup.FOUR_SEAM
    assert group_pitch("SI") is PitchGroup.MOVING_FASTBALL
    assert group_pitch("FT") is PitchGroup.MOVING_FASTBALL
    assert group_pitch("FC") is PitchGroup.MOVING_FASTBALL
    assert group_pitch("ST") is PitchGroup.SLIDER
    assert group_pitch("KC") is PitchGroup.CURVE
    assert group_pitch("FS") is PitchGroup.SPLITTER_FORK
    assert group_pitch("FO") is PitchGroup.SPLITTER_FORK
    assert group_pitch("KN") is None
    assert context_pitch("KN") == UNSUPPORTED_CONTEXT
    assert group_pitch("SC") is None
    assert context_pitch("SC") == UNSUPPORTED_CONTEXT
    assert group_pitch("AB") is None
    assert context_pitch("AB") is None
    assert group_pitch(None) is None
    assert len(PITCH_GROUPS) == 6


def test_raw_probabilities_merge_to_fixed_target_space() -> None:
    probabilities = normalize_group_probabilities(
        {
            "FF": 0.2,
            "SI": 0.2,
            "FC": 0.1,
            "SL": 0.2,
            "ST": 0.1,
            "FS": 0.1,
            "AB": 0.1,
        }
    )
    assert tuple(probabilities) == PITCH_GROUPS
    assert sum(probabilities.values()) == pytest.approx(1)
    assert probabilities[PitchGroup.MOVING_FASTBALL] == pytest.approx(1 / 3)
    assert probabilities[PitchGroup.SLIDER] == pytest.approx(1 / 3)
    assert probabilities[PitchGroup.SPLITTER_FORK] == pytest.approx(1 / 9)


def test_invalid_probability_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid probability"):
        normalize_group_probabilities({"FF": -0.1, "SL": 1.1})
