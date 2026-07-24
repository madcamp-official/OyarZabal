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
    assert group_pitch("ST") is PitchGroup.SLIDER
    assert group_pitch("KC") is PitchGroup.CURVE
    assert group_pitch("FS") is None
    assert context_pitch("FS") == UNSUPPORTED_CONTEXT
    assert group_pitch("AB") is None
    assert context_pitch("AB") is None
    assert group_pitch(None) is None
    assert len(PITCH_GROUPS) == 6


def test_raw_probabilities_merge_to_fixed_target_space() -> None:
    probabilities = normalize_group_probabilities(
        {"FF": 0.25, "SI": 0.25, "SL": 0.2, "ST": 0.1, "FS": 0.1, "AB": 0.1}
    )
    assert tuple(probabilities) == PITCH_GROUPS
    assert sum(probabilities.values()) == pytest.approx(1)
    assert probabilities[PitchGroup.SLIDER] == pytest.approx(0.375)


def test_invalid_probability_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid probability"):
        normalize_group_probabilities({"FF": -0.1, "SL": 1.1})
