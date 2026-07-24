"""Stable pitch-family mapping shared by training and replay."""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum


class PitchGroup(StrEnum):
    FOUR_SEAM = "FOUR_SEAM"
    SINKER = "SINKER"
    CUTTER = "CUTTER"
    SLIDER = "SLIDER"
    CURVE = "CURVE"
    CHANGEUP = "CHANGEUP"


PITCH_GROUPS = tuple(PitchGroup)

PITCH_GROUP_NAMES: dict[PitchGroup, str] = {
    PitchGroup.FOUR_SEAM: "포심",
    PitchGroup.SINKER: "싱커/투심",
    PitchGroup.CUTTER: "커터",
    PitchGroup.SLIDER: "슬라이더/스위퍼",
    PitchGroup.CURVE: "커브",
    PitchGroup.CHANGEUP: "체인지업",
}

_CODE_TO_GROUP: dict[str, PitchGroup] = {
    "FF": PitchGroup.FOUR_SEAM,
    "FA": PitchGroup.FOUR_SEAM,
    "SI": PitchGroup.SINKER,
    "FT": PitchGroup.SINKER,
    "FC": PitchGroup.CUTTER,
    "SL": PitchGroup.SLIDER,
    "ST": PitchGroup.SLIDER,
    "SV": PitchGroup.SLIDER,
    "CU": PitchGroup.CURVE,
    "KC": PitchGroup.CURVE,
    "CS": PitchGroup.CURVE,
    "CH": PitchGroup.CHANGEUP,
}

EXCLUDED_PITCH_CODES = frozenset({"AB", "IN", "PO", "UN"})
UNSUPPORTED_PITCH_CODES = frozenset({"FS", "FO", "SC", "KN", "EP"})
UNSUPPORTED_CONTEXT = "UNSUPPORTED"


def group_pitch(raw_code: object) -> PitchGroup | None:
    """Map a Statcast pitch code to the fixed target space."""
    if raw_code is None:
        return None
    code = str(raw_code).strip().upper()
    if not code or code == "NAN":
        return None
    return _CODE_TO_GROUP.get(code)


def context_pitch(raw_code: object) -> PitchGroup | str | None:
    """Map a pitch for sequence context, retaining unsupported valid pitches."""
    group = group_pitch(raw_code)
    if group is not None:
        return group
    if raw_code is None:
        return None
    code = str(raw_code).strip().upper()
    return UNSUPPORTED_CONTEXT if code in UNSUPPORTED_PITCH_CODES else None


def normalize_group_probabilities(
    raw_probabilities: Mapping[str, float],
) -> dict[PitchGroup, float]:
    """Merge raw Statcast probabilities and normalize them to seven groups."""
    grouped = {group: 0.0 for group in PITCH_GROUPS}
    for code, raw_value in raw_probabilities.items():
        group = group_pitch(code)
        if group is None:
            continue
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid probability for {code}: {raw_value}")
        grouped[group] += value

    total = sum(grouped.values())
    if total <= 0:
        raise ValueError("probability distribution has no supported pitch mass")
    return {group: value / total for group, value in grouped.items()}


def serialize_probabilities(
    probabilities: Mapping[PitchGroup | str, float],
) -> dict[str, float]:
    values = {
        str(group): float(probabilities.get(group, 0.0)) for group in PITCH_GROUPS
    }
    total = sum(values.values())
    if total <= 0:
        raise ValueError("probability distribution has zero mass")
    return {group: value / total for group, value in values.items()}
