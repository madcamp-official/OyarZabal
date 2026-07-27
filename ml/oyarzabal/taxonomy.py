"""Stable pitch-family mapping shared by training and replay."""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum


class PitchGroup(StrEnum):
    FOUR_SEAM = "FOUR_SEAM"
    MOVING_FASTBALL = "MOVING_FASTBALL"
    SLIDER = "SLIDER"
    CURVE = "CURVE"
    CHANGEUP = "CHANGEUP"
    SPLITTER_FORK = "SPLITTER_FORK"


class PitchFamily(StrEnum):
    FASTBALL = "FASTBALL"
    BREAKING = "BREAKING"
    OFFSPEED = "OFFSPEED"


PITCH_GROUPS = tuple(PitchGroup)
PITCH_FAMILIES = tuple(PitchFamily)

PITCH_GROUP_NAMES: dict[PitchGroup, str] = {
    PitchGroup.FOUR_SEAM: "포심",
    PitchGroup.MOVING_FASTBALL: "무빙 패스트볼",
    PitchGroup.SLIDER: "슬라이더 계열",
    PitchGroup.CURVE: "커브 계열",
    PitchGroup.CHANGEUP: "체인지업",
    PitchGroup.SPLITTER_FORK: "스플리터·포크",
}

PITCH_FAMILY_NAMES: dict[PitchFamily, str] = {
    PitchFamily.FASTBALL: "패스트볼 계열",
    PitchFamily.BREAKING: "브레이킹볼 계열",
    PitchFamily.OFFSPEED: "오프스피드 계열",
}

PITCH_GROUP_FAMILIES: dict[PitchGroup, PitchFamily] = {
    PitchGroup.FOUR_SEAM: PitchFamily.FASTBALL,
    PitchGroup.MOVING_FASTBALL: PitchFamily.FASTBALL,
    PitchGroup.SLIDER: PitchFamily.BREAKING,
    PitchGroup.CURVE: PitchFamily.BREAKING,
    PitchGroup.CHANGEUP: PitchFamily.OFFSPEED,
    PitchGroup.SPLITTER_FORK: PitchFamily.OFFSPEED,
}
PITCH_GROUP_FAMILY_LABELS = tuple(
    PITCH_FAMILIES.index(PITCH_GROUP_FAMILIES[group]) for group in PITCH_GROUPS
)

_CODE_TO_GROUP: dict[str, PitchGroup] = {
    "FF": PitchGroup.FOUR_SEAM,
    "SI": PitchGroup.MOVING_FASTBALL,
    "FT": PitchGroup.MOVING_FASTBALL,
    "FC": PitchGroup.MOVING_FASTBALL,
    "SL": PitchGroup.SLIDER,
    "ST": PitchGroup.SLIDER,
    "SV": PitchGroup.SLIDER,
    "CU": PitchGroup.CURVE,
    "KC": PitchGroup.CURVE,
    "CS": PitchGroup.CURVE,
    "CH": PitchGroup.CHANGEUP,
    "FS": PitchGroup.SPLITTER_FORK,
    "FO": PitchGroup.SPLITTER_FORK,
}

EXCLUDED_PITCH_CODES = frozenset({"AB", "IN", "PO"})
UNSUPPORTED_PITCH_CODES = frozenset({"FA", "KN", "SC", "EP", "UN"})
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
    """Merge raw Statcast probabilities and normalize them to six groups."""
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


def family_for_group(group: PitchGroup | str) -> PitchFamily:
    return PITCH_GROUP_FAMILIES[PitchGroup(group)]


def family_names() -> dict[str, str]:
    return {str(family): name for family, name in PITCH_FAMILY_NAMES.items()}


def group_families() -> dict[str, str]:
    return {
        str(group): str(family)
        for group, family in PITCH_GROUP_FAMILIES.items()
    }
