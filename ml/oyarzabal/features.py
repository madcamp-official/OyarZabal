"""Point-in-time feature construction with explicit lag boundaries."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .taxonomy import PITCH_GROUPS, PitchGroup, context_pitch, group_pitch

REPERTOIRE_PREFIXES = (
    "career_rate_",
    "season_rate_",
    "recent_100_rate_",
    "game_rate_",
    "game_delta_",
    "count_rate_",
    "stand_rate_",
    "transition_rate_",
)
REPERTOIRE_FEATURES = tuple(
    f"{prefix}{group}" for prefix in REPERTOIRE_PREFIXES for group in PITCH_GROUPS
)
PA_COUNT_FEATURES = tuple(f"pa_count_{group}" for group in PITCH_GROUPS)
V83_PITCH_METRICS = (
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
)
V83_RELEASE_METRICS = (
    "release_pos_x",
    "release_pos_z",
    "release_extension",
)
V83_CONTEXT_FEATURES = (
    *(
        f"v83_{group}_{metric}_{suffix}"
        for group in PITCH_GROUPS
        for metric in V83_PITCH_METRICS
        for suffix in ("season", "recent20", "delta")
    ),
    *(f"v83_{group}_physical_support" for group in PITCH_GROUPS),
    *(
        f"v83_{metric}_{suffix}"
        for metric in V83_RELEASE_METRICS
        for suffix in ("season", "recent20", "delta")
    ),
    "v83_release_support",
    "v83_catcher_support",
    "v83_battery_support",
    *(f"v83_catcher_rate_{group}" for group in PITCH_GROUPS),
    *(f"v83_battery_delta_{group}" for group in PITCH_GROUPS),
)

LEGACY_NUMERIC_FEATURES = (
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "base_1",
    "base_2",
    "base_3",
    "score_diff",
    "game_pitch_count",
    "n_thruorder_pitcher",
    "pitcher_days_since_prev_game",
    "prev_release_speed",
    "prev_plate_x",
    "prev_plate_z",
)

NUMERIC_FEATURES = (
    *LEGACY_NUMERIC_FEATURES,
    "pitch_number",
    "prev_pitch_streak",
    "pa_prev_pitch_streak",
    "pa_distinct_pitch_groups",
    "pa_fastball_share",
    "pa_breaking_share",
    "pa_offspeed_share",
    "game_distinct_pitch_groups",
    *PA_COUNT_FEATURES,
    *REPERTOIRE_FEATURES,
)

LEGACY_CATEGORICAL_FEATURES = (
    "pitcher",
    "batter",
    "stand",
    "p_throws",
    "inning_topbot",
    "prev_pitch_group",
    "prev_description",
)

CATEGORICAL_FEATURES = (
    *LEGACY_CATEGORICAL_FEATURES,
    "pa_prev_pitch_1",
    "pa_prev_pitch_2",
    "pa_prev_pitch_3",
    "pa_first_pitch_group",
)

GLOBAL_CATEGORICAL_FEATURES = tuple(
    name for name in CATEGORICAL_FEATURES if name not in {"pitcher", "batter"}
)
GLOBAL_NUMERIC_FEATURES = NUMERIC_FEATURES

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

SORT_COLUMNS = (
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
)


def _safe_column(frame: pd.DataFrame, name: str, default: object) -> pd.Series:
    if name in frame:
        return frame[name]
    return pd.Series(default, index=frame.index)


def _past_group_counts(
    rows: pd.DataFrame,
    keys: list[str],
    indicators: dict[object, pd.Series],
    supported: pd.Series,
) -> tuple[pd.Series, dict[object, pd.Series]]:
    total = supported.groupby([rows[key] for key in keys], sort=False).cumsum()
    total = total - supported
    counts = {}
    for group, indicator in indicators.items():
        count = indicator.groupby([rows[key] for key in keys], sort=False).cumsum()
        counts[group] = count - indicator
    return total, counts


def _smoothed_rates(
    rows: pd.DataFrame,
    prefix: str,
    total: pd.Series,
    counts: dict[object, pd.Series],
    backing: dict[object, pd.Series],
    strength: float,
) -> dict[object, pd.Series]:
    rates = {}
    for group in PITCH_GROUPS:
        rate = (counts[group] + strength * backing[group]) / (total + strength)
        rows[f"{prefix}{group}"] = rate.astype("float32")
        rates[group] = rate
    return rates


def _add_repertoire_features(rows: pd.DataFrame) -> None:
    supported = rows["target_group"].notna().astype("float64")
    indicators = {
        group: rows["target_group"].eq(group).astype("float64")
        for group in PITCH_GROUPS
    }

    global_total = supported.cumsum() - supported
    global_rates = {
        group: (indicator.cumsum() - indicator + 1.0)
        / (global_total + len(PITCH_GROUPS))
        for group, indicator in indicators.items()
    }

    career_total, career_counts = _past_group_counts(
        rows, ["pitcher"], indicators, supported
    )
    rows["career_support"] = career_total.astype("float32")
    career_rates = _smoothed_rates(
        rows,
        "career_rate_",
        career_total,
        career_counts,
        global_rates,
        50,
    )

    season_total, season_counts = _past_group_counts(
        rows, ["pitcher", "_season"], indicators, supported
    )
    season_rates = _smoothed_rates(
        rows,
        "season_rate_",
        season_total,
        season_counts,
        career_rates,
        30,
    )

    game_total, game_counts = _past_group_counts(
        rows, ["game_pk", "pitcher"], indicators, supported
    )
    game_rates = _smoothed_rates(
        rows,
        "game_rate_",
        game_total,
        game_counts,
        season_rates,
        20,
    )
    rows["game_distinct_pitch_groups"] = sum(
        count.gt(0).astype("int8") for count in game_counts.values()
    )
    for group in PITCH_GROUPS:
        rows[f"game_delta_{group}"] = (
            game_rates[group] - season_rates[group]
        ).astype("float32")

    recent_counts = {}
    for group, indicator in indicators.items():
        previous = indicator.groupby(rows["pitcher"], sort=False).shift(1).fillna(0)
        recent_counts[group] = (
            previous.groupby(rows["pitcher"], sort=False)
            .rolling(100, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
            .sort_index()
        )
    recent_total = sum(recent_counts.values())
    _smoothed_rates(
        rows,
        "recent_100_rate_",
        recent_total,
        recent_counts,
        career_rates,
        20,
    )

    for prefix, support_name, keys, strength in (
        (
            "count_rate_",
            "count_support",
            ["pitcher", "balls", "strikes"],
            20,
        ),
        ("stand_rate_", "stand_support", ["pitcher", "stand"], 30),
        (
            "transition_rate_",
            "transition_support",
            ["pitcher", "prev_pitch_group"],
            15,
        ),
    ):
        total, counts = _past_group_counts(rows, keys, indicators, supported)
        rows[support_name] = total.astype("float32")
        _smoothed_rates(
            rows,
            prefix,
            total,
            counts,
            career_rates,
            strength,
        )


def _add_plate_appearance_features(rows: pd.DataFrame) -> None:
    keys = ["game_pk", "at_bat_number"]
    supported = rows["target_group"].notna().astype("float64")
    indicators = {
        group: rows["target_group"].eq(group).astype("float64")
        for group in PITCH_GROUPS
    }
    _, counts = _past_group_counts(rows, keys, indicators, supported)
    for group, count in counts.items():
        rows[f"pa_count_{group}"] = count.astype("float32")

    rows["pa_distinct_pitch_groups"] = sum(
        count.gt(0).astype("int8") for count in counts.values()
    )
    previous_count = sum(counts.values())
    fastball_count = sum(
        counts[group]
        for group in (PitchGroup.FOUR_SEAM, PitchGroup.MOVING_FASTBALL)
    )
    breaking_count = sum(
        counts[group]
        for group in (PitchGroup.SLIDER, PitchGroup.CURVE)
    )
    offspeed_count = counts[PitchGroup.CHANGEUP] + counts[PitchGroup.SPLITTER_FORK]
    denominator = previous_count + 3
    rows["pa_fastball_share"] = ((fastball_count + 1) / denominator).astype(
        "float32"
    )
    rows["pa_breaking_share"] = ((breaking_count + 1) / denominator).astype(
        "float32"
    )
    rows["pa_offspeed_share"] = ((offspeed_count + 1) / denominator).astype(
        "float32"
    )

    pa_position = rows.groupby(keys, sort=False).cumcount()
    first_group = rows.groupby(keys, sort=False)["context_group"].transform("first")
    rows["pa_first_pitch_group"] = first_group.where(pa_position > 0, "UNKNOWN")

    previous_group = rows.groupby(keys, sort=False)["context_group"].shift(1)
    changed = rows["context_group"].ne(previous_group)
    rows["_pa_pitch_run"] = changed.groupby(
        [rows[key] for key in keys], sort=False
    ).cumsum()
    current_streak = (
        rows.groupby([*keys, "_pa_pitch_run"], sort=False).cumcount() + 1
    )
    rows["pa_prev_pitch_streak"] = (
        current_streak.groupby([rows[key] for key in keys], sort=False)
        .shift(1)
        .fillna(0)
    )


def _past_rolling_mean(
    values: pd.Series,
    keys: pd.Series,
    *,
    window: int,
) -> pd.Series:
    previous = values.groupby(keys, sort=False).shift(1)
    return (
        previous.groupby(keys, sort=False)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )


def _add_v83_context_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Add ID-free physical and catcher profiles using prior pitches only."""
    pitcher = rows["pitcher"]
    season_keys = [pitcher, rows["_season"]]
    target = rows["target_group"]
    pitcher_recent_key = pitcher
    features: dict[str, pd.Series] = {}

    for group in PITCH_GROUPS:
        is_group = target.eq(group)
        support = is_group.groupby(season_keys, sort=False).cumsum() - is_group
        features[f"v83_{group}_physical_support"] = support.astype("float32")
        for metric in V83_PITCH_METRICS:
            values = pd.to_numeric(
                _safe_column(rows, metric, np.nan),
                errors="coerce",
            )
            observed = is_group & values.notna()
            weighted = values.fillna(0).where(is_group, 0)
            total = weighted.groupby(season_keys, sort=False).cumsum() - weighted
            count = observed.groupby(season_keys, sort=False).cumsum() - observed
            season = total / count.replace(0, np.nan)
            recent_values = values.where(is_group)
            recent = _past_rolling_mean(
                recent_values,
                pitcher_recent_key,
                window=20,
            )
            features[f"v83_{group}_{metric}_season"] = season.astype("float32")
            features[f"v83_{group}_{metric}_recent20"] = recent.astype("float32")
            features[f"v83_{group}_{metric}_delta"] = (recent - season).astype(
                "float32"
            )

    release_observed = pd.Series(False, index=rows.index)
    for metric in V83_RELEASE_METRICS:
        values = pd.to_numeric(
            _safe_column(rows, metric, np.nan),
            errors="coerce",
        )
        observed = values.notna()
        release_observed |= observed
        weighted = values.fillna(0)
        total = weighted.groupby(season_keys, sort=False).cumsum() - weighted
        count = observed.groupby(season_keys, sort=False).cumsum() - observed
        season = total / count.replace(0, np.nan)
        recent = _past_rolling_mean(values, pitcher_recent_key, window=20)
        features[f"v83_{metric}_season"] = season.astype("float32")
        features[f"v83_{metric}_recent20"] = recent.astype("float32")
        features[f"v83_{metric}_delta"] = (recent - season).astype("float32")
    features["v83_release_support"] = (
        release_observed.groupby(season_keys, sort=False).cumsum() - release_observed
    ).astype("float32")

    catcher = pd.to_numeric(
        _safe_column(rows, "fielder_2", np.nan),
        errors="coerce",
    ).fillna(-1)
    supported = target.notna().astype("float64")
    indicators = {group: target.eq(group).astype("float64") for group in PITCH_GROUPS}
    catcher_total, catcher_counts = _past_group_counts(
        rows.assign(_v83_catcher=catcher),
        ["_v83_catcher"],
        indicators,
        supported,
    )
    battery_rows = rows.assign(_v83_catcher=catcher)
    battery_total, battery_counts = _past_group_counts(
        battery_rows,
        ["pitcher", "_v83_catcher"],
        indicators,
        supported,
    )
    features["v83_catcher_support"] = catcher_total.astype("float32")
    features["v83_battery_support"] = battery_total.astype("float32")
    for group in PITCH_GROUPS:
        catcher_rate = (catcher_counts[group] + 1) / (catcher_total + len(PITCH_GROUPS))
        battery_rate = (battery_counts[group] + 15 * rows[f"career_rate_{group}"]) / (
            battery_total + 15
        )
        features[f"v83_catcher_rate_{group}"] = catcher_rate.astype("float32")
        features[f"v83_battery_delta_{group}"] = (
            battery_rate - rows[f"career_rate_{group}"]
        ).astype("float32")
    return pd.concat([rows, pd.DataFrame(features, index=rows.index)], axis=1)


def prepare_pitch_rows(
    frames: Iterable[pd.DataFrame],
    *,
    include_v83: bool = False,
) -> pd.DataFrame:
    """Return target pitches with only pre-pitch and lagged information."""
    materialized = [frame for frame in frames if not frame.empty]
    if not materialized:
        raise ValueError("no pitch frames supplied")

    rows = pd.concat(materialized, ignore_index=True)
    del materialized
    required = {"pitch_type", "game_date", "game_pk", "pitcher", "batter"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"missing required Statcast columns: {sorted(missing)}")

    rows["game_date"] = pd.to_datetime(rows["game_date"], errors="coerce")
    rows["target_group"] = rows["pitch_type"].map(group_pitch)
    rows["context_group"] = rows["pitch_type"].map(context_pitch)
    rows = rows[rows["context_group"].notna() & rows["game_date"].notna()].copy()
    rows["at_bat_number"] = pd.to_numeric(
        _safe_column(rows, "at_bat_number", 0), errors="coerce"
    ).fillna(0)
    rows["pitch_number"] = pd.to_numeric(
        _safe_column(rows, "pitch_number", 0), errors="coerce"
    ).fillna(0)
    rows = rows.sort_values(list(SORT_COLUMNS), kind="stable").reset_index(drop=True)
    rows["_season"] = rows["game_date"].dt.year
    rows["balls"] = pd.to_numeric(
        _safe_column(rows, "balls", 0), errors="coerce"
    ).fillna(0)
    rows["strikes"] = pd.to_numeric(
        _safe_column(rows, "strikes", 0), errors="coerce"
    ).fillna(0)
    rows["stand"] = _safe_column(rows, "stand", "UNKNOWN").fillna("UNKNOWN")

    game_pitcher = [rows["game_pk"], rows["pitcher"]]
    rows["game_pitch_count"] = rows.groupby(
        ["game_pk", "pitcher"], sort=False
    ).cumcount()
    rows["prev_pitch_group"] = (
        rows.groupby(["game_pk", "pitcher"], sort=False)["context_group"]
        .shift(1)
        .fillna("UNKNOWN")
    )
    plate_appearance = ["game_pk", "at_bat_number"]
    for lag in range(1, 4):
        rows[f"pa_prev_pitch_{lag}"] = rows.groupby(plate_appearance, sort=False)[
            "context_group"
        ].shift(lag)
    _add_plate_appearance_features(rows)

    changed = rows["context_group"].ne(rows["prev_pitch_group"])
    rows["_pitch_run"] = changed.groupby(game_pitcher, sort=False).cumsum()
    rows["_current_pitch_streak"] = (
        rows.groupby(["game_pk", "pitcher", "_pitch_run"], sort=False).cumcount() + 1
    )
    rows["prev_pitch_streak"] = (
        rows.groupby(["game_pk", "pitcher"], sort=False)["_current_pitch_streak"]
        .shift(1)
        .fillna(0)
    )
    rows["prev_release_speed"] = (
        _safe_column(rows, "release_speed", np.nan)
        .groupby(game_pitcher, sort=False)
        .shift(1)
    )
    rows["prev_plate_x"] = (
        _safe_column(rows, "plate_x", np.nan).groupby(game_pitcher, sort=False).shift(1)
    )
    rows["prev_plate_z"] = (
        _safe_column(rows, "plate_z", np.nan).groupby(game_pitcher, sort=False).shift(1)
    )
    rows["prev_description"] = (
        _safe_column(rows, "description", "NONE")
        .groupby(game_pitcher, sort=False)
        .shift(1)
        .fillna("NONE")
    )

    _add_repertoire_features(rows)
    if include_v83:
        rows = _add_v83_context_features(rows)

    for base in (1, 2, 3):
        values = _safe_column(rows, f"on_{base}b", np.nan)
        rows[f"base_{base}"] = values.notna().astype("int8")

    bat_score = pd.to_numeric(
        _safe_column(rows, "bat_score", 0), errors="coerce"
    ).fillna(0)
    fld_score = pd.to_numeric(
        _safe_column(rows, "fld_score", 0), errors="coerce"
    ).fillna(0)
    fallback_score_diff = bat_score - fld_score
    score_diff = pd.to_numeric(
        _safe_column(rows, "bat_score_diff", np.nan), errors="coerce"
    )
    rows["score_diff"] = score_diff.fillna(fallback_score_diff)

    rows = rows[rows["target_group"].notna()].copy().reset_index(drop=True)
    rows["pitcher_id"] = pd.to_numeric(rows["pitcher"], errors="raise").astype("int64")
    rows["batter_id"] = pd.to_numeric(rows["batter"], errors="raise").astype("int64")

    numeric_features = (
        (*NUMERIC_FEATURES, *V83_CONTEXT_FEATURES) if include_v83 else NUMERIC_FEATURES
    )
    for name in numeric_features:
        rows[name] = pd.to_numeric(
            _safe_column(rows, name, np.nan), errors="coerce"
        ).astype("float32")
    for name in CATEGORICAL_FEATURES:
        rows[name] = _safe_column(rows, name, "UNKNOWN").fillna("UNKNOWN").astype(str)

    label_index = {group: index for index, group in enumerate(PITCH_GROUPS)}
    rows["target"] = rows["target_group"].map(label_index).astype("int8")
    return rows


def chronological_split(
    rows: pd.DataFrame, validation_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    dates = np.array(sorted(rows["game_date"].dt.normalize().unique()))
    if len(dates) < 2:
        raise ValueError("at least two game dates are required")
    split_index = max(
        1,
        min(len(dates) - 1, int(len(dates) * (1 - validation_fraction))),
    )
    cutoff = dates[split_index]
    train = rows[rows["game_date"].dt.normalize() < cutoff].copy()
    validation = rows[rows["game_date"].dt.normalize() >= cutoff].copy()
    if train.empty or validation.empty:
        raise ValueError("chronological split produced an empty partition")
    if train["game_date"].max() >= validation["game_date"].min():
        raise AssertionError("chronological split overlaps")
    return train, validation


def group_names() -> dict[str, str]:
    from .taxonomy import PITCH_GROUP_NAMES

    return {str(group): PITCH_GROUP_NAMES[group] for group in PITCH_GROUPS}
