import pandas as pd
import pytest
from oyarzabal.features import (
    GLOBAL_CATEGORICAL_FEATURES,
    V83_CONTEXT_FEATURES,
    chronological_split,
    prepare_pitch_rows,
)
from oyarzabal.taxonomy import PITCH_GROUPS, UNSUPPORTED_CONTEXT, PitchGroup
from pandas.testing import assert_frame_equal


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_date": ["2024-01-01", "2024-01-01", "2024-02-01"],
            "game_pk": [1, 1, 2],
            "at_bat_number": [1, 1, 1],
            "pitch_number": [1, 2, 1],
            "pitcher": [10, 10, 10],
            "batter": [20, 20, 30],
            "pitch_type": ["FF", "SL", "CH"],
            "release_speed": [95.0, 85.0, 82.0],
            "plate_x": [0.1, -0.2, 0.4],
            "plate_z": [2.5, 1.8, 2.0],
            "description": ["called_strike", "ball", "swinging_strike"],
            "balls": [0, 0, 0],
            "strikes": [0, 1, 0],
            "outs_when_up": [0, 0, 0],
            "inning": [1, 1, 1],
            "inning_topbot": ["Top", "Top", "Top"],
            "stand": ["R", "R", "L"],
            "p_throws": ["R", "R", "R"],
            "bat_score": [0, 0, 0],
            "fld_score": [0, 0, 0],
        }
    )


def test_lagged_features_never_use_current_pitch() -> None:
    rows = prepare_pitch_rows([_rows()])
    assert rows.loc[0, "prev_pitch_group"] == "UNKNOWN"
    assert rows.loc[1, "prev_pitch_group"] == str(PitchGroup.FOUR_SEAM)
    assert rows.loc[1, "prev_release_speed"] == 95.0
    assert rows.loc[2, "prev_pitch_group"] == "UNKNOWN"


def test_chronological_split_has_strict_boundary() -> None:
    rows = prepare_pitch_rows([_rows()])
    train, validation = chronological_split(rows, validation_fraction=0.5)
    assert train["game_date"].max() < validation["game_date"].min()
    assert set(train["game_pk"]).isdisjoint(validation["game_pk"])


def test_point_in_time_rates_do_not_use_current_or_future_targets() -> None:
    original = _rows()
    changed = original.copy()
    changed.loc[1:, "pitch_type"] = ["CU", "FC"]

    before = prepare_pitch_rows([original]).iloc[:2]
    after = prepare_pitch_rows([changed]).iloc[:2]
    rate_columns = [
        name
        for name in before
        if name.startswith(
            (
                "career_rate_",
                "season_rate_",
                "recent_100_rate_",
                "game_rate_",
                "game_delta_",
                "count_rate_",
                "stand_rate_",
                "transition_rate_",
                "pa_count_",
                "pa_distinct_",
                "pa_fastball_",
                "pa_breaking_",
                "pa_offspeed_",
            )
        )
    ]
    assert_frame_equal(
        before[rate_columns].reset_index(drop=True),
        after[rate_columns].reset_index(drop=True),
    )


def test_repertoire_probability_families_are_normalized() -> None:
    rows = prepare_pitch_rows([_rows()])
    for prefix in (
        "career_rate_",
        "season_rate_",
        "recent_100_rate_",
        "game_rate_",
        "count_rate_",
        "stand_rate_",
        "transition_rate_",
    ):
        columns = [f"{prefix}{group}" for group in PITCH_GROUPS]
        assert rows[columns].sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_unsupported_pitch_is_context_but_not_a_target() -> None:
    rows = _rows()
    unsupported = rows.iloc[[1]].copy()
    unsupported["pitch_number"] = 3
    unsupported["pitch_type"] = "KN"
    final = rows.iloc[[1]].copy()
    final["pitch_number"] = 4
    final["pitch_type"] = "CH"
    source = pd.concat([rows.iloc[:2], unsupported, final], ignore_index=True)

    prepared = prepare_pitch_rows([source])

    assert len(prepared) == 3
    last = prepared.iloc[-1]
    assert last["pa_prev_pitch_1"] == UNSUPPORTED_CONTEXT
    assert last["pa_prev_pitch_2"] == str(PitchGroup.SLIDER)
    assert last["pa_prev_pitch_3"] == str(PitchGroup.FOUR_SEAM)
    assert last["prev_pitch_streak"] == 1


def test_plate_appearance_and_game_context_use_only_prior_pitches() -> None:
    source = pd.DataFrame(
        {
            "game_date": ["2024-01-01"] * 5 + ["2024-02-01"],
            "game_pk": [1] * 5 + [2],
            "at_bat_number": [1, 1, 1, 1, 2, 1],
            "pitch_number": [1, 2, 3, 4, 1, 1],
            "pitcher": [10] * 6,
            "batter": [20] * 4 + [30, 40],
            "pitch_type": ["FF", "SL", "FF", "CH", "FF", "CU"],
            "balls": [0] * 6,
            "strikes": [0] * 6,
            "stand": ["R"] * 6,
        }
    )

    prepared = prepare_pitch_rows([source])
    fourth = prepared.iloc[3]
    next_batter = prepared.iloc[4]
    next_game = prepared.iloc[5]

    assert fourth["pa_count_FOUR_SEAM"] == 2
    assert fourth["pa_count_SLIDER"] == 1
    assert fourth["pa_distinct_pitch_groups"] == 2
    assert fourth["pa_first_pitch_group"] == str(PitchGroup.FOUR_SEAM)
    assert fourth["pa_prev_pitch_streak"] == 1
    assert sum(
        fourth[name]
        for name in (
            "pa_fastball_share",
            "pa_breaking_share",
            "pa_offspeed_share",
        )
    ) == pytest.approx(1)

    assert next_batter["pa_count_FOUR_SEAM"] == 0
    assert next_batter["pa_distinct_pitch_groups"] == 0
    assert next_batter["pa_first_pitch_group"] == "UNKNOWN"
    assert next_batter["pa_prev_pitch_streak"] == 0
    assert next_batter["game_distinct_pitch_groups"] == 3

    assert next_game["game_distinct_pitch_groups"] == 0
    game_rate_columns = [f"game_rate_{group}" for group in PITCH_GROUPS]
    assert prepared[game_rate_columns].sum(axis=1).tolist() == pytest.approx([1] * 6)


def test_global_model_features_do_not_expose_player_identity() -> None:
    assert "pitcher" not in GLOBAL_CATEGORICAL_FEATURES
    assert "batter" not in GLOBAL_CATEGORICAL_FEATURES
    assert {"stand", "p_throws", "prev_pitch_group"} <= set(
        GLOBAL_CATEGORICAL_FEATURES
    )
    prepared = prepare_pitch_rows([_rows()])
    assert prepared["pitcher_id"].tolist() == [10, 10, 10]
    assert prepared["batter_id"].tolist() == [20, 20, 30]


def test_gate_support_features_are_point_in_time() -> None:
    prepared = prepare_pitch_rows([_rows()])

    assert prepared["career_support"].tolist() == [0, 1, 2]
    assert prepared["count_support"].tolist() == [0, 0, 1]
    assert prepared["stand_support"].tolist() == [0, 1, 0]
    assert prepared["transition_support"].tolist() == [0, 0, 1]


def test_v83_physical_and_catcher_profiles_exclude_current_and_future() -> None:
    source = _rows()
    source["release_spin_rate"] = [2300.0, 2500.0, 1800.0]
    source["pfx_x"] = [0.1, -0.4, 0.2]
    source["pfx_z"] = [1.1, 0.2, 0.5]
    source["release_pos_x"] = [-2.0, -2.1, -2.2]
    source["release_pos_z"] = [5.8, 5.7, 5.6]
    source["release_extension"] = [6.2, 6.1, 6.0]
    source["fielder_2"] = [100, 100, 200]
    changed = source.copy()
    changed.loc[1:, "pitch_type"] = ["CU", "FS"]
    changed.loc[1:, "release_spin_rate"] = [999.0, 998.0]

    before = prepare_pitch_rows([source], include_v83=True)
    after = prepare_pitch_rows([changed], include_v83=True)

    assert_frame_equal(
        before.loc[:1, list(V83_CONTEXT_FEATURES)],
        after.loc[:1, list(V83_CONTEXT_FEATURES)],
    )
    assert before.loc[0, "v83_catcher_support"] == 0
    assert before.loc[1, "v83_catcher_support"] == 1


def test_sparse_bat_score_diff_column_falls_back_row_by_row() -> None:
    history = _rows().iloc[:2].copy()
    history["bat_score"] = [3, 4]
    history["fld_score"] = [1, 2]
    game = _rows().iloc[[2]].copy()
    game["bat_score_diff"] = 5

    prepared = prepare_pitch_rows([history, game])

    assert prepared["score_diff"].tolist() == [2, 2, 5]
