import numpy as np
import pandas as pd
import pytest
from oyarzabal.sequence import (
    HierarchicalSequenceTransformer,
    SequenceExampleBuilder,
    SequenceNormalizer,
    SequenceVocabulary,
    blend_log_probabilities,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_date": ["2025-04-01"] * 5,
            "game_pk": [1, 1, 1, 1, 2],
            "at_bat_number": [1, 1, 2, 2, 1],
            "pitch_number": [1, 2, 1, 2, 1],
            "pitch_type": ["FF", "SL", "CH", "CU", "SI"],
            "pitcher": [10] * 5,
            "batter": [20, 20, 30, 30, 40],
            "fielder_2": [100, 100, 200, 200, 200],
            "description": [
                "ball",
                "called_strike",
                "foul",
                "hit_into_play",
                "ball",
            ],
            "balls": [0, 1, 0, 0, 0],
            "strikes": [0, 0, 0, 1, 0],
            "stand": ["R", "R", "L", "L", "R"],
            "zone": [11, 5, 3, 7, 12],
            "release_speed": [95, 86, 88, 80, 94],
            "release_spin_rate": [2300, 2500, 1800, 2700, 2200],
            "pfx_x": [0.1, 0.2, -0.3, 0.4, -0.5],
            "pfx_z": [1.1, 0.2, 0.3, -0.4, 0.5],
            "release_pos_x": [-2.0, -2.1, -2.0, -2.2, -2.1],
            "release_pos_z": [5.8, 5.7, 5.8, 5.9, 5.8],
            "release_extension": [6.2, 6.1, 6.3, 6.0, 6.2],
            "plate_x": [0.1, -0.2, 0.3, -0.4, 0.5],
            "plate_z": [2.5, 2.1, 1.9, 2.7, 3.0],
            "sz_top": [3.5] * 5,
            "sz_bot": [1.5] * 5,
            "outs_when_up": [0, 0, 1, 1, 0],
            "inning": [1, 1, 1, 1, 2],
            "on_1b": [np.nan] * 5,
            "on_2b": [np.nan] * 5,
            "on_3b": [np.nan] * 5,
            "n_thruorder_pitcher": [1] * 5,
        }
    )


def _examples(rows: pd.DataFrame):
    vocabulary = SequenceVocabulary.fit(rows)
    return SequenceExampleBuilder(length=4).build(rows, vocabulary), vocabulary


def test_current_and_future_values_never_enter_current_sequence_input():
    original = _rows()
    changed = original.copy()
    changed.loc[2:, "pitch_type"] = ["KC", "FS", "FC"]
    changed.loc[2:, "release_speed"] = [60, 61, 62]
    changed.loc[2:, "release_spin_rate"] = [900, 901, 902]
    changed.loc[2:, "pfx_x"] = [9, 9, 9]
    changed.loc[2:, "release_pos_x"] = [9, 9, 9]
    changed.loc[2:, "plate_x"] = [9, 9, 9]
    changed.loc[2:, "zone"] = [14, 14, 14]

    before, vocabulary = _examples(original)
    after = SequenceExampleBuilder(length=4).build(changed, vocabulary)
    normalizer = SequenceNormalizer.fit(before, np.array([0, 1]))
    before_batch = before.batch(np.array([2]), normalizer)
    after_batch = after.batch(np.array([2]), normalizer)

    for name in (
        "token_categorical",
        "token_numeric",
        "history_flags",
        "padding_mask",
        "current_categorical",
        "current_numeric",
    ):
        np.testing.assert_array_equal(before_batch[name], after_batch[name])


def test_pa_game_boundaries_and_catcher_change_are_point_in_time():
    examples, vocabulary = _examples(_rows())
    third_history = examples.history_indices[2]
    valid = third_history[third_history >= 0]

    assert examples.history_flags[2, -2:, 0].tolist() == [0, 0]
    assert examples.history_flags[2, -2:, 1].tolist() == [1, 1]
    assert examples.source_catcher_ids[valid].tolist() == [
        vocabulary.catchers[100],
        vocabulary.catchers[100],
    ]
    fourth_history = examples.history_indices[3]
    assert examples.source_catcher_ids[fourth_history[-1]] == (
        vocabulary.catchers[200]
    )
    assert examples.history_flags[4, -4:, 0].tolist() == [0, 0, 0, 0]


def test_unknown_entities_fall_back_to_zero_without_entering_global_tokens():
    rows = _rows()
    vocabulary = SequenceVocabulary.fit(rows.iloc[:2])
    examples = SequenceExampleBuilder(length=4).build(rows, vocabulary)

    assert examples.pitcher_ids[0] != 0
    assert examples.batter_ids[2] == 0
    assert examples.catcher_ids[2] == 0
    assert examples.source_categorical.shape[1] == 6


def test_current_zone_is_target_only():
    rows = _rows()
    before, vocabulary = _examples(rows)
    changed = rows.copy()
    changed.loc[2, "zone"] = 14
    after = SequenceExampleBuilder(length=4).build(changed, vocabulary)
    normalizer = SequenceNormalizer.fit(before, np.array([0, 1]))

    before_batch = before.batch(np.array([2]), normalizer)
    after_batch = after.batch(np.array([2]), normalizer)
    assert before_batch["target_zone"].item() != after_batch["target_zone"].item()
    for name in before_batch:
        if name != "target_zone":
            np.testing.assert_array_equal(before_batch[name], after_batch[name])


def test_zero_sequence_blend_is_exact_global_fallback():
    global_probabilities = np.array([[0.4, 0.2, 0.1, 0.1, 0.1, 0.1]])
    sequence_probabilities = np.full((1, 6), 1 / 6)

    np.testing.assert_allclose(
        blend_log_probabilities(
            global_probabilities,
            sequence_probabilities,
            0,
        ),
        global_probabilities,
    )


def test_transformer_is_causal_and_joint_probabilities_sum_to_one():
    torch = pytest.importorskip("torch")
    rows = _rows()
    vocabulary = SequenceVocabulary.fit(rows)
    examples = SequenceExampleBuilder(length=16).build(rows, vocabulary)
    normalizer = SequenceNormalizer.fit(examples, np.array([0, 1, 2]))
    numpy_batch = examples.batch(np.array([2, 3]), normalizer)
    batch = {
        name: torch.as_tensor(value)
        for name, value in numpy_batch.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        batch = {
            name: value.repeat((1024,) + (1,) * (value.ndim - 1))
            for name, value in batch.items()
        }
    batch = {name: value.to(device) for name, value in batch.items()}
    model = HierarchicalSequenceTransformer(
        len(vocabulary.descriptions) + 1,
        length=16,
    ).to(device).eval()
    with torch.no_grad():
        output = model(batch)
    mask = model.causal_mask(device)

    assert mask[0, 1]
    assert not mask[1, 0]
    expected_rows = len(batch["target_group"])
    assert output["family_logits"].shape == (expected_rows, 3)
    assert output["child_logits"].shape == (expected_rows, 3, 2)
    assert output["zone_logits"].shape == (expected_rows, 6, 14)
    assert torch.isfinite(output["group_probabilities"]).all()
    torch.testing.assert_close(
        output["group_probabilities"].sum(dim=1),
        torch.ones(expected_rows, device=device),
    )
    assert torch.isfinite(model.loss(output, batch, location_weight=0.25))
    assert torch.isfinite(model.loss(output, batch, location_weight=0))
