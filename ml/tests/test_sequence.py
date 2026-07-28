import numpy as np
import pandas as pd
import pytest
from oyarzabal.features import prepare_pitch_rows
from oyarzabal.sequence import (
    HierarchicalSequenceTransformer,
    SequenceExampleBuilder,
    SequenceNormalizer,
    SequenceVocabulary,
    blend_log_probabilities,
    inverse_sqrt_weights,
)
from oyarzabal.v83_sequence import (
    FittedV83Expert,
    GlobalConditionedSequenceResidual,
    apply_hierarchical_calibration,
    fit_hierarchical_calibration,
    hierarchical_residual_probabilities,
    load_v83_expert,
    mild_family_weights,
    save_v83_expert,
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


def test_missing_numeric_values_have_distinct_observation_masks():
    rows = _rows()
    rows.loc[0, "release_spin_rate"] = np.nan
    examples, _ = _examples(rows)
    normalizer = SequenceNormalizer.fit(examples, np.array([0, 1, 2]))
    batch = examples.batch(np.array([1]), normalizer)

    spin = list(examples.token_numeric_columns).index("release_spin_rate")
    assert batch["token_numeric"][0, -1, spin] == pytest.approx(0)
    assert batch["token_observed"][0, -1, spin] == 0
    assert batch["token_observed"][0, -1, 0] == 1


def test_repertoire_is_point_in_time_and_excludes_current_pitch():
    original = _rows()
    changed = original.copy()
    changed.loc[2:, "pitch_type"] = ["KC", "FS", "FC"]
    before, vocabulary = _examples(original)
    after = SequenceExampleBuilder(length=4).build(changed, vocabulary)
    before = before.with_repertoire(prepare_pitch_rows([original]))
    after = after.with_repertoire(prepare_pitch_rows([changed]))

    np.testing.assert_allclose(
        before.repertoire_context[:3],
        after.repertoire_context[:3],
    )
    assert before.repertoire_context.shape == (5, 18)


def test_inverse_sqrt_weights_are_fold_local_mean_one_and_bounded():
    training = np.array([0] * 12 + [1] * 8 + [2] * 6 + [3] * 4 + [4] * 3 + [5])
    changed_outside_fold = np.concatenate([training, np.full(100, 5)])

    weights = inverse_sqrt_weights(training, 6)
    outside_ignored = inverse_sqrt_weights(changed_outside_fold[: len(training)], 6)

    np.testing.assert_allclose(weights, outside_ignored)
    assert weights.mean() == pytest.approx(1)
    assert weights.min() >= 0.5
    assert weights.max() <= 3


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
    np.testing.assert_allclose(
        blend_log_probabilities(
            np.repeat(global_probabilities, 2, axis=0),
            np.repeat(sequence_probabilities, 2, axis=0),
            np.array([0, 0.25]),
        )[0],
        global_probabilities[0],
    )


def test_hierarchical_residual_scale_zero_is_exact_global():
    global_probabilities = np.array(
        [[0.30, 0.20, 0.18, 0.12, 0.15, 0.05]],
        dtype=np.float64,
    )
    family_delta = np.array([[2.0, -1.0, 0.5]])
    child_delta = np.array([[[1.0, -1.0], [0.5, -0.5], [-2.0, 2.0]]])

    unchanged = hierarchical_residual_probabilities(
        global_probabilities,
        family_delta,
        child_delta,
        0,
    )
    adjusted = hierarchical_residual_probabilities(
        global_probabilities,
        family_delta,
        child_delta,
        0.5,
    )

    np.testing.assert_array_equal(unchanged, global_probabilities)
    np.testing.assert_allclose(adjusted.sum(axis=1), 1)
    assert not np.allclose(adjusted, global_probabilities)


def test_mild_family_weights_are_mean_one_and_bounded():
    labels = np.array([0] * 100 + [1] * 40 + [2] * 10)
    weights = mild_family_weights(labels)

    assert weights.mean() == pytest.approx(1)
    assert weights.min() >= 0.75
    assert weights.max() <= 1.5


def test_hierarchical_calibration_is_fit_only_from_supplied_rows():
    actual = np.array([0, 0, 2, 2, 4, 4])
    biased = np.array(
        [
            [0.20, 0.30, 0.15, 0.10, 0.15, 0.10],
            [0.22, 0.28, 0.15, 0.10, 0.15, 0.10],
        ]
        * 3
    )
    before = -np.log(biased[np.arange(6), actual]).mean()
    parameters = fit_hierarchical_calibration(actual, biased)
    adjusted = apply_hierarchical_calibration(biased, parameters)
    after = -np.log(adjusted[np.arange(6), actual]).mean()

    assert parameters.shape == (9,)
    assert after < before
    np.testing.assert_allclose(adjusted.sum(axis=1), 1)


def test_transformer_is_causal_and_joint_probabilities_sum_to_one():
    torch = pytest.importorskip("torch")
    rows = _rows()
    vocabulary = SequenceVocabulary.fit(rows)
    examples = SequenceExampleBuilder(length=16).build(
        rows, vocabulary
    ).with_repertoire(prepare_pitch_rows([rows]))
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
        current_numeric_width=examples.current_numeric.shape[1],
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


def test_v83_zero_initialized_residual_returns_global_probabilities():
    torch = pytest.importorskip("torch")
    rows = _rows()
    vocabulary = SequenceVocabulary.fit(rows)
    examples = (
        SequenceExampleBuilder(length=4)
        .build(rows, vocabulary)
        .with_repertoire(prepare_pitch_rows([rows]))
    )
    normalizer = SequenceNormalizer.fit(examples, np.array([0, 1, 2]))
    batch = {
        name: torch.as_tensor(value)
        for name, value in examples.batch(np.array([2, 3]), normalizer).items()
    }
    global_probabilities = torch.tensor(
        [
            [0.30, 0.20, 0.18, 0.12, 0.15, 0.05],
            [0.20, 0.25, 0.25, 0.10, 0.10, 0.10],
        ]
    )
    batch["global_probabilities"] = global_probabilities
    model = GlobalConditionedSequenceResidual(
        len(vocabulary.descriptions) + 1,
        length=4,
        current_numeric_width=examples.current_numeric.shape[1],
    ).eval()

    with torch.no_grad():
        output = model(batch)

    torch.testing.assert_close(
        output["group_probabilities"],
        global_probabilities,
    )
    torch.testing.assert_close(
        output["group_probabilities"].sum(dim=1),
        torch.ones(2),
    )


def test_v83_checkpoint_round_trip_preserves_predictions(tmp_path):
    torch = pytest.importorskip("torch")
    rows = _rows()
    vocabulary = SequenceVocabulary.fit(rows)
    examples = SequenceExampleBuilder(length=4).build(rows, vocabulary)
    normalizer = SequenceNormalizer.fit(examples, np.array([0, 1, 2]))
    model = GlobalConditionedSequenceResidual(
        len(vocabulary.descriptions) + 1,
        length=4,
        current_numeric_width=examples.current_numeric.shape[1],
    )
    fitted = FittedV83Expert(
        model,
        normalizer,
        1.0,
        1,
        0.1,
        0.2,
        np.ones(3, dtype=np.float32),
    )
    path = tmp_path / "sequence.pt"
    save_v83_expert(fitted, path)
    restored = load_v83_expert(path)

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[name])
