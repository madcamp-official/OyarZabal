import numpy as np
import pandas as pd
import pytest
from oyarzabal.features import prepare_pitch_rows
from oyarzabal.sequence import SequenceExampleBuilder, SequenceVocabulary
from oyarzabal.sequence_gate import (
    discrete_gate_scales,
    gate_feature_frame,
    gate_targets,
)


def _rows() -> pd.DataFrame:
    groups = ["FF", "SL", "CH", "CU", "SI", "FS"] * 3
    return pd.DataFrame(
        {
            "game_date": ["2025-04-01"] * len(groups),
            "game_pk": [1] * len(groups),
            "at_bat_number": np.repeat(np.arange(1, 10), 2),
            "pitch_number": np.tile([1, 2], 9),
            "pitch_type": groups,
            "pitcher": [10] * len(groups),
            "batter": np.repeat(np.arange(20, 29), 2),
            "description": ["ball"] * len(groups),
            "balls": [0] * len(groups),
            "strikes": [0] * len(groups),
            "stand": ["R"] * len(groups),
        }
    )


def _examples():
    rows = _rows()
    vocabulary = SequenceVocabulary.fit(rows)
    return SequenceExampleBuilder(length=16).build(
        rows, vocabulary
    ).with_repertoire(prepare_pitch_rows([rows]))


def test_gate_features_are_id_free_and_probability_aligned():
    examples = _examples()
    indices = np.arange(len(examples))
    global_probabilities = np.full((len(indices), 6), 1 / 6)
    sequence_probabilities = global_probabilities.copy()
    sequence_probabilities[:, 0] += 0.1
    sequence_probabilities[:, 1:] -= 0.02

    features = gate_feature_frame(
        examples,
        indices,
        global_probabilities,
        sequence_probabilities,
    )

    assert len(features) == len(indices)
    assert not any(
        "pitcher" in name or "batter" in name or "catcher" in name
        for name in features
    )
    assert features.isna().sum().sum() == 0


def test_gate_targets_and_three_scales():
    global_probabilities = np.array(
        [
            [0.4, 0.2, 0.1, 0.1, 0.1, 0.1],
            [0.4, 0.2, 0.1, 0.1, 0.1, 0.1],
        ]
    )
    sequence_probabilities = np.array(
        [
            [0.6, 0.1, 0.1, 0.1, 0.05, 0.05],
            [0.6, 0.1, 0.1, 0.1, 0.05, 0.05],
        ]
    )
    assert gate_targets(
        np.array([0, 1]),
        global_probabilities,
        sequence_probabilities,
        0.2,
    ).tolist() == [1, 0]

    scales = discrete_gate_scales(
        np.array([0.39, 0.4, 0.59, 0.6]),
        0.2,
        0.4,
    )
    np.testing.assert_allclose(scales, [0, 0.1, 0.1, 0.2])
    assert discrete_gate_scales(np.array([0.1]), 0.2, 0.4).item() == 0
    with pytest.raises(ValueError):
        discrete_gate_scales(np.array([0.5]), 0.2, 0.45)
