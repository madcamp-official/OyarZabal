import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from oyarzabal.features import (
    V9_GAME_STATE_FEATURES,
    V9_PHYSICAL_FEATURES,
    V9_STRATEGY_FEATURES,
)
from oyarzabal.residual import apply_correction
from oyarzabal.v9a import (
    _comparison_metrics,
    game_state_matrix,
    predict_game_state_correction,
    train_game_state_expert,
)


def _training_rows() -> pd.DataFrame:
    rows = 120
    dates = pd.date_range("2023-01-01", periods=20).repeat(6)
    return pd.DataFrame(
        {
            "game_date": dates,
            "target": np.tile(np.arange(6), 20),
            "game_pitch_count": np.arange(rows) % 100,
            "v9_game_mix_js": np.linspace(0, 0.2, rows),
            "v9_FOUR_SEAM_release_speed_game": np.linspace(92, 97, rows),
        }
    )


def _base(rows: int) -> np.ndarray:
    probabilities = np.full((rows, 6), 0.1)
    probabilities[:, 0] = 0.5
    return probabilities


def test_v9a_feature_contract_has_no_raw_entity_ids() -> None:
    forbidden = {"pitcher", "pitcher_id", "batter", "batter_id", "fielder_2"}

    assert not forbidden.intersection(V9_GAME_STATE_FEATURES)
    assert set(V9_STRATEGY_FEATURES).isdisjoint(V9_PHYSICAL_FEATURES)


def test_v85_comparison_requires_the_same_2026_fingerprint(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "v85.json"
    artifact.write_text(
        json.dumps(
            {
                "retrospectiveStatus": "complete",
                "retrospective2026": {
                    "rowFingerprint": "same",
                    "stages": {"E_safe_alpha_caps": {"accuracy": 0.5}},
                },
            }
        )
    )

    assert _comparison_metrics(
        artifact,
        expected_fingerprint="same",
    ) == {"accuracy": 0.5}
    with pytest.raises(ValueError, match="fingerprint"):
        _comparison_metrics(artifact, expected_fingerprint="different")


def test_physical_drop_keeps_strategy_and_masks_physical_values() -> None:
    rows = _training_rows().iloc[:2]
    names = (
        "game_pitch_count",
        "v9_game_mix_js",
        "v9_FOUR_SEAM_release_speed_game",
    )

    normal = game_state_matrix(rows, names)
    stress = game_state_matrix(rows, names, drop_physical=True)

    np.testing.assert_array_equal(stress[:, :2], normal[:, :2])
    assert np.isnan(stress[:, 2]).all()


def test_game_state_expert_round_trips_and_outputs_valid_probabilities() -> None:
    rows = _training_rows()
    base = _base(len(rows))
    feature_names = ("game_pitch_count", "v9_game_mix_js")

    fitted = train_game_state_expert(
        rows,
        base,
        feature_names,
        n_estimators=5,
    )
    restored = pickle.loads(pickle.dumps(fitted))
    correction = predict_game_state_correction(restored, rows)
    probabilities = apply_correction(base, correction, 0.1)

    assert restored.feature_names == feature_names
    assert correction.shape == base.shape
    assert np.isfinite(correction).all()
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
