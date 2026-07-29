import pickle

import numpy as np
import pandas as pd
import pytest
from oyarzabal.residual import (
    V85_RESIDUAL_FEATURES,
    FittedResidual,
    gate_feature_frame,
    gate_targets,
    residual_feature_frame,
)
from oyarzabal.v85 import (
    _apply_final,
    _cancellation,
    _hash_json,
    _joint_registry,
    _safe_registry,
    _select_common_scale,
)
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pitcher_id": [1, 2],
            "player_name": ["A", "B"],
            "balls": [0, 1],
            "strikes": [0, 1],
            "stand": ["R", "L"],
            "pa_prev_pitch_1": ["FOUR_SEAM", "SLIDER"],
            "count_support": [100, 100],
            "stand_support": [100, 100],
            "transition_support": [100, 100],
            "target": [0, 1],
            "game_pk": [10, 11],
        }
    )


def _probabilities() -> np.ndarray:
    return np.array(
        [
            [0.4, 0.2, 0.15, 0.10, 0.10, 0.05],
            [0.2, 0.4, 0.15, 0.10, 0.10, 0.05],
        ],
        dtype=float,
    )


def test_v85_residual_features_exclude_sequence_context() -> None:
    frame = residual_feature_frame(_rows(), V85_RESIDUAL_FEATURES)

    assert tuple(frame.columns) == ("pitcher_id", "count_bucket", "stand")
    assert "pa_prev_pitch_1" not in frame


def test_gate_target_uses_supplied_v84_base_and_reference_scale() -> None:
    base = _probabilities()
    correction = np.zeros_like(base)
    correction[:, 0] = 2
    correction[:, 1] = -2

    small = gate_targets(
        np.array([0, 1]),
        base,
        correction,
        reference_scale=0.25,
    )
    large = gate_targets(
        np.array([0, 1]),
        base,
        correction,
        reference_scale=1.0,
    )

    np.testing.assert_array_equal(small, [1, 0])
    np.testing.assert_array_equal(large, [1, 0])
    small_features = gate_feature_frame(
        _rows(),
        base,
        correction,
        reference_scale=0.25,
    )
    large_features = gate_feature_frame(
        _rows(),
        base,
        correction,
        reference_scale=1.0,
    )
    assert not np.allclose(
        small_features["global_reference_js"],
        large_features["global_reference_js"],
    )


def test_shadow_fallback_is_bit_exact_and_registry_has_no_multiplier() -> None:
    rows = _rows()
    base = _probabilities()
    correction = np.ones_like(base)
    registry = {
        1: {
            "pitcherId": 1,
            "status": "shadow",
            "safeAlpha": 0.0,
        },
        2: {
            "pitcherId": 2,
            "status": "shadow",
            "safeAlpha": 0.0,
        },
    }

    output, routing, applied = _apply_final(
        rows,
        base,
        correction,
        np.zeros(len(rows)),
        registry,
        np.zeros(len(rows)),
        np.ones(len(rows)),
        [None, None],
    )

    np.testing.assert_array_equal(output, base)
    np.testing.assert_array_equal(applied, 0)
    assert routing["fallbackExact"] is True
    assert all("scaleMultiplier" not in entry for entry in registry.values())


def test_safe_registry_enforces_support_and_alpha_bounds() -> None:
    rows = pd.concat([_rows().iloc[[0]]] * 100, ignore_index=True)
    rows["target"] = 0
    base = np.tile(_probabilities()[0], (len(rows), 1))
    correction = np.zeros_like(base)
    correction[:, 0] = 0.1
    correction[:, 1] = -0.1

    registry = _safe_registry(
        rows,
        base,
        correction,
        np.full(len(rows), 0.1),
        (1, 2),
        {1: "A", 2: "B"},
    )

    assert 0 <= registry[1]["safeAlpha"] <= 1
    assert registry[1]["status"] in {"full", "limited", "shadow"}
    assert registry[2]["status"] == "shadow"
    assert registry[2]["failureReasons"] == ["insufficient_support"]


def test_cancellation_diagnostic_partitions_every_row() -> None:
    rows = _rows()
    global_probabilities = _probabilities()
    base = global_probabilities.copy()
    base[:, 0] += 0.05
    base[:, 1] -= 0.05
    final = base.copy()
    final[:, 0] -= 0.02
    final[:, 1] += 0.02

    result = _cancellation(rows, global_probabilities, base, final)

    assert sum(result["quadrants"].values()) == len(rows)
    assert np.isfinite(result["meanLogitDeltaCosine"])
    assert result["cancelledUtility"] >= 0
    assert result["repairedUtility"] >= 0


def test_invalid_residual_feature_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown residual features"):
        residual_feature_frame(_rows(), ("pitcher_id", "future_pitch"))


def test_common_scale_leaves_relative_distribution_safety_to_final_stage() -> None:
    rows = pd.concat([_rows().iloc[[0]]] * 600, ignore_index=True)
    rows["target"] = np.tile(np.arange(6), 100)
    prototype = np.full((6, 6), 0.1)
    np.fill_diagonal(prototype, 0.5)
    base = np.tile(prototype, (100, 1))
    correction = np.zeros_like(base)
    correction[np.arange(len(rows)), rows["target"].to_numpy()] = 0.2

    selected = _select_common_scale(rows, base, correction)

    assert selected["rawImproved"] is True
    assert selected["accepted"] is True
    assert selected["scale"] in {0.25, 0.5, 0.75, 1.0}


def test_joint_registry_uses_minimum_safe_alpha_without_multiplier() -> None:
    first = {
        1: {"safeAlpha": 1.0, "support": 100, "failureReasons": []},
        2: {"safeAlpha": 0.5, "support": 100, "failureReasons": []},
    }
    second = {
        1: {"safeAlpha": 0.5, "support": 200, "failureReasons": []},
        2: {
            "safeAlpha": 0.0,
            "support": 200,
            "failureReasons": ["unsafe"],
        },
    }

    registry = _joint_registry(first, second, {1: "A", 2: "B"})

    assert registry[1]["status"] == "limited"
    assert registry[1]["safeAlpha"] == 0.5
    assert registry[2]["status"] == "shadow"
    assert registry[2]["safeAlpha"] == 0
    assert all("scaleMultiplier" not in entry for entry in registry.values())


def test_v85_model_contract_round_trips_and_hashes_deterministically() -> None:
    fitted = FittedResidual(
        encoder=OneHotEncoder(),
        model=XGBClassifier(),
        tree_count=10,
        device="cpu",
        feature_names=V85_RESIDUAL_FEATURES,
    )

    restored = pickle.loads(pickle.dumps(fitted))

    assert restored.feature_names == V85_RESIDUAL_FEATURES
    first = {"seeds": [8401, 8402, 8403], "scale": 0.25}
    same = {"scale": 0.25, "seeds": [8401, 8402, 8403]}
    changed = {"scale": 0.25, "seeds": [8401, 8402, 8404]}
    assert _hash_json(first) == _hash_json(same)
    assert _hash_json(first) != _hash_json(changed)


def test_final_probability_contract_caps_large_correction() -> None:
    rows = _rows()
    base = _probabilities()
    correction = np.full_like(base, -10.0)
    correction[:, 5] = 10.0
    registry = {
        pitcher_id: {
            "pitcherId": pitcher_id,
            "status": "full",
            "safeAlpha": 1.0,
        }
        for pitcher_id in (1, 2)
    }

    output, routing, applied = _apply_final(
        rows,
        base,
        correction,
        np.full(len(rows), 0.5),
        registry,
        np.ones(len(rows)),
        np.ones(len(rows)),
        [None, None],
    )

    assert np.isfinite(output).all()
    np.testing.assert_allclose(output.sum(axis=1), 1)
    assert np.max(np.abs(output - base)) <= 0.20 + 1e-8
    assert np.all(applied > 0)
    assert sum(routing["capCounts"].values()) > 0
