from datetime import date

import numpy as np
import pandas as pd
import pytest
from oyarzabal.hybrid import (
    RegistryEntry,
    apply_logit_bias,
    apply_pooled_residual_by_pitcher,
    apply_reliability_gated_residual,
    blend_by_pitcher,
    blend_probabilities,
    fit_logit_bias,
    personalize_by_pitcher,
    personalizer_passes,
    select_blend_weight,
    select_personalizer_strength,
    serialize_registry_entry,
    specialist_eligibility,
)


def test_blend_weight_selection_requires_improvement() -> None:
    actual = np.array([0, 1, 0, 1])
    global_probabilities = np.array(
        [
            [0.55, 0.45, 0, 0, 0, 0],
            [0.55, 0.45, 0, 0, 0, 0],
            [0.55, 0.45, 0, 0, 0, 0],
            [0.55, 0.45, 0, 0, 0, 0],
        ]
    )
    specialist_probabilities = np.array(
        [
            [0.9, 0.1, 0, 0, 0, 0],
            [0.1, 0.9, 0, 0, 0, 0],
            [0.9, 0.1, 0, 0, 0, 0],
            [0.1, 0.9, 0, 0, 0, 0],
        ]
    )

    selected = select_blend_weight(
        actual,
        global_probabilities,
        specialist_probabilities,
        weights=(0.0, 0.5, 1.0),
    )

    assert selected["accepted"] is True
    assert selected["specialistWeight"] == 1.0
    assert selected["metrics"]["logLoss"] < selected["globalMetrics"]["logLoss"]


def test_router_blends_enabled_specialist_and_falls_back_to_global() -> None:
    global_probabilities = np.array([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]])
    specialist = {10: np.array([[0.2, 0.8], [0.4, 0.6]])}
    registry = {
        10: RegistryEntry(
            pitcher_id=10,
            enabled=True,
            specialist_weight=0.75,
            model="specialist-10",
        ),
        20: RegistryEntry(
            pitcher_id=20,
            enabled=False,
            specialist_weight=1.0,
            model="specialist-20",
        ),
    }

    probabilities, sources = blend_by_pitcher(
        np.array([10, 20, 10]),
        global_probabilities,
        specialist,
        registry,
    )

    assert probabilities[0] == pytest.approx([0.3, 0.7])
    assert probabilities[1] == pytest.approx(global_probabilities[1])
    assert probabilities[2] == pytest.approx([0.5, 0.5])
    assert sources == ["hybrid", "global", "hybrid"]
    assert probabilities.sum(axis=1) == pytest.approx([1, 1, 1])


def test_specialist_eligibility_checks_volume_recency_and_repertoire() -> None:
    rows = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                ["2023-01-01", "2024-01-01", "2024-02-01", "2024-03-01"]
            ),
            "target": [0, 0, 1, 2],
        }
    )
    result = specialist_eligibility(
        rows,
        min_pitches=4,
        min_recent_pitches=3,
        min_pitch_groups=3,
        min_group_share=0.2,
    )
    assert result["eligible"] is True

    too_small = specialist_eligibility(rows.iloc[:2], min_pitches=4)
    assert too_small["eligible"] is False
    assert "pitch_count" in too_small["reasons"]


def test_blend_rejects_invalid_weight() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        blend_probabilities(np.array([[1, 0]]), np.array([[0, 1]]), 1.1)


def test_registry_serialization_uses_public_schema_names() -> None:
    entry = RegistryEntry(
        10,
        True,
        0.75,
        "specialist-10.pkl",
        "2025-12-31",
        reliability=0.2,
        reliability_components={"n": 1_000, "pAll": 0.7, "pRecent": 0.6},
        status="limited",
        scale_multiplier=0.4,
    )
    payload = serialize_registry_entry(entry, name="Pitcher")
    assert payload["pitcherId"] == 10
    assert payload["globalWeight"] == 0.25
    assert payload["specialistWeight"] == 0.75
    assert payload["reliability"] == 0.2
    assert payload["reliabilityComponents"]["pRecent"] == 0.6
    assert payload["status"] == "limited"
    assert payload["scaleMultiplier"] == 0.4
    assert "pitcher_id" not in payload


def test_logit_bias_personalizer_shrinks_and_routes_by_pitcher() -> None:
    calibration_probabilities = np.tile(
        np.array([[0.6, 0.4, 0, 0, 0, 0]]), (6, 1)
    )
    calibration_actual = np.array([1, 1, 1, 1, 0, 0])
    light_shrinkage = fit_logit_bias(
        calibration_actual, calibration_probabilities, prior_strength=1
    )
    heavy_shrinkage = fit_logit_bias(
        calibration_actual, calibration_probabilities, prior_strength=100
    )
    evaluation = np.array([[0.6, 0.4, 0, 0, 0, 0]])

    adjusted = apply_logit_bias(evaluation, light_shrinkage)
    conservative = apply_logit_bias(evaluation, heavy_shrinkage)

    assert adjusted[0, 1] > evaluation[0, 1]
    assert conservative[0, 1] < adjusted[0, 1]
    assert adjusted.sum(axis=1) == pytest.approx([1])

    registry = {
        10: RegistryEntry(
            pitcher_id=10,
            enabled=True,
            specialist_weight=0.5,
            model="logit-bias",
            logit_bias=tuple(light_shrinkage),
            personalizer_strength=1,
        )
    }
    routed, sources = personalize_by_pitcher(
        np.array([10, 20]),
        np.vstack([evaluation, evaluation]),
        registry,
    )
    assert routed[0, 1] > evaluation[0, 1]
    assert routed[1] == pytest.approx(evaluation[0])
    assert sources == ["hybrid", "global"]


def test_personalizer_strength_is_selected_on_separate_rows() -> None:
    calibration_probabilities = np.tile(
        np.array([[0.6, 0.4, 0, 0, 0, 0]]), (8, 1)
    )
    evaluation_probabilities = calibration_probabilities.copy()
    actual = np.array([1, 1, 1, 1, 0, 1, 0, 1])
    evaluation_probabilities[actual == 0, :2] = [0.9, 0.1]

    selected = select_personalizer_strength(
        actual,
        calibration_probabilities,
        actual,
        evaluation_probabilities,
        strengths=(1, 10),
    )

    assert selected["accepted"] is True
    assert selected["personalizerStrength"] in {1.0, 10.0}
    assert len(selected["logitBias"]) == 6


def test_personalizer_final_gate_rejects_accuracy_and_f1_regression() -> None:
    global_metrics = {
        "accuracy": 0.448,
        "macroF1": 0.238,
        "logLoss": 1.184,
        "zeroRecallClasses": ["MOVING_FASTBALL"],
        "actualDistribution": {"MOVING_FASTBALL": 0.003},
        "majorityPredictionGap": 0.23,
    }
    candidate_metrics = {
        "accuracy": 0.433,
        "macroF1": 0.237,
        "logLoss": 1.170,
        "zeroRecallClasses": ["MOVING_FASTBALL"],
        "actualDistribution": {"MOVING_FASTBALL": 0.003},
        "majorityPredictionGap": 0.08,
    }

    assert personalizer_passes(global_metrics, candidate_metrics) is False


def test_pooled_residual_routes_only_enabled_pitchers() -> None:
    global_probabilities = np.full((2, 6), 1 / 6)
    correction = np.array([[2, -2, 0, 0, 0, 0], [2, -2, 0, 0, 0, 0]])
    registry = {
        10: RegistryEntry(
            pitcher_id=10,
            enabled=True,
            specialist_weight=0.5,
            model="pooled-residual.pkl",
            status="active",
            residual_scale=0.5,
        )
    }

    adjusted, sources = apply_pooled_residual_by_pitcher(
        np.array([10, 20]),
        global_probabilities,
        correction,
        registry,
    )

    assert adjusted[0, 0] > global_probabilities[0, 0]
    assert adjusted[1] == pytest.approx(global_probabilities[1])
    assert sources == ["pooled-residual", "global"]


def test_reliability_router_exposes_scale_components_and_fallbacks() -> None:
    rows = pd.DataFrame(
        {
            "pitcher_id": [10, 20, 30],
            "count_support": [30, 30, 30],
            "stand_support": [30, 30, 30],
            "transition_support": [30, 30, 30],
            "pa_prev_pitch_1": ["FOUR_SEAM", "FOUR_SEAM", "FOUR_SEAM"],
        }
    )
    global_probabilities = np.full((3, 6), 1 / 6)
    correction = np.tile([2, -2, 0, 0, 0, 0], (3, 1))
    registry = {
        10: RegistryEntry(
            pitcher_id=10,
            enabled=True,
            specialist_weight=0,
            model="pooled-residual.pkl",
            status="full",
            reliability=0.4,
        ),
        20: RegistryEntry(
            pitcher_id=20,
            enabled=True,
            specialist_weight=0,
            model="pooled-residual.pkl",
            status="limited",
            reliability=0.4,
            scale_multiplier=0.25,
        ),
        30: RegistryEntry(
            pitcher_id=30,
            enabled=False,
            specialist_weight=0,
            model="",
            status="shadow",
            reliability=0.4,
            scale_multiplier=0,
        ),
    }

    adjusted, sources, routing = apply_reliability_gated_residual(
        rows,
        global_probabilities,
        correction,
        np.array([0.5, 0.5, 0.5]),
        registry,
    )

    assert adjusted[0, 0] > global_probabilities[0, 0]
    assert adjusted[1, 0] > global_probabilities[1, 0]
    assert adjusted[1, 0] < adjusted[0, 0]
    assert adjusted[2] == pytest.approx(global_probabilities[2])
    assert sources == [
        "reliability-gated-residual",
        "reliability-gated-residual",
        "global",
    ]
    assert routing[0]["pitcherReliability"] == 0.4
    assert routing[0]["contextGate"] == 0.5
    assert routing[0]["effectiveScale"] == pytest.approx(0.2)
    assert routing[0]["registryTier"] == "full"
    assert routing[1]["effectiveScale"] == pytest.approx(0.05)
    assert routing[1]["registryTier"] == "limited"
    assert routing[1]["scaleMultiplier"] == 0.25
    assert routing[2]["hardGateReason"] == "registry_inactive"
    assert routing[2]["registryTier"] == "shadow"


def test_stale_limited_pitcher_uses_existing_decay() -> None:
    rows = pd.DataFrame(
        {
            "pitcher_id": [10],
            "count_support": [30],
            "stand_support": [30],
            "transition_support": [30],
            "pa_prev_pitch_1": ["FOUR_SEAM"],
        }
    )
    global_probabilities = np.full((1, 6), 1 / 6)
    registry = {
        10: RegistryEntry(
            pitcher_id=10,
            enabled=True,
            specialist_weight=0,
            model="pooled-residual.pkl",
            data_cutoff="2025-01-01",
            status="limited",
            reliability=0.4,
            scale_multiplier=0.5,
            stale=True,
        )
    }

    _, _, routing = apply_reliability_gated_residual(
        rows,
        global_probabilities,
        np.array([[2, -2, 0, 0, 0, 0]]),
        np.array([1.0]),
        registry,
        prediction_dates=[date(2026, 1, 1)],
    )

    assert routing[0]["pitcherReliability"] == pytest.approx(0.075, abs=1e-4)
    assert routing[0]["effectiveScale"] == pytest.approx(0.0375, abs=1e-4)


def test_tier_boosts_change_matching_tiers_and_cap_at_half() -> None:
    rows = pd.DataFrame(
        {
            "pitcher_id": [10, 20],
            "count_support": [30, 30],
            "stand_support": [30, 30],
            "transition_support": [30, 30],
            "pa_prev_pitch_1": ["FOUR_SEAM", "FOUR_SEAM"],
        }
    )
    global_probabilities = np.full((2, 6), 1 / 6)
    registry = {
        10: RegistryEntry(
            pitcher_id=10,
            enabled=True,
            specialist_weight=0,
            model="pooled-residual.pkl",
            status="full",
            reliability=0.4,
        ),
        20: RegistryEntry(
            pitcher_id=20,
            enabled=True,
            specialist_weight=0,
            model="pooled-residual.pkl",
            status="limited",
            reliability=0.5,
            scale_multiplier=1,
        ),
    }

    _, _, routing = apply_reliability_gated_residual(
        rows,
        global_probabilities,
        np.array([[0.1, -0.1, 0, 0, 0, 0]] * 2),
        np.ones(2),
        registry,
        full_tier_boost=3,
        limited_scale_boost=4,
    )

    assert routing[0]["effectiveScale"] == pytest.approx(0.5)
    assert routing[0]["limitedScaleBoost"] == 1
    assert routing[0]["fullTierBoost"] == 3
    assert routing[1]["effectiveScale"] == pytest.approx(0.5)
    assert routing[1]["limitedScaleBoost"] == 4
    assert routing[1]["limitedTierBoost"] == 4


def test_reliability_and_gate_controls_increase_scale_with_half_cap() -> None:
    rows = pd.DataFrame(
        {
            "pitcher_id": [10],
            "count_support": [30],
            "stand_support": [30],
            "transition_support": [30],
            "pa_prev_pitch_1": ["FOUR_SEAM"],
        }
    )
    registry = {
        10: RegistryEntry(
            pitcher_id=10,
            enabled=True,
            specialist_weight=0,
            model="pooled-residual.pkl",
            status="full",
            reliability=0.4,
        )
    }

    _, _, routing = apply_reliability_gated_residual(
        rows,
        np.full((1, 6), 1 / 6),
        np.array([[0.1, -0.1, 0, 0, 0, 0]]),
        np.array([0.25]),
        registry,
        reliability_scale_boost=1.5,
        context_gate_power=0.5,
    )

    assert routing[0]["adjustedPitcherReliability"] == pytest.approx(0.5)
    assert routing[0]["adjustedContextGate"] == pytest.approx(0.5)
    assert routing[0]["effectiveScale"] == pytest.approx(0.25)
