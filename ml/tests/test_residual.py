import numpy as np
import pandas as pd
import pytest
from oyarzabal.residual import (
    apply_correction,
    apply_dynamic_correction,
    compute_pitcher_reliability,
    count_bucket,
    effective_residual_scale,
    gate_feature_frame,
    gate_targets,
    hard_safety_mask,
    pitcher_metrics_pass,
    predict_context_gate,
    predict_correction,
    provisional_scale,
    reliability_score,
    residual_feature_frame,
    train_final_residual,
    train_gate,
)


def _rows(size: int = 120) -> pd.DataFrame:
    targets = np.arange(size) % 6
    return pd.DataFrame(
        {
            "pitcher_id": np.where(np.arange(size) % 2, 10, 20),
            "balls": np.arange(size) % 4,
            "strikes": np.arange(size) % 3,
            "stand": np.where(np.arange(size) % 2, "L", "R"),
            "pa_prev_pitch_1": np.array(
                ["FOUR_SEAM", "SLIDER", "UNKNOWN"]
            )[np.arange(size) % 3],
            "target": targets,
        }
    )


def test_count_bucket_has_five_exclusive_states() -> None:
    rows = pd.DataFrame(
        {
            "balls": [0, 3, 0, 2, 1],
            "strikes": [0, 2, 2, 0, 1],
        }
    )
    assert count_bucket(rows).tolist() == [
        "first",
        "full",
        "pitcher_ahead",
        "batter_ahead",
        "even",
    ]


def test_residual_model_produces_additive_corrections() -> None:
    rows = _rows()
    global_probabilities = np.full((len(rows), 6), 1 / 6)
    fitted = train_final_residual(
        rows,
        global_probabilities,
        n_estimators=3,
    )
    correction = predict_correction(fitted, rows)
    adjusted = apply_correction(global_probabilities, correction, 0.5)

    assert residual_feature_frame(rows).columns.tolist() == [
        "pitcher_id",
        "count_bucket",
        "stand",
        "pa_prev_pitch_1",
    ]
    assert correction.shape == global_probabilities.shape
    assert correction.mean(axis=1) == pytest.approx(np.zeros(len(rows)))
    assert adjusted.sum(axis=1) == pytest.approx(np.ones(len(rows)))


def test_provisional_scale_is_capped_and_decays() -> None:
    assert provisional_scale(0, base_reliability=0.4) == 0.15
    assert provisional_scale(365, base_reliability=0.4) == pytest.approx(0.075)
    assert provisional_scale(730, base_reliability=0.4) == pytest.approx(0.0375)


def test_pitcher_gate_does_not_reuse_aggregate_macro_f1_gate() -> None:
    global_metrics = {
        "logLoss": 1.2,
        "accuracy": 0.5,
        "macroF1": 0.5,
        "zeroRecallClasses": [],
        "actualDistribution": {},
        "majorityPredictionGap": 0.0,
    }
    candidate_metrics = {
        "logLoss": 1.1,
        "accuracy": 0.496,
        "macroF1": 0.2,
        "zeroRecallClasses": [],
        "actualDistribution": {},
        "majorityPredictionGap": 0.3,
    }

    assert pitcher_metrics_pass(global_metrics, candidate_metrics) is False


def test_reliability_is_bounded_monotonic_and_recent_sensitive() -> None:
    small = reliability_score(100, 0.8, 0.8)
    large = reliability_score(10_000, 0.8, 0.8)
    recent_drop = reliability_score(10_000, 0.8, 0.2)

    assert 0 <= small < large <= 0.5
    assert recent_drop < large
    assert reliability_score(10_000, 1, 1) < 0.5


def test_dynamic_scale_and_caps_preserve_probability_rows() -> None:
    global_probabilities = np.array([[0.5, 0.3, 0.2]])
    correction = np.array([[8.0, -4.0, -4.0]])
    scale = effective_residual_scale(0.4, 0.75, hard_safety_pass=True)
    probabilities, applied_scale, cap_reason = apply_dynamic_correction(
        global_probabilities,
        correction,
        np.array([scale]),
        js_cap=0.05,
        probability_shift_cap=0.20,
    )

    assert scale == pytest.approx(0.3)
    assert 0 < applied_scale[0] < scale
    assert cap_reason == ["js_and_probability_shift"]
    assert probabilities.sum(axis=1) == pytest.approx([1])
    assert np.max(np.abs(probabilities - global_probabilities)) <= 0.200001


def test_pitcher_reliability_uses_recent_window() -> None:
    rows = pd.DataFrame(
        {
            "pitcher_id": [10] * 8,
            "game_pk": [1, 1, 2, 2, 3, 3, 4, 4],
            "game_date": pd.to_datetime(
                ["2024-01-01"] * 2
                + ["2024-02-01"] * 2
                + ["2024-10-01"] * 2
                + ["2024-11-01"] * 2
            ),
            "target": [0, 1] * 4,
        }
    )
    global_probabilities = np.full((8, 2), 0.5)
    reference = np.array(
        [
            [0.8, 0.2],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.8, 0.2],
        ]
    )

    result = compute_pitcher_reliability(
        rows,
        global_probabilities,
        reference,
        samples=50,
        seed=9,
    )[10]

    assert result["pAll"] > result["pRecent"]
    assert result["reliability"] == pytest.approx(
        reliability_score(8, result["pAll"], result["pRecent"])
    )


def test_recent_window_is_relative_to_each_pitchers_latest_game() -> None:
    rows = pd.DataFrame(
        {
            "pitcher_id": [10, 10, 20, 20],
            "game_pk": [1, 2, 3, 4],
            "game_date": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-10-01", "2024-11-01"]
            ),
            "target": [0, 0, 0, 0],
        }
    )
    global_probabilities = np.full((4, 2), 0.5)
    reference = np.tile([0.8, 0.2], (4, 1))

    result = compute_pitcher_reliability(
        rows,
        global_probabilities,
        reference,
        samples=20,
        seed=3,
    )

    assert result[10]["recentSupport"] == 2
    assert result[20]["recentSupport"] == 2
    assert result[10]["pRecent"] > 0.9
    assert result[20]["pRecent"] > 0.9


def test_binary_gate_excludes_identity_and_predicts_probabilities() -> None:
    rows = _rows(240)
    rows["game_date"] = pd.to_datetime(
        np.where(np.arange(len(rows)) < 180, "2024-01-01", "2024-02-01")
    )
    rows["p_throws"] = "R"
    for name in (
        "outs_when_up",
        "inning",
        "base_1",
        "base_2",
        "base_3",
        "score_diff",
        "game_pitch_count",
        "n_thruorder_pitcher",
        "pitcher_days_since_prev_game",
        "pitch_number",
        "career_support",
        "count_support",
        "stand_support",
        "transition_support",
    ):
        rows[name] = np.arange(len(rows)) % 40
    global_probabilities = np.full((len(rows), 6), 1 / 6)
    correction = np.zeros_like(global_probabilities)
    correction[np.arange(len(rows)), rows["target"].to_numpy()] = np.where(
        np.arange(len(rows)) % 2,
        2,
        -2,
    )
    frame = gate_feature_frame(rows, global_probabilities, correction)

    assert "pitcher_id" not in frame
    assert "batter_id" not in frame
    assert set(
        gate_targets(rows["target"].to_numpy(), global_probabilities, correction)
    ) == {0, 1}

    fitted = train_gate(
        rows.iloc[:180],
        global_probabilities[:180],
        correction[:180],
        n_estimators=3,
    )
    predicted = predict_context_gate(
        fitted,
        rows.iloc[180:],
        global_probabilities[180:],
        correction[180:],
    )
    assert predicted.shape == (60,)
    assert ((predicted >= 0) & (predicted <= 1)).all()


def test_hard_safety_requires_supported_context() -> None:
    rows = pd.DataFrame(
        {
            "count_support": [19, 20, 20, 20],
            "stand_support": [20, 19, 20, 20],
            "transition_support": [20, 20, 19, 0],
            "pa_prev_pitch_1": ["FOUR_SEAM", "FOUR_SEAM", "FOUR_SEAM", "UNKNOWN"],
        }
    )
    passed, reasons = hard_safety_mask(rows)

    assert passed.tolist() == [False, False, False, True]
    assert reasons == [
        "count_support",
        "stand_support",
        "transition_support",
        None,
    ]
