import numpy as np
import pandas as pd
import pytest
from oyarzabal.residual import (
    apply_correction,
    count_bucket,
    pitcher_metrics_pass,
    predict_correction,
    provisional_scale,
    residual_feature_frame,
    train_final_residual,
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
    assert provisional_scale(0) == 0.25
    assert provisional_scale(365) == pytest.approx(0.25)
    assert provisional_scale(730) == pytest.approx(0.25)
    assert provisional_scale(1_095) == pytest.approx(0.125)


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

    assert pitcher_metrics_pass(global_metrics, candidate_metrics) is True
