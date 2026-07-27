import numpy as np
import pandas as pd
import pytest
from oyarzabal.modeling import (
    apply_temperature,
    candidate_specs,
    class_sample_weights,
    select_candidate,
    time_fold_masks,
)


def test_sqrt_class_weights_have_unit_sample_mean_and_are_bounded() -> None:
    labels = np.array([0, 0, 0, 0, 1, 2, 3, 4, 5])
    light = class_sample_weights(labels, "light")
    weights = class_sample_weights(labels, "sqrt")
    assert light.mean() == pytest.approx(1)
    assert weights.mean() == pytest.approx(1)
    assert weights.min() >= 0.5
    assert weights.max() <= 3
    assert light[-1] < weights[-1]
    assert light[0] > weights[0]
    assert weights[-1] > weights[0]


def test_temperature_preserves_argmax_and_probability_mass() -> None:
    probabilities = np.array([[0.8, 0.1, 0.1], [0.2, 0.5, 0.3]])
    adjusted = apply_temperature(probabilities, 1.7)
    assert adjusted.argmax(axis=1).tolist() == [0, 1]
    assert adjusted.sum(axis=1).tolist() == pytest.approx([1, 1])


def test_candidate_selection_enforces_balance_gates_and_tie_breaks() -> None:
    reference = {
        "accuracy": 0.52,
        "hierarchicalAccuracy": 0.60,
        "macroF1": 0.30,
        "logLoss": 1.20,
        "zeroRecallClasses": ["CURVE"],
        "majorityPredictionGap": 0.30,
    }
    rejected = {
        "name": "accurate-but-collapsed",
        "metrics": {
            "accuracy": 0.53,
            "hierarchicalAccuracy": 0.61,
            "macroF1": 0.31,
            "logLoss": 1.10,
            "zeroRecallClasses": ["CURVE"],
            "majorityPredictionGap": 0.25,
        },
    }
    lower_loss = {
        "name": "lower-loss",
        "metrics": {
            "accuracy": 0.518,
            "hierarchicalAccuracy": 0.598,
            "macroF1": 0.34,
            "logLoss": 1.10,
            "zeroRecallClasses": [],
            "majorityPredictionGap": 0.10,
        },
    }
    tied_better_f1 = {
        "name": "balanced",
        "metrics": {
            "accuracy": 0.516,
            "hierarchicalAccuracy": 0.596,
            "macroF1": 0.38,
            "logLoss": 1.104,
            "zeroRecallClasses": [],
            "majorityPredictionGap": 0.12,
        },
    }

    selected = select_candidate(reference, [rejected, lower_loss, tied_better_f1])

    assert selected["name"] == "balanced"


def test_candidate_matrix_and_time_folds_are_fixed() -> None:
    specs = candidate_specs()
    assert len(specs) == 24
    assert {spec.feature_set for spec in specs} == {"legacy", "enriched"}
    assert {spec.weight_mode for spec in specs} == {"none", "light", "sqrt"}

    rows = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                ["2021-12-31", "2022-01-01", "2023-01-01", "2024-01-01"]
            )
        }
    )
    folds = time_fold_masks(rows)
    assert [fold.name for fold, _, _ in folds] == [
        "through-2021-to-2022",
        "through-2022-to-2023",
        "through-2023-to-2024",
    ]
    for _, train_mask, evaluation_mask in folds:
        assert (
            rows.loc[train_mask, "game_date"].max()
            < rows.loc[evaluation_mask, "game_date"].min()
        )
