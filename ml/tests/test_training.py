from pathlib import Path

import pandas as pd
import pytest
from oyarzabal.training import (
    global_specs,
    load_residual_tuning_manifest,
    specialist_specs,
    validation_folds,
)


def test_validation_folds_are_strictly_time_ordered() -> None:
    rows = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                ["2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"]
            )
        }
    )
    folds = validation_folds(rows)
    assert [year for year, _, _ in folds] == [2023, 2024, 2025]
    for _, train, evaluation in folds:
        assert rows.loc[train, "game_date"].max() < rows.loc[
            evaluation, "game_date"
        ].min()


def test_validation_folds_require_every_evaluation_year() -> None:
    rows = pd.DataFrame(
        {"game_date": pd.to_datetime(["2022-01-01", "2023-01-01", "2024-01-01"])}
    )
    with pytest.raises(ValueError, match="2025"):
        validation_folds(rows)


def test_specialist_candidate_matrix_is_deliberately_small() -> None:
    specs = specialist_specs()
    assert len(specs) == 4
    assert {spec.max_depth for spec in specs} == {3, 4}
    assert {spec.weight_mode for spec in specs} == {"none", "sqrt"}
    assert {spec.min_child_weight for spec in specs} == {8}


def test_global_candidate_matrix_focuses_on_weight_and_depth() -> None:
    specs = global_specs()
    assert len(specs) == 6
    assert {spec.feature_set for spec in specs} == {"global"}
    assert {spec.max_depth for spec in specs} == {4, 6}
    assert {spec.weight_mode for spec in specs} == {"none", "light", "sqrt"}
    assert {spec.min_child_weight for spec in specs} == {8}


def test_residual_tuning_manifest_freezes_small_temporal_grid() -> None:
    manifest = load_residual_tuning_manifest(
        Path("config/v7-residual-tuning.json")
    )

    assert manifest["selectionYear"] == 2024
    assert manifest["confirmationYear"] == 2025
    assert manifest["forbiddenProspectiveStart"] == "2026-07-26"
    assert len(manifest["reliabilityScaleBoosts"]) == 3
    assert len(manifest["contextGatePowers"]) == 3
    assert len(manifest["limitedScaleBoosts"]) == 3
    assert manifest["opened2026Policy"]["selectionUse"] is False
