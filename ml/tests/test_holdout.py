import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from oyarzabal.holdout import (
    COMPARISON_COHORT_ID,
    V5_EVALUATION_PITCHER_IDS,
    _registry,
    _scale_summary,
    evaluate_frozen_holdout,
    evaluation_sample_fingerprint,
    frozen_rows,
    load_prospective_manifest,
    promotion_sample_is_prospective,
    prospective_metric_failure_reasons,
    prospective_sample_status,
)


def _pitch(path: Path, game_date: str, game_pk: int) -> None:
    pd.DataFrame(
        {
            "pitch_type": ["FF"],
            "game_date": [game_date],
            "game_pk": [game_pk],
            "pitcher": [10],
            "batter": [20],
        }
    ).to_parquet(path)


def test_frozen_rows_keeps_2026_out_of_history(tmp_path: Path) -> None:
    history = tmp_path / "history"
    holdout = tmp_path / "holdout"
    history.mkdir()
    holdout.mkdir()
    _pitch(history / "2025.parquet", "2025-09-01", 1)
    _pitch(holdout / "2026.parquet", "2026-04-01", 2)

    rows = frozen_rows(history, holdout)
    assert rows["game_date"].dt.year.tolist() == [2026]

    _pitch(history / "leak.parquet", "2026-01-01", 3)
    with pytest.raises(ValueError, match="training history"):
        frozen_rows(history, holdout)


def test_registry_rejects_any_post_2025_training_data() -> None:
    payload = {
        "dataCutoff": "2025-12-31",
        "specialists": {
            "10": {
                "pitcherId": 10,
                "enabled": True,
                "specialistWeight": 0.5,
                "model": "pooled-residual.pkl",
                "dataCutoff": "2026-01-01",
                "status": "active",
                "residualScale": 0.5,
            }
        },
    }
    with pytest.raises(ValueError, match="post-2025"):
        _registry(json.loads(json.dumps(payload)))


def test_registry_reads_v6_reliability_metadata() -> None:
    payload = {
        "schemaVersion": 5,
        "dataCutoff": "2025-11-01",
        "specialists": {
            "10": {
                "pitcherId": 10,
                "enabled": True,
                "specialistWeight": 0,
                "model": "pooled-residual.pkl",
                "dataCutoff": "2025-09-01",
                "status": "active",
                "residualScale": None,
                "reliability": 0.2,
                "reliabilityComponents": {
                    "n": 1_000,
                    "supportCoefficient": 0.5,
                    "pAll": 0.8,
                    "pRecent": 0.5,
                },
                "selectionRank": None,
                "support": {"2025": 500},
            }
        },
    }

    entry = _registry(payload)[10]

    assert entry.reliability == 0.2
    assert entry.reliability_components["pRecent"] == 0.5

    payload["dataCutoff"] = "2026-01-01"
    payload["specialists"]["10"]["dataCutoff"] = "2025-12-31"
    with pytest.raises(ValueError, match="post-2025"):
        _registry(json.loads(json.dumps(payload)))


def test_registry_reads_v7_tiers_and_scale_multipliers() -> None:
    payload = {
        "schemaVersion": 7,
        "decisionRule": "family-sum-then-child",
        "dataCutoff": "2025-11-01",
        "specialists": {
            "10": {
                "pitcherId": 10,
                "enabled": True,
                "specialistWeight": 0,
                "model": "pooled-residual.pkl",
                "dataCutoff": "2025-09-01",
                "status": "limited",
                "reliability": 0.2,
                "scaleMultiplier": 0.4,
                "stale": True,
                "incrementalValidation": {"strict2025Passed": False},
            },
            "20": {
                "pitcherId": 20,
                "enabled": False,
                "specialistWeight": 0,
                "model": "",
                "dataCutoff": "2025-09-01",
                "status": "shadow",
                "reliability": 0.1,
                "scaleMultiplier": 0,
                "stale": False,
            },
        },
    }

    registry = _registry(payload)

    assert registry[10].status == "limited"
    assert registry[10].scale_multiplier == 0.4
    assert registry[10].stale is True
    assert registry[10].incremental_validation["strict2025Passed"] is False
    assert registry[20].enabled is False
    assert registry[20].scale_multiplier == 0


def test_v5_evaluation_cohort_is_frozen_and_sample_hash_ignores_row_order() -> None:
    assert COMPARISON_COHORT_ID == "v5-enabled-pitchers-v1"
    assert len(V5_EVALUATION_PITCHER_IDS) == 30
    rows = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2026-04-02", "2026-04-01"]),
            "game_pk": [2, 1],
            "at_bat_number": [1, 1],
            "pitch_number": [1, 1],
            "pitcher_id": [543037, 458681],
            "batter_id": [20, 21],
            "target": [0, 1],
        }
    )

    assert evaluation_sample_fingerprint(rows) == evaluation_sample_fingerprint(
        rows.iloc[::-1]
    )


def test_only_post_cutoff_rows_are_prospective() -> None:
    opened = pd.DataFrame({"game_date": pd.to_datetime(["2026-07-25"])})
    future = pd.DataFrame({"game_date": pd.to_datetime(["2026-07-26"])})

    assert promotion_sample_is_prospective(opened) is False
    assert promotion_sample_is_prospective(future) is True


def test_v7_prospective_manifest_freezes_candidates_and_policy() -> None:
    manifest = load_prospective_manifest(
        Path("config/v7-prospective.json")
    )

    assert manifest["deploymentStatus"] == "active"
    assert manifest["deploymentBasis"] == "taxonomy-product-generation"
    assert manifest["evaluationPurpose"] == "post-deployment-performance-certification"
    assert manifest["prospectiveStart"] == "2026-07-26"
    assert [
        (candidate["id"], candidate["limitedScaleBoost"])
        for candidate in manifest["candidates"]
    ] == [
        ("v7.2-reliability-1.5-gate-0.5", 1.0),
    ]
    assert (
        manifest["metricPolicy"]["maximumDistributionRegression"]
        == 0.005
    )


def test_prospective_sample_status_excludes_opened_2026_rows() -> None:
    manifest = load_prospective_manifest(
        Path("config/v7-prospective.json")
    )
    rows = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                ["2026-07-25", "2026-07-26", "2026-08-24"]
            )
        }
    )

    status = prospective_sample_status(rows, manifest)

    assert status["rows"]["game_date"].dt.day.tolist() == [26, 24]
    assert status["rowCount"] == 2
    assert status["days"] == 30
    assert status["dateAndRowThresholdsMet"] is False


def test_prospective_metric_gate_includes_family_and_distribution_tolerance() -> None:
    manifest = load_prospective_manifest(
        Path("config/v7-prospective.json")
    )
    policy = manifest["metricPolicy"]
    reference = {
        "accuracy": 0.50,
        "familyAccuracy": 0.65,
        "hierarchicalAccuracy": 0.575,
        "macroF1": 0.45,
        "zeroRecallClasses": [],
        "actualDistribution": {"FOUR_SEAM": 1.0},
        "maxClassShareError": 0.05,
        "totalVariationDistance": 0.06,
        "maxClassCalibrationError": 0.04,
    }
    candidate = {
        **reference,
        "familyAccuracy": 0.644,
        "totalVariationDistance": 0.066,
    }

    reasons = prospective_metric_failure_reasons(
        reference,
        candidate,
        {"ciLower": 0.001},
        policy,
    )

    assert "family_accuracy_drop_gt_tolerance" in reasons
    assert "total_variation_distance_failed" in reasons


def test_prospective_evaluation_does_not_open_metrics_before_start(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    candidate.mkdir()
    reference.mkdir()
    registry = json.dumps({"dataCutoff": "2025-11-01"}).encode()
    candidate_files = {
        "global.pkl": b"global",
        "pooled-residual.pkl": b"residual",
        "context-gate.pkl": b"gate",
        "registry.json": registry,
    }
    reference_files = {
        "global.pkl": b"global",
        "pooled-residual.pkl": b"v5-residual",
        "registry.json": registry,
    }
    for directory, files in (
        (candidate, candidate_files),
        (reference, reference_files),
    ):
        for name, content in files.items():
            (directory / name).write_bytes(content)
    manifest = {
        "prospectiveStart": "2026-07-26",
        "candidates": [
            {
                "id": "v7.0",
                "limitedScaleBoost": 1.0,
                "promotionPriority": 1,
            }
        ],
        "firstLook": {
            "minimumDays": 30,
            "minimumRows": 100_000,
            "minimumIntervenedRows": 15_000,
        },
        "candidateModelSha256": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in candidate_files.items()
        },
        "referenceModelSha256": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in reference_files.items()
        },
    }

    result = evaluate_frozen_holdout(
        pd.DataFrame(
            {"game_date": pd.to_datetime(["2026-07-25"])}
        ),
        candidate,
        reference_model_directory=reference,
        prospective_manifest=manifest,
    )

    validation = result["prospectiveValidation"]
    assert validation["rowCount"] == 0
    assert validation["firstLookConsumed"] is False
    assert validation["selectedCandidateId"] is None
    assert "metrics" not in validation["candidates"]["v7.0"]


def test_scale_summary_reports_applied_distribution() -> None:
    summary = _scale_summary(np.array([0, 0.1, 0.2, 0.3]))

    assert summary["nonZero"] == 3
    assert summary["nonZeroMean"] == pytest.approx(0.2)
    assert summary["nonZeroP50"] == pytest.approx(0.2)
