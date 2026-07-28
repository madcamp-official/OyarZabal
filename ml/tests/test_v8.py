from copy import deepcopy

from oyarzabal.v8 import V74_PERSONALIZATION, _candidate_passes


def _metrics() -> dict[str, object]:
    return {
        "logLoss": 1.2,
        "accuracy": 0.45,
        "familyAccuracy": 0.57,
        "hierarchicalAccuracy": 0.51,
        "macroF1": 0.42,
        "totalVariationDistance": 0.08,
        "maxClassCalibrationError": 0.04,
        "zeroRecallClasses": [],
        "actualDistribution": {
            "FOUR_SEAM": 0.3,
            "MOVING_FASTBALL": 0.25,
            "SLIDER": 0.2,
            "CURVE": 0.1,
            "CHANGEUP": 0.12,
            "SPLITTER_FORK": 0.03,
        },
    }


def test_v8_keeps_v74_personalization_controls_and_rejects_unsafe_blend():
    assert V74_PERSONALIZATION == {
        "reliabilityScaleBoost": 1.5,
        "contextGatePower": 0.5,
        "fullTierBoost": 4.0,
        "limitedTierBoost": 4.0,
        "shadowTierBoost": 0.0,
        "scaleCap": 0.5,
        "jsDivergenceCap": 0.05,
        "classProbabilityShiftCap": 0.2,
    }
    baseline = _metrics()
    candidate = deepcopy(baseline)
    candidate["logLoss"] = 1.19
    assert _candidate_passes(baseline, candidate)

    candidate["macroF1"] = 0.414
    assert not _candidate_passes(baseline, candidate)
