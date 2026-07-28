from oyarzabal.v83 import _failure_reasons


def test_v83_failure_reasons_report_balance_regressions():
    reference = {
        "logLoss": 1.1,
        "accuracy": 0.49,
        "familyAccuracy": 0.59,
        "hierarchicalAccuracy": 0.54,
        "macroF1": 0.46,
        "totalVariationDistance": 0.08,
        "maxClassCalibrationError": 0.05,
    }
    candidate = {
        **reference,
        "logLoss": 1.07,
        "macroF1": 0.42,
        "totalVariationDistance": 0.16,
        "zeroRecallClasses": [],
        "actualDistribution": {},
    }

    reasons = _failure_reasons(reference, candidate, {"ciLower": 0.01})

    assert reasons == [
        "macro_f1_drop_gt_0.5pp",
        "tvd_regression_gt_0.5pp",
    ]
