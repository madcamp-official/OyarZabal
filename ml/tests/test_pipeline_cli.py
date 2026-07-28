import pandas as pd
import pytest
from oyarzabal.pipeline import _parsed_probabilities, parse_args


def test_demo_cli_requires_external_inputs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        parse_args([])

    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "--history" in message
    assert "--game" in message
    assert "--predictions" in message


def test_parsed_probabilities_accepts_grouped_prediction_json() -> None:
    probabilities = _parsed_probabilities(
        pd.Series(
            [
                (
                    '{"FOUR_SEAM": 0.4, "MOVING_FASTBALL": 0.2, '
                    '"SLIDER": 0.15, "CURVE": 0.1, '
                    '"CHANGEUP": 0.1, "SPLITTER_FORK": 0.05}'
                )
            ]
        )
    )

    assert probabilities.shape == (1, 6)
    assert probabilities[0].sum() == pytest.approx(1.0)
    assert probabilities[0, 0] == pytest.approx(0.4)
