import pytest
from oyarzabal.pipeline import parse_args


def test_demo_cli_requires_external_inputs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        parse_args([])

    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "--history" in message
    assert "--game" in message
    assert "--predictions" in message
