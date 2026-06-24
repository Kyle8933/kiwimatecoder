from typer.testing import CliRunner

from kiwimatecoder import main
from kiwimatecoder.updater import build_update_command


def test_update_flag_invokes_updater(monkeypatch):
    calls = []

    def fake_update(console):
        calls.append(console)
        return 0

    monkeypatch.setattr(main, "run_update", fake_update)

    result = CliRunner().invoke(main.app, ["-update"])

    assert result.exit_code == 0
    assert len(calls) == 1


def test_update_command_invokes_updater(monkeypatch):
    calls = []

    def fake_update(console):
        calls.append(console)
        return 0

    monkeypatch.setattr(main, "run_update", fake_update)

    result = CliRunner().invoke(main.app, ["update"])

    assert result.exit_code == 0
    assert len(calls) == 1


def test_build_update_command_uses_current_python():
    command = build_update_command()

    assert command[-3:] == ["install", "--upgrade", "kiwimatecoder"]
    assert command[1:4] == ["-m", "pip", "install"]
