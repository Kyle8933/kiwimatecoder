import pytest
from typer.testing import CliRunner

from kiwimatecoder import __version__, config, main
from kiwimatecoder.updater import build_update_command


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    """Point config storage at a temp dir and clear provider env vars."""
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_update_flag_invokes_updater(monkeypatch):
    calls = []

    def fake_update(console):
        calls.append(console)
        return 0

    monkeypatch.setattr(main, "run_update", fake_update)

    result = CliRunner().invoke(main.app, ["-update"])

    assert result.exit_code == 0
    assert len(calls) == 1


def test_update_long_flag_invokes_updater(monkeypatch):
    calls = []

    def fake_update(console):
        calls.append(console)
        return 0

    monkeypatch.setattr(main, "run_update", fake_update)

    result = CliRunner().invoke(main.app, ["--update"])

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


def test_version_flag_prints_version():
    result = CliRunner().invoke(main.app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_version_command_prints_version():
    result = CliRunner().invoke(main.app, ["version"])

    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_build_update_command_uses_current_python():
    command = build_update_command()

    assert command[1:4] == ["-m", "pip", "install"]
    assert "--upgrade" in command
    assert "--force-reinstall" in command
    assert command[-1].startswith("git+https://")
    assert "kiwimatecoder.git" in command[-1]


def test_setup_non_interactive_saves_key():
    result = CliRunner().invoke(
        main.app, ["setup", "--provider", "openai", "--key", "sk-openai"]
    )

    assert result.exit_code == 0
    assert result.output and "API key saved" in result.output
    assert config.get_key("openai") == "sk-openai"
    assert config.get_selected_provider_id() == "openai"


def test_setup_unknown_provider_exits_nonzero():
    result = CliRunner().invoke(
        main.app, ["setup", "--provider", "nope", "--key", "sk-x"]
    )

    assert result.exit_code == 1
    assert "Unknown provider" in result.output


def test_bare_config_shows_orientation():
    result = CliRunner().invoke(main.app, ["config"])

    assert result.exit_code == 0
    assert "Manage KiwiMateCoder configuration" in result.output
    assert "config key set" in result.output


def test_config_key_set_saves_key():
    result = CliRunner().invoke(
        main.app, ["config", "key", "set", "openai", "sk-openai"]
    )

    assert result.exit_code == 0
    assert config.get_key("openai") == "sk-openai"


def test_config_key_remove():
    config.set_key("openai", "sk-openai")
    result = CliRunner().invoke(main.app, ["config", "key", "remove", "openai"])

    assert result.exit_code == 0
    assert config.get_key("openai") is None


def test_config_provider_use_sets_default():
    result = CliRunner().invoke(main.app, ["config", "provider", "use", "openai"])

    assert result.exit_code == 0
    assert config.get_selected_provider_id() == "openai"


def test_config_provider_use_unknown_fails():
    result = CliRunner().invoke(main.app, ["config", "provider", "use", "nope"])

    assert result.exit_code == 1
    assert "Unknown provider" in result.output


def test_config_model_set_and_reset():
    result = CliRunner().invoke(
        main.app, ["config", "model", "set", "gpt-test"]
    )
    assert result.exit_code == 0
    assert config.load_config().get("selected_model") == "gpt-test"

    result = CliRunner().invoke(main.app, ["config", "model", "reset"])
    assert result.exit_code == 0
    assert config.load_config().get("selected_model") is None


def test_config_mode_set_and_reset():
    result = CliRunner().invoke(main.app, ["config", "mode", "set", "plan"])
    assert result.exit_code == 0
    assert config.get_default_mode() == "plan"

    result = CliRunner().invoke(main.app, ["config", "mode", "reset"])
    assert result.exit_code == 0
    assert config.get_default_mode() == "ask"


def test_config_show_prints_summary():
    config.set_key("openai", "sk-openai")
    result = CliRunner().invoke(main.app, ["config", "show"])

    assert result.exit_code == 0
    assert "Provider:" in result.output
    assert "Key:" in result.output


def test_legacy_set_key_alias_still_works():
    result = CliRunner().invoke(
        main.app, ["config", "set-key", "--provider", "openai", "sk-alias"]
    )

    assert result.exit_code == 0
    assert config.get_key("openai") == "sk-alias"


def test_legacy_check_alias_still_works():
    result = CliRunner().invoke(main.app, ["config", "check"])

    assert result.exit_code == 0
    assert "Provider:" in result.output
