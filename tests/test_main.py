import pytest
from typer.testing import CliRunner

from kiwimatecoder import __version__, config, main, ui
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


# ---------------------------------------------------------------------------
# Local providers
# ---------------------------------------------------------------------------


def test_setup_local_provider_needs_no_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setattr(main, "probe", lambda provider: False)
    config.set_selected_model("stale-model")

    result = CliRunner().invoke(main.app, ["setup", "--provider", "ollama"])

    assert result.exit_code == 0
    assert "needs no API key" in result.output
    assert config.get_selected_provider_id() == "ollama"
    # A stale model from another provider must not carry over.
    assert config.load_config()["selected_model"] is None
    assert config.get_key("ollama") is None


def test_setup_local_provider_warns_when_server_not_detected(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setattr(main, "probe", lambda provider: False)

    result = CliRunner().invoke(main.app, ["setup", "--provider", "ollama"])

    assert result.exit_code == 0
    assert "No server answered" in result.output


def test_setup_local_provider_with_explicit_key_uses_the_key_path(monkeypatch):
    result = CliRunner().invoke(
        main.app, ["setup", "--provider", "ollama", "--key", "optional-key"]
    )

    assert result.exit_code == 0
    assert config.get_key("ollama") == "optional-key"


def test_ask_with_local_provider_and_no_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    captured = {}

    async def fake_stream(prompt, api_key, model=None, provider=None):
        captured["api_key"] = api_key
        captured["model"] = model
        captured["provider"] = provider

    monkeypatch.setattr(main, "stream_response", fake_stream)
    monkeypatch.setattr(
        main, "resolve_default_model", lambda provider: "llama3.1:8b"
    )

    result = CliRunner().invoke(main.app, ["ask", "hi", "--provider", "ollama"])

    assert result.exit_code == 0
    assert captured["api_key"] == ""
    assert captured["model"] == "llama3.1:8b"
    assert captured["provider"].id == "ollama"


def test_setup_unsloth_without_key_prompts_instead_of_skipping(monkeypatch):
    """Unsloth is local but enforces auth, so setup must ask for its key."""
    monkeypatch.delenv("UNSLOTH_API_KEY", raising=False)

    result = CliRunner().invoke(main.app, ["setup", "--provider", "unsloth"])

    # Empty stdin cancels the key prompt; nothing is selected or stored.
    # (A cancelled setup exits 1, matching the picker-cancel convention.)
    assert result.exit_code == 1
    assert "nothing changed" in result.output
    assert "needs no API key" not in result.output
    assert config.get_selected_provider_id() == "openrouter"
    assert config.get_key("unsloth") is None


def test_setup_unsloth_with_key_saves_and_selects(monkeypatch):
    monkeypatch.delenv("UNSLOTH_API_KEY", raising=False)

    result = CliRunner().invoke(
        main.app, ["setup", "--provider", "unsloth", "--key", "sk-unsloth-test"]
    )

    assert result.exit_code == 0
    assert "API key saved" in result.output
    assert config.get_key("unsloth") == "sk-unsloth-test"
    assert config.get_selected_provider_id() == "unsloth"


def test_ask_with_unsloth_and_no_key_exits(monkeypatch):
    monkeypatch.delenv("UNSLOTH_API_KEY", raising=False)

    result = CliRunner().invoke(main.app, ["ask", "hi", "--provider", "unsloth"])

    assert result.exit_code == 1
    assert "No API key" in result.output


def test_ask_with_unsloth_and_key_proceeds(monkeypatch):
    monkeypatch.setenv("UNSLOTH_API_KEY", "sk-unsloth-test")
    captured = {}

    async def fake_stream(prompt, api_key, model=None, provider=None):
        captured["api_key"] = api_key
        captured["provider"] = provider

    monkeypatch.setattr(main, "stream_response", fake_stream)
    monkeypatch.setattr(
        main, "resolve_default_model", lambda provider: "unsloth/Qwen3.6-27B-GGUF"
    )

    result = CliRunner().invoke(main.app, ["ask", "hi", "--provider", "unsloth"])

    assert result.exit_code == 0
    assert captured["api_key"] == "sk-unsloth-test"
    assert captured["provider"].id == "unsloth"


def test_continue_resumes_autosave(tmp_path, monkeypatch):
    from kiwimatecoder import repl
    from kiwimatecoder.session import Session, save_session

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    monkeypatch.setattr("kiwimatecoder.session._sessions_dir", lambda: sessions_dir)
    saved = Session(
        provider_id="anthropic",
        model="claude-sonnet-5",
        workspace_root=tmp_path,
        messages=[{"role": "user", "content": "hi"}],
    )
    save_session(saved, "last")

    captured = {}
    monkeypatch.setattr(repl, "run", lambda session: captured.setdefault("session", session))

    result = CliRunner().invoke(main.app, ["--continue"])

    assert result.exit_code == 0
    assert captured["session"].provider_id == "anthropic"
    assert captured["session"].model == "claude-sonnet-5"


def test_continue_without_saved_session_starts_fresh(tmp_path, monkeypatch):
    from kiwimatecoder import repl

    monkeypatch.setattr(
        "kiwimatecoder.session._sessions_dir", lambda: tmp_path / "sessions"
    )
    captured = {}
    monkeypatch.setattr(
        repl, "run", lambda session: captured.setdefault("session", session)
    )

    result = CliRunner().invoke(main.app, ["--continue"])

    assert result.exit_code == 0
    assert "Starting fresh" in result.output
    assert captured["session"].provider_id == "openrouter"


def test_resume_missing_session_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "kiwimatecoder.session._sessions_dir", lambda: tmp_path / "sessions"
    )

    result = CliRunner().invoke(main.app, ["--resume", "nope"])

    assert result.exit_code == 1
    assert "Could not resume" in result.output


def test_doctor_command_runs():
    result = CliRunner().invoke(main.app, ["doctor"])

    assert result.exit_code == 0
    assert "diagnostics" in result.output.lower()


def test_config_sampling_set_show_reset():
    runner = CliRunner()

    result = runner.invoke(main.app, ["config", "sampling", "set", "temperature=0.2"])
    assert result.exit_code == 0
    assert config.get_sampling() == {"temperature": 0.2}

    result = runner.invoke(main.app, ["config", "sampling", "show"])
    assert result.exit_code == 0
    assert "0.2" in result.output

    result = runner.invoke(main.app, ["config", "sampling", "reset"])
    assert result.exit_code == 0
    assert config.get_sampling() == {}


def test_config_sampling_rejects_invalid():
    result = CliRunner().invoke(main.app, ["config", "sampling", "set", "temperature=9"])

    assert result.exit_code == 1
    assert config.get_sampling() == {}


def test_config_style_roundtrip():
    runner = CliRunner()

    result = runner.invoke(main.app, ["config", "style", "set", "concise"])
    assert result.exit_code == 0
    assert config.get_output_style() == "concise"

    result = runner.invoke(main.app, ["config", "style", "show"])
    assert "concise" in result.output

    result = runner.invoke(main.app, ["config", "style", "set", "fancy"])
    assert result.exit_code == 1


def test_config_prompt_set_show_clear():
    runner = CliRunner()

    result = runner.invoke(main.app, ["config", "prompt", "set", "Always use type hints."])
    assert result.exit_code == 0
    assert config.get_system_prompt() == "Always use type hints."

    result = runner.invoke(main.app, ["config", "prompt", "show"])
    assert "Always use type hints." in result.output

    result = runner.invoke(main.app, ["config", "prompt", "clear"])
    assert result.exit_code == 0
    assert config.get_system_prompt() is None


def test_config_permissions_roundtrip():
    runner = CliRunner()

    config.persist_always_allowed_tool("run_bash")
    result = runner.invoke(main.app, ["config", "permissions", "list"])
    assert "run_bash" in result.output

    result = runner.invoke(main.app, ["config", "permissions", "remove", "run_bash"])
    assert result.exit_code == 0
    assert config.get_always_allowed_tools() == []


def test_config_commands_roundtrip():
    runner = CliRunner()

    result = runner.invoke(main.app, ["config", "commands", "deny", r"rm -rf"])
    assert result.exit_code == 0
    assert config.get_command_rules()["deny"] == [r"rm -rf"]

    result = runner.invoke(main.app, ["config", "commands", "list"])
    assert "rm -rf" in result.output

    result = runner.invoke(main.app, ["config", "commands", "remove", "deny", r"rm -rf"])
    assert result.exit_code == 0
    assert config.get_command_rules()["deny"] == []

    result = runner.invoke(main.app, ["config", "commands", "clear"])
    assert result.exit_code == 0
    assert config.get_command_rules() == {"allow": [], "deny": []}


def test_config_show_mentions_new_settings():
    result = CliRunner().invoke(main.app, ["config", "show"])

    assert result.exit_code == 0
    assert "Output style" in result.output
    assert "Sampling" in result.output
    assert "Always-allowed tools" in result.output


def test_config_trusted_workspace_roundtrip():
    runner = CliRunner()

    result = runner.invoke(main.app, ["config", "trusted-workspace", "on"])
    assert result.exit_code == 0
    assert config.get_trusted_workspace() is True

    result = runner.invoke(main.app, ["config", "trusted-workspace"])
    assert "on" in result.output

    result = runner.invoke(main.app, ["config", "trusted-workspace", "off"])
    assert result.exit_code == 0
    assert config.get_trusted_workspace() is False


def test_config_verify_and_budget_roundtrip():
    runner = CliRunner()

    result = runner.invoke(main.app, ["config", "verify", "set", "pytest -q"])
    assert result.exit_code == 0
    assert config.get_verify_command() == "pytest -q"

    result = runner.invoke(main.app, ["config", "budget", "tokens", "1000"])
    assert result.exit_code == 0
    assert config.get_budget() == {"max_tokens": 1000}

    result = runner.invoke(main.app, ["config", "budget", "clear"])
    assert result.exit_code == 0
    assert config.get_budget() == {}


def test_config_web_roundtrip():
    runner = CliRunner()

    result = runner.invoke(main.app, ["config", "web", "max-chars", "60000"])
    assert result.exit_code == 0
    assert config.get_web()["max_chars"] == 60000

    result = runner.invoke(main.app, ["config", "web", "timeout", "9"])
    assert result.exit_code == 0
    assert config.get_web()["timeout"] == 9.0

    result = runner.invoke(main.app, ["config", "web", "allow-local", "on"])
    assert result.exit_code == 0
    assert config.get_web()["allow_local"] is True

    result = runner.invoke(main.app, ["config", "web", "show"])
    assert result.exit_code == 0
    assert "60000" in result.output
    assert "on" in result.output


def test_config_web_rejects_invalid_values():
    result = CliRunner().invoke(main.app, ["config", "web", "max-chars", "5"])

    assert result.exit_code == 1
    assert config.get_web()["max_chars"] == 50000

    result = CliRunner().invoke(main.app, ["config", "web", "allow-local", "maybe"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Profiles and config validation
# ---------------------------------------------------------------------------


def test_config_cache_roundtrip():
    runner = CliRunner()

    result = runner.invoke(main.app, ["config", "cache"])
    assert result.exit_code == 0
    assert "off" in result.output

    result = runner.invoke(main.app, ["config", "cache", "on"])
    assert result.exit_code == 0
    assert config.get_prompt_cache() is True

    result = runner.invoke(main.app, ["config", "cache"])
    assert "on" in result.output

    result = runner.invoke(main.app, ["config", "cache", "off"])
    assert result.exit_code == 0
    assert config.get_prompt_cache() is False

    result = runner.invoke(main.app, ["config", "cache", "maybe"])
    assert result.exit_code == 1


def test_config_profile_save_list_show_use_remove_roundtrip():
    runner = CliRunner()
    config.set_selected_provider("openai")
    config.set_selected_model("gpt-test")

    result = runner.invoke(main.app, ["config", "profile", "save", "work"])
    assert result.exit_code == 0
    assert config.get_profile("work")["provider"] == "openai"

    result = runner.invoke(main.app, ["config", "profile", "list"])
    assert result.exit_code == 0
    assert "work" in result.output

    result = runner.invoke(main.app, ["config", "profile", "show", "work"])
    assert result.exit_code == 0
    assert "gpt-test" in result.output

    config.set_selected_provider("deepseek")
    result = runner.invoke(main.app, ["config", "profile", "use", "work"])
    assert result.exit_code == 0
    assert config.get_selected_provider_id() == "openai"

    result = runner.invoke(main.app, ["config", "profile", "remove", "work"])
    assert result.exit_code == 0
    assert config.get_profile("work") is None


def test_config_profile_use_and_show_unknown_exit_nonzero():
    runner = CliRunner()

    result = runner.invoke(main.app, ["config", "profile", "use", "nope"])
    assert result.exit_code == 1

    result = runner.invoke(main.app, ["config", "profile", "show", "nope"])
    assert result.exit_code == 1


def test_config_profile_rename():
    config.save_profile("one", {"mode": "plan"})

    result = CliRunner().invoke(main.app, ["config", "profile", "rename", "one", "two"])

    assert result.exit_code == 0
    assert config.get_profile("two") == {"mode": "plan"}


def test_config_validate_clean_and_broken():
    runner = CliRunner()

    result = runner.invoke(main.app, ["config", "validate"])
    assert result.exit_code == 0
    assert "valid" in result.output.lower()

    cfg = config.load_config()
    cfg["default_mode"] = "bogus"
    config.save_config(cfg)

    result = runner.invoke(main.app, ["config", "validate"])
    assert result.exit_code == 1
    assert "default_mode" in result.output


def test_launch_with_profile_applies_session_without_persisting(monkeypatch):
    from kiwimatecoder import repl

    config.set_selected_provider("openai")
    config.set_selected_model("gpt-profile")
    config.set_default_mode("plan")
    config.save_profile("work")
    config.set_selected_provider("deepseek")
    config.set_selected_model(None)
    config.set_default_mode("ask")

    captured = {}
    monkeypatch.setattr(
        repl, "run", lambda session: captured.setdefault("session", session)
    )

    result = CliRunner().invoke(main.app, ["--profile", "work"])

    assert result.exit_code == 0
    assert captured["session"].provider_id == "openai"
    assert captured["session"].model == "gpt-profile"
    assert captured["session"].mode.value == "plan"
    # The overlay must not touch persisted config.
    assert config.get_selected_provider_id() == "deepseek"
    assert config.get_default_mode() == "ask"


def test_launch_with_unknown_profile_exits_nonzero(monkeypatch):
    from kiwimatecoder import repl

    monkeypatch.setattr(repl, "run", lambda session: None)

    result = CliRunner().invoke(main.app, ["--profile", "nope"])

    assert result.exit_code == 1
    assert "Unknown profile" in result.output


def test_profile_with_resume_overlays_mode_and_model(tmp_path, monkeypatch):
    from kiwimatecoder import repl
    from kiwimatecoder.session import Session, save_session

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    monkeypatch.setattr("kiwimatecoder.session._sessions_dir", lambda: sessions_dir)
    save_session(
        Session(
            provider_id="anthropic",
            model="claude-old",
            workspace_root=tmp_path,
            messages=[{"role": "user", "content": "hi"}],
        ),
        "keep",
    )
    config.save_profile("work", {"mode": "plan", "model": "claude-new"})

    captured = {}
    monkeypatch.setattr(
        repl, "run", lambda session: captured.setdefault("session", session)
    )

    result = CliRunner().invoke(main.app, ["--resume", "keep", "--profile", "work"])

    assert result.exit_code == 0
    assert captured["session"].provider_id == "anthropic"
    assert captured["session"].model == "claude-new"
    assert captured["session"].mode.value == "plan"


# ---------------------------------------------------------------------------
# `config ui` CLI
# ---------------------------------------------------------------------------


def test_config_ui_roundtrip():
    runner = CliRunner()

    result = runner.invoke(main.app, ["config", "ui", "color", "never"])
    assert result.exit_code == 0
    assert config.get_ui()["color"] == "never"

    result = runner.invoke(main.app, ["config", "ui", "output", "compact"])
    assert result.exit_code == 0
    assert config.get_ui()["output_mode"] == "compact"

    result = runner.invoke(main.app, ["config", "ui", "ascii", "on"])
    assert result.exit_code == 0
    assert config.get_ui()["ascii"] is True

    result = runner.invoke(main.app, ["config", "ui", "theme", "ocean"])
    assert result.exit_code == 0
    assert config.get_ui()["theme"] == "ocean"

    result = runner.invoke(main.app, ["config", "ui", "show"])
    assert result.exit_code == 0
    assert "ocean" in result.output
    assert "never" in result.output
    assert "compact" in result.output


def test_config_ui_ascii_off_roundtrip():
    runner = CliRunner()
    runner.invoke(main.app, ["config", "ui", "ascii", "on"])

    result = runner.invoke(main.app, ["config", "ui", "ascii", "off"])

    assert result.exit_code == 0
    assert config.get_ui()["ascii"] is False


def test_config_ui_rejects_invalid_values():
    runner = CliRunner()

    for args in (
        ["config", "ui", "color", "rainbow"],
        ["config", "ui", "output", "loud"],
        ["config", "ui", "ascii", "maybe"],
        ["config", "ui", "theme", "neon"],
    ):
        result = runner.invoke(main.app, args)
        assert result.exit_code == 1, args

    assert config.get_ui() == ui.UI_DEFAULTS


def test_config_confirmations_use_ascii_glyph_when_enabled():
    runner = CliRunner()
    runner.invoke(main.app, ["config", "ui", "ascii", "on"])

    result = runner.invoke(main.app, ["config", "key", "set", "openai", "sk-openai"])

    assert result.exit_code == 0
    assert "[ok]" in result.output
    assert "✓" not in result.output


def test_config_show_mentions_ui_settings():
    result = CliRunner().invoke(main.app, ["config", "show"])

    assert result.exit_code == 0
    assert "UI:" in result.output
    assert "theme" in result.output
    assert "output" in result.output
