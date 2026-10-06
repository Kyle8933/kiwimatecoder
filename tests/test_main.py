import io

import pytest
from rich.console import Console
from typer.testing import CliRunner

from kiwimatecoder import __version__, catalog, config, main, ui
from kiwimatecoder.providers import REGISTRY
from kiwimatecoder.updater import build_update_command


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    """Point config storage at a temp dir and clear provider env vars."""
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for provider in REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)


def _install_fetch(monkeypatch, model_ids, calls=None):
    """Patch the network fetch to return ``model_ids`` newest-first."""

    def fake_fetch(provider, api_key=None, **kwargs):
        if calls is not None:
            calls.append(provider.id)
        return [
            catalog.RemoteModel(model_id, float(len(model_ids) - index))
            for index, model_id in enumerate(model_ids)
        ]

    monkeypatch.setattr(config.catalog, "fetch_models", fake_fetch)


def _forbid_fetch(monkeypatch):
    def fake_fetch(provider, api_key=None, **kwargs):
        raise AssertionError("the network must not be touched here")

    monkeypatch.setattr(config.catalog, "fetch_models", fake_fetch)


def _flat(text: str) -> str:
    """Collapse Rich's line wrapping so assertions can match whole sentences."""
    return " ".join(text.split())


def _capture_console(monkeypatch) -> io.StringIO:
    """Route main's console to a buffer for functions called directly."""
    buffer = io.StringIO()
    monkeypatch.setattr(
        main, "console", Console(file=buffer, force_terminal=False, width=200)
    )
    return buffer


def test_update_flag_invokes_updater(monkeypatch):
    calls = []

    def fake_update(console, ref=None):
        calls.append(console)
        return 0

    monkeypatch.setattr(main, "run_update", fake_update)

    result = CliRunner().invoke(main.app, ["-update"])

    assert result.exit_code == 0
    assert len(calls) == 1


def test_update_long_flag_invokes_updater(monkeypatch):
    calls = []

    def fake_update(console, ref=None):
        calls.append(console)
        return 0

    monkeypatch.setattr(main, "run_update", fake_update)

    result = CliRunner().invoke(main.app, ["--update"])

    assert result.exit_code == 0
    assert len(calls) == 1


def test_update_command_invokes_updater(monkeypatch):
    calls = []

    def fake_update(console, ref=None):
        calls.append(console)
        return 0

    monkeypatch.setattr(main, "run_update", fake_update)

    result = CliRunner().invoke(main.app, ["update"])

    assert result.exit_code == 0
    assert len(calls) == 1


def test_update_command_passes_ref(monkeypatch):
    refs = []

    def fake_update(console, ref=None):
        refs.append(ref)
        return 0

    monkeypatch.setattr(main, "run_update", fake_update)

    result = CliRunner().invoke(main.app, ["update", "--ref", "v0.2.0"])

    assert result.exit_code == 0
    assert refs == ["v0.2.0"]


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


def test_config_provider_add_accepts_auth_flags():
    result = CliRunner().invoke(
        main.app,
        [
            "config",
            "provider",
            "add",
            "my-azure",
            "My Azure",
            "https://my-resource.openai.azure.com/openai/v1",
            "my-deployment",
            "--key-env",
            "AZURE_OPENAI_API_KEY",
            "--key-header",
            "api-key",
            "--key-prefix",
            "",
            "--api-version",
            "2024-10-21",
        ],
    )

    assert result.exit_code == 0
    provider = config.get_provider_config("my-azure")
    assert provider.key_header == "api-key"
    assert provider.key_prefix == ""
    assert provider.api_version == "2024-10-21"
    assert config.get_provider_model("my-azure") == "my-deployment"


def test_config_provider_add_requires_a_model_and_prints_it():
    runner = CliRunner()
    base = ["config", "provider", "add", "my-llm", "My LLM", "https://llm.example/v1"]

    missing = runner.invoke(main.app, base)
    assert missing.exit_code == 2
    assert config.load_config()["providers"] == {}

    result = runner.invoke(main.app, [*base, "my-model"])
    assert result.exit_code == 0
    assert "with model my-model." in _flat(result.output)
    assert config.get_provider_model("my-llm") == "my-model"
    # The model is a per-provider choice, not a field of the provider entry.
    assert "default_model" not in config.load_config()["providers"]["my-llm"]


def test_config_provider_edit_changes_the_model_and_keeps_the_old_flag():
    runner = CliRunner()
    config.add_provider("my-llm", "My LLM", "https://llm.example/v1", "first")

    result = runner.invoke(
        main.app, ["config", "provider", "edit", "my-llm", "--model", "second"]
    )
    assert result.exit_code == 0
    assert config.get_provider_model("my-llm") == "second"

    result = runner.invoke(
        main.app, ["config", "provider", "edit", "my-llm", "--default-model", "third"]
    )
    assert result.exit_code == 0
    assert config.get_provider_model("my-llm") == "third"


def test_config_provider_list_shows_experimental_name_and_chosen_model(monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    config.set_provider_model("kiwimate", "kiwimate-mini-1-0")

    result = CliRunner().invoke(main.app, ["config", "provider", "list"])

    assert result.exit_code == 0
    rows = {
        cells[0]: cells[1:]
        for cells in (
            [cell.strip() for cell in line.strip("│┃ ").split("│")]
            for line in result.output.splitlines()
            if line.startswith("│")
        )
    }
    assert rows["kiwimate"][:2] == ["KiwiMate (experimental)", "kiwimate-mini-1-0"]
    # No model chosen: cloud providers show a dash, local servers list theirs.
    assert rows["openai"][:2] == ["OpenAI", "—"]
    assert rows["ollama"][1] == "(from server)"
    assert "(experimental)" not in rows["openrouter"][0]


def test_config_model_set_and_reset():
    # A stale global pin (older config, profile) must not outrank the choice.
    cfg = config.load_config()
    cfg["selected_model"] = "stale-pin"
    config.save_config(cfg)
    assert config.get_provider_model("openrouter") == "stale-pin"

    result = CliRunner().invoke(
        main.app, ["config", "model", "set", "gpt-test"]
    )
    assert result.exit_code == 0
    assert "Model for OpenRouter set to gpt-test." in _flat(result.output)
    # The choice belongs to the active provider, not a global pin.
    assert config.get_provider_model("openrouter") == "gpt-test"
    assert config.load_config()["provider_models"] == {"openrouter": "gpt-test"}
    assert config.load_config().get("selected_model") is None

    result = CliRunner().invoke(main.app, ["config", "model", "reset"])
    assert result.exit_code == 0
    output = _flat(result.output)
    assert "Model choice for OpenRouter cleared." in output
    assert "You will be asked to choose one next time." in output
    assert config.get_provider_model("openrouter") == ""
    assert config.load_config()["provider_models"] == {}


def test_config_model_reset_for_a_local_provider_explains_the_fallback():
    config.set_provider_model("ollama", "qwen3:8b")

    result = CliRunner().invoke(main.app, ["config", "model", "reset", "-p", "ollama"])

    assert result.exit_code == 0
    output = _flat(result.output)
    assert "Model choice for Ollama (local) cleared." in output
    assert "Sessions use the first model the server lists." in output
    assert config.get_provider_model("ollama") == ""


def test_config_model_set_and_reset_for_another_provider():
    runner = CliRunner()
    config.set_provider_model("openrouter", "or-model")

    result = runner.invoke(
        main.app, ["config", "model", "set", "gpt-x", "--provider", "openai"]
    )
    assert result.exit_code == 0
    assert "Model for OpenAI set to gpt-x." in _flat(result.output)
    assert config.get_provider_model("openai") == "gpt-x"
    # The primary's choice is untouched.
    assert config.get_provider_model("openrouter") == "or-model"

    result = runner.invoke(main.app, ["config", "model", "reset", "-p", "openai"])
    assert result.exit_code == 0
    assert "Model choice for OpenAI cleared." in _flat(result.output)
    assert config.get_provider_model("openai") == ""
    assert config.get_provider_model("openrouter") == "or-model"


def test_config_model_set_and_reset_reject_unknown_provider():
    runner = CliRunner()

    result = runner.invoke(
        main.app, ["config", "model", "set", "m", "--provider", "nope"]
    )
    assert result.exit_code == 1
    assert "Unknown provider" in result.output

    result = runner.invoke(main.app, ["config", "model", "reset", "--provider", "nope"])
    assert result.exit_code == 1
    assert "Unknown provider" in result.output
    assert config.load_config()["provider_models"] == {}


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


def test_config_show_reports_the_chosen_model():
    runner = CliRunner()

    result = runner.invoke(main.app, ["config", "show"])
    assert result.exit_code == 0
    assert "Model: (none chosen)" in result.output

    config.set_provider_model("openrouter", "vendor/chosen")
    result = runner.invoke(main.app, ["config", "show"])
    assert "Model: vendor/chosen" in result.output

    # Local servers list their own models when nothing was chosen.
    config.set_selected_provider("ollama")
    result = runner.invoke(main.app, ["config", "show"])
    assert "Model: (from server)" in result.output


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
    # A global pin (older config, profile) applies to whichever provider is
    # primary, so switching to Ollama must drop it rather than inherit it.
    cfg = config.load_config()
    cfg["selected_model"] = "stale-pin"
    config.save_config(cfg)

    result = CliRunner().invoke(main.app, ["setup", "--provider", "ollama"])

    assert result.exit_code == 0
    assert "needs no API key" in result.output
    assert config.get_selected_provider_id() == "ollama"
    # Models are chosen per provider: the previous provider's model must not
    # carry over to Ollama, and stays recorded for the previous provider.
    assert config.load_config()["selected_model"] is None
    assert config.get_provider_model("ollama") == ""
    assert config.get_provider_model("openrouter") == "stale-model"
    # A stopped server has nothing to list, so no picker; say what happens.
    assert "No model chosen" in result.output
    assert config.get_key("ollama") is None


def test_setup_local_provider_accepts_a_model_flag(monkeypatch):
    monkeypatch.setattr(main, "probe", lambda provider: False)
    monkeypatch.setattr(
        main,
        "_interactive_select_model",
        lambda *args, **kwargs: pytest.fail("--model must skip the picker"),
    )

    result = CliRunner().invoke(
        main.app, ["setup", "--provider", "ollama", "--model", "qwen3:8b"]
    )

    assert result.exit_code == 0
    assert config.get_selected_provider_id() == "ollama"
    assert config.get_provider_model("ollama") == "qwen3:8b"


def test_setup_local_provider_offers_the_picker_only_when_the_server_runs(
    monkeypatch,
):
    monkeypatch.setattr(main, "_stdin_is_tty", lambda: True)
    asked: list[str] = []

    def picker(provider, current=""):
        asked.append(provider.id)
        return "qwen3:8b"

    monkeypatch.setattr(main, "_interactive_select_model", picker)
    runner = CliRunner()

    # Stopped server: nothing real to list, so no picker even on a terminal.
    monkeypatch.setattr(main, "probe", lambda provider: False)
    result = runner.invoke(main.app, ["setup", "--provider", "ollama"])
    assert result.exit_code == 0
    assert asked == []
    assert config.get_provider_model("ollama") == ""

    # Running server: the user picks from what it lists.
    monkeypatch.setattr(main, "probe", lambda provider: True)
    result = runner.invoke(main.app, ["setup", "--provider", "ollama"])
    assert result.exit_code == 0
    assert asked == ["ollama"]
    assert "Model set to qwen3:8b." in _flat(result.output)
    assert config.get_provider_model("ollama") == "qwen3:8b"


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
    # Nothing chosen: a local server's first listed model is used.
    _install_fetch(monkeypatch, ["qwen3:8b", "llama3.1:8b"])

    result = CliRunner().invoke(main.app, ["ask", "hi", "--provider", "ollama"])

    assert result.exit_code == 0
    assert captured["api_key"] == ""
    assert captured["model"] == "qwen3:8b"
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


@pytest.mark.parametrize("terminal", [True, False])
def test_setup_key_prompt_hides_input_only_on_a_terminal(monkeypatch, terminal):
    seen: dict[str, object] = {}

    def fake_input(prompt="", **kwargs):
        seen["prompt"] = prompt
        seen.update(kwargs)
        return "  sk-typed  "

    monkeypatch.setattr(main.console, "input", fake_input)
    monkeypatch.setattr(ui, "hide_typed_secrets", lambda: terminal)

    assert main._interactive_api_key() == "sk-typed"
    assert seen == {"prompt": "key> ", "password": terminal}


def test_setup_key_prompt_cancel_returns_none(monkeypatch):
    def cancelled(prompt="", **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(main.console, "input", cancelled)

    assert main._interactive_api_key() is None


def test_setup_reads_a_piped_key_from_stdin_as_before(monkeypatch):
    # No terminal: nothing is echoed, so the key is read from stdin, not from a
    # keyboard that a script cannot reach.
    monkeypatch.delenv("UNSLOTH_API_KEY", raising=False)
    monkeypatch.setattr(main, "_interactive_select_model", lambda *a, **k: "unsloth-model")
    _forbid_fetch(monkeypatch)

    result = CliRunner().invoke(
        main.app, ["setup", "--provider", "unsloth"], input="sk-unsloth-piped\n"
    )

    assert config.get_key("unsloth") == "sk-unsloth-piped", result.output


@pytest.mark.parametrize("value", ["nan", "inf", "1e999"])
def test_cli_budget_cost_rejects_non_finite_limits(value):
    result = CliRunner().invoke(main.app, ["config", "budget", "cost", value])

    assert result.exit_code == 1
    assert "finite number" in result.output
    assert config.get_budget() == {}


@pytest.mark.parametrize("action", ["cost", "tokens"])
def test_cli_budget_error_echoes_a_bracketed_value_literally(action):
    # The message quotes the rejected value; Rich must not parse it as markup
    # (an unmatched "[/x]" used to raise MarkupError).
    result = CliRunner().invoke(main.app, ["config", "budget", action, "[/x]"])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)  # a clean exit, not a crash
    assert "[/x]" in result.output


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
        captured["model"] = model
        captured["provider"] = provider

    monkeypatch.setattr(main, "stream_response", fake_stream)
    _install_fetch(monkeypatch, ["unsloth/Qwen3.6-27B-GGUF"])

    result = CliRunner().invoke(main.app, ["ask", "hi", "--provider", "unsloth"])

    assert result.exit_code == 0
    assert captured["api_key"] == "sk-unsloth-test"
    assert captured["model"] == "unsloth/Qwen3.6-27B-GGUF"
    assert captured["provider"].id == "unsloth"


def test_ask_with_a_stopped_local_server_and_no_model_exits_1(monkeypatch):
    def offline(provider, api_key=None, **kwargs):
        raise catalog.CatalogFetchError("connection refused")

    monkeypatch.setattr(config.catalog, "fetch_models", offline)
    called: list[object] = []

    async def fake_stream(*args, **kwargs):
        called.append(args)

    monkeypatch.setattr(main, "stream_response", fake_stream)

    result = CliRunner().invoke(main.app, ["ask", "hi", "--provider", "ollama"])

    # The suggested models would be a hidden default, so none is used.
    assert result.exit_code == 1
    assert "No model chosen for Ollama (local)" in _flat(result.output)
    assert called == []


def test_ask_without_a_chosen_model_exits_1(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    _forbid_fetch(monkeypatch)
    called: list[object] = []

    async def fake_stream(*args, **kwargs):
        called.append(args)

    monkeypatch.setattr(main, "stream_response", fake_stream)

    result = CliRunner().invoke(main.app, ["ask", "hi", "--provider", "openai"])

    assert result.exit_code == 1
    assert "No model chosen for OpenAI" in _flat(result.output)
    assert called == []


def test_ask_uses_the_chosen_model_and_model_flag_wins(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    config.set_provider_model("openai", "gpt-chosen")
    models: list[str | None] = []

    async def fake_stream(prompt, api_key, model=None, provider=None):
        models.append(model)

    monkeypatch.setattr(main, "stream_response", fake_stream)
    runner = CliRunner()

    result = runner.invoke(main.app, ["ask", "hi", "--provider", "openai"])
    assert result.exit_code == 0
    result = runner.invoke(
        main.app, ["ask", "hi", "--provider", "openai", "--model", "gpt-flag"]
    )
    assert result.exit_code == 0

    assert models == ["gpt-chosen", "gpt-flag"]


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
    monkeypatch.setattr(main, "_stdin_is_tty", lambda: True)

    result = CliRunner().invoke(main.app, ["--continue"])

    assert result.exit_code == 0
    assert captured["session"].provider_id == "anthropic"
    assert captured["session"].model == "claude-sonnet-5"


def test_continue_without_saved_session_starts_fresh(tmp_path, monkeypatch):
    from kiwimatecoder import repl

    monkeypatch.setattr(
        "kiwimatecoder.session._sessions_dir", lambda: tmp_path / "sessions"
    )
    config.set_provider_model("openrouter", "test-model")
    captured = {}
    monkeypatch.setattr(
        repl, "run", lambda session: captured.setdefault("session", session)
    )
    monkeypatch.setattr(main, "_stdin_is_tty", lambda: True)

    result = CliRunner().invoke(main.app, ["--continue"])

    assert result.exit_code == 0
    assert "Starting fresh" in result.output
    assert captured["session"].provider_id == "openrouter"
    assert captured["session"].model == "test-model"


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
    assert "none chosen" in result.output


def test_doctor_checks_the_chosen_model(monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    config.set_provider_model("openrouter", "vendor/chosen")

    result = CliRunner().invoke(main.app, ["doctor"])

    assert result.exit_code == 0
    assert "none chosen" not in result.output
    assert "vendor/chosen" in _flat(result.output)


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
    monkeypatch.setattr(main, "_stdin_is_tty", lambda: True)

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
    monkeypatch.setattr(main, "_stdin_is_tty", lambda: True)

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

    result = runner.invoke(main.app, ["config", "ui", "keybindings", "vim"])
    assert result.exit_code == 0
    assert config.get_ui()["keybindings"] == "vim"

    result = runner.invoke(main.app, ["config", "ui", "notify", "bell"])
    assert result.exit_code == 0
    assert config.get_ui()["notify"] == "bell"

    result = runner.invoke(main.app, ["config", "ui", "notify-after", "5"])
    assert result.exit_code == 0
    assert config.get_ui()["notify_after_seconds"] == 5

    result = runner.invoke(main.app, ["config", "ui", "spinner", "off"])
    assert result.exit_code == 0
    assert config.get_ui()["spinner"] == "off"

    result = runner.invoke(main.app, ["config", "ui", "show"])
    assert result.exit_code == 0
    assert "ocean" in result.output
    assert "never" in result.output
    assert "compact" in result.output
    assert "vim" in result.output
    assert "bell" in result.output
    assert "off" in result.output


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
        ["config", "ui", "locale", "kl"],
        ["config", "ui", "keybindings", "nano"],
        ["config", "ui", "notify", "loud"],
        ["config", "ui", "notify-after", "soon"],
        ["config", "ui", "spinner", "sometimes"],
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


# ---------------------------------------------------------------------------
# Choosing a model (there are no default models)
# ---------------------------------------------------------------------------


def test_setup_model_flag_saves_the_model_without_the_picker(monkeypatch):
    # Even on a terminal, --model wins over the picker.
    monkeypatch.setattr(main, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(
        main,
        "_interactive_select_model",
        lambda *args, **kwargs: pytest.fail("--model must skip the picker"),
    )

    result = CliRunner().invoke(
        main.app,
        ["setup", "--provider", "openai", "--key", "sk-openai", "--model", "gpt-5.5"],
    )

    assert result.exit_code == 0
    assert "Model set to gpt-5.5." in _flat(result.output)
    assert config.get_provider_model("openai") == "gpt-5.5"
    assert config.get_selected_provider_id() == "openai"
    assert config.get_key("openai") == "sk-openai"


def test_setup_key_without_model_on_non_tty_says_no_model_chosen(monkeypatch):
    monkeypatch.setattr(main, "_stdin_is_tty", lambda: False)
    monkeypatch.setattr(
        main,
        "_interactive_select_model",
        lambda *args, **kwargs: pytest.fail("no picker without a terminal"),
    )

    result = CliRunner().invoke(
        main.app, ["setup", "--provider", "openai", "--key", "sk-openai"]
    )

    assert result.exit_code == 0
    output = _flat(result.output)
    assert "API key saved" in output
    assert "No model chosen yet." in output
    # The fix-it command must not ask for the key again.
    assert "kiwimatecoder config model set <model> --provider openai" in output
    assert config.get_key("openai") == "sk-openai"
    assert config.get_provider_model("openai") == ""


def test_setup_kiwimate_prints_the_experimental_note(monkeypatch):
    monkeypatch.setattr(main, "_stdin_is_tty", lambda: False)

    result = CliRunner().invoke(
        main.app,
        [
            "setup",
            "--provider",
            "kiwimate",
            "--key",
            "sk-km-test",
            "-m",  # the short spelling of --model
            "kiwimate-small-1-0",
        ],
    )

    assert result.exit_code == 0
    output = _flat(result.output)
    assert "KiwiMate is experimental." in output
    assert "chat only for now" in output
    assert config.get_key("kiwimate") == "sk-km-test"
    assert config.get_selected_provider_id() == "kiwimate"
    assert config.get_provider_model("kiwimate") == "kiwimate-small-1-0"


def test_setup_for_a_regular_provider_has_no_experimental_note(monkeypatch):
    monkeypatch.setattr(main, "_stdin_is_tty", lambda: False)

    result = CliRunner().invoke(
        main.app,
        ["setup", "--provider", "openai", "--key", "sk-openai", "--model", "gpt-5.5"],
    )

    assert result.exit_code == 0
    assert "experimental" not in result.output


def test_setup_model_keeps_existing_choice_when_picker_cancelled(monkeypatch):
    config.set_provider_model("openai", "gpt-old")
    seen: list[tuple[str, str]] = []

    def cancelled_picker(provider, current=""):
        seen.append((provider.id, current))
        return None

    monkeypatch.setattr(main, "_interactive_select_model", cancelled_picker)
    monkeypatch.setattr(main, "_stdin_is_tty", lambda: True)
    buffer = _capture_console(monkeypatch)

    main._setup_model(config.get_provider_config("openai"), None)

    assert seen == [("openai", "gpt-old")]
    assert config.get_provider_model("openai") == "gpt-old"
    assert "Keeping model gpt-old." in _flat(buffer.getvalue())


def test_setup_model_saves_the_picked_model(monkeypatch):
    monkeypatch.setattr(
        main, "_interactive_select_model", lambda provider, current="": "gpt-picked"
    )
    monkeypatch.setattr(main, "_stdin_is_tty", lambda: True)
    buffer = _capture_console(monkeypatch)

    main._setup_model(config.get_provider_config("openai"), None)

    assert config.get_provider_model("openai") == "gpt-picked"
    assert "Model set to gpt-picked." in _flat(buffer.getvalue())


def test_setup_model_skips_the_picker_when_not_interactive(monkeypatch):
    monkeypatch.setattr(main, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(
        main,
        "_interactive_select_model",
        lambda *args, **kwargs: pytest.fail("interactive=False must not prompt"),
    )
    buffer = _capture_console(monkeypatch)

    main._setup_model(config.get_provider_config("ollama"), None, interactive=False)

    assert config.get_provider_model("ollama") == ""
    assert "first model the server lists" in _flat(buffer.getvalue())


def test_interactive_select_provider_puts_experimental_providers_in_their_own_box(
    monkeypatch,
):
    import kiwimatecoder.pickers as pickers

    captured: dict = {}

    def fake_grouped_choice(message, *, groups, **kwargs):
        captured["message"] = message
        captured["groups"] = groups
        captured.update(kwargs)
        return "kiwimate"

    monkeypatch.setattr(pickers, "grouped_choice", fake_grouped_choice)
    config.set_key("openai", "sk-openai")
    config.set_provider_model("openai", "gpt-5.5")
    providers = config.list_provider_configs()

    chosen = main._interactive_select_provider(
        providers, "openrouter", {"ollama": True, "lmstudio": False}
    )

    assert chosen == "kiwimate"
    assert captured["default"] == "openrouter"
    groups = captured["groups"]
    assert len(groups) == 2
    title, experimental = groups[0]
    assert title == "Experimental"
    assert [value for value, _label in experimental] == ["kiwimate"]
    assert experimental[0][1] == (
        "KiwiMate — needs an API key · kiwimate.net · chat only, no tool use yet"
    )
    rest_title, rest = groups[1]
    assert rest_title is None
    rest_ids = [value for value, _label in rest]
    assert "kiwimate" not in rest_ids
    assert rest_ids == [p.id for p in providers if not p.experimental]
    labels = dict(rest)
    assert labels["openai"] == "OpenAI — key set · gpt-5.5"
    assert labels["openrouter"] == "OpenRouter — needs an API key"
    assert labels["ollama"].startswith("Ollama (local) — running (")
    assert labels["lmstudio"].startswith("LM Studio (local) — not detected (")


def test_interactive_select_provider_returns_none_on_cancel(monkeypatch):
    import kiwimatecoder.pickers as pickers

    def cancelled(message, *, groups, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(pickers, "grouped_choice", cancelled)

    assert (
        main._interactive_select_provider(config.list_provider_configs(), "openrouter")
        is None
    )


def _patch_model_picker(monkeypatch, answer, typed=None):
    """Patch prompt_toolkit's ``choice`` and the console's ``input``.

    ``answer`` receives the offered options and returns the picked value (or
    raises to cancel). Returns the dict the fake picker records into.
    """
    import prompt_toolkit.shortcuts as shortcuts

    captured: dict = {}

    def fake_choice(*, message, options, default=None, **kwargs):
        captured["message"] = message
        captured["options"] = list(options)
        captured["default"] = default
        return answer(options)

    monkeypatch.setattr(shortcuts, "choice", fake_choice)
    _capture_console(monkeypatch)

    def fake_input(prompt=""):
        captured["input_prompt"] = prompt
        if isinstance(typed, BaseException):
            raise typed
        if typed is None:
            pytest.fail("the typed-id prompt must not be shown")
        return typed

    monkeypatch.setattr(main.console, "input", fake_input)
    return captured


def test_interactive_select_model_returns_a_typed_id(monkeypatch):
    config.set_key("openai", "sk-openai")
    calls: list[str] = []
    _install_fetch(monkeypatch, ["gpt-new", "gpt-old"], calls)
    captured = _patch_model_picker(
        monkeypatch, answer=lambda options: main._TYPE_MODEL_ID, typed="  my/custom-model  "
    )

    picked = main._interactive_select_model(config.get_provider_config("openai"))

    assert picked == "my/custom-model"
    # The live listing is refetched (a key may have just been saved).
    assert calls == ["openai"]
    values = [value for value, _label in captured["options"]]
    # The free-text row comes first so it is reachable with long listings,
    # but the cursor starts on the top listed model.
    assert values == [main._TYPE_MODEL_ID, "gpt-new", "gpt-old"]
    assert captured["options"][0][1] == "Type a model id…"
    assert captured["default"] == "gpt-new"
    assert "Choose a model for OpenAI" in captured["message"]


def test_interactive_select_model_returns_a_listed_model(monkeypatch):
    config.set_key("openai", "sk-openai")
    _install_fetch(monkeypatch, ["gpt-new", "gpt-old"])
    captured = _patch_model_picker(monkeypatch, answer=lambda options: "gpt-old")

    picked = main._interactive_select_model(
        config.get_provider_config("openai"), "gpt-old"
    )

    assert picked == "gpt-old"
    assert captured["default"] == "gpt-old"


def test_interactive_select_model_returns_none_on_cancel(monkeypatch):
    config.set_key("openai", "sk-openai")
    _install_fetch(monkeypatch, ["gpt-new"])

    def cancel(options):
        raise KeyboardInterrupt

    _patch_model_picker(monkeypatch, answer=cancel)

    assert main._interactive_select_model(config.get_provider_config("openai")) is None


def test_interactive_select_model_returns_none_when_typing_is_cancelled(monkeypatch):
    config.set_key("openai", "sk-openai")
    _install_fetch(monkeypatch, ["gpt-new"])
    _patch_model_picker(
        monkeypatch, answer=lambda options: main._TYPE_MODEL_ID, typed=EOFError()
    )

    assert main._interactive_select_model(config.get_provider_config("openai")) is None


def test_interactive_select_model_returns_none_for_an_empty_typed_id(monkeypatch):
    config.set_key("openai", "sk-openai")
    _install_fetch(monkeypatch, ["gpt-new"])
    _patch_model_picker(
        monkeypatch, answer=lambda options: main._TYPE_MODEL_ID, typed="  "
    )

    assert main._interactive_select_model(config.get_provider_config("openai")) is None


def _launch(monkeypatch, picker):
    """Start the interactive session with the REPL and model picker patched."""
    from kiwimatecoder import repl

    captured: dict = {}
    monkeypatch.setattr(
        repl, "run", lambda session: captured.setdefault("session", session)
    )
    monkeypatch.setattr(main, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(main, "_interactive_select_model", picker)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    _forbid_fetch(monkeypatch)
    result = CliRunner().invoke(main.app, [])
    return result, captured


def test_launch_without_a_model_asks_for_one_and_persists_it(monkeypatch):
    asked: list[tuple[str, str]] = []

    def picker(provider, current=""):
        asked.append((provider.id, current))
        return "vendor/picked"

    result, captured = _launch(monkeypatch, picker)

    assert result.exit_code == 0
    assert asked == [("openrouter", "")]
    assert "No model chosen for OpenRouter yet." in _flat(result.output)
    assert captured["session"].model == "vendor/picked"
    assert config.get_provider_model("openrouter") == "vendor/picked"


def test_launch_with_a_chosen_model_skips_the_picker(monkeypatch):
    config.set_provider_model("openrouter", "vendor/chosen")

    result, captured = _launch(
        monkeypatch,
        lambda *args, **kwargs: pytest.fail("a chosen model must not prompt"),
    )

    assert result.exit_code == 0
    assert captured["session"].model == "vendor/chosen"
    assert "No model chosen" not in result.output


def test_launch_continues_without_a_model_when_the_picker_is_cancelled(monkeypatch):
    result, captured = _launch(monkeypatch, lambda provider, current="": None)

    assert result.exit_code == 0
    assert captured["session"].model == ""
    assert "Choose a model with /model before chatting." in _flat(result.output)
    assert config.get_provider_model("openrouter") == ""


def test_launch_quick_start_setup_model_is_used_without_asking_again(monkeypatch):
    from kiwimatecoder import repl

    captured: dict = {}
    asked: list[str] = []

    def picker(provider, current=""):
        asked.append(provider.id)
        return "vendor/from-setup"

    monkeypatch.setattr(
        repl, "run", lambda session: captured.setdefault("session", session)
    )
    monkeypatch.setattr(main, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(main, "_prompt_yes_no", lambda question: True)
    monkeypatch.setattr(main, "_interactive_api_key", lambda: "sk-or-quickstart")
    monkeypatch.setattr(main, "_interactive_select_model", picker)
    _forbid_fetch(monkeypatch)

    # No key yet: the launch offers the quick-start setup, which saves a key
    # and asks for a model.
    result = CliRunner().invoke(main.app, [])

    assert result.exit_code == 0
    assert "Quick start" in result.output
    assert config.get_key("openrouter") == "sk-or-quickstart"
    # Asked once, during setup: the launch rereads config and uses that choice.
    assert asked == ["openrouter"]
    assert captured["session"].model == "vendor/from-setup"
    assert "No model chosen" not in result.output
