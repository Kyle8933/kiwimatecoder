import io

import pytest
from rich.console import Console

from kiwimatecoder import catalog, config
from kiwimatecoder.commands import (
    CommandResult,
    SelectionPrompt,
    dispatch,
    slash_argument_completions,
    slash_command_completions,
)
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.providers import REGISTRY


def _console():
    return Console(file=io.StringIO(), force_terminal=False, width=120)


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    for provider in REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)
    monkeypatch.delenv("LOCAL_API_KEY", raising=False)


def test_context_add_lists_and_deduplicates(session):
    (session.workspace_root / "README.md").write_text("hello\n")
    console = _console()

    assert (
        dispatch("/context add README.md README.md", session, console)
        == CommandResult.CONTINUE
    )

    assert session.context_files == ["README.md"]


def test_context_add_glob(session):
    (session.workspace_root / "a.py").write_text("print('a')\n")
    (session.workspace_root / "b.txt").write_text("b\n")
    console = _console()

    dispatch("/context add *.py", session, console)

    assert session.context_files == ["a.py"]


def test_context_rejects_binary_files(session):
    (session.workspace_root / "image.bin").write_bytes(b"\x00\x01")
    console = _console()

    dispatch("/context add image.bin", session, console)

    assert session.context_files == []


def test_context_remove_and_clear(session):
    (session.workspace_root / "a.py").write_text("print('a')\n")
    (session.workspace_root / "b.py").write_text("print('b')\n")
    console = _console()
    dispatch("/context add *.py", session, console)

    dispatch("/context remove a.py", session, console)
    assert session.context_files == ["b.py"]

    dispatch("/context clear", session, console)
    assert session.context_files == []


def test_slash_command_completions_include_core_commands():
    completions = {command for command, _ in slash_command_completions("")}

    assert {"/help", "/model", "/provider", "/mode", "/context", "/config", "/cost"} <= completions


def test_bare_model_command_selects_from_current_provider(session):
    config.set_model_filter("openrouter", "allow", ["model-a", "model-b"])
    prompts: list[SelectionPrompt] = []

    def select(prompt: SelectionPrompt) -> str:
        prompts.append(prompt)
        return "model-b"

    result = dispatch("/model", session, _console(), selector=select)

    assert result == CommandResult.CONTINUE
    assert session.model == "model-b"
    assert prompts[0].title == "Select model"
    assert [option.value for option in prompts[0].options] == ["model-a", "model-b"]
    assert "openrouter" in prompts[0].text


def test_bare_model_command_offers_full_catalog_without_filter(session):
    prompts: list[SelectionPrompt] = []

    def select(prompt: SelectionPrompt) -> str:
        prompts.append(prompt)
        return prompt.options[0].value

    dispatch("/model", session, _console(), selector=select)

    offered = [option.value for option in prompts[0].options]
    provider = REGISTRY["openrouter"]
    assert offered[0] == provider.default_model
    assert set(provider.models) <= set(offered)
    assert len(offered) > 1


def test_cancelled_model_selection_leaves_model_unchanged(session):
    result = dispatch("/model", session, _console(), selector=lambda prompt: None)

    assert result == CommandResult.CONTINUE
    assert session.model == "test-model"


def test_bare_provider_and_mode_commands_are_interactive(session):
    def select(prompt: SelectionPrompt) -> str:
        if prompt.title == "Select provider":
            return "openai"
        return "plan"

    dispatch("/provider", session, _console(), selector=select)
    dispatch("/mode", session, _console(), selector=select)

    assert session.provider_id == "openai"
    assert session.model == REGISTRY["openai"].default_model
    assert session.mode is PermissionMode.PLAN


def test_explicit_choice_does_not_open_selector(session):
    def fail_if_called(prompt: SelectionPrompt) -> str:
        raise AssertionError("selector should not be called")

    dispatch("/model custom-model", session, _console(), selector=fail_if_called)

    assert session.model == "custom-model"


def test_config_provider_key_and_model_filter_workflow(session):
    console = _console()

    dispatch(
        '/config provider add local "Local Models" http://localhost:1234/v1 local-code LOCAL_API_KEY',
        session,
        console,
    )
    dispatch("/config key set local sk-local", session, console)
    dispatch("/config provider use local", session, console)
    dispatch("/config models allow local-code local-fast", session, console)

    assert config.get_provider_config("local").name == "Local Models"
    assert config.get_key("local") == "sk-local"
    assert session.provider_id == "local"
    assert session.model == "local-code"
    assert config.list_visible_models("local") == ["local-code", "local-fast"]

    dispatch("/config provider remove local", session, console)

    assert session.provider_id == "openrouter"
    with pytest.raises(KeyError):
        config.get_provider_config("local")


def test_config_key_remove(session):
    console = _console()
    config.set_key("openai", "sk-openai")

    dispatch("/config key remove openai", session, console)

    assert config.get_key("openai") is None


# ---------------------------------------------------------------------------
# Live model catalogs
# ---------------------------------------------------------------------------


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


def _output(console) -> str:
    return console.file.getvalue()


def test_bare_model_command_refreshes_the_catalog(session, monkeypatch):
    config.set_key("openrouter", "sk-test")
    calls: list[str] = []
    _install_fetch(monkeypatch, ["vendor/new", "vendor/stable"], calls)
    prompts: list[SelectionPrompt] = []

    def select(prompt: SelectionPrompt) -> str:
        prompts.append(prompt)
        return "vendor/new"

    dispatch("/model", session, _console(), selector=select)

    assert calls == ["openrouter"]
    assert [option.value for option in prompts[0].options] == [
        "vendor/new",
        "vendor/stable",
    ]
    assert session.model == "vendor/new"


def test_model_refresh_reports_new_and_deprecated_models(session, monkeypatch):
    config.set_key("openrouter", "sk-test")
    session.model = "anthropic/claude-opus-4-8"  # a curated id the provider drops
    _install_fetch(monkeypatch, ["anthropic/claude-sonnet-5", "vendor/brand-new"])
    console = _console()

    assert dispatch("/model refresh", session, console) == CommandResult.CONTINUE

    output = _output(console)
    assert "vendor/brand-new" in output
    assert "Deprecated" in output
    assert "anthropic/claude-opus-4-8" in output
    # The retired model is called out rather than silently left selected.
    assert "no longer offered" in output
    assert config.list_visible_models("openrouter") == [
        "anthropic/claude-sonnet-5",
        "vendor/brand-new",
    ]


def test_model_refresh_survives_a_failing_provider(session, monkeypatch):
    config.set_key("openrouter", "sk-test")

    def fake_fetch(provider, api_key=None, **kwargs):
        raise catalog.CatalogFetchError("connection refused")

    monkeypatch.setattr(config.catalog, "fetch_models", fake_fetch)
    console = _console()

    dispatch("/model refresh", session, console)

    output = _output(console)
    assert "connection refused" in output
    assert REGISTRY["openrouter"].default_model in output


def test_model_list_uses_the_cache_without_fetching(session, monkeypatch):
    config.set_key("openrouter", "sk-test")
    _install_fetch(monkeypatch, ["vendor/one"])
    config.get_model_catalog("openrouter", refresh=True)

    def fail(provider, api_key=None, **kwargs):
        raise AssertionError("no network for /model list")

    monkeypatch.setattr(config.catalog, "fetch_models", fail)
    console = _console()

    dispatch("/model list", session, console)

    assert "vendor/one" in _output(console)


def test_setting_a_model_by_name_still_works(session, monkeypatch):
    config.set_key("openrouter", "sk-test")

    def fail(provider, api_key=None, **kwargs):
        raise AssertionError("setting a model must not hit the network")

    monkeypatch.setattr(config.catalog, "fetch_models", fail)

    dispatch("/model some/unlisted-model", session, _console())

    assert session.model == "some/unlisted-model"


def test_config_models_refresh_updates_the_catalog(session, monkeypatch):
    config.set_key("openrouter", "sk-test")
    _install_fetch(monkeypatch, ["vendor/one", "vendor/two"])
    console = _console()

    dispatch("/config models refresh", session, console)

    assert "vendor/two" in _output(console)
    assert config.list_visible_models("openrouter") == ["vendor/one", "vendor/two"]


def test_model_argument_completions_include_refresh(session):
    values = {
        value for value, _ in slash_argument_completions("model", "", session)
    }

    assert {"refresh", "list"} <= values


# ---------------------------------------------------------------------------
# /config mode (default permission mode)
# ---------------------------------------------------------------------------


def test_config_mode_set_and_show_persists(session):
    console = _console()

    dispatch("/config mode set plan", session, console)

    assert config.get_default_mode() == "plan"
    assert "plan" in _output(console)

    dispatch("/config mode show", session, console)
    assert "Default mode: plan" in _output(console)


def test_config_mode_accepts_aliases(session):
    dispatch("/config mode set auto", session, _console())
    assert config.get_default_mode() == "auto-accept"


def test_config_mode_rejects_unknown(session):
    console = _console()
    dispatch("/config mode set bogus", session, console)
    assert config.get_default_mode() == "ask"
    assert "Unknown mode" in _output(console)


def test_config_mode_reset_restores_default(session):
    dispatch("/config mode set plan", session, _console())
    dispatch("/config mode reset", session, _console())
    assert config.get_default_mode() == "ask"


# ---------------------------------------------------------------------------
# /config provider edit
# ---------------------------------------------------------------------------


def test_config_provider_edit_updates_fields(session):
    console = _console()
    dispatch(
        '/config provider add local "Local Models" http://localhost:1234/v1 local-code LOCAL_API_KEY',
        session,
        console,
    )

    dispatch(
        "/config provider edit local "
        'name="Local Models 2" default_model=local-fast',
        session,
        console,
    )

    provider = config.get_provider_config("local")
    assert provider.name == "Local Models 2"
    assert provider.default_model == "local-fast"
    assert provider.base_url == "http://localhost:1234/v1"
    assert provider.key_env == "LOCAL_API_KEY"


def test_config_provider_edit_rejects_unknown_field(session, monkeypatch):
    config.add_provider("local", "Local", "http://localhost:1234/v1", "local-code")
    console = _console()

    dispatch("/config provider edit local nope=x", session, console)

    assert "Unknown provider field" in _output(console)
    assert config.get_provider_config("local").name == "Local"


def test_config_provider_edit_rejects_builtin(session):
    console = _console()

    dispatch("/config provider edit openai name=hi", session, console)

    assert "built in" in _output(console)


def test_config_provider_edit_requires_pairs(session):
    config.add_provider("local", "Local", "http://localhost:1234/v1", "local-code")
    console = _console()

    dispatch("/config provider edit local name", session, console)

    assert "Expected field=value" in _output(console)


# ---------------------------------------------------------------------------
# bare /config interactive menu
# ---------------------------------------------------------------------------


def test_bare_config_opens_interactive_menu(session):
    prompts: list[SelectionPrompt] = []

    def select(prompt: SelectionPrompt) -> str:
        prompts.append(prompt)
        return "keys"

    dispatch("/config", session, _console(), selector=select)

    assert len(prompts) == 1
    assert prompts[0].title == "Configure KiwiMateCoder"
    values = [option.value for option in prompts[0].options]
    assert {"show", "providers", "keys", "model", "models", "mode", "help"} <= set(
        values
    )


def test_bare_config_menu_selection_dispatches_section(session):
    def select(prompt: SelectionPrompt) -> str:
        return "keys"

    config.set_key("openai", "sk-openai")
    console = _console()
    dispatch("/config", session, console, selector=select)

    assert "openai" in _output(console)
    assert "configured" in _output(console)
