import io

import pytest
from rich.console import Console

from kiwimatecoder import catalog, config, ui
from kiwimatecoder.commands import (
    CommandResult,
    MultiSelectionPrompt,
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
    assert config.load_config().get("selected_model") == "model-b"
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


def test_cancelled_model_selection_does_not_change_saved_default(session):
    config.set_selected_model("already-saved")

    dispatch("/model", session, _console(), selector=lambda prompt: None)

    assert session.model == "test-model"
    assert config.load_config().get("selected_model") == "already-saved"


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
    assert config.load_config().get("selected_model") == "custom-model"


def test_bare_provider_command_opens_multi_checklist(session):
    config.set_selected_model("stale-from-openrouter")
    prompts: list[MultiSelectionPrompt] = []

    def select(prompt: MultiSelectionPrompt) -> list[str]:
        prompts.append(prompt)
        return ["openai", "deepseek"]

    result = dispatch("/provider", session, _console(), multi_selector=select)

    assert result == CommandResult.CONTINUE
    assert prompts[0].title == "Select active providers"
    assert "openrouter" in {option.value for option in prompts[0].options}
    assert session.provider_id == "openai"
    assert session.model == REGISTRY["openai"].default_model
    assert session.active_provider_ids == ["openai", "deepseek"]
    assert config.get_active_provider_ids() == ["openai", "deepseek"]
    assert config.load_config().get("selected_model") is None


def test_bare_provider_checklist_marks_current_selection(session):
    config.set_active_providers(["openrouter", "openai"])
    session.set_active_providers(["openrouter", "openai"])
    prompts: list[MultiSelectionPrompt] = []

    def select(prompt: MultiSelectionPrompt) -> list[str]:
        prompts.append(prompt)
        return list(prompt.selected)

    dispatch("/provider", session, _console(), multi_selector=select)

    assert list(prompts[0].selected) == ["openrouter", "openai"]


def test_bare_provider_checklist_rejects_empty_selection(session):
    console = _console()
    result = dispatch(
        "/provider", session, console, multi_selector=lambda prompt: []
    )

    assert result == CommandResult.CONTINUE
    assert "invalid choice" in _output(console).lower()


def test_provider_command_with_id_resets_roster_to_single(session):
    console = _console()
    config.set_active_providers(["openrouter", "openai"])
    config.set_selected_model("stale-from-openrouter")

    dispatch("/provider deepseek", session, console)

    assert session.provider_id == "deepseek"
    assert config.get_active_provider_ids() == ["deepseek"]
    assert config.load_config().get("selected_model") is None


def test_explicit_provider_id_does_not_open_checklist(session):
    def fail_if_called(prompt: MultiSelectionPrompt) -> list[str]:
        raise AssertionError("checklist should not be called")

    dispatch("/provider openai", session, _console(), multi_selector=fail_if_called)

    assert session.provider_id == "openai"


def test_reapplying_provider_checklist_keeps_current_model(session):
    session.model = "my-custom-model"
    config.set_selected_model("my-custom-model")
    session.allow_always("run_bash")

    dispatch(
        "/provider",
        session,
        _console(),
        multi_selector=lambda prompt: ["openrouter", "openai"],
    )

    assert session.provider_id == "openrouter"
    assert session.model == "my-custom-model"
    assert session.is_always_allowed("run_bash")
    assert session.active_provider_ids == ["openrouter", "openai"]
    assert config.load_config().get("selected_model") == "my-custom-model"


def test_provider_checklist_lists_current_roster_first(session):
    config.set_active_providers(["openrouter", "openai"])
    session.set_active_providers(["openrouter", "openai"])
    prompts: list[MultiSelectionPrompt] = []

    def select(prompt: MultiSelectionPrompt) -> list[str]:
        prompts.append(prompt)
        return list(prompt.selected)

    dispatch("/provider", session, _console(), multi_selector=select)

    values = [option.value for option in prompts[0].options]
    assert values[:2] == ["openrouter", "openai"]
    assert "(primary)" in prompts[0].options[0].label
    assert "fallback" in prompts[0].options[1].label.lower()


def test_config_provider_remove_updates_session_fallback_roster(session):
    config.add_provider("local", "Local", "http://localhost:1234/v1", "local-code")
    config.set_active_providers(["openrouter", "local"])
    session.set_active_providers(["openrouter", "local"])
    session.model = "keep-me"

    dispatch("/config provider remove local", session, _console())

    assert session.provider_id == "openrouter"
    assert session.active_provider_ids == ["openrouter"]
    assert session.model == "keep-me"


def test_config_show_uses_session_roster(session):
    session.set_active_providers(["openai", "deepseek"])
    console = _console()

    dispatch("/config", session, console)

    output = _output(console)
    assert "openai" in output
    assert "deepseek" in output


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
    assert config.load_config().get("selected_model") is None
    assert session.model == "test-model"


def test_setting_a_model_by_name_still_works(session, monkeypatch):
    config.set_key("openrouter", "sk-test")

    def fail(provider, api_key=None, **kwargs):
        raise AssertionError("setting a model must not hit the network")

    monkeypatch.setattr(config.catalog, "fetch_models", fail)

    dispatch("/model some/unlisted-model", session, _console())

    assert session.model == "some/unlisted-model"
    assert config.load_config().get("selected_model") == "some/unlisted-model"


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


def test_model_search_offers_matches_and_selects(session, monkeypatch):
    config.set_key("openrouter", "sk-test")
    _install_fetch(monkeypatch, ["vendor/foo", "vendor/bar", "other/baz"])
    prompts: list[SelectionPrompt] = []

    def select(prompt: SelectionPrompt) -> str:
        prompts.append(prompt)
        return "vendor/bar"

    dispatch("/model search vendor", session, _console(), selector=select)

    assert session.model == "vendor/bar"
    assert config.load_config().get("selected_model") == "vendor/bar"
    assert prompts[0].title == "Search: vendor"
    assert [option.value for option in prompts[0].options] == [
        "vendor/foo",
        "vendor/bar",
    ]


def test_model_search_no_match(session, monkeypatch):
    config.set_key("openrouter", "sk-test")
    _install_fetch(monkeypatch, ["vendor/one"])
    console = _console()

    dispatch("/model search nope", session, console)

    assert "No models matching 'nope'" in _output(console)
    assert session.model == "test-model"


def test_model_search_requires_a_term(session):
    console = _console()

    dispatch("/model search", session, console)

    assert "Usage: /model search <term>" in _output(console)


def test_model_search_without_selector_prints_matches(session, monkeypatch):
    config.set_key("openrouter", "sk-test")
    _install_fetch(monkeypatch, ["vendor/one", "vendor/two"])
    console = _console()

    dispatch("/model search vendor", session, console)

    output = _output(console)
    assert "vendor/one" in output
    assert "vendor/two" in output
    assert session.model == "test-model"


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


def test_config_provider_add_accepts_azure_auth_fields(session):
    console = _console()

    dispatch(
        '/config provider add my-azure "My Azure" '
        "https://my-resource.openai.azure.com/openai/v1 deploy "
        "AZURE_OPENAI_API_KEY key_header=api-key key_prefix= "
        "api_version=2024-10-21",
        session,
        console,
    )

    provider = config.get_provider_config("my-azure")
    assert provider.key_header == "api-key"
    assert provider.key_prefix == ""
    assert provider.api_version == "2024-10-21"


def test_config_provider_edit_updates_auth_fields(session):
    config.add_provider(
        "my-azure",
        "My Azure",
        "https://my-resource.openai.azure.com/openai/v1",
        "deploy",
    )
    console = _console()

    dispatch(
        "/config provider edit my-azure key_header=api-key key_prefix= "
        "api_version=2024-10-21",
        session,
        console,
    )

    provider = config.get_provider_config("my-azure")
    assert provider.key_header == "api-key"
    assert provider.key_prefix == ""
    assert provider.api_version == "2024-10-21"


def test_config_provider_add_rejects_unknown_auth_field(session):
    console = _console()

    dispatch(
        '/config provider add weird "Weird" https://w.example/v1 m nope=1',
        session,
        console,
    )

    assert "Unknown provider field" in _output(console)
    with pytest.raises(KeyError):
        config.get_provider_config("weird")


# ---------------------------------------------------------------------------
# bare /config interactive menu
# ---------------------------------------------------------------------------


def test_bare_config_opens_interactive_menu(session):
    prompts: list[SelectionPrompt] = []

    def select(prompt: SelectionPrompt) -> str | None:
        prompts.append(prompt)
        return "help"

    dispatch("/config", session, _console(), selector=select)

    assert len(prompts) == 1
    assert prompts[0].title == "Configure KiwiMateCoder"
    values = [option.value for option in prompts[0].options]
    assert {"show", "providers", "keys", "model", "models", "mode", "help"} <= set(
        values
    )


def test_bare_config_menu_selection_lists_providers(session):
    def select(prompt: SelectionPrompt) -> str:
        return "providers"

    console = _console()
    dispatch("/config", session, console, selector=select)

    assert "openrouter" in _output(console)
    assert "openai" in _output(console)


def test_bare_config_interactive_key_set(session):
    config.add_provider("local", "Local", "http://localhost:1234/v1", "local-code")
    selections = iter(["keys", "local", "set"])

    def select(prompt: SelectionPrompt) -> str:
        return next(selections)

    console = _console()

    def prompt_input(_msg: str) -> str:
        return "sk-local-new"

    dispatch("/config", session, console, selector=select, prompt_input=prompt_input)

    assert config.get_key("local") == "sk-local-new"
    assert "local" in _output(console)
    assert "…-new" in _output(console)


def test_bare_config_interactive_key_remove(session):
    config.set_key("openai", "sk-openai")
    selections = iter(["keys", "openai", "remove"])

    def select(prompt: SelectionPrompt) -> str:
        return next(selections)

    console = _console()
    dispatch("/config", session, console, selector=select)

    assert config.get_key("openai") is None
    assert "Removed stored API key" in _output(console)


def test_bare_config_interactive_key_set_warns_on_env_override(session, monkeypatch):
    config.set_key("openai", "old")
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    selections = iter(["keys", "openai", "set"])

    def select(prompt: SelectionPrompt) -> str:
        return next(selections)

    console = _console()
    dispatch(
        "/config",
        session,
        console,
        selector=select,
        prompt_input=lambda _msg: "sk-new",
    )

    assert config.load_config()["keys"]["openai"] == "sk-new"
    assert "takes precedence" in _output(console)


# ---------------------------------------------------------------------------
# Local providers
# ---------------------------------------------------------------------------


def test_switch_to_local_provider_without_key(session, monkeypatch):
    _install_fetch(monkeypatch, ["qwen3:8b", "llama3.1:8b"])
    console = _console()

    assert dispatch("/provider ollama", session, console) == CommandResult.CONTINUE

    assert session.provider_id == "ollama"
    # The model was resolved live from the server, not from a static default.
    assert session.model == "qwen3:8b"
    assert "qwen3:8b" in _output(console)


def test_switch_to_offline_local_provider_uses_curated_fallback(session, monkeypatch):
    def fake_fetch(provider, api_key=None, **kwargs):
        raise catalog.CatalogFetchError("connection refused")

    monkeypatch.setattr(config.catalog, "fetch_models", fake_fetch)
    console = _console()

    dispatch("/provider ollama", session, console)

    assert session.provider_id == "ollama"
    assert session.model == REGISTRY["ollama"].models[0]


def test_provider_table_shows_local_default_as_from_server(session):
    console = _console()

    dispatch("/provider", session, console)

    output = _output(console)
    assert "Ollama (local)" in output
    assert "(from server)" in output


def test_config_provider_list_marks_local_kind(session):
    console = _console()

    dispatch("/config provider list", session, console)

    output = _output(console)
    assert "Ollama (local)" in output
    assert "LM Studio (local)" in output
    assert "local" in output  # the type column


def test_doctor_command_reports_diagnostics(session):
    console = _console()

    result = dispatch("/doctor", session, console)

    assert result == CommandResult.CONTINUE
    output = _output(console)
    assert "diagnostics" in output.lower()
    assert "openrouter" in output


def test_config_permissions_list_and_remove(session):
    config.persist_always_allowed_tool("run_bash")
    session.allow_always("run_bash")
    console = _console()

    dispatch("/config permissions list", session, console)
    assert "run_bash" in _output(console)

    dispatch("/config permissions remove run_bash", session, console)
    assert config.get_always_allowed_tools() == []
    assert not session.is_always_allowed("run_bash")


def test_config_permissions_clear(session):
    config.persist_always_allowed_tool("run_bash")
    session.allow_always("run_bash")
    console = _console()

    dispatch("/config permissions clear", session, console)

    assert config.get_always_allowed_tools() == []
    assert session.always_allowed == set()


def test_config_commands_allow_deny_and_clear(session):
    console = _console()

    dispatch("/config commands deny rm -rf", session, console)
    assert config.get_command_rules()["deny"] == ["rm -rf"]
    assert session.command_rules["deny"] == ["rm -rf"]

    dispatch("/config commands allow ^pytest", session, console)
    dispatch("/config commands list", session, console)
    output = _output(console)
    assert "rm -rf" in output
    assert "^pytest" in output

    dispatch("/config commands remove deny rm -rf", session, console)
    assert config.get_command_rules()["deny"] == []

    dispatch("/config commands clear", session, console)
    assert config.get_command_rules() == {"allow": [], "deny": []}
    assert session.command_rules == {"allow": [], "deny": []}


def test_config_sampling_set_show_reset(session):
    console = _console()

    dispatch("/config sampling set temperature=0.2 max_tokens=4096", session, console)
    assert config.get_sampling() == {"temperature": 0.2, "max_tokens": 4096}

    dispatch("/config sampling show", session, console)
    assert "0.2" in _output(console)

    dispatch("/config sampling reset", session, console)
    assert config.get_sampling() == {}


def test_config_sampling_rejects_bad_values(session):
    console = _console()

    dispatch("/config sampling set temperature=9", session, console)

    output = _output(console)
    assert "between 0 and 2" in output
    assert config.get_sampling() == {}


def test_config_style_updates_session(session):
    console = _console()

    dispatch("/config style set concise", session, console)

    assert session.output_style == "concise"
    assert config.get_output_style() == "concise"


def test_config_style_rejects_unknown(session):
    console = _console()

    dispatch("/config style set fancy", session, console)

    assert session.output_style == "default"
    assert "Unknown output style" in _output(console)


def test_config_prompt_set_show_clear(session):
    console = _console()

    dispatch("/config prompt set Always use type hints.", session, console)
    assert session.custom_system_prompt == "Always use type hints."
    assert config.get_system_prompt() == "Always use type hints."

    dispatch("/config prompt", session, console)
    assert "Always use type hints." in _output(console)

    dispatch("/config prompt clear", session, console)
    assert session.custom_system_prompt is None
    assert config.get_system_prompt() is None


def test_undo_command_restores_file(session):
    target = session.workspace_root / "a.txt"
    target.write_text("v0")
    session.checkpoint(["a.txt"], "edit_file a.txt")
    target.write_text("v1")
    console = _console()

    result = dispatch("/undo", session, console)

    assert result == CommandResult.CONTINUE
    assert target.read_text() == "v0"
    assert "a.txt" in _output(console)


def test_undo_command_without_checkpoints(session):
    console = _console()

    dispatch("/undo", session, console)

    assert "No checkpoints" in _output(console)


def test_checkpoints_command_lists(session):
    (session.workspace_root / "a.txt").write_text("x")
    session.checkpoint(["a.txt"], "edit_file a.txt")
    console = _console()

    dispatch("/checkpoints", session, console)

    output = _output(console)
    assert "edit_file a.txt" in output
    assert "Checkpoints" in output


def test_export_command_writes_markdown(session):
    session.messages.append({"role": "user", "content": "hello"})
    console = _console()

    dispatch("/export notes.md", session, console)

    destination = session.workspace_root / "notes.md"
    assert destination.is_file()
    assert "hello" in destination.read_text()


def test_fork_command_saves_copy(session):
    session.messages.append({"role": "user", "content": "branch me"})
    console = _console()

    dispatch("/fork mybranch", session, console)

    assert config.CONFIG_DIR.joinpath("sessions", "mybranch.json").is_file()
    assert "mybranch" in _output(console)



def test_dry_run_command_toggles(session):
    console = _console()

    dispatch("/dry-run on", session, console)
    assert session.dry_run is True

    dispatch("/dry-run toggle", session, console)
    assert session.dry_run is False

    dispatch("/dry-run", session, console)
    assert "off" in _output(console)


def test_config_trust_toggles(session):
    console = _console()

    dispatch("/config trust on", session, console)
    assert session.trusted_workspace is True
    assert config.get_trusted_workspace() is True

    dispatch("/config trust off", session, console)
    assert session.trusted_workspace is False
    assert config.get_trusted_workspace() is False

    dispatch("/config trust", session, console)
    assert "off" in _output(console)


def test_todos_command_lists_tasks(session):
    session.todos = [{"content": "ship it", "status": "in_progress"}]
    console = _console()

    dispatch("/todos", session, console)

    output = _output(console)
    assert "ship it" in output
    assert "in progress" in output


def test_compact_command_trims_history(session):
    for index in range(12):
        session.messages.append(
            {"role": "user", "content": f"turn {index} " + "x" * 2000}
        )
        session.messages.append(
            {"role": "assistant", "content": f"answer {index} " + "y" * 2000}
        )
    console = _console()

    dispatch("/compact 2000", session, console)

    output = _output(console)
    assert "Compacted" in output
    assert len(session.messages) < 24


def test_config_verify_set_and_clear(session):
    console = _console()

    dispatch("/config verify set pytest -q", session, console)
    assert session.verify_command == "pytest -q"
    assert config.get_verify_command() == "pytest -q"

    dispatch("/config verify clear", session, console)
    assert session.verify_command == ""
    assert config.get_verify_command() == ""


def test_config_budget_set_show_clear(session):
    console = _console()

    dispatch("/config budget tokens 5000", session, console)
    assert config.get_budget() == {"max_tokens": 5000}

    dispatch("/config budget cost 2.5", session, console)
    assert config.get_budget() == {"max_tokens": 5000, "max_cost_usd": 2.5}

    dispatch("/config budget show", session, console)
    assert "5000" in _output(console)

    dispatch("/config budget clear", session, console)
    assert config.get_budget() == {}


def test_config_profile_save_list_show_use_remove(session):
    console = _console()
    config.set_selected_model("captured-model")
    session.model = "captured-model"

    dispatch("/config profile save work", session, console)
    assert config.get_profile("work")["model"] == "captured-model"

    list_console = _console()
    dispatch("/config profile list", session, list_console)
    assert "work" in _output(list_console)

    show_console = _console()
    dispatch("/config profile show work", session, show_console)
    assert "captured-model" in _output(show_console)

    config.set_selected_model("other")
    session.model = "other"
    use_console = _console()
    dispatch("/config profile use work", session, use_console)

    assert session.model == "captured-model"
    assert config.load_config()["selected_model"] == "captured-model"
    assert "work" in _output(use_console)

    dispatch("/config profile remove work", session, console)
    assert config.get_profile("work") is None


def test_config_profile_unknown_is_reported(session):
    console = _console()

    dispatch("/config profile use nope", session, console)

    assert "Unknown profile" in _output(console)


def test_config_action_descriptions_include_profile():
    from kiwimatecoder.commands import _CONFIG_ACTION_DESCRIPTIONS

    assert "profile" in _CONFIG_ACTION_DESCRIPTIONS


def test_config_cache_toggles(session):
    console = _console()

    dispatch("/config cache on", session, console)
    assert config.get_prompt_cache() is True

    dispatch("/config cache", session, console)
    assert "on" in _output(console)

    dispatch("/config cache off", session, console)
    assert config.get_prompt_cache() is False


# ---------------------------------------------------------------------------
# Custom prompt templates
# ---------------------------------------------------------------------------


def _write_template(session, name, text):
    commands_dir = session.workspace_root / ".kiwimatecoder" / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    (commands_dir / f"{name}.md").write_text(text)


def test_templates_command_lists_custom_templates(session):
    _write_template(session, "review", "# Review the diff\nCheck for bugs.")
    console = _console()

    assert dispatch("/templates", session, console) == CommandResult.CONTINUE

    output = _output(console)
    assert "/review" in output
    assert "Review the diff" in output
    assert ".md" in output


def test_templates_command_without_templates(session):
    console = _console()

    dispatch("/templates", session, console)

    assert "No custom command templates" in _output(console)


def test_help_includes_custom_commands_group(session):
    _write_template(session, "review", "# Review the change")
    console = _console()

    dispatch("/help", session, console)

    output = _output(console)
    assert "Custom commands" in output
    assert "/review" in output


# ---------------------------------------------------------------------------
# /config ui
# ---------------------------------------------------------------------------


def test_config_ui_show_prints_current_settings(session):
    console = _console()

    dispatch("/config ui show", session, console)

    output = _output(console)
    assert "UI:" in output
    assert "default" in output
    assert "normal" in output


def test_config_ui_bare_prints_current_settings(session):
    console = _console()

    dispatch("/config ui", session, console)

    assert "UI:" in _output(console)


def test_config_ui_updates_settings(session):
    console = _console()

    dispatch("/config ui color never", session, console)
    assert config.get_ui()["color"] == "never"

    dispatch("/config ui output compact", session, console)
    assert config.get_ui()["output_mode"] == "compact"

    dispatch("/config ui ascii on", session, console)
    assert config.get_ui()["ascii"] is True

    dispatch("/config ui theme ocean", session, console)
    assert config.get_ui()["theme"] == "ocean"

    dispatch("/config ui keybindings vim", session, console)
    assert config.get_ui()["keybindings"] == "vim"

    dispatch("/config ui notify bell", session, console)
    assert config.get_ui()["notify"] == "bell"

    dispatch("/config ui notify-after 5", session, console)
    assert config.get_ui()["notify_after_seconds"] == 5

    dispatch("/config ui spinner off", session, console)
    assert config.get_ui()["spinner"] == "off"

    output = _output(console)
    assert "ocean" in output
    assert "Restart the session" in output


def test_config_ui_rejects_invalid_values(session):
    console = _console()

    dispatch("/config ui color rainbow", session, console)
    dispatch("/config ui output loud", session, console)
    dispatch("/config ui ascii maybe", session, console)
    dispatch("/config ui theme neon", session, console)
    dispatch("/config ui keybindings nano", session, console)
    dispatch("/config ui notify loud", session, console)
    dispatch("/config ui notify-after soon", session, console)
    dispatch("/config ui spinner sometimes", session, console)

    assert config.get_ui() == ui.UI_DEFAULTS


def test_config_action_descriptions_include_ui():
    from kiwimatecoder.commands import _CONFIG_ACTION_DESCRIPTIONS

    assert "ui" in _CONFIG_ACTION_DESCRIPTIONS
