import io

import pytest
from rich.console import Console

from kiwimatecoder import config
from kiwimatecoder.commands import (
    CommandResult,
    SelectionPrompt,
    dispatch,
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
