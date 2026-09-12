from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from kiwimatecoder import config, memory, tools
from kiwimatecoder.agent import Agent
from kiwimatecoder.commands import (
    dispatch,
    slash_argument_completions,
    slash_command_completions,
)
from kiwimatecoder.main import app as cli_app
from kiwimatecoder.prompts import build_system_prompt
from kiwimatecoder.tools.memory import _recall, _remember, preview


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "home")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "home" / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "home" / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def _output(console: Console) -> str:
    return console.file.getvalue()


# ---------------------------------------------------------------------------
# memory module
# ---------------------------------------------------------------------------


def test_memory_paths(session):
    assert memory.memory_path("project", session.workspace_root) == (
        session.workspace_root / ".kiwimatecoder" / "memory.md"
    )
    assert memory.memory_path("user", session.workspace_root) == (
        config.CONFIG_DIR / "memory.md"
    )


def test_project_append_read_clear_roundtrip(session):
    path = memory.append_memory("project", session.workspace_root, "Uses Ruff")

    assert path.is_file()
    assert "- Uses Ruff\n" in path.read_text()
    memory.append_memory("project", session.workspace_root, "Runs pytest -q")

    text = memory.read_memory("project", session.workspace_root)
    assert "- Uses Ruff" in text
    assert "- Runs pytest -q" in text

    assert memory.clear_memory("project", session.workspace_root) is True
    assert memory.read_memory("project", session.workspace_root) == ""
    assert memory.clear_memory("project", session.workspace_root) is False


def test_user_scope_roundtrip(session):
    path = memory.append_memory("user", session.workspace_root, "Prefers concise answers")

    assert path == config.CONFIG_DIR / "memory.md"
    assert memory.read_memory("user", session.workspace_root) == (
        "- Prefers concise answers"
    )
    assert memory.clear_memory("user", session.workspace_root) is True


def test_append_creates_parent_directories(session):
    directory = session.workspace_root / ".kiwimatecoder"
    assert not directory.exists()

    memory.append_memory("project", session.workspace_root, "fact")

    assert directory.is_dir()


def test_append_rejects_empty_and_oversized(session):
    for text in ("", "   ", "\n"):
        with pytest.raises(ValueError, match="non-empty"):
            memory.append_memory("project", session.workspace_root, text)

    with pytest.raises(ValueError, match="too large"):
        memory.append_memory("project", session.workspace_root, "x" * 20_000)

    exact = "y" * memory.MAX_ENTRY_BYTES
    path = memory.append_memory("project", session.workspace_root, exact)
    assert path.is_file()


def test_read_missing_binary_and_truncated(session):
    assert memory.read_memory("project", session.workspace_root) == ""

    path = memory.memory_path("project", session.workspace_root)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"bin\x00ary")
    assert memory.read_memory("project", session.workspace_root) == ""

    path.write_text("a" * 100)
    text = memory.read_memory("project", session.workspace_root, max_bytes=10)
    assert text.startswith("a" * 10)
    assert "[truncated: showing 10 of 100 bytes]" in text


def test_invalid_scope_rejected(session):
    with pytest.raises(ValueError, match="Unknown memory scope"):
        memory.memory_path("global", session.workspace_root)
    with pytest.raises(ValueError, match="Unknown memory scope"):
        memory.append_memory("global", session.workspace_root, "x")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_memory_config_defaults_and_roundtrip():
    assert config.get_memory() == {"enabled": True, "max_bytes": 16384}

    settings = config.set_memory(max_bytes=2048)
    assert settings["max_bytes"] == 2048
    assert config.load_config()["memory"]["max_bytes"] == 2048

    config.set_memory(enabled=False)
    assert config.get_memory() == {"enabled": False, "max_bytes": 2048}


@pytest.mark.parametrize("value", [10, 2_000_000, "junk"])
def test_set_memory_rejects_bad_max_bytes(value):
    with pytest.raises(ValueError):
        config.set_memory(max_bytes=value)


def test_set_memory_rejects_non_bool_enabled():
    with pytest.raises(ValueError, match="true or false"):
        config.set_memory(enabled="on")  # type: ignore[arg-type]


def test_get_memory_normalizes_bad_stored_values():
    config.save_config({"memory": {"enabled": "yes", "max_bytes": "lots"}})

    assert config.get_memory() == {"enabled": True, "max_bytes": 16384}


def test_validate_memory_section():
    clean = config.load_config()
    assert [
        issue for issue in config.validate_config(clean) if issue["key"].startswith("memory")
    ] == []

    cfg = config.load_config()
    cfg["memory"] = {"enabled": "yes", "max_bytes": 5}
    keys = {issue["key"] for issue in config.validate_config(cfg)}
    assert {"memory.enabled", "memory.max_bytes"} <= keys

    cfg["memory"] = ["nope"]
    issues = config.validate_config(cfg)
    assert any(
        issue["level"] == "error" and issue["key"] == "memory" for issue in issues
    )


# ---------------------------------------------------------------------------
# prompt section
# ---------------------------------------------------------------------------


def test_memory_section_renders_both_scopes(session):
    memory.append_memory("project", session.workspace_root, "Project uses Ruff")
    memory.append_memory("user", session.workspace_root, "User prefers pytest")

    section = memory.memory_section(session.workspace_root)

    assert '<kiwi_memory scope="project">' in section
    assert '<kiwi_memory scope="user">' in section
    assert "Project uses Ruff" in section
    assert "User prefers pytest" in section
    assert "remember tool" in section
    assert "never store secrets" in section


def test_memory_section_omitted_when_disabled(session):
    memory.append_memory("project", session.workspace_root, "Project uses Ruff")
    config.set_memory(enabled=False)

    assert memory.memory_section(session.workspace_root) == ""


def test_memory_section_caps_total_bytes(session):
    config.set_memory(max_bytes=1024)
    memory.append_memory("project", session.workspace_root, "P" * 900)
    memory.append_memory("user", session.workspace_root, "U" * 900)

    section = memory.memory_section(session.workspace_root)

    assert "[truncated" in section
    assert "U" * 200 not in section


def test_system_prompt_includes_and_omits_memory(session):
    memory.append_memory("project", session.workspace_root, "Uses Ruff for linting")

    prompt = build_system_prompt(session)["content"]
    assert "Persistent memory follows" in prompt
    assert "Uses Ruff for linting" in prompt

    config.set_memory(enabled=False)
    disabled = build_system_prompt(session)["content"]
    assert "Persistent memory follows" not in disabled
    assert "Uses Ruff for linting" not in disabled


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


def test_remember_tool_writes_and_returns_path(session):
    result = _remember({"content": "Uses Ruff", "scope": "project"}, session)

    assert result.ok
    path = memory.memory_path("project", session.workspace_root)
    assert str(path) in result.content
    assert path.read_text() == "- Uses Ruff\n"


def test_remember_tool_validation(session):
    assert not _remember({"content": ""}, session).ok
    assert not _remember({"content": "x" * 20_000}, session).ok
    bad_scope = _remember({"content": "fact", "scope": "global"}, session)
    assert not bad_scope.ok
    assert "Unknown memory scope" in bad_scope.content


def test_recall_tool_lists_and_filters(session):
    memory.append_memory("project", session.workspace_root, "Prefers pytest")
    memory.append_memory("user", session.workspace_root, "Lives in Auckland")

    everything = _recall({}, session)
    assert everything.ok
    assert "Prefers pytest" in everything.content
    assert "Lives in Auckland" in everything.content

    filtered = _recall({"query": "PYTEST"}, session)
    assert "Prefers pytest" in filtered.content
    assert "Auckland" not in filtered.content

    missing = _recall({"query": "nothing-matches"}, session)
    assert "No memory entries match" in missing.content


def test_memory_tools_registered_with_correct_flags():
    remember = tools.get_tool("remember")
    recall = tools.get_tool("recall")
    assert remember is not None and recall is not None
    assert remember.writes is True
    assert remember.needs_approval is True
    assert recall.writes is False
    assert recall.runs is False
    assert recall.needs_approval is False

    read_only_names = {
        schema["function"]["name"] for schema in tools.tool_schemas(read_only=True)
    }
    assert "recall" in read_only_names
    assert "remember" not in read_only_names


def test_remember_preview_shows_path_and_bullet(session):
    text = preview({"content": "key sk-or-abcdefghijkl", "scope": "project"}, session)

    assert str(memory.memory_path("project", session.workspace_root)) in text
    assert "- " in text
    assert "sk-or-abcdefghijkl" not in text
    assert "[REDACTED]" in text
    assert "(empty; will be rejected)" in preview({"content": ""}, session)


def test_agent_memory_call_summaries(session):
    agent = Agent(session, Console(quiet=True), MagicMock())

    assert agent._format_call_summary("remember", {"content": "fact"}) == (
        "memory [dim]remember[/dim]"
    )
    assert agent._format_call_summary("recall", {}) == "memory [dim]recall[/dim]"
    assert agent._format_call_summary("recall", {"query": "ruff"}) == (
        "memory [dim]recall ruff[/dim]"
    )


# ---------------------------------------------------------------------------
# slash command
# ---------------------------------------------------------------------------


def test_memory_slash_list_add_and_clear(session):
    console = _console()
    dispatch("/memory list", session, console)
    flat = _output(console).replace("\n", "")
    assert "Memory:" in flat
    assert ".kiwimatecoder" in flat
    assert str(config.CONFIG_DIR) in flat
    assert "memory.md" in flat
    assert "(empty)" in flat

    dispatch("/memory add project fact", session, _console())
    dispatch("/memory add-user user fact", session, _console())
    assert "- project fact" in memory.read_memory("project", session.workspace_root)
    assert "- user fact" in memory.read_memory("user", session.workspace_root)

    listing = _console()
    dispatch("/memory", session, listing)
    assert "project fact" in _output(listing)
    assert "user fact" in _output(listing)

    cleared = _console()
    dispatch("/memory clear user", session, cleared)
    assert "Cleared user memory" in _output(cleared)
    assert memory.read_memory("user", session.workspace_root) == ""


def test_memory_slash_usage_and_errors(session):
    console = _console()
    dispatch("/memory add", session, console)
    dispatch("/memory add-user", session, console)
    dispatch("/memory clear global", session, console)
    dispatch("/memory dance", session, console)

    output = _output(console)
    assert "Usage: /memory add <text>" in output
    assert "Usage: /memory add-user <text>" in output
    assert "Unknown memory scope 'global'" in output
    assert "Usage: /memory" in output


def test_memory_slash_oversized_entry_is_reported(session):
    console = _console()

    dispatch(f"/memory add {'x' * 20_000}", session, console)

    assert "too large" in _output(console)


def test_memory_help_and_completions(session):
    help_console = _console()
    dispatch("/help", session, help_console)
    assert "/memory" in _output(help_console)

    assert ("/memory", "Show or edit persistent project and user memory.") in (
        slash_command_completions("mem")
    )
    values = {value for value, _desc in slash_argument_completions("memory", "")}
    assert {"list", "add", "add-user", "clear"} <= values


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_config_memory_show_and_toggle():
    result = CliRunner().invoke(cli_app, ["config", "memory", "show"])
    assert result.exit_code == 0
    assert "Memory" in result.output

    result = CliRunner().invoke(cli_app, ["config", "memory", "max-bytes", "2048"])
    assert result.exit_code == 0
    assert config.get_memory()["max_bytes"] == 2048

    result = CliRunner().invoke(cli_app, ["config", "memory", "enable", "off"])
    assert result.exit_code == 0
    assert config.get_memory()["enabled"] is False

    result = CliRunner().invoke(cli_app, ["config", "memory", "max-bytes", "5"])
    assert result.exit_code == 1
    assert "between" in result.output
