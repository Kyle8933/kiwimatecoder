"""Coverage for previously untested slash commands."""
import io

import pytest
from rich.console import Console

from kiwimatecoder import config
from kiwimatecoder.commands import CommandResult, dispatch
from kiwimatecoder.permissions import PermissionMode


def _console():
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def _output(console) -> str:
    return console.file.getvalue()


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    from kiwimatecoder.providers import REGISTRY

    for provider in REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)
    monkeypatch.delenv("LOCAL_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# /tools, /files, /cost, session persistence slash commands
# ---------------------------------------------------------------------------


def test_tools_lists_read_only_and_approval_tools(session):
    session.allow_always("run_bash")
    console = _console()

    assert dispatch("/tools", session, console) == CommandResult.CONTINUE

    output = _output(console)
    assert "Available Tools" in output
    assert "read_file" in output
    assert "run_bash" in output
    assert "write_file" in output
    assert "read-only" in output.lower() or "Read-only" in output or "read-only" in output
    # always-allowed write tool shows as always allowed
    assert "always allowed" in output.lower()
    # other approval tools still need approval
    assert "needs approval" in output.lower()


def test_files_empty_session_prints_dim_message(session):
    console = _console()

    assert dispatch("/files", session, console) == CommandResult.CONTINUE
    assert "No files changed" in _output(console)


def test_files_lists_existing_missing_and_directory(session):
    existing = session.workspace_root / "touched.py"
    existing.write_text("print('hi')\n")
    (session.workspace_root / "subdir").mkdir()
    session.touched_files = ["touched.py", "subdir", "gone.py"]
    console = _console()

    assert dispatch("/files", session, console) == CommandResult.CONTINUE

    output = _output(console)
    assert "touched.py" in output
    assert "exists" in output.lower()
    assert "subdir" in output
    assert "directory" in output.lower()
    assert "gone.py" in output
    assert "deleted" in output.lower() or "missing" in output.lower()


def test_files_shows_kb_size_for_larger_files(session):
    big = session.workspace_root / "big.bin"
    big.write_bytes(b"x" * 2048)
    session.touched_files = ["big.bin"]
    console = _console()

    dispatch("/files", session, console)

    assert "KB" in _output(console)


def test_cost_reports_token_totals(session):
    session.prompt_tokens = 1_000
    session.completion_tokens = 250
    session.messages = [{"role": "user", "content": "hi"}]
    console = _console()

    assert dispatch("/cost", session, console) == CommandResult.CONTINUE

    output = _output(console)
    assert "1,000" in output
    assert "250" in output
    assert "1,250" in output
    assert "Estimated Cost" in output


def test_help_lists_core_commands(session):
    console = _console()

    assert dispatch("/help", session, console) == CommandResult.CONTINUE

    output = _output(console)
    assert "/model" in output
    assert "/provider" in output
    assert "/config" in output
    assert "kiwimatecoder setup" in output


def test_clear_resets_conversation_history(session):
    session.messages = [{"role": "user", "content": "keep me?"}]
    session.prompt_tokens = 10
    console = _console()

    assert dispatch("/clear", session, console) == CommandResult.CONTINUE

    assert session.messages == []
    assert "cleared" in _output(console).lower()


def test_exit_and_quit_return_exit(session):
    console = _console()

    assert dispatch("/exit", session, console) == CommandResult.EXIT
    assert "Goodbye" in _output(console)

    console2 = _console()
    assert dispatch("/quit", session, console2) == CommandResult.EXIT


def test_mode_rejects_unknown_value(session):
    console = _console()

    assert dispatch("/mode nonsense", session, console) == CommandResult.CONTINUE
    assert session.mode is PermissionMode.ASK
    output = _output(console).lower()
    assert "invalid" in output or "unknown" in output


def test_context_list_shows_empty_and_pinned(session):
    console = _console()
    assert dispatch("/context", session, console) == CommandResult.CONTINUE
    assert "No pinned context" in _output(console)

    pinned = session.workspace_root / "notes.md"
    pinned.write_text("pinned\n")
    session.context_files = ["notes.md"]
    console2 = _console()
    assert dispatch("/context list", session, console2) == CommandResult.CONTINUE
    output = _output(console2)
    assert "notes.md" in output
    assert "bytes" in output


def test_config_model_show_set_and_reset(session):
    console = _console()

    dispatch("/config model show", session, console)
    assert "test-model" in _output(console)

    dispatch("/config model set my-default", session, _console())
    assert session.model == "my-default"
    assert config.load_config().get("selected_model") == "my-default"

    usage = _console()
    dispatch("/config model set", session, usage)  # usage error path
    assert "Usage:" in _output(usage)

    dispatch("/config model reset", session, _console())
    assert session.model == session.provider.default_model
    assert config.load_config().get("selected_model") is None


def test_config_model_unknown_action(session):
    console = _console()
    dispatch("/config model dance", session, console)
    assert "Unknown model config action" in _output(console)


def test_config_help_prints_usage(session):
    console = _console()
    dispatch("/config help", session, console)
    output = _output(console)
    assert "/config" in output
    assert "key" in output.lower() or "provider" in output.lower()


def test_save_load_and_sessions_roundtrip(session, tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    monkeypatch.setattr("kiwimatecoder.session._sessions_dir", lambda: sessions_dir)

    session.messages = [{"role": "user", "content": "remember this"}]
    session.prompt_tokens = 42
    session.completion_tokens = 7
    console = _console()

    assert dispatch("/save draft-session", session, console) == CommandResult.CONTINUE
    assert "draft-session" in _output(console)
    assert (sessions_dir / "draft-session.json").is_file() or any(
        sessions_dir.iterdir()
    )

    list_console = _console()
    assert dispatch("/sessions", session, list_console) == CommandResult.CONTINUE
    assert "draft-session" in _output(list_console) or "Saved Sessions" in _output(
        list_console
    )

    # Wipe local state, then load back
    session.messages = []
    session.prompt_tokens = 0
    session.completion_tokens = 0
    load_console = _console()
    assert dispatch("/load draft-session", session, load_console) == CommandResult.CONTINUE
    assert session.messages == [{"role": "user", "content": "remember this"}]
    assert session.prompt_tokens == 42
    assert "Loaded session" in _output(load_console)


def test_load_without_name_lists_sessions(session, tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    monkeypatch.setattr("kiwimatecoder.session._sessions_dir", lambda: sessions_dir)
    console = _console()

    assert dispatch("/load", session, console) == CommandResult.CONTINUE
    assert "No saved sessions" in _output(console)


def test_load_missing_session_prints_error(session, tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    monkeypatch.setattr("kiwimatecoder.session._sessions_dir", lambda: sessions_dir)
    console = _console()

    assert dispatch("/load does-not-exist", session, console) == CommandResult.CONTINUE
    assert "Failed to load" in _output(console)
