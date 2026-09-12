"""CLI interface assets: non-TTY fallback, completions, and the man page."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kiwimatecoder import config, main
from kiwimatecoder.client import Done, TextDelta

ROOT = Path(__file__).resolve().parents[1]
STREAM_CHAT = "kiwimatecoder.client.UnifiedClient.stream_chat"


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Non-TTY fallback
# ---------------------------------------------------------------------------


def test_bare_invocation_without_tty_exits_2_with_guidance(monkeypatch):
    from kiwimatecoder import repl

    launched: list[object] = []
    monkeypatch.setattr(repl, "run", lambda session: launched.append(session))

    result = CliRunner().invoke(main.app, [])

    assert result.exit_code == 2
    assert "Interactive session needs a TTY" in result.stderr
    assert "kiwimatecoder -p" in result.stderr
    assert "-p -" in result.stderr
    assert launched == []


def test_continue_without_tty_exits_2(tmp_path, monkeypatch):
    from kiwimatecoder import repl

    monkeypatch.setattr(
        "kiwimatecoder.session._sessions_dir", lambda: tmp_path / "sessions"
    )
    launched: list[object] = []
    monkeypatch.setattr(repl, "run", lambda session: launched.append(session))

    result = CliRunner().invoke(main.app, ["--continue"])

    assert result.exit_code == 2
    assert "Interactive session needs a TTY" in result.stderr
    assert launched == []


def test_print_dash_still_reads_stdin_without_tty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "get_key", lambda provider_id: "test-key")
    captured: list[list[dict[str, object]]] = []

    async def stream(*args, **kwargs):
        captured.append(args[1])
        yield TextDelta(text="stdin ok")
        yield Done(finish_reason="stop")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(STREAM_CHAT, stream)
        result = CliRunner().invoke(main.app, ["-p", "-"], input="hello stdin")

    assert result.exit_code == 0
    assert "stdin ok" in result.stdout
    user_messages = [
        message for message in captured[0] if message.get("role") == "user"
    ]
    assert user_messages[-1]["content"] == "hello stdin"


# ---------------------------------------------------------------------------
# Completions
# ---------------------------------------------------------------------------


def test_help_advertises_shell_completion_flags():
    result = CliRunner().invoke(main.app, ["--help"])

    assert result.exit_code == 0
    assert "--install-completion" in result.stdout
    assert "--show-completion" in result.stdout


def test_show_completion_emits_a_script(monkeypatch):
    import typer.completion

    monkeypatch.setattr(typer.completion, "_get_shell_name", lambda: "bash")

    result = CliRunner().invoke(main.app, ["--show-completion"])

    assert result.exit_code == 0
    assert "_COMPLETE=complete_bash" in result.stdout or "COMPREPLY" in result.stdout


# ---------------------------------------------------------------------------
# Man page
# ---------------------------------------------------------------------------


def test_man_page_has_expected_sections_and_commands():
    text = (ROOT / "docs" / "kiwimatecoder.1").read_text(encoding="utf-8")

    for marker in (
        ".SH NAME",
        ".SH SYNOPSIS",
        ".SH DESCRIPTION",
        ".SH OPTIONS",
        ".SH COMMANDS",
        ".SH FILES",
        ".SH EXIT STATUS",
    ):
        assert marker in text
    for command in ("setup", "ask", "doctor", "update", "version", "config"):
        assert f".SS \\fB{command}\\fP" in text
    assert r"\fB\-\-print\fP, \fB\-p\fP" in text
    assert "kiwimatecoder \\- agentic AI coding assistant CLI" in text


def test_committed_man_page_matches_the_generator():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_man.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 0, result.stderr
