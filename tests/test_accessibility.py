"""Accessibility guarantees: plain mode, spinner suppression, and the audit."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

from kiwimatecoder import config, main, ui
from kiwimatecoder.agent import Agent
from kiwimatecoder.client import Done, TextDelta, ToolCallDelta
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.session import Session
from tests.conftest import track_console

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    ui.apply_plain_mode(False)
    yield
    ui.apply_plain_mode(False)


@pytest.fixture
def agent_session(tmp_path):
    return Session(
        provider_id="openrouter",
        model="test-model",
        mode=PermissionMode.AUTO,
        workspace_root=tmp_path,
    )


async def _run_tool_turn(session: Session, console: Console, spinner: str) -> None:
    (session.workspace_root / "hello.txt").write_text("hi")
    agent = Agent(session, console, lambda *args: True, spinner=spinner)
    rounds = [
        [
            ToolCallDelta(
                index=0,
                id="call_read",
                name="read_file",
                args_fragment='{"path": "hello.txt"}',
            ),
            Done(finish_reason="tool_calls"),
        ],
        [TextDelta(text="done"), Done(finish_reason="stop")],
    ]
    state = {"round": 0}

    async def mock_stream(*args, **kwargs):
        events = rounds[min(state["round"], len(rounds) - 1)]
        state["round"] += 1
        for event in events:
            yield event

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn("read hello.txt")


async def test_spinner_off_suppresses_all_status(agent_session):
    console = Console(quiet=True)
    log = track_console(console)

    await _run_tool_turn(agent_session, console, "off")

    status_events = [event for event in log if event[0] in {"start", "stop"}]
    assert status_events == []


async def test_spinner_auto_keeps_status(agent_session):
    console = Console(quiet=True)
    log = track_console(console)

    await _run_tool_turn(agent_session, console, "auto")

    assert any(event[0] == "start" for event in log)


# ---------------------------------------------------------------------------
# --plain
# ---------------------------------------------------------------------------


def test_plain_flag_sets_process_overrides_without_persisting():
    result = CliRunner().invoke(main.app, ["--plain", "config", "ui", "show"])

    assert result.exit_code == 0
    assert ui.runtime_overrides() == {
        "ascii": True,
        "color": "never",
        "output_mode": "compact",
        "spinner": "off",
    }
    assert ui.glyph("check") == "[ok]"
    assert ui.spinner_enabled() is False
    assert ui.resolve_color() is False

    # Nothing was written to the config file.
    stored = config.get_ui()
    assert stored["ascii"] is False
    assert stored["spinner"] == "auto"
    assert stored["color"] == "auto"
    assert stored["output_mode"] == "normal"


def test_plain_flag_is_cleared_by_the_next_invocation():
    runner = CliRunner()
    runner.invoke(main.app, ["--plain", "config", "ui", "show"])

    runner.invoke(main.app, ["config", "ui", "show"])

    assert ui.runtime_overrides() == {}
    assert ui.glyph("check") == "✓"


def test_plain_output_uses_ascii_glyphs():
    result = CliRunner().invoke(
        main.app, ["--plain", "config", "key", "set", "openai", "sk-plain"]
    )

    assert result.exit_code == 0
    assert "[ok]" in result.output
    assert "✓" not in result.output


def test_plain_mode_does_not_persist_through_unsets():
    runner = CliRunner()
    runner.invoke(main.app, ["--plain", "config", "ui", "show"])
    runner.invoke(main.app, ["config", "ui", "show"])

    stored = config.get_ui()
    assert stored["ascii"] is False
    assert stored["output_mode"] == "normal"


# ---------------------------------------------------------------------------
# Audit assets
# ---------------------------------------------------------------------------


def test_accessibility_doc_covers_the_audit_scope():
    text = (ROOT / "docs" / "accessibility.md").read_text(encoding="utf-8").lower()

    for marker in (
        "no_color",
        "ui.color",
        "ui.ascii",
        "ui.output_mode",
        "ui.spinner",
        "--plain",
        "screen-reader",
        "high contrast",
        "reduced motion",
        "i18n",
        "follow-up",
    ):
        assert marker in text, marker


def test_readme_links_the_accessibility_audit():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "docs/accessibility.md" in readme
    assert "--plain" in readme
