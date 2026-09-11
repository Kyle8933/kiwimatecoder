from __future__ import annotations

import io
import json
import shlex
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from kiwimatecoder import audit, config, hooks
from kiwimatecoder.agent import Agent
from kiwimatecoder.client import Done, TextDelta, ToolCallDelta
from kiwimatecoder.events import POST_TOOL, PRE_TOOL, SESSION_START, EventBus
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.session import Session


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    home = tmp_path / "config-home"
    monkeypatch.setattr(config, "CONFIG_DIR", home)
    monkeypatch.setattr(config, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", home / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    return home


@pytest.fixture
def hook_session(tmp_path):
    return Session(
        provider_id="openrouter",
        model="test-model",
        mode=PermissionMode.ASK,
        workspace_root=tmp_path,
    )


def _script_command(workspace: Path, name: str, body: str) -> str:
    script = workspace / f"{name}.py"
    script.write_text(body)
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"


def _mock_tool_stream(tool: str, arguments: dict[str, Any], reply: str = "done"):
    round_1 = [
        ToolCallDelta(
            index=0,
            id="call_1",
            name=tool,
            args_fragment=json.dumps(arguments),
        ),
        Done(finish_reason="tool_calls"),
    ]
    round_2 = [TextDelta(text=reply), Done(finish_reason="stop")]
    calls = {"n": 0}

    async def mock_stream(*args, **kwargs):
        events = round_1 if calls["n"] == 0 else round_2
        calls["n"] += 1
        for event in events:
            yield event

    return mock_stream


# ---------------------------------------------------------------------------
# run_hooks
# ---------------------------------------------------------------------------


def test_run_hooks_without_configuration_is_empty(hook_session):
    assert hooks.run_hooks("pre_tool", session=hook_session) == []


def test_run_hooks_exports_environment_and_cwd(hook_session):
    workspace = hook_session.workspace_root
    marker = workspace / "env.txt"
    command = _script_command(
        workspace,
        "dump_env",
        "import os, pathlib\n"
        f"pathlib.Path({str(marker)!r}).write_text('|'.join([\n"
        "    os.environ['KIWI_EVENT'],\n"
        "    os.environ.get('KIWI_TOOL_NAME', ''),\n"
        "    os.environ['KIWI_WORKSPACE'],\n"
        "    os.path.realpath(os.getcwd()),\n"
        "]))\n",
    )
    config.add_hook("pre_tool", command)

    results = hooks.run_hooks(
        "pre_tool", session=hook_session, tool_name="write_file"
    )

    assert [result.ok for result in results] == [True]
    fields = marker.read_text().split("|")
    assert fields[0] == "pre_tool"
    assert fields[1] == "write_file"
    assert Path(fields[2]).resolve() == workspace.resolve()
    assert Path(fields[3]).resolve() == workspace.resolve()


def test_run_hooks_nonzero_exit_is_reported(hook_session):
    config.add_hook("pre_tool", "exit 3")

    results = hooks.run_hooks("pre_tool", session=hook_session)

    assert len(results) == 1
    result = results[0]
    assert result.exit_code == 3
    assert result.timed_out is False
    assert result.ok is False
    assert result.blocked is True


def test_run_hooks_timeout_is_captured(hook_session, monkeypatch):
    monkeypatch.setattr(hooks, "HOOK_TIMEOUT_SECONDS", 0.2)
    config.add_hook("pre_tool", "sleep 5")

    result = hooks.run_hooks("pre_tool", session=hook_session)[0]

    assert result.timed_out is True
    assert result.exit_code == -1
    assert result.ok is False
    assert result.blocked is True
    assert "timed out" in result.output


def test_run_hooks_redacts_secret_looking_tool_args(hook_session):
    workspace = hook_session.workspace_root
    marker = workspace / "args.txt"
    command = _script_command(
        workspace,
        "dump_args",
        "import os, pathlib\n"
        f"pathlib.Path({str(marker)!r}).write_text(os.environ['KIWI_TOOL_ARGS'])\n",
    )
    config.add_hook("post_tool", command)
    secret = "sk-or-abcdefghijklmnop123456"

    hooks.run_hooks(
        "post_tool", session=hook_session, tool_args={"api_key": secret}
    )

    written = marker.read_text()
    assert secret not in written
    assert "[REDACTED]" in written


def test_run_hooks_exports_tool_ok(hook_session):
    workspace = hook_session.workspace_root
    marker = workspace / "ok.txt"
    command = _script_command(
        workspace,
        "dump_ok",
        "import os, pathlib\n"
        f"pathlib.Path({str(marker)!r}).write_text("
        "os.environ.get('KIWI_TOOL_OK', 'missing'))\n",
    )
    config.add_hook("post_tool", command)

    hooks.run_hooks(
        "post_tool", session=hook_session, tool_name="write_file", ok=True
    )
    assert marker.read_text() == "true"

    hooks.run_hooks(
        "post_tool", session=hook_session, tool_name="write_file", ok=False
    )
    assert marker.read_text() == "false"


def test_run_hooks_stays_silent_without_console(hook_session, capsys):
    config.add_hook("session_start", "echo invisible")

    results = hooks.run_hooks("session_start", session=hook_session)

    assert results[0].output == "invisible\n"
    assert capsys.readouterr().out == ""


def test_run_hooks_prints_output_with_console(hook_session):
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    config.add_hook("session_start", "echo visible")

    hooks.run_hooks("session_start", session=hook_session, console=console)

    output = buf.getvalue()
    assert "visible" in output
    assert "session_start" in output


def test_run_hooks_uses_given_config_mapping(hook_session):
    cfg = {"hooks": {"session_end": ["echo from-cfg"]}}

    results = hooks.run_hooks("session_end", session=hook_session, cfg=cfg)

    assert results[0].output.strip() == "from-cfg"


def test_repl_lifecycle_hooks_emit_and_never_raise(hook_session):
    from kiwimatecoder import repl

    config.add_hook("session_start", "exit 1")
    bus = EventBus()
    seen: list[tuple[str, dict[str, Any]]] = []
    bus.subscribe(SESSION_START, lambda event: seen.append((event.name, event.payload)))

    repl._run_lifecycle_hooks(bus, SESSION_START, hook_session)  # noqa: SLF001

    assert [name for name, _ in seen] == [SESSION_START]
    assert seen[0][1]["workspace"] == str(hook_session.workspace_root)


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_failing_pre_tool_hook_blocks_write(hook_session):
    config.add_hook("pre_tool", "echo block reason >&2; exit 7")
    agent = Agent(
        hook_session,
        Console(quiet=True),
        MagicMock(return_value=True),
        bus=EventBus(),
    )

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=_mock_tool_stream(
                "write_file", {"path": "blocked.txt", "content": "hi"}
            ),
        ),
    ):
        await agent.run_turn("write a file")

    assert not (hook_session.workspace_root / "blocked.txt").exists()
    tool_message = next(
        message for message in hook_session.messages if message.get("role") == "tool"
    )
    assert "blocked" in tool_message["content"].lower()
    assert "block reason" in tool_message["content"]

    entries = [
        json.loads(line)
        for line in audit.audit_log_path().read_text(encoding="utf-8").splitlines()
    ]
    blocked_entry = next(entry for entry in entries if entry["tool"] == "write_file")
    assert blocked_entry["decision"] == "hook_blocked"


@pytest.mark.anyio
async def test_successful_pre_tool_hook_runs_before_write(hook_session):
    config.add_hook("pre_tool", "echo ran > pre.marker")
    agent = Agent(
        hook_session,
        Console(quiet=True),
        MagicMock(return_value=True),
        bus=EventBus(),
    )

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=_mock_tool_stream(
                "write_file", {"path": "wrote.txt", "content": "hi"}
            ),
        ),
    ):
        await agent.run_turn("write a file")

    assert (hook_session.workspace_root / "pre.marker").exists()
    assert (hook_session.workspace_root / "wrote.txt").read_text() == "hi"


@pytest.mark.anyio
async def test_post_tool_hook_sees_success(hook_session):
    config.add_hook("post_tool", 'echo "$KIWI_TOOL_OK" > post.marker')
    agent = Agent(
        hook_session,
        Console(quiet=True),
        MagicMock(return_value=True),
        bus=EventBus(),
    )

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=_mock_tool_stream(
                "write_file", {"path": "wrote.txt", "content": "hi"}
            ),
        ),
    ):
        await agent.run_turn("write a file")

    marker = hook_session.workspace_root / "post.marker"
    assert marker.read_text().strip() == "true"


@pytest.mark.anyio
async def test_agent_emits_pre_and_post_tool_events(hook_session):
    (hook_session.workspace_root / "hello.txt").write_text("hi")
    bus = EventBus()
    seen: list[tuple[str, dict[str, Any]]] = []
    bus.subscribe(PRE_TOOL, lambda event: seen.append((event.name, event.payload)))
    bus.subscribe(POST_TOOL, lambda event: seen.append((event.name, event.payload)))
    agent = Agent(
        hook_session,
        Console(quiet=True),
        MagicMock(return_value=True),
        bus=bus,
    )

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=_mock_tool_stream("read_file", {"path": "hello.txt"}),
        ),
    ):
        await agent.run_turn("read the file")

    assert [name for name, _ in seen] == [PRE_TOOL, POST_TOOL]
    assert seen[0][1]["tool"] == "read_file"
    post = seen[1][1]
    assert post["tool"] == "read_file"
    assert post["ok"] is True
    assert post["decision"] == "allowed"
    assert isinstance(post["duration_ms"], int)


@pytest.mark.anyio
async def test_hooks_do_not_run_for_dry_run(hook_session):
    hook_session.dry_run = True
    config.add_hook("pre_tool", "echo ran > dry.marker")
    agent = Agent(
        hook_session,
        Console(quiet=True),
        MagicMock(return_value=True),
        bus=EventBus(),
    )

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=_mock_tool_stream(
                "write_file", {"path": "dry.txt", "content": "hi"}
            ),
        ),
    ):
        await agent.run_turn("write a file")

    assert not (hook_session.workspace_root / "dry.marker").exists()
    assert not (hook_session.workspace_root / "dry.txt").exists()