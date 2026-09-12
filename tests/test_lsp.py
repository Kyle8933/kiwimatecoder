from __future__ import annotations

import io
import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

from kiwimatecoder import commands, config, lsp, main, tools
from kiwimatecoder.agent import Agent
from kiwimatecoder.client import Done, TextDelta, ToolCallDelta
from kiwimatecoder.commands import CommandResult
from kiwimatecoder.lsp import manager as lsp_manager_module
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.session import Session

FAKE_SERVER = r'''
import json
import sys


def send(message):
    body = json.dumps(message).encode("utf-8")
    header = b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n"
    sys.stdout.buffer.write(header + body)
    sys.stdout.buffer.flush()


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        name, _, value = line.decode("ascii", "replace").partition(":")
        headers[name.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    return json.loads(sys.stdin.buffer.read(length).decode("utf-8"))


def diagnostics_for(uri, text):
    if "bad" in text:
        return [
            {
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 3},
                },
                "severity": 1,
                "source": "fake",
                "message": "undefined name 'bad'",
            }
        ]
    return []


def main():
    while True:
        message = read_message()
        if message is None:
            return
        method = message.get("method")
        request_id = message.get("id")
        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"capabilities": {"textDocumentSync": 1}},
                }
            )
        elif method == "initialized":
            pass
        elif method == "textDocument/didOpen":
            doc = message["params"]["textDocument"]
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "textDocument/publishDiagnostics",
                    "params": {
                        "uri": doc["uri"],
                        "version": doc["version"],
                        "diagnostics": diagnostics_for(doc["uri"], doc.get("text", "")),
                    },
                }
            )
        elif method == "textDocument/didChange":
            doc = message["params"]["textDocument"]
            text = message["params"]["contentChanges"][-1].get("text", "")
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "textDocument/publishDiagnostics",
                    "params": {
                        "uri": doc["uri"],
                        "version": doc["version"],
                        "diagnostics": diagnostics_for(doc["uri"], text),
                    },
                }
            )
        elif method == "textDocument/definition":
            uri = message["params"]["textDocument"]["uri"]
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": [
                        {
                            "uri": uri,
                            "range": {
                                "start": {"line": 4, "character": 2},
                                "end": {"line": 4, "character": 6},
                            },
                        }
                    ],
                }
            )
        elif method == "textDocument/references":
            uri = message["params"]["textDocument"]["uri"]
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": [
                        {
                            "uri": uri,
                            "range": {
                                "start": {"line": 1, "character": 0},
                                "end": {"line": 1, "character": 4},
                            },
                        },
                        {
                            "uri": uri,
                            "range": {
                                "start": {"line": 8, "character": 4},
                                "end": {"line": 8, "character": 8},
                            },
                        },
                    ],
                }
            )
        elif method == "shutdown":
            send({"jsonrpc": "2.0", "id": request_id, "result": None})
        elif method == "exit":
            return
        elif request_id is not None:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "unknown method"},
                }
            )


main()
'''

SILENT_SERVER = "import sys\nfor _ in sys.stdin.buffer:\n    pass\n"


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=140)


def _output(console: Console) -> str:
    return console.file.getvalue()


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    config_dir = tmp_path / "cfg"
    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "CONFIG_FILE", config_dir / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", config_dir / "config")
    return config_dir


@pytest.fixture(autouse=True)
def reset_lsp_manager() -> Iterator[None]:
    yield
    # Read through the implementation module: tests monkeypatch the package
    # level ``lsp.get_manager`` with fakes that have no ``shutdown``.
    manager = lsp_manager_module.get_manager()
    if manager is not None:
        manager.shutdown()
    lsp_manager_module.set_manager(None)


def _use_fake_server(tmp_path: Path) -> Path:
    server = tmp_path / "fake_lsp_server.py"
    server.write_text(FAKE_SERVER)
    config.set_lsp(
        enabled=True,
        timeout=5.0,
        servers={
            "fake": {
                "command": sys.executable,
                "args": [str(server)],
                "extensions": [".py"],
            }
        },
    )
    return server


class _FakeLspManager:
    def __init__(self, diagnostics=None, error=None) -> None:
        self.diagnostics = diagnostics or []
        self.error = error
        self.calls: list[tuple[str, float | None]] = []

    def diagnostics_for(self, path, timeout=None):
        self.calls.append((str(path), timeout))
        if self.error is not None:
            raise self.error
        return self.diagnostics


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_lsp_config_defaults_roundtrip_and_validation():
    defaults = config.get_lsp()
    assert defaults["enabled"] is False
    assert defaults["timeout"] == 10.0
    assert defaults["diagnostics_after_edits"] is True
    assert defaults["servers"] == {}

    settings = config.set_lsp(
        enabled=True,
        timeout=5,
        diagnostics_after_edits=False,
        servers={
            "python": {
                "command": "pyright-langserver",
                "args": ["--stdio"],
                "extensions": ["py", ".pyi"],
            }
        },
    )
    assert settings["enabled"] is True
    assert settings["timeout"] == 5.0
    assert settings["diagnostics_after_edits"] is False
    assert settings["servers"]["python"]["extensions"] == [".py", ".pyi"]
    assert config.get_lsp() == settings

    with pytest.raises(ValueError):
        config.set_lsp(timeout=0.5)
    with pytest.raises(ValueError):
        config.set_lsp(timeout=61)
    with pytest.raises(ValueError):
        config.set_lsp(timeout="soon")
    with pytest.raises(ValueError):
        config.set_lsp(servers={"x": {"command": ""}})
    with pytest.raises(ValueError):
        config.set_lsp(servers={"x": {"command": "clangd", "args": "nope"}})
    with pytest.raises(ValueError):
        config.set_lsp(servers={"x": {"command": "clangd", "extensions": "py"}})
    with pytest.raises(ValueError):
        config.set_lsp(servers="nope")  # type: ignore[arg-type]


def test_validate_config_flags_bad_lsp_section():
    cfg = config.load_config()
    cfg["lsp"] = {
        "enabled": "yes",
        "diagnostics_after_edits": "no",
        "timeout": 999,
        "servers": {"x": {"command": ""}, "y": ["not", "an", "object"]},
    }

    issues = config.validate_config(cfg)

    error_keys = {issue["key"] for issue in issues if issue["level"] == "error"}
    assert {
        "lsp.enabled",
        "lsp.diagnostics_after_edits",
        "lsp.timeout",
        "lsp.servers.x",
        "lsp.servers.y",
    } <= error_keys

    cfg["lsp"] = ["nope"]
    assert any(
        issue["key"] == "lsp" and issue["level"] == "error"
        for issue in config.validate_config(cfg)
    )
    assert config.validate_config(config.load_config()) == []


def test_effective_servers_merge_presets_and_overrides():
    config.set_lsp(
        servers={
            "python": {
                "command": "custom-pyright",
                "args": ["--stdio", "--x"],
                "extensions": [],
            },
            "wat": {"command": "wat-ls", "extensions": ["wat"]},
        }
    )

    specs = lsp.effective_servers()

    assert specs["python"].command == "custom-pyright"
    assert specs["python"].args == ("--stdio", "--x")
    assert specs["python"].extensions == (".py", ".pyi")
    assert specs["wat"].command == "wat-ls"
    assert specs["wat"].extensions == (".wat",)
    assert specs["go"].command == "gopls"


def test_manager_routes_by_extension():
    manager = lsp.LspManager()
    assert manager.language_for("a.py") == "python"
    assert manager.language_for("a.PYI") == "python"
    assert manager.language_for("a.go") == "go"
    assert manager.language_for("a.rs") == "rust"
    assert manager.language_for("a.cpp") == "c/cpp"
    assert manager.language_for("a.txt") is None
    assert manager.language_for("Makefile") is None


# ---------------------------------------------------------------------------
# Fake-server end to end
# ---------------------------------------------------------------------------


def test_manager_e2e_with_fake_server(tmp_path):
    _use_fake_server(tmp_path)
    sample = tmp_path / "sample.py"
    sample.write_text("x = bad\n")
    manager = lsp.LspManager(root=tmp_path)
    lsp.set_manager(manager)

    diagnostics = manager.diagnostics_for(sample, timeout=5.0)
    assert diagnostics == [
        {
            "line": 1,
            "character": 1,
            "severity": "error",
            "message": "undefined name 'bad'",
            "source": "fake",
        }
    ]

    # A later edit is picked up: the client must wait for the new publish.
    sample.write_text("x = 1\n")
    assert manager.diagnostics_for(sample, timeout=5.0) == []

    resolved = sample.resolve()
    assert manager.definition(sample, 1, 1) == [
        {"path": str(resolved), "line": 5, "character": 3}
    ]
    assert manager.references(sample, 1, 1) == [
        {"path": str(resolved), "line": 2, "character": 1},
        {"path": str(resolved), "line": 9, "character": 5},
    ]

    client = manager.clients["fake"]
    manager.shutdown()
    assert manager.clients == {}
    assert not client.running


def test_manager_records_missing_command_failure(tmp_path):
    config.set_lsp(
        enabled=True,
        servers={
            "ghost": {
                "command": "kiwimatecoder-missing-lsp-binary",
                "extensions": [".py"],
            }
        },
    )
    sample = tmp_path / "a.py"
    sample.write_text("x = 1\n")
    manager = lsp.LspManager(root=tmp_path)

    with pytest.raises(lsp.LspError, match="not found on PATH"):
        manager.diagnostics_for(sample, timeout=1.0)
    assert manager.failure_for("ghost")
    assert manager.clients == {}

    with pytest.raises(lsp.LspError, match="not found on PATH"):
        manager.definition(sample, 1, 1, timeout=1.0)


def test_client_initialize_timeout_is_bounded(tmp_path):
    server = tmp_path / "silent_lsp.py"
    server.write_text(SILENT_SERVER)
    client = lsp.LspClient(
        sys.executable,
        [str(server)],
        root=tmp_path,
        initialize_timeout=0.2,
        request_timeout=0.2,
    )
    try:
        with pytest.raises(lsp.LspTimeout):
            client.initialize()
    finally:
        client.close()
    assert not client.running


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_lsp_tools_are_read_only():
    for name in ("lsp_diagnostics", "lsp_definition", "lsp_references"):
        tool = tools.get_tool(name)
        assert tool is not None
        assert tool.writes is False
        assert tool.runs is False
        assert tool.needs_approval is False


def test_tools_report_disabled(session):
    config.set_lsp(enabled=False)

    result = tools.dispatch("lsp_diagnostics", {"path": "a.py"}, session)

    assert not result.ok
    assert "LSP is disabled" in result.content
    assert "/lsp on" in result.content


def test_tools_report_no_matching_server(session, tmp_path):
    config.set_lsp(enabled=True)
    (tmp_path / "notes.txt").write_text("hello")

    result = tools.dispatch("lsp_diagnostics", {"path": "notes.txt"}, session)

    assert not result.ok
    assert "No language server" in result.content


def test_lsp_tools_with_fake_server(session, tmp_path):
    _use_fake_server(tmp_path)
    (tmp_path / "sample.py").write_text("x = bad\n")
    lsp.set_manager(lsp.LspManager(root=tmp_path))

    result = tools.dispatch("lsp_diagnostics", {"path": "sample.py"}, session)
    assert result.ok
    assert "sample.py:1:1 error undefined name 'bad' (fake)" in result.content

    result = tools.dispatch(
        "lsp_definition",
        {"path": "sample.py", "line": 1, "character": 1},
        session,
    )
    assert result.ok
    assert "sample.py:5:3" in result.content

    result = tools.dispatch(
        "lsp_references",
        {"path": "sample.py", "line": 1, "character": 1},
        session,
    )
    assert result.ok
    assert result.content.count("sample.py:") == 2


def test_lsp_diagnostics_reports_clean_file(session, tmp_path):
    _use_fake_server(tmp_path)
    (tmp_path / "clean.py").write_text("x = 1\n")
    lsp.set_manager(lsp.LspManager(root=tmp_path))

    result = tools.dispatch("lsp_diagnostics", {"path": "clean.py"}, session)

    assert result.ok
    assert result.content == "No diagnostics."


def test_lsp_diagnostics_uses_recent_touched_files(session, tmp_path):
    _use_fake_server(tmp_path)
    for index in range(6):
        (tmp_path / f"f{index}.py").write_text("x = bad\n")
    session.touched_files = [f"f{index}.py" for index in range(6)]
    lsp.set_manager(lsp.LspManager(root=tmp_path))

    result = tools.dispatch("lsp_diagnostics", {}, session)

    assert result.ok
    assert "f1.py:1:1" in result.content
    assert "f5.py:1:1" in result.content
    assert "f0.py" not in result.content


def test_lsp_position_validation(session):
    config.set_lsp(enabled=True)

    result = tools.dispatch(
        "lsp_definition", {"path": "a.py", "line": 0, "character": 1}, session
    )
    assert not result.ok
    assert "1-based" in result.content

    result = tools.dispatch(
        "lsp_references", {"path": "a.py", "line": "x", "character": 1}, session
    )
    assert not result.ok
    assert "integers" in result.content


# ---------------------------------------------------------------------------
# Slash command and CLI
# ---------------------------------------------------------------------------


def test_lsp_slash_status_and_toggle(session):
    console = _console()

    assert commands.dispatch("/lsp", session, console) == CommandResult.CONTINUE
    output = _output(console)
    assert "LSP: off" in output
    assert "Running clients: none" in output

    on_console = _console()
    assert commands.dispatch("/lsp on", session, on_console) == CommandResult.CONTINUE
    assert config.get_lsp()["enabled"] is True
    assert "LSP: on" in _output(on_console)

    assert commands.dispatch("/lsp off", session, _console()) == CommandResult.CONTINUE
    assert config.get_lsp()["enabled"] is False

    usage = _console()
    commands.dispatch("/lsp nonsense", session, usage)
    assert "Usage: /lsp" in _output(usage)


def test_lsp_slash_restart_stops_clients(session, tmp_path):
    _use_fake_server(tmp_path)
    (tmp_path / "sample.py").write_text("x = bad\n")
    manager = lsp.LspManager(root=tmp_path)
    lsp.set_manager(manager)
    manager.diagnostics_for(tmp_path / "sample.py", timeout=5.0)
    client = manager.clients["fake"]

    console = _console()
    assert (
        commands.dispatch("/lsp restart", session, console) == CommandResult.CONTINUE
    )

    assert manager.clients == {}
    assert not client.running
    assert "restarted" in _output(console).lower()


def test_cli_config_lsp_roundtrip():
    runner = CliRunner()

    result = runner.invoke(main.app, ["config", "lsp", "enable", "on"])
    assert result.exit_code == 0, result.output
    assert config.get_lsp()["enabled"] is True

    result = runner.invoke(main.app, ["config", "lsp", "after-edits", "off"])
    assert result.exit_code == 0, result.output
    assert config.get_lsp()["diagnostics_after_edits"] is False

    result = runner.invoke(main.app, ["config", "lsp", "show"])
    assert result.exit_code == 0, result.output
    assert "LSP: on" in result.output

    result = runner.invoke(main.app, ["config", "lsp", "enable", "maybe"])
    assert result.exit_code == 1
    assert config.get_lsp()["enabled"] is True


# ---------------------------------------------------------------------------
# Post-edit diagnostics in the agent
# ---------------------------------------------------------------------------


def _agent_session(tmp_path: Path) -> Session:
    return Session(
        provider_id="openrouter",
        model="test-model",
        mode=PermissionMode.AUTO,
        workspace_root=tmp_path,
    )


def _write_round_trip(path: str, content: str):
    """A stream factory that writes ``path`` then answers with text."""
    rounds = [
        [
            ToolCallDelta(
                index=0,
                id="call_write",
                name="write_file",
                args_fragment=(
                    '{"path": "%s", "content": "%s"}' % (path, content)
                ),
            ),
            Done(finish_reason="tool_calls"),
        ],
        [TextDelta(text="done"), Done(finish_reason="stop")],
    ]
    state = {"n": 0}

    async def mock_stream(*args, **kwargs):
        stream = rounds[min(state["n"], len(rounds) - 1)]
        state["n"] += 1
        for event in stream:
            yield event

    return mock_stream


async def _run_write_turn(agent: Agent, path: str, content: str) -> list[dict]:
    mock_stream = _write_round_trip(path, content)
    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn(f"write {path}")
    return agent.session.messages


@pytest.mark.anyio
async def test_agent_appends_lsp_diagnostics_after_edit(tmp_path, monkeypatch):
    config.set_lsp(enabled=True, diagnostics_after_edits=True)
    session = _agent_session(tmp_path)
    agent = Agent(session, Console(quiet=True), MagicMock(return_value=True))
    fake = _FakeLspManager(
        diagnostics=[
            {
                "line": 2,
                "character": 3,
                "severity": "error",
                "message": "boom",
                "source": "fake",
            }
        ]
    )
    monkeypatch.setattr(lsp, "get_manager", lambda: fake)

    messages = await _run_write_turn(agent, "out.py", "print(1)")

    tool_message = next(m for m in messages if m.get("role") == "tool")
    assert "LSP:" in tool_message["content"]
    assert "out.py:2:3 error boom (fake)" in tool_message["content"]
    assert fake.calls
    assert fake.calls[0][0] == str(tmp_path / "out.py")
    assert fake.calls[0][1] == 3.0


@pytest.mark.anyio
async def test_agent_silently_skips_failing_lsp(tmp_path, monkeypatch):
    config.set_lsp(enabled=True, diagnostics_after_edits=True)
    session = _agent_session(tmp_path)
    agent = Agent(session, Console(quiet=True), MagicMock(return_value=True))
    fake = _FakeLspManager(error=RuntimeError("server exploded"))
    monkeypatch.setattr(lsp, "get_manager", lambda: fake)

    messages = await _run_write_turn(agent, "out.py", "print(1)")

    tool_message = next(m for m in messages if m.get("role") == "tool")
    assert "LSP:" not in tool_message["content"]
    assert "Created out.py" in tool_message["content"]
    assert (tmp_path / "out.py").exists()
    assert fake.calls


@pytest.mark.anyio
async def test_agent_skips_lsp_when_disabled_or_after_edits_off(tmp_path, monkeypatch):
    session = _agent_session(tmp_path)
    agent = Agent(session, Console(quiet=True), MagicMock(return_value=True))
    fake = _FakeLspManager(
        diagnostics=[
            {
                "line": 1,
                "character": 1,
                "severity": "error",
                "message": "boom",
                "source": "fake",
            }
        ]
    )
    monkeypatch.setattr(lsp, "get_manager", lambda: fake)

    config.set_lsp(enabled=False)
    messages = await _run_write_turn(agent, "one.py", "print(1)")
    tool_message = next(m for m in messages if m.get("role") == "tool")
    assert "LSP:" not in tool_message["content"]
    assert fake.calls == []

    config.set_lsp(enabled=True, diagnostics_after_edits=False)
    messages = await _run_write_turn(agent, "two.py", "print(2)")
    tool_message = next(m for m in messages if m.get("role") == "tool")
    assert "LSP:" not in tool_message["content"]
    assert fake.calls == []
