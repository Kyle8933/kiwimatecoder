from __future__ import annotations

import io
import json
import sys
from collections.abc import Iterator

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

from kiwimatecoder import commands, config, main, mcp, tools
from kiwimatecoder.commands import CommandResult


def _console():
    return Console(file=io.StringIO(), force_terminal=False, width=140)


def _output(console) -> str:
    return console.file.getvalue()


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    config_dir = tmp_path / "cfg"
    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "CONFIG_FILE", config_dir / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", config_dir / "config")
    return config_dir


@pytest.fixture(autouse=True)
def restore_state() -> Iterator[None]:
    """Restore the tool registry and drop any session MCP manager."""
    original_tools = list(tools.TOOLS.all())
    name_sources = {
        name: source for source, names in tools.TOOLS.sources().items() for name in names
    }
    yield
    manager = mcp.get_manager()
    if manager is not None:
        manager.shutdown()
    mcp.set_manager(None)
    for name in list(tools.TOOLS):
        tools.TOOLS.unregister(name, force=True)
    for tool in original_tools:
        tools.TOOLS.register(tool, source=name_sources.get(tool.name, "builtin"))


FAKE_SERVER = r'''
import json
import sys

TOOLS = [
    {
        "name": "read_thing",
        "description": "Read a thing.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "write thing",
        "description": "Write a thing.",
        "inputSchema": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
        },
        "annotations": {},
    },
]


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" not in request:
            continue
        request_id = request["id"]
        method = request.get("method")
        params = request.get("params") or {}
        if method == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}, "resources": {}},
                "serverInfo": {"name": "fake", "version": "1.0"},
            }
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            arguments = params.get("arguments") or {}
            result = {
                "content": [
                    {"type": "text", "text": "called " + str(params.get("name"))},
                    {"type": "text", "text": json.dumps(arguments, sort_keys=True)},
                ],
                "isError": False,
            }
        elif method == "resources/list":
            result = {
                "resources": [
                    {
                        "uri": "file:///notes.txt",
                        "name": "notes",
                        "mimeType": "text/plain",
                    }
                ]
            }
        elif method == "resources/read":
            result = {
                "contents": [
                    {
                        "uri": params.get("uri"),
                        "mimeType": "text/plain",
                        "text": "resource body",
                    }
                ]
            }
        else:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "unknown method"},
                }
            )
            continue
        send({"jsonrpc": "2.0", "id": request_id, "result": result})


main()
'''


def _fake_server(tmp_path) -> str:
    path = tmp_path / "fake_mcp_server.py"
    path.write_text(FAKE_SERVER)
    return str(path)


def _stdio_spec(tmp_path) -> dict:
    return {"command": sys.executable, "args": [_fake_server(tmp_path)]}


def test_stdio_manager_registers_tools_resources_and_shutdown(session, tmp_path):
    config.set_mcp_server("fake", _stdio_spec(tmp_path))
    manager = mcp.McpManager(console=_console())

    result = manager.load()

    assert result.servers == ["fake"]
    assert result.failed == []
    assert result.tools == ["mcp__fake__read_thing", "mcp__fake__write_thing"]

    read_tool = tools.get_tool("mcp__fake__read_thing")
    write_tool = tools.get_tool("mcp__fake__write_thing")
    assert read_tool is not None and write_tool is not None
    assert read_tool.needs_approval is False
    assert write_tool.needs_approval is True
    assert write_tool.runs is False

    read_result = tools.dispatch("mcp__fake__read_thing", {"path": "x"}, session)
    assert read_result.ok
    assert "called read_thing" in read_result.content
    assert '"path": "x"' in read_result.content

    write_result = tools.dispatch("mcp__fake__write_thing", {"value": "v"}, session)
    assert write_result.ok
    assert "called write thing" in write_result.content

    client = manager.client("fake")
    assert isinstance(client, mcp.StdioMcpClient)
    resources = client.list_resources()
    assert resources[0]["uri"] == "file:///notes.txt"
    assert client.read_resource("file:///notes.txt") == "resource body"
    assert manager.resource_count("fake") == 1

    manager.shutdown()

    assert tools.get_tool("mcp__fake__read_thing") is None
    assert tools.get_tool("mcp__fake__write_thing") is None
    assert manager.client("fake") is None
    assert not client.running


def test_failure_isolation_keeps_other_servers(session, tmp_path):
    config.set_mcp_servers(
        {
            "good": _stdio_spec(tmp_path),
            "bad": {"command": str(tmp_path / "definitely-not-a-real-command-xyz")},
        }
    )
    console = _console()
    manager = mcp.McpManager(console=console)

    result = manager.load()

    assert result.servers == ["good"]
    assert [name for name, _ in result.failed] == ["bad"]
    assert tools.get_tool("mcp__good__read_thing") is not None
    assert tools.get_tool("mcp__good__write_thing") is not None
    assert "MCP server 'bad' failed" in _output(console)
    manager.shutdown()


def test_disabled_server_is_skipped(session, tmp_path):
    config.set_mcp_server("off", {**_stdio_spec(tmp_path), "disabled": True})
    manager = mcp.McpManager(console=_console())

    result = manager.load()

    assert result.servers == []
    assert result.failed == []
    assert tools.get_tool("mcp__off__read_thing") is None


def test_tool_error_becomes_failed_tool_result(session, tmp_path):
    script = tmp_path / "error_server.py"
    script.write_text(
        """
import json
import sys

def send(message):
    sys.stdout.write(json.dumps(message) + "\\n")
    sys.stdout.flush()

for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request.get("method")
    if method == "initialize":
        result = {"capabilities": {"tools": {}}, "serverInfo": {}}
    elif method == "tools/list":
        result = {"tools": [{"name": "boom", "inputSchema": {"type": "object"}}]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": "kaboom"}], "isError": True}
    else:
        result = {}
    send({"jsonrpc": "2.0", "id": request["id"], "result": result})
"""
    )
    config.set_mcp_server("error", {"command": sys.executable, "args": [str(script)]})
    manager = mcp.McpManager(console=_console())
    manager.load()

    outcome = tools.dispatch("mcp__error__boom", {}, session)

    assert outcome.ok is False
    assert "kaboom" in outcome.content
    manager.shutdown()


def test_http_client_json_and_sse_parsing():
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append({"payload": payload, "headers": dict(request.headers)})
        method = payload["method"]
        if method == "initialize":
            body = {
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "http-fake", "version": "1"},
                },
            }
            return httpx.Response(200, json=body)
        if method == "tools/call":
            body = {
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {
                    "content": [{"type": "text", "text": "sse says hi"}],
                    "isError": False,
                },
            }
            return httpx.Response(
                200,
                text=f"event: message\ndata: {json.dumps(body)}\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": payload.get("id"), "result": {}}
        )

    transport = httpx.MockTransport(handler)
    client = mcp.HttpMcpClient(
        "https://mcp.example/rpc",
        headers={"Authorization": "Bearer tok"},
        transport=transport,
    )

    info = client.initialize()
    assert info["serverInfo"]["name"] == "http-fake"
    assert client.call_tool("t", {"a": 1}) == "sse says hi"
    client.close()

    assert requests[0]["headers"]["accept"] == "application/json, text/event-stream"
    assert requests[0]["headers"]["authorization"] == "Bearer tok"
    assert requests[0]["payload"]["params"]["clientInfo"]["name"] == "kiwimatecoder"
    assert requests[1]["payload"]["method"] == "notifications/initialized"
    assert "id" not in requests[1]["payload"]


def test_http_client_error_status_raises():
    transport = httpx.MockTransport(lambda request: httpx.Response(500, text="boom"))
    client = mcp.HttpMcpClient("https://mcp.example/rpc", transport=transport)

    with pytest.raises(mcp.McpError):
        client.initialize()


def test_protocol_flatten_content_summarizes_non_text():
    flattened = mcp.flatten_content(
        [
            {"type": "text", "text": "hello"},
            {"type": "image", "data": "..."},
            {"type": "audio", "data": "..."},
            {"type": "resource", "resource": {"uri": "x", "blob": "..."}},
        ]
    )

    assert flattened.splitlines() == [
        "hello",
        "[image content]",
        "[audio content]",
        "[resource content]",
    ]


def test_mcp_command_lists_servers_and_tools(session, tmp_path):
    config.set_mcp_server("fake", _stdio_spec(tmp_path))
    config.set_mcp_server("off", {"command": "srv", "disabled": True})
    manager = mcp.McpManager(console=_console())
    mcp.set_manager(manager)
    manager.load()

    console = _console()
    assert commands.dispatch("/mcp list", session, console) == CommandResult.CONTINUE

    output = _output(console)
    assert "MCP servers" in output
    assert "fake" in output and "off" in output
    assert "disabled" in output
    assert "mcp__fake__read_thing" in output
    assert "mcp__fake__write_thing" in output
    assert "read-only" in output
    assert "needs approval" in output
    manager.shutdown()


def test_mcp_command_without_manager_shows_config(session, tmp_path):
    config.set_mcp_server("fake", _stdio_spec(tmp_path))
    console = _console()

    assert commands.dispatch("/mcp", session, console) == CommandResult.CONTINUE

    output = _output(console)
    assert "fake" in output
    assert "not connected" in output


def test_mcp_command_reload_reconnects(session, tmp_path):
    config.set_mcp_server("fake", _stdio_spec(tmp_path))
    manager = mcp.McpManager(console=_console())
    mcp.set_manager(manager)
    manager.load()
    console = _console()

    assert commands.dispatch("/mcp reload", session, console) == CommandResult.CONTINUE

    assert "MCP reloaded" in _output(console)
    assert manager.is_connected("fake")
    assert tools.get_tool("mcp__fake__read_thing") is not None


def test_mcp_command_reload_without_manager(session, tmp_path):
    config.set_mcp_server("fake", _stdio_spec(tmp_path))
    console = _console()

    assert commands.dispatch("/mcp reload", session, console) == CommandResult.CONTINUE

    manager = mcp.get_manager()
    assert manager is not None
    assert manager.is_connected("fake")


def test_config_mcp_cli_roundtrip(isolate_config):
    runner = CliRunner()

    added = runner.invoke(
        main.app, ["config", "mcp", "add", "local", "--command", "python"]
    )
    assert added.exit_code == 0
    assert config.get_mcp_servers()["local"]["command"] == "python"

    with_args = runner.invoke(
        main.app,
        ["config", "mcp", "add", "argsy", "--command", "python", "--args=-m fake"],
    )
    assert with_args.exit_code == 0
    assert config.get_mcp_servers()["argsy"]["args"] == ["-m", "fake"]

    remote = runner.invoke(
        main.app,
        [
            "config",
            "mcp",
            "add",
            "remote",
            "--url",
            "https://host/mcp",
            "-H",
            "Authorization: Bearer tok",
        ],
    )
    assert remote.exit_code == 0
    assert config.get_mcp_servers()["remote"]["headers"] == {
        "Authorization": "Bearer tok"
    }

    listing = runner.invoke(main.app, ["config", "mcp", "list"])
    assert listing.exit_code == 0
    assert "local" in listing.output
    assert "remote" in listing.output

    removed = runner.invoke(main.app, ["config", "mcp", "remove", "local"])
    assert removed.exit_code == 0
    assert "local" not in config.get_mcp_servers()


def test_config_mcp_cli_rejects_invalid_spec(isolate_config):
    runner = CliRunner()

    result = runner.invoke(
        main.app,
        ["config", "mcp", "add", "both", "--command", "x", "--url", "https://y"],
    )

    assert result.exit_code == 1
    assert config.get_mcp_servers() == {}

    invalid_header = runner.invoke(
        main.app,
        [
            "config",
            "mcp",
            "add",
            "remote",
            "--url",
            "https://y",
            "-H",
            "not-a-header",
        ],
    )
    assert invalid_header.exit_code == 1
