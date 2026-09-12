"""In-process tests for the ACP server (fake queues, no real stdio/network)."""

from __future__ import annotations

import asyncio
import io
import threading
from typing import Any, Callable

import pytest
from rich.console import Console
from typer.testing import CliRunner

from kiwimatecoder import config, main
from kiwimatecoder.acp.protocol import decode_lines, encode
from kiwimatecoder.acp.server import AcpServer, prompt_text
from kiwimatecoder.client import Done, TextDelta, ToolCallDelta
from kiwimatecoder.commands import CommandResult, dispatch

STREAM_CHAT = "kiwimatecoder.client.UnifiedClient.stream_chat"


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    """Point config storage at a temp dir and clear provider env vars."""
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for name in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def _scripted_stream(rounds: list[list[Any]], calls: list[Any] | None = None):
    """Async stream generator yielding one scripted round per call."""
    state = {"n": 0}

    async def stream(*args: Any, **kwargs: Any):
        if calls is not None:
            calls.append(args)
        index = state["n"]
        state["n"] += 1
        for event in rounds[min(index, len(rounds) - 1)]:
            yield event

    return stream


class Harness:
    """Drive an AcpServer through in-memory asyncio queues."""

    def __init__(self, server: AcpServer) -> None:
        self.server = server
        self.inbox: asyncio.Queue[str] = asyncio.Queue()
        self.outbox: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task[int] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(
            self.server.serve(self.inbox.get, self.outbox.put)
        )
        await asyncio.sleep(0)

    async def send(self, message: dict[str, Any]) -> None:
        await self.inbox.put(encode(message))

    async def send_raw(self, line: str) -> None:
        await self.inbox.put(line)

    async def receive(self, timeout: float = 5.0) -> dict[str, Any]:
        line = await asyncio.wait_for(self.outbox.get(), timeout)
        messages, _remainder = decode_lines(line)
        assert messages, f"not a JSON-RPC line: {line!r}"
        return messages[0]

    async def receive_until(
        self, predicate: Callable[[dict[str, Any]], bool], timeout: float = 5.0
    ) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise AssertionError(f"no matching message; got {collected!r}")
            line = await asyncio.wait_for(self.outbox.get(), remaining)
            messages, _remainder = decode_lines(line)
            for message in messages:
                collected.append(message)
                if predicate(message):
                    return collected

    async def request(
        self, request_id: int, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        await self.send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            }
        )
        collected = await self.receive_until(
            lambda m: m.get("id") == request_id and ("result" in m or "error" in m)
        )
        return collected[-1]

    async def initialize(self, request_id: int = 1, capabilities: Any = None):
        params: dict[str, Any] = {"protocolVersion": 1}
        if capabilities is not None:
            params["clientCapabilities"] = capabilities
        response = await self.request(request_id, "initialize", params)
        return response["result"]

    async def new_session(self, request_id: int = 2, cwd: Any = None) -> str:
        params = {"cwd": str(cwd)} if cwd is not None else {}
        response = await self.request(request_id, "session/new", params)
        assert "result" in response, response
        return response["result"]["sessionId"]

    async def run_prompt(
        self,
        session_id: str,
        request_id: int,
        text: str,
        responder: Callable[[dict[str, Any]], str] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Send a prompt, answering permission requests via ``responder``."""
        await self.send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": text}],
                },
            }
        )
        updates: list[dict[str, Any]] = []
        while True:
            message = await self.receive()
            if message.get("method") == "session/request_permission":
                if responder is not None:
                    option_id = responder(message["params"])
                    await self.send(
                        {
                            "jsonrpc": "2.0",
                            "id": message["id"],
                            "result": {
                                "outcome": {
                                    "outcome": "selected",
                                    "optionId": option_id,
                                }
                            },
                        }
                    )
                continue
            if message.get("method") == "session/update":
                updates.append(message["params"]["update"])
                continue
            if message.get("id") == request_id:
                return message, updates

    async def stop(self) -> None:
        if self._task is None:
            return
        await self.inbox.put("")
        await asyncio.wait_for(self._task, 10)
        self._task = None


@pytest.fixture
async def make_harness(tmp_path):
    created: list[Harness] = []

    async def factory(**kwargs: Any) -> Harness:
        kwargs.setdefault("permission_timeout", 1.0)
        server = AcpServer(**kwargs)
        harness = Harness(server)
        await harness.start()
        created.append(harness)
        return harness

    yield factory
    for harness in created:
        await harness.stop()


# ---------------------------------------------------------------------------
# Handshake and sessions
# ---------------------------------------------------------------------------


async def test_initialize_handshake_reports_capabilities(make_harness):
    harness = await make_harness()

    result = await harness.initialize(
        capabilities={"fs": {"readTextFile": True, "writeTextFile": True}}
    )

    assert result["protocolVersion"] == 1
    assert result["agentCapabilities"] == {
        "loadSession": False,
        "promptCapabilities": {"image": True},
    }
    assert result["agentInfo"]["name"] == "kiwimatecoder"
    assert result["agentInfo"]["version"]
    assert harness.server._fs_read is True
    assert harness.server._fs_write is True


async def test_session_new_rejects_missing_workspace(make_harness, tmp_path):
    harness = await make_harness()

    response = await harness.request(
        3, "session/new", {"cwd": str(tmp_path / "does-not-exist")}
    )

    assert response["error"]["code"] == -32602
    assert "not a directory" in response["error"]["message"]


async def test_unknown_method_returns_method_not_found(make_harness):
    harness = await make_harness()

    response = await harness.request(4, "does/not/exist")

    assert response["error"]["code"] == -32601


async def test_malformed_line_does_not_crash_server(make_harness):
    harness = await make_harness()

    await harness.send_raw("this is not json\n")

    parse_error = await harness.receive()
    assert parse_error["error"]["code"] == -32700
    assert parse_error["id"] is None

    result = await harness.initialize(request_id=2)
    assert result["agentInfo"]["name"] == "kiwimatecoder"


async def test_prompt_without_text_is_invalid(make_harness, tmp_path):
    harness = await make_harness()
    session_id = await harness.new_session(cwd=tmp_path)

    response = await harness.request(
        5, "session/prompt", {"sessionId": session_id, "prompt": []}
    )

    assert response["error"]["code"] == -32602


def test_prompt_text_concatenates_and_counts_images():
    text, images = prompt_text(
        [
            {"type": "text", "text": "first"},
            {"type": "image", "mimeType": "image/png", "data": "..."},
            {"type": "text", "text": "second"},
        ]
    )

    assert text == "first\nsecond"
    assert images == 1
    assert prompt_text("nonsense") == ("", 0)


# ---------------------------------------------------------------------------
# Prompt -> session/update translation
# ---------------------------------------------------------------------------


async def test_prompt_runs_read_tool_and_streams_ordered_updates(
    make_harness, tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "get_key", lambda provider_id: "test-key")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "hello.txt").write_text("file content")
    calls: list[Any] = []
    rounds = [
        [
            TextDelta(text="Checking "),
            ToolCallDelta(
                index=0,
                id="tc1",
                name="read_file",
                args_fragment='{"path": "hello.txt"}',
            ),
            Done(finish_reason="tool_calls"),
        ],
        [TextDelta(text="It says: file content"), Done(finish_reason="stop")],
    ]
    monkeypatch.setattr(STREAM_CHAT, _scripted_stream(rounds, calls))
    harness = await make_harness()
    await harness.initialize()
    session_id = await harness.new_session(cwd=workspace)

    response, updates = await harness.run_prompt(session_id, 10, "read hello.txt")

    assert response["result"] == {"stopReason": "end_turn"}
    assert [update["sessionUpdate"] for update in updates] == [
        "agent_message_chunk",
        "tool_call",
        "tool_call_update",
        "agent_message_chunk",
    ]
    assert updates[0]["content"]["text"] == "Checking "
    tool_call = updates[1]
    assert tool_call["kind"] == "read"
    assert tool_call["status"] == "in_progress"
    assert "hello.txt" in tool_call["title"]
    tool_update = updates[2]
    assert tool_update["status"] == "completed"
    assert tool_update["toolCallId"] == tool_call["toolCallId"]
    assert updates[3]["content"]["text"] == "It says: file content"
    # The read tool really ran: its result reached the second model call.
    second_request = calls[1][1]
    assert any(
        message.get("role") == "tool"
        and "file content" in str(message.get("content"))
        for message in second_request
    )


# ---------------------------------------------------------------------------
# Permission bridge
# ---------------------------------------------------------------------------


async def test_permission_allow_once_writes_file(
    make_harness, tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "get_key", lambda provider_id: "test-key")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    rounds = [
        [
            ToolCallDelta(
                index=0,
                id="w1",
                name="write_file",
                args_fragment='{"path": "out.txt", "content": "hello"}',
            ),
            Done(finish_reason="tool_calls"),
        ],
        [TextDelta(text="done"), Done(finish_reason="stop")],
    ]
    monkeypatch.setattr(STREAM_CHAT, _scripted_stream(rounds))
    harness = await make_harness()
    await harness.initialize()
    session_id = await harness.new_session(cwd=workspace)
    requests: list[dict[str, Any]] = []

    def responder(params: dict[str, Any]) -> str:
        requests.append(params)
        return "allow_once"

    response, _updates = await harness.run_prompt(
        session_id, 10, "write out.txt", responder=responder
    )

    assert response["result"] == {"stopReason": "end_turn"}
    assert (workspace / "out.txt").read_text() == "hello"
    assert requests
    assert requests[0]["sessionId"] == session_id
    assert requests[0]["toolCall"]["title"].startswith("write_file")
    assert {option["optionId"] for option in requests[0]["options"]} == {
        "allow_once",
        "allow_always",
        "reject_once",
    }


async def test_permission_reject_denies_and_reports(
    make_harness, tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "get_key", lambda provider_id: "test-key")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    calls: list[Any] = []
    rounds = [
        [
            ToolCallDelta(
                index=0,
                id="w1",
                name="write_file",
                args_fragment='{"path": "out.txt", "content": "hello"}',
            ),
            Done(finish_reason="tool_calls"),
        ],
        [TextDelta(text="done"), Done(finish_reason="stop")],
    ]
    monkeypatch.setattr(STREAM_CHAT, _scripted_stream(rounds, calls))
    harness = await make_harness()
    await harness.initialize()
    session_id = await harness.new_session(cwd=workspace)

    response, _updates = await harness.run_prompt(
        session_id, 10, "write out.txt", responder=lambda params: "reject_once"
    )

    assert response["result"] == {"stopReason": "end_turn"}
    assert not (workspace / "out.txt").exists()
    second_request = calls[1][1]
    tool_messages = [
        message for message in second_request if message.get("role") == "tool"
    ]
    assert tool_messages
    assert "Denied by user." in tool_messages[-1]["content"]


async def test_permission_allow_always_persists_tool(
    make_harness, tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "get_key", lambda provider_id: "test-key")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    rounds = [
        [
            ToolCallDelta(
                index=0,
                id="w1",
                name="write_file",
                args_fragment='{"path": "out.txt", "content": "hello"}',
            ),
            Done(finish_reason="tool_calls"),
        ],
        [TextDelta(text="done"), Done(finish_reason="stop")],
    ]
    monkeypatch.setattr(STREAM_CHAT, _scripted_stream(rounds))
    harness = await make_harness()
    await harness.initialize()
    session_id = await harness.new_session(cwd=workspace)

    await harness.run_prompt(
        session_id, 10, "write out.txt", responder=lambda params: "allow_always"
    )

    assert (workspace / "out.txt").read_text() == "hello"
    assert "write_file" in config.get_always_allowed_tools()


async def test_permission_timeout_denies(make_harness, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "get_key", lambda provider_id: "test-key")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    rounds = [
        [
            ToolCallDelta(
                index=0,
                id="w1",
                name="write_file",
                args_fragment='{"path": "out.txt", "content": "hello"}',
            ),
            Done(finish_reason="tool_calls"),
        ],
        [TextDelta(text="done"), Done(finish_reason="stop")],
    ]
    monkeypatch.setattr(STREAM_CHAT, _scripted_stream(rounds))
    harness = await make_harness(permission_timeout=0.25)
    await harness.initialize()
    session_id = await harness.new_session(cwd=workspace)

    # No responder: the editor never answers, so the request times out.
    response, _updates = await harness.run_prompt(session_id, 10, "write out.txt")

    assert response["result"] == {"stopReason": "end_turn"}
    assert not (workspace / "out.txt").exists()


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


async def test_cancel_returns_cancelled_and_preserves_partial_text(
    make_harness, tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "get_key", lambda provider_id: "test-key")
    release = threading.Event()

    async def stream(*args: Any, **kwargs: Any):
        yield TextDelta(text="partial answer")
        await asyncio.to_thread(release.wait, 5)
        yield Done(finish_reason="stop")

    monkeypatch.setattr(STREAM_CHAT, stream)
    harness = await make_harness()
    await harness.initialize()
    session_id = await harness.new_session(cwd=tmp_path)
    await harness.send(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": "keep going"}],
            },
        }
    )

    first = await harness.receive()
    assert first["method"] == "session/update"
    assert first["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
    assert first["params"]["update"]["content"]["text"] == "partial answer"

    await harness.send(
        {
            "jsonrpc": "2.0",
            "method": "session/cancel",
            "params": {"sessionId": session_id},
        }
    )
    response = await harness.receive_until(lambda m: m.get("id") == 10)
    # release the blocked worker thread so the test leaves no stragglers
    release.set()

    assert response[-1]["result"] == {"stopReason": "cancelled"}


# ---------------------------------------------------------------------------
# Optional filesystem delegation
# ---------------------------------------------------------------------------


async def test_fs_delegation_uses_client_when_advertised(make_harness, tmp_path):
    harness = await make_harness()
    await harness.initialize(
        capabilities={"fs": {"readTextFile": True, "writeTextFile": True}}
    )
    session_id = await harness.new_session(cwd=tmp_path)

    reader = asyncio.create_task(
        harness.server.read_text_file(session_id, "/remote/file.txt")
    )
    request = await harness.receive()
    assert request["method"] == "fs/read_text_file"
    assert request["params"]["path"] == "/remote/file.txt"
    await harness.send(
        {"jsonrpc": "2.0", "id": request["id"], "result": {"content": "from client"}}
    )
    assert await reader == "from client"

    writer = asyncio.create_task(
        harness.server.write_text_file(session_id, "/remote/out.txt", "data")
    )
    request = await harness.receive()
    assert request["method"] == "fs/write_text_file"
    assert request["params"]["content"] == "data"
    await harness.send({"jsonrpc": "2.0", "id": request["id"], "result": {}})
    await writer


async def test_fs_falls_back_to_local_without_capabilities(make_harness, tmp_path):
    target = tmp_path / "local.txt"
    target.write_text("local content")
    harness = await make_harness()
    await harness.initialize()
    session_id = await harness.new_session(cwd=tmp_path)

    assert await harness.server.read_text_file(session_id, str(target)) == "local content"


# ---------------------------------------------------------------------------
# Config and CLI surface
# ---------------------------------------------------------------------------


def test_acp_config_crud_and_defaults():
    assert config.get_acp()["permission_timeout"] == 300

    assert config.set_acp(permission_timeout=42)["permission_timeout"] == 42
    assert config.load_config()["acp"]["permission_timeout"] == 42

    with pytest.raises(ValueError):
        config.set_acp(permission_timeout=0)
    with pytest.raises(ValueError):
        config.set_acp(permission_timeout="soon")
    with pytest.raises(ValueError):
        config.set_acp(permission_timeout=99999)


def test_validate_config_flags_bad_acp_section():
    cfg = config.load_config()
    cfg["acp"] = {"permission_timeout": "nope"}

    issues = config.validate_config(cfg)

    assert any(issue["key"] == "acp.permission_timeout" for issue in issues)


def test_acp_cli_command_and_config_subcommands():
    help_result = CliRunner().invoke(main.app, ["--help"])
    assert help_result.exit_code == 0
    assert "acp" in help_result.stdout

    sub_help = CliRunner().invoke(main.app, ["config", "acp", "--help"])
    assert sub_help.exit_code == 0
    assert "timeout" in sub_help.stdout

    show = CliRunner().invoke(main.app, ["config", "acp", "show"])
    assert show.exit_code == 0
    assert "300s" in show.stdout

    set_result = CliRunner().invoke(main.app, ["config", "acp", "timeout", "45"])
    assert set_result.exit_code == 0
    assert config.get_acp()["permission_timeout"] == 45

    bad = CliRunner().invoke(main.app, ["config", "acp", "timeout", "0"])
    assert bad.exit_code == 1


def test_slash_config_acp_show_and_timeout(session):
    def output(console: Console) -> str:
        return console.file.getvalue()  # type: ignore[union-attr]

    console = Console(file=io.StringIO(), force_terminal=False, width=120)
    assert dispatch("/config acp show", session, console) == CommandResult.CONTINUE
    assert "300" in output(console)

    console = Console(file=io.StringIO(), force_terminal=False, width=120)
    assert dispatch("/config acp timeout 12", session, console) == CommandResult.CONTINUE
    assert config.get_acp()["permission_timeout"] == 12
    assert "12" in output(console)
