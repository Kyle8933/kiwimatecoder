from __future__ import annotations

import asyncio
import io
import json
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from kiwimatecoder.agent import Agent
from kiwimatecoder.client import AssembledToolCall, Done, ProviderError, TextDelta, ToolCallDelta
from kiwimatecoder.permissions import ApprovalResult, PermissionMode
from kiwimatecoder.session import Session
from tests.conftest import track_console


@pytest.fixture(autouse=True)
def isolate_config_home(tmp_path, monkeypatch):
    from kiwimatecoder import config

    home = tmp_path / "config-home"
    monkeypatch.setattr(config, "CONFIG_DIR", home)
    monkeypatch.setattr(config, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", home / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)


@pytest.fixture
def agent_session(tmp_path):
    return Session(
        provider_id="openrouter",
        model="test-model",
        mode=PermissionMode.AUTO,
        workspace_root=tmp_path,
    )


@pytest.mark.anyio
async def test_agent_run_turn_pure_text(agent_session):
    console = Console(quiet=True)
    confirm = MagicMock(return_value=True)
    agent = Agent(agent_session, console, confirm)

    mock_events = [
        TextDelta(text="Hello "),
        TextDelta(text="world!"),
        Done(finish_reason="stop"),
    ]

    async def mock_stream(*args, **kwargs):
        for event in mock_events:
            yield event

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn("Hi")

    assert len(agent_session.messages) == 2
    assert agent_session.messages[0] == {"role": "user", "content": "Hi"}
    assert agent_session.messages[1] == {
        "role": "assistant",
        "content": "Hello world!",
    }


@pytest.mark.anyio
async def test_agent_run_turn_with_tool_call(agent_session):
    test_file = agent_session.workspace_root / "hello.txt"
    test_file.write_text("file content")

    console = Console(quiet=True)
    confirm = MagicMock(return_value=True)
    agent = Agent(agent_session, console, confirm)

    # First turn calls read_file, second turn responds with text
    round_1 = [
        ToolCallDelta(
            index=0,
            id="call_read",
            name="read_file",
            args_fragment='{"path": "hello.txt"}',
        ),
        Done(finish_reason="tool_calls"),
    ]
    round_2 = [
        TextDelta(text="The file has: file content"),
        Done(finish_reason="stop"),
    ]

    calls_count = 0

    async def mock_stream(*args, **kwargs):
        nonlocal calls_count
        calls_count += 1
        stream = round_1 if calls_count == 1 else round_2
        for event in stream:
            yield event

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn("Read hello.txt")

    assert len(agent_session.messages) == 4
    assert agent_session.messages[0]["role"] == "user"
    assert agent_session.messages[1]["role"] == "assistant"
    assert "tool_calls" in agent_session.messages[1]
    assert agent_session.messages[2]["role"] == "tool"
    assert "file content" in agent_session.messages[2]["content"]
    assert agent_session.messages[3]["role"] == "assistant"
    assert "The file has" in agent_session.messages[3]["content"]


@pytest.mark.anyio
async def test_agent_run_turn_denied_permission(agent_session):
    agent_session.mode = PermissionMode.ASK
    console = Console(quiet=True)
    confirm = MagicMock(return_value=False)
    agent = Agent(agent_session, console, confirm)

    round_1 = [
        ToolCallDelta(
            index=0,
            id="call_write",
            name="write_file",
            args_fragment='{"path": "out.txt", "content": "hello"}',
        ),
        Done(finish_reason="tool_calls"),
    ]
    round_2 = [
        TextDelta(text="Write was denied."),
        Done(finish_reason="stop"),
    ]

    calls = 0

    async def mock_stream(*args, **kwargs):
        nonlocal calls
        calls += 1
        stream = round_1 if calls == 1 else round_2
        for event in stream:
            yield event

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn("Write out.txt")

    assert not (agent_session.workspace_root / "out.txt").exists()
    tool_res = next(
        m for m in agent_session.messages if m.get("role") == "tool"
    )
    assert "denied by user" in tool_res["content"].lower()


@pytest.mark.anyio
async def test_agent_provider_error_handled(agent_session):
    console = Console(quiet=True)
    confirm = MagicMock(return_value=True)
    agent = Agent(agent_session, console, confirm)

    async def mock_stream(*args, **kwargs):
        raise ProviderError("API rate limit")
        yield Done()

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn("Hello")

    # Should not crash, user message appended
    assert len(agent_session.messages) == 1


@pytest.mark.anyio
async def test_agent_continues_past_former_step_limit(agent_session):
    console = Console(quiet=True)
    confirm = MagicMock(return_value=True)
    agent = Agent(agent_session, console, confirm)
    rounds = {"n": 0}

    async def mock_stream(*args, **kwargs):
        rounds["n"] += 1
        if rounds["n"] <= 30:
            yield ToolCallDelta(
                index=0,
                id=f"call_{rounds['n']}",
                name="search",
                args_fragment='{"pattern": "test"}',
            )
            yield Done(finish_reason="tool_calls")
            return
        yield TextDelta(text="done")
        yield Done(finish_reason="stop")

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn("Keep going")

    assert rounds["n"] == 31
    assert agent_session.messages[-1]["content"] == "done"
    assert not any(
        "step limit" in str(message.get("content") or "").lower()
        for message in agent_session.messages
    )


def test_agent_client_allows_keyless_local_provider(agent_session):
    agent_session.provider_id = "ollama"
    agent = Agent(agent_session, Console(quiet=True), MagicMock())

    with patch("kiwimatecoder.config.get_key", return_value=None):
        client = agent._client()

    assert client.provider.id == "ollama"
    assert client.api_key == ""


def test_agent_client_prefers_a_key_when_one_is_set_for_local(agent_session):
    agent_session.provider_id = "ollama"
    agent = Agent(agent_session, Console(quiet=True), MagicMock())

    with patch("kiwimatecoder.config.get_key", return_value="optional-key"):
        client = agent._client()

    assert client.api_key == "optional-key"


def test_agent_client_requires_key_for_cloud_provider(agent_session):
    agent = Agent(agent_session, Console(quiet=True), MagicMock())

    with (
        patch("kiwimatecoder.config.get_key", return_value=None),
        pytest.raises(ProviderError, match="No API key"),
    ):
        agent._client()


def test_agent_client_requires_key_for_key_requiring_local(agent_session):
    """Unsloth is local but enforces auth: no key → the friendly error, not a 401."""
    agent_session.provider_id = "unsloth"
    agent = Agent(agent_session, Console(quiet=True), MagicMock())

    with (
        patch("kiwimatecoder.config.get_key", return_value=None),
        pytest.raises(ProviderError, match="No API key"),
    ):
        agent._client()


def test_agent_client_passes_prompt_cache_setting(agent_session):
    from kiwimatecoder import config

    agent = Agent(agent_session, Console(quiet=True), MagicMock())

    with patch("kiwimatecoder.config.get_key", return_value="dummy_key"):
        assert agent._client().prompt_cache is False

        config.set_prompt_cache(True)
        client = agent._client()

    assert client.prompt_cache is True


# ---------------------------------------------------------------------------
# Active-provider failover
# ---------------------------------------------------------------------------


class _FakeStream:
    """A UnifiedClient stand-in recording the provider id and model it got."""

    def __init__(self, provider_id: str, fail: bool = False):
        self.provider_id = provider_id
        self.fail = fail

    async def stream_chat(self, messages, tools, model):
        if self.fail:
            raise ProviderError(f"{self.provider_id} down")
        yield TextDelta(text=f"hi from {self.provider_id}")
        yield Done(finish_reason="stop")


@pytest.mark.anyio
async def test_agent_stream_once_falls_back_to_next_provider(agent_session):
    agent_session.set_active_providers(["openrouter", "openai"])
    agent_session.models["openai"] = "gpt-5.6-sol"
    agent = Agent(agent_session, Console(quiet=True), MagicMock(return_value=True))

    attempts: list[tuple[str, str]] = []

    def fake_client(provider_id: str | None = None):
        attempts.append((provider_id, ""))
        return _FakeStream(provider_id, fail=(provider_id == "openrouter"))

    with patch("kiwimatecoder.agent.Agent._client", side_effect=fake_client):
        msg, calls = await agent._stream_once()

    assert msg["content"] == "hi from openai"
    assert [pid for pid, _ in attempts] == ["openrouter", "openai"]


@pytest.mark.anyio
async def test_agent_stream_once_announces_primary_failure(agent_session):
    agent_session.set_active_providers(["openrouter", "openai"])
    buf = io.StringIO()
    agent = Agent(
        agent_session, Console(file=buf, force_terminal=False, width=120), MagicMock()
    )

    def fake_client(provider_id: str | None = None):
        return _FakeStream(provider_id, fail=(provider_id == "openrouter"))

    with patch("kiwimatecoder.agent.Agent._client", side_effect=fake_client):
        msg, _calls = await agent._stream_once()

    output = buf.getvalue().lower()
    assert msg["content"] == "hi from openai"
    assert "openrouter" in output
    assert "trying" in output
    assert "openai" in output


@pytest.mark.anyio
async def test_agent_stream_once_raises_when_all_providers_fail(agent_session):
    agent_session.set_active_providers(["openrouter", "openai"])
    agent = Agent(agent_session, Console(quiet=True), MagicMock(return_value=True))

    def fake_client(provider_id: str | None = None):
        return _FakeStream(provider_id, fail=True)

    with (
        patch("kiwimatecoder.agent.Agent._client", side_effect=fake_client),
        pytest.raises(ProviderError, match="openrouter down") as caught,
    ):
        await agent._stream_once()

    assert "openai down" in str(caught.value)


@pytest.mark.anyio
async def test_agent_stream_once_skips_provider_without_key(agent_session):
    agent_session.set_active_providers(["openrouter", "openai"])
    agent = Agent(agent_session, Console(quiet=True), MagicMock(return_value=True))

    attempts: list[str] = []

    def fake_client(provider_id: str | None = None):
        attempts.append(provider_id)
        if provider_id == "openrouter":
            raise ProviderError("No API key for OpenAI")
        return _FakeStream(provider_id)

    with patch("kiwimatecoder.agent.Agent._client", side_effect=fake_client):
        msg, calls = await agent._stream_once()

    assert msg["content"] == "hi from openai"
    assert attempts == ["openrouter", "openai"]


def test_session_model_for_uses_override_for_fallback(agent_session):
    agent_session.set_active_providers(["openrouter", "openai"])
    agent_session.models["openai"] = "custom-openai-model"

    assert agent_session.model_for("openrouter") == agent_session.model
    assert agent_session.model_for("openai") == "custom-openai-model"


# ---------------------------------------------------------------------------
# Per-turn model routing
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_agent_routes_short_prompt_to_simple_model(agent_session):
    from kiwimatecoder import config

    config.set_model_routing(enabled=True, simple_model="cheap-model")
    console = Console(quiet=True)
    agent = Agent(agent_session, console, MagicMock(return_value=True))
    seen: list[str] = []

    async def mock_stream(messages, tools, model):
        seen.append(model)
        yield TextDelta(text="ok")
        yield Done(finish_reason="stop")

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn("add a docstring")

    assert seen == ["cheap-model"]
    # The override must not mutate the session's model.
    assert agent_session.model == "test-model"


@pytest.mark.anyio
async def test_agent_keeps_model_for_complex_prompt(agent_session):
    from kiwimatecoder import config

    config.set_model_routing(enabled=True, simple_model="cheap-model")
    agent = Agent(agent_session, Console(quiet=True), MagicMock(return_value=True))
    seen: list[str] = []

    async def mock_stream(messages, tools, model):
        seen.append(model)
        yield TextDelta(text="ok")
        yield Done(finish_reason="stop")

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn("please refactor the parser")

    assert seen == ["test-model"]


@pytest.mark.anyio
async def test_agent_routing_override_applies_to_primary_only(agent_session):
    from kiwimatecoder import config

    agent_session.set_active_providers(["openrouter", "openai"])
    agent_session.models["openai"] = "openai-model"
    config.set_model_routing(enabled=True, simple_model="cheap-model")
    agent = Agent(agent_session, Console(quiet=True), MagicMock(return_value=True))
    seen: list[tuple[str, str]] = []

    class RecordingStream:
        def __init__(self, provider_id: str, fail: bool):
            self.provider_id = provider_id
            self.fail = fail

        async def stream_chat(self, messages, tools, model):
            seen.append((self.provider_id, model))
            if self.fail:
                raise ProviderError(f"{self.provider_id} down")
            yield TextDelta(text=f"hi from {self.provider_id}")
            yield Done(finish_reason="stop")

    def fake_client(provider_id: str | None = None):
        return RecordingStream(provider_id or "", fail=provider_id == "openrouter")

    with patch("kiwimatecoder.agent.Agent._client", side_effect=fake_client):
        msg, _calls = await agent._stream_once(model_override="cheap-model")

    assert msg["content"] == "hi from openai"
    assert seen == [
        ("openrouter", "cheap-model"),
        ("openai", "openai-model"),
    ]


@pytest.mark.anyio
async def test_agent_prints_routed_line(agent_session):
    from kiwimatecoder import config

    config.set_model_routing(enabled=True, simple_model="cheap-model")
    buf = io.StringIO()
    agent = Agent(
        agent_session, Console(file=buf, force_terminal=False, width=120), MagicMock()
    )

    async def mock_stream(messages, tools, model):
        yield TextDelta(text="ok")
        yield Done(finish_reason="stop")

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn("hi")

    assert "routed to cheap-model" in buf.getvalue()


# ---------------------------------------------------------------------------
# Thinking / working status
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_agent_shows_thinking_until_first_text(agent_session):
    """The thinking spinner must stop before the first streamed token prints."""
    console = Console(file=io.StringIO(), force_terminal=False, width=120)
    log = track_console(console)
    agent = Agent(agent_session, console, MagicMock(return_value=True))

    async def mock_stream(*args, **kwargs):
        yield TextDelta(text="Hello")
        yield Done(finish_reason="stop")

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn("Hi")

    start_idx = next(
        (
            i
            for i, event in enumerate(log)
            if event[0] == "start" and "Thinking" in event[1]
        ),
        None,
    )
    stop_idx = next(
        (
            i
            for i, event in enumerate(log)
            if event[0] == "stop" and "Thinking" in event[1]
        ),
        None,
    )
    hello_idx = next(
        (
            i
            for i, event in enumerate(log)
            if event[0] == "print" and "Hello" in event[1]
        ),
        None,
    )
    assert start_idx is not None, f"expected Thinking status, got {log}"
    assert stop_idx is not None, f"expected Thinking status to stop, got {log}"
    assert hello_idx is not None, f"expected streamed text, got {log}"
    assert start_idx < stop_idx < hello_idx


@pytest.mark.anyio
async def test_agent_shows_working_status_while_tool_runs(agent_session):
    """A tool's working spinner must stop before the ✓/✗ result line prints."""
    test_file = agent_session.workspace_root / "hello.txt"
    test_file.write_text("file content")

    console = Console(file=io.StringIO(), force_terminal=False, width=120)
    log = track_console(console)
    agent = Agent(agent_session, console, MagicMock(return_value=True))

    round_1 = [
        ToolCallDelta(
            index=0,
            id="call_read",
            name="read_file",
            args_fragment='{"path": "hello.txt"}',
        ),
        Done(finish_reason="tool_calls"),
    ]
    round_2 = [
        TextDelta(text="The file has: file content"),
        Done(finish_reason="stop"),
    ]
    calls_count = 0

    async def mock_stream(*args, **kwargs):
        nonlocal calls_count
        calls_count += 1
        stream = round_1 if calls_count == 1 else round_2
        for event in stream:
            yield event

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn("Read hello.txt")

    start_idx = next(
        (
            i
            for i, event in enumerate(log)
            if event[0] == "start"
            and "read_file" in event[1]
            and "hello.txt" in event[1]
        ),
        None,
    )
    stop_idx = next(
        (
            i
            for i, event in enumerate(log)
            if event[0] == "stop"
            and "read_file" in event[1]
            and "hello.txt" in event[1]
        ),
        None,
    )
    check_idx = next(
        (i for i, event in enumerate(log) if event[0] == "print" and "✓" in event[1]),
        None,
    )
    assert start_idx is not None, f"expected tool working status, got {log}"
    assert stop_idx is not None, f"expected tool status to stop, got {log}"
    assert check_idx is not None, f"expected tool result line, got {log}"
    assert start_idx < stop_idx < check_idx

    thinking_starts = [
        i for i, event in enumerate(log) if event[0] == "start" and "Thinking" in event[1]
    ]
    assert len(thinking_starts) >= 2, f"expected Thinking between tool rounds, got {log}"
    assert thinking_starts[0] < start_idx < thinking_starts[1]


@pytest.mark.anyio
async def test_agent_stops_thinking_before_provider_error(agent_session):
    """A failed model call must stop Thinking before the error line prints."""
    console = Console(file=io.StringIO(), force_terminal=False, width=120)
    log = track_console(console)
    agent = Agent(agent_session, console, MagicMock(return_value=True))

    async def mock_stream(*args, **kwargs):
        raise ProviderError("API rate limit")
        yield Done()

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn("Hello")

    stop_idx = next(
        (
            i
            for i, event in enumerate(log)
            if event[0] == "stop" and "Thinking" in event[1]
        ),
        None,
    )
    error_idx = next(
        (
            i
            for i, event in enumerate(log)
            if event[0] == "print" and "API rate limit" in event[1]
        ),
        None,
    )
    assert stop_idx is not None, f"expected Thinking status to stop, got {log}"
    assert error_idx is not None, f"expected error line, got {log}"
    assert stop_idx < error_idx


@pytest.mark.anyio
async def test_agent_dry_run_previews_without_executing(agent_session):
    agent_session.dry_run = True
    agent_session.mode = PermissionMode.ASK
    console = Console(quiet=True)
    confirm = MagicMock(return_value=False)
    agent = Agent(agent_session, console, confirm)

    round_1 = [
        ToolCallDelta(
            index=0,
            id="call_write",
            name="write_file",
            args_fragment='{"path": "new.txt", "content": "hi"}',
        ),
        Done(finish_reason="tool_calls"),
    ]
    round_2 = [TextDelta(text="done"), Done(finish_reason="stop")]
    calls = {"n": 0}

    async def mock_stream(*args, **kwargs):
        calls["n"] += 1
        for event in round_1 if calls["n"] == 1 else round_2:
            yield event

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn("write a file")

    assert not (agent_session.workspace_root / "new.txt").exists()
    assert "DRY RUN" in agent_session.messages[2]["content"]
    confirm.assert_not_called()
    assert agent_session.checkpoints == []


@pytest.mark.anyio
async def test_agent_checkpoints_file_before_write(agent_session):
    existing = agent_session.workspace_root / "existing.txt"
    existing.write_text("before")
    console = Console(quiet=True)
    agent = Agent(agent_session, console, MagicMock(return_value=True))

    round_1 = [
        ToolCallDelta(
            index=0,
            id="call_write",
            name="write_file",
            args_fragment='{"path": "existing.txt", "content": "after"}',
        ),
        Done(finish_reason="tool_calls"),
    ]
    round_2 = [TextDelta(text="done"), Done(finish_reason="stop")]
    calls = {"n": 0}

    async def mock_stream(*args, **kwargs):
        calls["n"] += 1
        for event in round_1 if calls["n"] == 1 else round_2:
            yield event

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn("write the file")

    assert existing.read_text() == "after"
    assert [item.paths for item in agent_session.checkpoints] == [["existing.txt"]]

    agent_session.undo_checkpoints()

    assert existing.read_text() == "before"


@pytest.mark.anyio
async def test_agent_runs_read_only_tools_in_parallel(agent_session):
    import threading
    import time

    from kiwimatecoder import tools as tools_module

    (agent_session.workspace_root / "a.txt").write_text("a")
    (agent_session.workspace_root / "b.txt").write_text("b")
    console = Console(quiet=True)
    agent = Agent(agent_session, console, MagicMock(return_value=True))

    round_1 = [
        ToolCallDelta(
            index=0, id="call_a", name="read_file", args_fragment='{"path": "a.txt"}'
        ),
        ToolCallDelta(
            index=1, id="call_b", name="read_file", args_fragment='{"path": "b.txt"}'
        ),
        Done(finish_reason="tool_calls"),
    ]
    round_2 = [TextDelta(text="done"), Done(finish_reason="stop")]
    calls = {"n": 0}

    async def mock_stream(*args, **kwargs):
        calls["n"] += 1
        for event in round_1 if calls["n"] == 1 else round_2:
            yield event

    active = {"count": 0, "max": 0}
    lock = threading.Lock()
    real_execute = tools_module.TOOLS["read_file"].execute

    def slow_execute(args, session):
        with lock:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
        time.sleep(0.05)
        with lock:
            active["count"] -= 1
        return real_execute(args, session)

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch.object(
            tools_module.TOOLS["read_file"], "execute", side_effect=slow_execute
        ),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn("read both")

    assert active["max"] == 2
    tool_messages = [m for m in agent_session.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_a", "call_b"]


@pytest.mark.anyio
async def test_agent_keeps_writes_sequential(agent_session):
    console = Console(quiet=True)
    agent = Agent(agent_session, console, MagicMock(return_value=True))

    calls = [
        AssembledToolCall("call_a", "write_file", '{"path": "a.txt", "content": "a"}'),
        AssembledToolCall("call_b", "write_file", '{"path": "b.txt", "content": "b"}'),
    ]

    assert agent._can_run_parallel(calls) is False  # noqa: SLF001


@pytest.mark.anyio
async def test_agent_auto_verify_runs_after_edits(agent_session):
    agent_session.mode = PermissionMode.AUTO
    agent_session.verify_command = "echo verified"
    console = Console(quiet=True)
    agent = Agent(agent_session, console, MagicMock(return_value=True))

    round_1 = [
        ToolCallDelta(
            index=0,
            id="call_write",
            name="write_file",
            args_fragment='{"path": "new.txt", "content": "hi"}',
        ),
        Done(finish_reason="tool_calls"),
    ]
    rounds = [
        round_1,
        [TextDelta(text="wrote it"), Done(finish_reason="stop")],
        [TextDelta(text="all good"), Done(finish_reason="stop")],
    ]
    calls = {"n": 0}

    async def mock_stream(*args, **kwargs):
        events = rounds[calls["n"]]
        calls["n"] += 1
        for event in events:
            yield event

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn("edit something")

    contents = [str(message.get("content")) for message in agent_session.messages]
    assert any("[auto-verify]" in content for content in contents)
    assert any("verified" in content for content in contents)
    assert calls["n"] == 3


@pytest.mark.anyio
async def test_agent_budget_blocks_before_streaming(agent_session):
    from kiwimatecoder import config

    config.set_budget(max_tokens=10)
    agent_session.prompt_tokens = 50
    console = Console(quiet=True)
    called = {"n": 0}

    async def mock_stream(*args, **kwargs):
        called["n"] += 1
        yield Done(finish_reason="stop")

    agent = Agent(agent_session, console, MagicMock(return_value=True))
    with patch(
        "kiwimatecoder.client.UnifiedClient.stream_chat", side_effect=mock_stream
    ):
        await agent.run_turn("hello")

    assert called["n"] == 0
    assert agent_session.messages == []


@pytest.mark.anyio
async def test_agent_warns_when_near_budget(agent_session):
    from kiwimatecoder import config

    config.set_budget(max_tokens=100)
    agent_session.prompt_tokens = 90
    console = Console(quiet=True)
    log = track_console(console)

    async def mock_stream(*args, **kwargs):
        yield TextDelta(text="hi")
        yield Done(finish_reason="stop")

    agent = Agent(agent_session, console, MagicMock(return_value=True))
    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat", side_effect=mock_stream
        ),
    ):
        await agent.run_turn("hello")

    assert any("Budget warning" in text for _, text in log)


# ---------------------------------------------------------------------------
# Hunk-level partial approvals
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_agent_applies_partial_hunk_selection_to_write(agent_session):
    from kiwimatecoder import audit

    agent_session.mode = PermissionMode.ASK
    old_text = "".join(f"line{i}\n" for i in range(1, 21))
    new_text = old_text.replace("line1\n", "LINE1\n", 1).replace(
        "line20\n", "LINE20\n", 1
    )
    target = agent_session.workspace_root / "multi.txt"
    target.write_text(old_text)

    console = Console(quiet=True)
    confirm = MagicMock(
        return_value=ApprovalResult(allowed=True, selected_hunks=(2,))
    )
    agent = Agent(agent_session, console, confirm)

    round_1 = [
        ToolCallDelta(
            index=0,
            id="call_write",
            name="write_file",
            args_fragment=json.dumps({"path": "multi.txt", "content": new_text}),
        ),
        Done(finish_reason="tool_calls"),
    ]
    round_2 = [TextDelta(text="done"), Done(finish_reason="stop")]
    calls = {"n": 0}

    async def mock_stream(*args, **kwargs):
        calls["n"] += 1
        for event in round_1 if calls["n"] == 1 else round_2:
            yield event

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn("update the file")

    assert target.read_text() == old_text.replace("line20\n", "LINE20\n", 1)
    assert [item.paths for item in agent_session.checkpoints] == [["multi.txt"]]
    tool_message = next(m for m in agent_session.messages if m.get("role") == "tool")
    assert "hunks: 2" in tool_message["content"]

    entries = [
        json.loads(line)
        for line in audit.audit_log_path().read_text(encoding="utf-8").splitlines()
    ]
    partial = next(entry for entry in entries if entry["tool"] == "write_file")
    assert partial["decision"] == "allowed_partial"
    assert partial["hunks"] == [2]


@pytest.mark.anyio
async def test_agent_applies_partial_hunk_selection_to_edit(agent_session):
    agent_session.mode = PermissionMode.ASK
    old_text = (
        "".join(f"line{i}\n" for i in range(1, 21))
        .replace("line5", "TODO", 1)
        .replace("line15", "TODO", 1)
    )
    target = agent_session.workspace_root / "edits.txt"
    target.write_text(old_text)

    console = Console(quiet=True)
    confirm = MagicMock(
        return_value=ApprovalResult(allowed=True, selected_hunks=(1,))
    )
    agent = Agent(agent_session, console, confirm)

    round_1 = [
        ToolCallDelta(
            index=0,
            id="call_edit",
            name="edit_file",
            args_fragment=json.dumps(
                {
                    "path": "edits.txt",
                    "old_string": "TODO",
                    "new_string": "DONE",
                    "replace_all": True,
                }
            ),
        ),
        Done(finish_reason="tool_calls"),
    ]
    round_2 = [TextDelta(text="done"), Done(finish_reason="stop")]
    calls = {"n": 0}

    async def mock_stream(*args, **kwargs):
        calls["n"] += 1
        for event in round_1 if calls["n"] == 1 else round_2:
            yield event

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn("rename the TODOs")

    lines = target.read_text().splitlines()
    assert lines[4] == "DONE"
    assert lines[14] == "TODO"


@pytest.mark.anyio
async def test_agent_reports_unapplicable_hunk_selection(agent_session):
    agent_session.mode = PermissionMode.ASK
    target = agent_session.workspace_root / "new.txt"
    console = Console(quiet=True)
    confirm = MagicMock(
        return_value=ApprovalResult(allowed=True, selected_hunks=(3,))
    )
    agent = Agent(agent_session, console, confirm)

    round_1 = [
        ToolCallDelta(
            index=0,
            id="call_write",
            name="write_file",
            args_fragment=json.dumps(
                {"path": "new.txt", "content": "alpha\nbeta\n"}
            ),
        ),
        Done(finish_reason="tool_calls"),
    ]
    round_2 = [TextDelta(text="done"), Done(finish_reason="stop")]
    calls = {"n": 0}

    async def mock_stream(*args, **kwargs):
        calls["n"] += 1
        for event in round_1 if calls["n"] == 1 else round_2:
            yield event

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn("write the file")

    assert not target.exists()
    assert agent_session.checkpoints == []
    tool_message = next(m for m in agent_session.messages if m.get("role") == "tool")
    assert "could not be applied" in tool_message["content"]


# ---------------------------------------------------------------------------
# Message steering and interrupt recovery
# ---------------------------------------------------------------------------


def test_agent_drain_steering_returns_false_on_empty_queue(agent_session):
    agent = Agent(agent_session, Console(quiet=True), MagicMock())

    assert agent._drain_steering() is False  # noqa: SLF001
    assert agent_session.messages == []


def test_agent_drain_steering_appends_every_queued_message(agent_session):
    agent_session.steering.append("first")
    agent_session.steering.append("second")
    agent = Agent(agent_session, Console(quiet=True), MagicMock())

    assert agent._drain_steering() is True  # noqa: SLF001
    assert agent_session.messages == [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ]
    assert agent._drain_steering() is False  # noqa: SLF001
    assert len(agent_session.messages) == 2


@pytest.mark.anyio
async def test_agent_injects_queued_steering_message(agent_session):
    console = Console(quiet=True)
    agent = Agent(agent_session, console, MagicMock(return_value=True))
    agent_session.steering.append("Also check the tests.")

    calls = {"n": 0}

    async def mock_stream(*args, **kwargs):
        calls["n"] += 1
        yield TextDelta(text=f"response {calls['n']}")
        yield Done(finish_reason="stop")

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn("First request")

    assert calls["n"] == 2
    assert [message["role"] for message in agent_session.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert agent_session.messages[0]["content"] == "First request"
    assert agent_session.messages[1]["content"] == "response 1"
    assert agent_session.messages[2]["content"] == "Also check the tests."
    assert agent_session.messages[3]["content"] == "response 2"


@pytest.mark.anyio
async def test_agent_preserves_partial_assistant_on_cancel(agent_session):
    console = Console(quiet=True)
    agent = Agent(agent_session, console, MagicMock(return_value=True))
    delivered = asyncio.Event()
    blocker = asyncio.Event()

    async def mock_stream(*args, **kwargs):
        yield TextDelta(text="partial answer")
        delivered.set()
        await blocker.wait()
        yield Done(finish_reason="stop")

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        task = asyncio.create_task(agent.run_turn("Do something"))
        await asyncio.wait_for(delivered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert agent_session.messages[-1] == {
        "role": "assistant",
        "content": "partial answer",
    }
