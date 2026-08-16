from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from kiwimatecoder.agent import MAX_TOOL_ROUNDS, Agent
from kiwimatecoder.client import Done, ProviderError, TextDelta, ToolCallDelta
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.session import Session


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
async def test_agent_step_limit_reached(agent_session):
    console = Console(quiet=True)
    confirm = MagicMock(return_value=True)
    agent = Agent(agent_session, console, confirm)

    tool_call_stream = [
        ToolCallDelta(
            index=0,
            id="call_inf",
            name="search",
            args_fragment='{"pattern": "test"}',
        ),
        Done(finish_reason="tool_calls"),
    ]

    async def mock_stream(*args, **kwargs):
        for event in tool_call_stream:
            yield event

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await agent.run_turn("Loop forever")

    # The agent should stop at MAX_TOOL_ROUNDS (25 rounds)
    # 1 user message + 25 rounds * (1 assistant + 1 tool) = 51 messages
    assert len(agent_session.messages) == 1 + (MAX_TOOL_ROUNDS * 2)
    last_tool_msg = agent_session.messages[-1]
    assert last_tool_msg["role"] == "tool"
    assert "step limit" in last_tool_msg["content"].lower()


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
