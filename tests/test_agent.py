from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from kiwimatecoder.agent import Agent
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
