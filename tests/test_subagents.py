"""Subagents: config, the task tool, and the nested agent loop."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

from kiwimatecoder import config, tools
from kiwimatecoder.agent import Agent
from kiwimatecoder.client import (
    AssembledToolCall,
    Done,
    ProviderError,
    TextDelta,
    ToolCallDelta,
    Usage,
)
from kiwimatecoder.commands import dispatch, slash_argument_completions
from kiwimatecoder.main import app as cli_app
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.prompts import build_system_prompt
from kiwimatecoder.session import Session


@pytest.fixture(autouse=True)
def isolate_config_home(tmp_path, monkeypatch):
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


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def _call(name: str, arguments: str, call_id: str = "call_1") -> AssembledToolCall:
    return AssembledToolCall(id=call_id, name=name, arguments=arguments)


class StreamScript:
    """Deterministic ``UnifiedClient.stream_chat`` replacement."""

    def __init__(self, rounds: list[list[object]]) -> None:
        self.rounds = list(rounds)
        self.calls = 0
        self.schemas: list[list[dict[str, object]]] = []
        self.messages: list[list[dict[str, object]]] = []
        self.models: list[str] = []

    async def __call__(self, messages, schemas, model):
        self.calls += 1
        self.messages.append(list(messages))
        self.schemas.append(list(schemas))
        self.models.append(model)
        for event in self.rounds.pop(0):
            yield event


def _patches(script: StreamScript):
    return (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=script,
        ),
    )


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_task_tool_is_registered_and_approval_gated():
    tool = tools.get_tool("task")

    assert tool is not None
    assert tool.writes is False
    assert tool.runs is True
    assert tool.needs_approval is True
    assert "subagent" in tool.description.lower()
    assert "cannot ask the user" in tool.description.lower()


def test_task_schema_requires_description_and_prompt():
    schema = tools.get_tool("task").schema()["function"]

    assert schema["name"] == "task"
    assert schema["parameters"]["required"] == ["description", "prompt"]
    assert "model" in schema["parameters"]["properties"]


def test_subagent_cannot_call_task_tool():
    child = Session(
        provider_id="openrouter",
        model="test-model",
        mode=PermissionMode.AUTO,
        workspace_root=MagicMock(),
        subagent=True,
    )
    agent = Agent(child, Console(quiet=True), MagicMock(return_value=True))

    message, edited = agent._run_tool_call(
        _call("task", '{"description": "nested", "prompt": "nope"}')
    )

    assert edited is False
    assert "nested subagents" in message["content"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_subagents_defaults_roundtrip_and_validation():
    assert config.get_subagents() == {
        "enabled": True,
        "max_steps": 20,
        "model": "",
    }

    settings = config.set_subagents(max_steps=5, model=" cheap-model ")
    assert settings == {"enabled": True, "max_steps": 5, "model": "cheap-model"}

    assert config.set_subagents(enabled=False)["enabled"] is False
    assert config.get_subagents()["enabled"] is False

    for bad in (0, 101, "many"):
        with pytest.raises(ValueError):
            config.set_subagents(max_steps=bad)
    with pytest.raises(ValueError):
        config.set_subagents(enabled="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        config.set_subagents(model=7)  # type: ignore[arg-type]

    assert config.get_subagents()["max_steps"] == 5


def test_get_subagents_tolerates_malformed_stored_values():
    stored = {
        "enabled": "nope",
        "max_steps": 9999,
        "model": 12,
    }

    assert config.get_subagents({"subagents": stored}) == {
        "enabled": True,
        "max_steps": 20,
        "model": "",
    }


def test_validate_config_flags_bad_subagents():
    issues = config.validate_config(
        {
            "subagents": {
                "enabled": "yes",
                "max_steps": 0,
                "model": 5,
            }
        }
    )
    keys = {issue["key"] for issue in issues if issue["level"] == "error"}

    assert "subagents.enabled" in keys
    assert "subagents.max_steps" in keys
    assert "subagents.model" in keys


def test_validate_config_accepts_default_config():
    assert config.validate_config(config.load_config()) == []


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def test_system_prompt_mentions_task_when_enabled(agent_session):
    prompt = build_system_prompt(agent_session)["content"]

    assert "task tool" in prompt
    assert "cannot ask the user" in prompt


def test_system_prompt_omits_task_note_for_subagents(agent_session):
    agent_session.subagent = True

    prompt = build_system_prompt(agent_session)["content"]

    assert "task tool" not in prompt


def test_system_prompt_omits_task_note_when_disabled(agent_session):
    config.set_subagents(enabled=False)

    prompt = build_system_prompt(agent_session)["content"]

    assert "task tool" not in prompt


def test_task_hidden_from_schemas_when_disabled(agent_session):
    agent = Agent(agent_session, Console(quiet=True), MagicMock())

    assert "task" in {s["function"]["name"] for s in agent._tool_schemas()}

    config.set_subagents(enabled=False)

    assert "task" not in {s["function"]["name"] for s in agent._tool_schemas()}


# ---------------------------------------------------------------------------
# Child session isolation
# ---------------------------------------------------------------------------


async def test_subagent_runs_in_isolated_child_session(agent_session):
    script = StreamScript(
        [[TextDelta(text="report done"), Done(finish_reason="stop")]]
    )
    agent = Agent(agent_session, Console(quiet=True), MagicMock(return_value=True))

    with patch("kiwimatecoder.config.get_key", return_value="dummy_key"), patch(
        "kiwimatecoder.client.UnifiedClient.stream_chat", side_effect=script
    ):
        result = await agent._run_task(
            {"description": "look", "prompt": "Find the auth flow"}
        )

    assert result.ok
    assert result.content.startswith("[task result]")
    assert "report done" in result.content
    assert "1 step(s)" not in result.content  # no tool batches ran
    assert "0 step(s)" in result.content

    # The parent conversation is untouched.
    assert agent_session.messages == []

    # The child sees the prompt plus a system message, not the parent history.
    child_messages = script.messages[0]
    assert child_messages[0]["role"] == "system"
    assert "Subagents:" not in child_messages[0]["content"]
    assert child_messages[-1] == {
        "role": "user",
        "content": "Find the auth flow",
    }

    # No recursion and no interactive user for the child.
    names = {schema["function"]["name"] for schema in script.schemas[0]}
    assert "task" not in names
    assert "ask_user" not in names
    assert "read_file" in names
    assert script.models[0] == "test-model"


async def test_subagent_model_override_is_used(agent_session):
    script = StreamScript(
        [[TextDelta(text="ok"), Done(finish_reason="stop")]]
    )
    agent = Agent(agent_session, Console(quiet=True), MagicMock(return_value=True))

    with patch("kiwimatecoder.config.get_key", return_value="dummy_key"), patch(
        "kiwimatecoder.client.UnifiedClient.stream_chat", side_effect=script
    ):
        await agent._run_task(
            {
                "description": "look",
                "prompt": "Find it",
                "model": "cheap-model",
            }
        )

    assert script.models[0] == "cheap-model"


async def test_subagent_usage_counts_toward_the_parent(agent_session):
    script = StreamScript(
        [
            [
                Usage(prompt_tokens=30, completion_tokens=10),
                TextDelta(text="ok"),
                Done(finish_reason="stop"),
            ]
        ]
    )
    agent = Agent(agent_session, Console(quiet=True), MagicMock(return_value=True))

    with patch("kiwimatecoder.config.get_key", return_value="dummy_key"), patch(
        "kiwimatecoder.client.UnifiedClient.stream_chat", side_effect=script
    ):
        await agent._run_task({"description": "look", "prompt": "Find it"})

    assert agent_session.prompt_tokens == 30
    assert agent_session.completion_tokens == 10


# ---------------------------------------------------------------------------
# End-to-end through the parent turn
# ---------------------------------------------------------------------------


async def test_parent_receives_task_report_as_tool_message(agent_session):
    (agent_session.workspace_root / "hello.txt").write_text("file content")
    script = StreamScript(
        [
            [
                ToolCallDelta(
                    index=0,
                    id="call_task",
                    name="task",
                    args_fragment=json.dumps(
                        {"description": "read hello", "prompt": "Read hello.txt"}
                    ),
                ),
                Done(finish_reason="tool_calls"),
            ],
            [
                ToolCallDelta(
                    index=0,
                    id="call_read",
                    name="read_file",
                    args_fragment='{"path": "hello.txt"}',
                ),
                Done(finish_reason="tool_calls"),
            ],
            [
                TextDelta(text="The file says file content"),
                Done(finish_reason="stop"),
            ],
            [TextDelta(text="All done."), Done(finish_reason="stop")],
        ]
    )
    agent = Agent(agent_session, Console(quiet=True), MagicMock(return_value=True))

    with patch("kiwimatecoder.config.get_key", return_value="dummy_key"), patch(
        "kiwimatecoder.client.UnifiedClient.stream_chat", side_effect=script
    ):
        await agent.run_turn("delegate reading hello")

    assert script.calls == 4
    roles = [message["role"] for message in agent_session.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    tool_message = agent_session.messages[2]
    assert tool_message["tool_call_id"] == "call_task"
    assert tool_message["content"].startswith("[task result]")
    assert "The file says file content" in tool_message["content"]
    assert "1 step(s)" in tool_message["content"]


async def test_subagent_edits_are_checkpointed_for_the_parent(agent_session):
    script = StreamScript(
        [
            [
                ToolCallDelta(
                    index=0,
                    id="call_write",
                    name="write_file",
                    args_fragment=json.dumps(
                        {"path": "out.txt", "content": "hello"}
                    ),
                ),
                Done(finish_reason="tool_calls"),
            ],
            [TextDelta(text="wrote out.txt"), Done(finish_reason="stop")],
        ]
    )
    agent = Agent(agent_session, Console(quiet=True), MagicMock(return_value=True))

    with patch("kiwimatecoder.config.get_key", return_value="dummy_key"), patch(
        "kiwimatecoder.client.UnifiedClient.stream_chat", side_effect=script
    ):
        result = await agent._run_task(
            {"description": "write", "prompt": "Write out.txt"}
        )

    assert result.ok
    assert (agent_session.workspace_root / "out.txt").read_text() == "hello"
    assert len(agent_session.checkpoints) == 1
    assert agent_session.checkpoints[0].paths == ["out.txt"]
    assert "out.txt" in agent_session.touched_files


async def test_task_call_is_denied_in_ask_mode(agent_session):
    agent_session.mode = PermissionMode.ASK
    script = StreamScript(
        [
            [
                ToolCallDelta(
                    index=0,
                    id="call_task",
                    name="task",
                    args_fragment=json.dumps(
                        {"description": "look", "prompt": "Find the auth flow"}
                    ),
                ),
                Done(finish_reason="tool_calls"),
            ],
            [TextDelta(text="Understood."), Done(finish_reason="stop")],
        ]
    )
    confirm = MagicMock(return_value=False)
    agent = Agent(agent_session, Console(quiet=True), confirm)

    with patch("kiwimatecoder.config.get_key", return_value="dummy_key"), patch(
        "kiwimatecoder.client.UnifiedClient.stream_chat", side_effect=script
    ):
        await agent.run_turn("delegate reading hello")

    assert confirm.call_count == 1
    assert script.calls == 2  # no child stream ran
    tool_message = next(
        message
        for message in agent_session.messages
        if message["role"] == "tool"
    )
    assert "denied" in tool_message["content"].lower()


# ---------------------------------------------------------------------------
# Caps and budgets
# ---------------------------------------------------------------------------


async def test_subagent_step_cap_stops_runaway_loop(agent_session):
    config.set_subagents(max_steps=2)
    script = StreamScript(
        [
            [
                ToolCallDelta(
                    index=0,
                    id="call_a",
                    name="search",
                    args_fragment='{"pattern": "x"}',
                ),
                Done(finish_reason="tool_calls"),
            ],
            [
                ToolCallDelta(
                    index=0,
                    id="call_b",
                    name="search",
                    args_fragment='{"pattern": "y"}',
                ),
                Done(finish_reason="tool_calls"),
            ],
            [TextDelta(text="never reached"), Done(finish_reason="stop")],
        ]
    )
    agent = Agent(agent_session, Console(quiet=True), MagicMock(return_value=True))

    with patch("kiwimatecoder.config.get_key", return_value="dummy_key"), patch(
        "kiwimatecoder.client.UnifiedClient.stream_chat", side_effect=script
    ):
        result = await agent._run_task(
            {"description": "runaway", "prompt": "search forever"}
        )

    assert script.calls == 2
    assert "step limit (2)" in result.content
    assert "partial report" in result.content
    assert "2 step(s)" in result.content
    assert "never reached" not in result.content


async def test_subagent_inherits_remaining_parent_budget(agent_session):
    config.set_budget(max_tokens=1000)
    agent_session.prompt_tokens = 900
    script = StreamScript(
        [
            [
                Usage(prompt_tokens=60, completion_tokens=60),
                ToolCallDelta(
                    index=0,
                    id="call_a",
                    name="list_dir",
                    args_fragment='{"path": "."}',
                ),
                Done(finish_reason="tool_calls"),
            ],
            [TextDelta(text="never reached"), Done(finish_reason="stop")],
        ]
    )
    agent = Agent(agent_session, Console(quiet=True), MagicMock(return_value=True))

    with patch("kiwimatecoder.config.get_key", return_value="dummy_key"), patch(
        "kiwimatecoder.client.UnifiedClient.stream_chat", side_effect=script
    ):
        result = await agent._run_task(
            {"description": "expensive", "prompt": "spend tokens"}
        )

    assert script.calls == 1
    assert "inherited token budget" in result.content
    assert "never reached" not in result.content
    assert agent_session.total_tokens == 1020


async def test_subagent_refuses_when_parent_budget_is_spent(agent_session):
    config.set_budget(max_tokens=1000)
    agent_session.prompt_tokens = 1000
    script = StreamScript([[TextDelta(text="nope"), Done(finish_reason="stop")]])
    agent = Agent(agent_session, Console(quiet=True), MagicMock(return_value=True))

    with patch("kiwimatecoder.config.get_key", return_value="dummy_key"), patch(
        "kiwimatecoder.client.UnifiedClient.stream_chat", side_effect=script
    ):
        result = await agent._run_task({"description": "x", "prompt": "y"})

    assert result.ok is False
    assert "budget" in result.content.lower()
    assert script.calls == 0


# ---------------------------------------------------------------------------
# Plan mode and disabled config
# ---------------------------------------------------------------------------


async def test_subagent_in_parent_plan_mode_gets_only_read_only_tools(agent_session):
    agent_session.mode = PermissionMode.PLAN
    script = StreamScript(
        [[TextDelta(text="planned"), Done(finish_reason="stop")]]
    )
    agent = Agent(agent_session, Console(quiet=True), MagicMock(return_value=True))

    with patch("kiwimatecoder.config.get_key", return_value="dummy_key"), patch(
        "kiwimatecoder.client.UnifiedClient.stream_chat", side_effect=script
    ):
        await agent._run_task({"description": "plan", "prompt": "Look around"})

    names = {schema["function"]["name"] for schema in script.schemas[0]}
    assert names
    for name in names:
        tool = tools.get_tool(name)
        assert tool is not None
        assert tool.needs_approval is False
    assert "write_file" not in names
    assert "run_bash" not in names
    assert "task" not in names
    assert "ask_user" not in names


async def test_subagent_disabled_returns_clear_error(agent_session):
    config.set_subagents(enabled=False)
    script = StreamScript([[TextDelta(text="nope"), Done(finish_reason="stop")]])
    agent = Agent(agent_session, Console(quiet=True), MagicMock(return_value=True))

    with patch("kiwimatecoder.config.get_key", return_value="dummy_key"), patch(
        "kiwimatecoder.client.UnifiedClient.stream_chat", side_effect=script
    ):
        result = await agent._run_task({"description": "x", "prompt": "y"})

    assert result.ok is False
    assert "disabled" in result.content.lower()
    assert "/config subagents enable on" in result.content
    assert script.calls == 0


async def test_subagent_provider_error_returns_error_result(agent_session):
    async def failing_stream(messages, schemas, model):
        raise ProviderError("rate limited")
        yield  # pragma: no cover - keeps this an async generator

    agent = Agent(agent_session, Console(quiet=True), MagicMock(return_value=True))

    with patch("kiwimatecoder.config.get_key", return_value="dummy_key"), patch(
        "kiwimatecoder.client.UnifiedClient.stream_chat",
        side_effect=failing_stream,
    ):
        result = await agent._run_task({"description": "x", "prompt": "y"})

    assert result.ok is False
    assert "Subagent failed" in result.content
    assert "rate limited" in result.content


# ---------------------------------------------------------------------------
# Output prefix and summaries
# ---------------------------------------------------------------------------


def test_agent_summary_for_task(agent_session):
    agent = Agent(agent_session, Console(quiet=True), MagicMock())

    assert (
        agent._format_call_summary("task", {"description": "find auth"})
        == "task [dim]find auth[/dim]"
    )


async def test_child_output_is_prefixed_with_task_tag(agent_session):
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    script = StreamScript(
        [[TextDelta(text="child report"), Done(finish_reason="stop")]]
    )
    agent = Agent(agent_session, console, MagicMock(return_value=True))

    with patch("kiwimatecoder.config.get_key", return_value="dummy_key"), patch(
        "kiwimatecoder.client.UnifiedClient.stream_chat", side_effect=script
    ):
        await agent._run_task({"description": "look", "prompt": "Investigate"})

    output = buf.getvalue()
    assert "[task]" in output
    assert "child report" in output


# ---------------------------------------------------------------------------
# Slash command and CLI
# ---------------------------------------------------------------------------


def test_slash_config_subagents_show_enable_and_max_steps(agent_session):
    console = _console()
    dispatch("/config subagents show", agent_session, console)
    assert "Subagents: on" in console.file.getvalue()
    assert "max steps 20" in console.file.getvalue()

    console = _console()
    dispatch("/config subagents enable off", agent_session, console)
    assert config.get_subagents()["enabled"] is False

    console = _console()
    dispatch("/config subagents max-steps 7", agent_session, console)
    assert config.get_subagents()["max_steps"] == 7

    console = _console()
    dispatch("/config subagents max-steps 0", agent_session, console)
    assert "between 1 and 100" in console.file.getvalue()
    assert config.get_subagents()["max_steps"] == 7

    console = _console()
    dispatch("/config subagents model cheap-model", agent_session, console)
    assert config.get_subagents()["model"] == "cheap-model"


def test_config_help_and_completions_include_subagents():
    console = _console()
    dispatch("/config help", Session("openrouter", "m"), console)
    assert "/config subagents" in console.file.getvalue()

    choices = dict(slash_argument_completions("config", "sub", None))
    assert "subagents" in choices


def test_cli_config_subagents_roundtrip():
    runner = CliRunner()

    result = runner.invoke(cli_app, ["config", "subagents", "show"])
    assert result.exit_code == 0
    assert "Subagents: on" in result.output

    result = runner.invoke(cli_app, ["config", "subagents", "enable", "off"])
    assert result.exit_code == 0
    assert config.get_subagents()["enabled"] is False

    result = runner.invoke(cli_app, ["config", "subagents", "max-steps", "7"])
    assert result.exit_code == 0
    assert config.get_subagents()["max_steps"] == 7

    result = runner.invoke(cli_app, ["config", "subagents", "max-steps", "0"])
    assert result.exit_code == 1
    assert config.get_subagents()["max_steps"] == 7

    result = runner.invoke(cli_app, ["config", "subagents", "model", "cheap"])
    assert result.exit_code == 0
    assert config.get_subagents()["model"] == "cheap"
