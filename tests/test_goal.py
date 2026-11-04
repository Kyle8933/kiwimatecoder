"""Tests for /goal: the state machine, the update_goal tool, the agent hooks, the
slash command, persistence, and the REPL round loop."""

from __future__ import annotations

import asyncio
import io
import json
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from kiwimatecoder import config, goal as goal_module, repl, tools
from kiwimatecoder.agent import Agent
from kiwimatecoder.client import Done, ProviderError, TextDelta, ToolCallDelta
from kiwimatecoder.commands import CommandResult, dispatch, slash_argument_completions
from kiwimatecoder.goal import (
    ACHIEVED,
    ACTIVE,
    BLOCKED,
    DEFAULT_MAX_ROUNDS,
    PAUSED,
    Goal,
    continuation_prompt,
    finish_round,
    new_goal,
    opening_prompt,
    parse_goal_command,
    pause,
    resume,
    start_round,
)
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.session import Session


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=200)


def _output(console: Console) -> str:
    return console.file.getvalue()


@pytest.fixture(autouse=True)
def isolate_config_home(tmp_path, monkeypatch):
    home = tmp_path / "config-home"
    monkeypatch.setattr(config, "CONFIG_DIR", home)
    monkeypatch.setattr(config, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", home / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    from kiwimatecoder.providers import REGISTRY

    for provider in REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("arg", "verb"),
    [("", "show"), ("  ", "show"), ("pause", "pause"), ("RESUME", "resume"), ("clear", "clear")],
)
def test_bare_words_are_verbs(arg, verb):
    assert parse_goal_command(arg).verb == verb


def test_an_objective_is_set_with_the_default_round_limit():
    command = parse_goal_command("make the parser handle nested tables")

    assert command.verb == "set"
    assert command.objective == "make the parser handle nested tables"
    assert command.max_rounds == DEFAULT_MAX_ROUNDS


def test_rounds_flag_sets_the_limit():
    command = parse_goal_command("--rounds 25 fix the flaky test")

    assert (command.verb, command.objective, command.max_rounds) == ("set", "fix the flaky test", 25)


def test_double_dash_makes_a_verb_word_an_objective():
    assert parse_goal_command("-- clear the cache").objective == "clear the cache"
    assert parse_goal_command("-- clear the cache").verb == "set"


def test_a_flag_like_word_that_is_not_the_flag_stays_in_the_objective():
    assert parse_goal_command("--roundtrip the API").objective == "--roundtrip the API"


@pytest.mark.parametrize(
    "arg",
    [
        "--rounds",
        "--rounds 0 fix it",
        "--rounds 101 fix it",
        "--rounds many fix it",
        "--rounds ² fix it",  # str.isdigit() is true for superscripts; int() is not
        "--rounds 5",
        "--",
        "-- ",
    ],
)
def test_bad_round_arguments_are_rejected(arg):
    with pytest.raises(ValueError):
        parse_goal_command(arg)


def test_an_overlong_objective_is_rejected():
    with pytest.raises(ValueError, match="at most"):
        parse_goal_command("x" * (goal_module.MAX_OBJECTIVE_CHARS + 1))


# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #


def test_a_normal_stop_keeps_the_goal_going_until_the_round_limit():
    goal = new_goal("ship it", max_rounds=2)

    start_round(goal)
    finish_round(goal, "stop")
    assert goal.due and goal.status == ACTIVE

    start_round(goal)
    finish_round(goal, "stop")
    assert not goal.due
    assert goal.status == PAUSED
    assert goal.outcome == "reached the 2-round limit"


@pytest.mark.parametrize(
    ("stop_reason", "explanation"),
    [
        ("interrupted", "you interrupted it"),
        ("budget", "the budget is spent"),
        ("provider_error", "the provider returned an error"),
        ("max_turns", "the tool-call limit was reached"),
        (None, "the round ended early"),
    ],
)
def test_an_abnormal_end_pauses_with_a_reason(stop_reason, explanation):
    goal = new_goal("x")
    start_round(goal)

    finish_round(goal, stop_reason)

    assert goal.status == PAUSED
    assert goal.outcome == explanation
    assert not goal.due


def test_a_status_reported_during_the_round_is_not_overridden():
    goal = new_goal("x")
    start_round(goal)
    goal.status = ACHIEVED
    goal.outcome = "tests pass"

    finish_round(goal, "stop")

    assert goal.status == ACHIEVED and goal.outcome == "tests pass" and not goal.due


def test_the_first_round_gets_the_objective_and_later_rounds_get_the_continuation():
    goal = new_goal("fix the build", max_rounds=5)

    first = start_round(goal)
    goal.progress = "parser compiles"
    second = start_round(goal)

    assert first == opening_prompt(goal)
    assert "Goal: fix the build" in first
    assert "update_goal" in first
    assert second == continuation_prompt(goal)
    assert "round 2 of 5" in second
    assert "Last progress note: parser compiles" in second


def test_resume_gives_a_fresh_round_allowance():
    goal = new_goal("x", max_rounds=3)
    goal.rounds = 3
    pause(goal, "reached the 3-round limit")

    resume(goal)

    assert goal.status == ACTIVE and goal.rounds == 0 and goal.due and goal.outcome == ""


def test_round_limits_are_clamped():
    assert new_goal("x", max_rounds=0).max_rounds == 1
    assert new_goal("x", max_rounds=10_000).max_rounds == goal_module.MAX_ROUNDS_LIMIT


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_a_goal_round_trips_through_a_dict():
    goal = new_goal("refactor the loader", max_rounds=7)
    start_round(goal)
    goal.progress = "half done"
    pause(goal, "you interrupted it")

    restored = Goal.from_dict(goal.to_dict())

    assert restored is not None
    assert restored.objective == "refactor the loader"
    assert restored.max_rounds == 7
    assert restored.rounds == 1
    assert restored.progress == "half done"
    assert restored.status == PAUSED
    assert restored.outcome == "you interrupted it"
    assert not restored.due


def test_a_goal_that_was_running_is_restored_as_paused():
    restored = Goal.from_dict({"objective": "x", "status": ACTIVE, "max_rounds": 4})

    assert restored is not None
    assert restored.status == PAUSED
    assert restored.outcome == "restored from a saved session"
    assert not restored.due


@pytest.mark.parametrize(
    "data",
    [None, "nope", {}, {"objective": "  "}, {"objective": ["not", "text"]}],
)
def test_malformed_saved_goals_are_dropped(data):
    assert Goal.from_dict(data) is None


def test_a_bad_saved_status_and_round_limit_are_repaired():
    restored = Goal.from_dict(
        {"objective": "x", "status": "weird", "max_rounds": "lots", "rounds": -3}
    )

    assert restored is not None
    assert restored.status == PAUSED
    assert restored.max_rounds == DEFAULT_MAX_ROUNDS
    assert restored.rounds == 0


def test_the_session_saves_and_loads_its_goal(tmp_path):
    session = Session(provider_id="openrouter", model="m", workspace_root=tmp_path)
    session.goal = new_goal("keep the docs in sync", max_rounds=3)

    data = json.loads(json.dumps(session.to_dict()))
    loaded = Session.from_dict(data)

    assert loaded.goal is not None
    assert loaded.goal.objective == "keep the docs in sync"
    assert loaded.goal.max_rounds == 3
    assert loaded.goal.status == PAUSED


def test_sessions_saved_before_goals_existed_load_without_one(tmp_path):
    session = Session(provider_id="openrouter", model="m", workspace_root=tmp_path)
    data = session.to_dict()
    data.pop("goal")

    assert Session.from_dict(data).goal is None


# --------------------------------------------------------------------------- #
# update_goal tool
# --------------------------------------------------------------------------- #


def _session(tmp_path, goal=None) -> Session:
    return Session(
        provider_id="openrouter",
        model="m",
        mode=PermissionMode.AUTO,
        workspace_root=tmp_path,
        goal=goal,
    )


def test_update_goal_needs_an_active_goal(tmp_path):
    session = _session(tmp_path)

    result = tools.dispatch("update_goal", {"status": "achieved", "summary": "done"}, session)

    assert not result.ok
    assert "No goal is active" in result.content


def test_update_goal_refuses_a_paused_goal(tmp_path):
    goal = new_goal("x")
    pause(goal, "by you")
    session = _session(tmp_path, goal)

    result = tools.dispatch("update_goal", {"status": "achieved", "summary": "done"}, session)

    assert not result.ok
    assert goal.status == PAUSED


@pytest.mark.parametrize(
    "args",
    [
        {"status": "finished", "summary": "x"},
        {"status": "achieved", "summary": "   "},
        {"status": "achieved"},
    ],
)
def test_update_goal_validates_its_arguments(tmp_path, args):
    session = _session(tmp_path, new_goal("x"))

    result = tools.dispatch("update_goal", args, session)

    assert not result.ok
    assert session.goal.status == ACTIVE


def test_update_goal_records_progress_without_ending_the_goal(tmp_path):
    session = _session(tmp_path, new_goal("x"))

    result = tools.dispatch(
        "update_goal", {"status": "in_progress", "summary": "wrote the parser"}, session
    )

    assert result.ok
    assert session.goal.progress == "wrote the parser"
    assert session.goal.status == ACTIVE


def test_update_goal_can_mark_the_goal_achieved_or_blocked(tmp_path):
    session = _session(tmp_path, new_goal("x"))

    tools.dispatch("update_goal", {"status": "achieved", "summary": "tests pass"}, session)
    assert session.goal.status == ACHIEVED
    assert session.goal.outcome == "tests pass"

    session.goal = new_goal("y")
    result = tools.dispatch(
        "update_goal", {"status": "blocked", "summary": "need the API key"}, session
    )
    assert result.ok
    assert session.goal.status == BLOCKED
    assert session.goal.outcome == "need the API key"


def test_update_goal_is_registered_as_a_read_only_tool():
    tool = tools.get_tool("update_goal")

    assert tool is not None
    assert not tool.needs_approval


# --------------------------------------------------------------------------- #
# Agent: stop reasons and when the model is offered update_goal
# --------------------------------------------------------------------------- #


def _names(agent) -> set[str]:
    return {schema["function"]["name"] for schema in agent._tool_schemas()}


def test_update_goal_is_offered_only_while_a_goal_is_active(tmp_path):
    session = _session(tmp_path)
    agent = Agent(session, _console(), MagicMock(return_value=True))
    assert "update_goal" not in _names(agent)

    session.goal = new_goal("x")
    assert "update_goal" in _names(agent)

    pause(session.goal, "by you")
    assert "update_goal" not in _names(agent)


def _stream(rounds):
    calls = 0

    async def stream_chat(*_args, **_kwargs):
        nonlocal calls
        events = rounds[min(calls, len(rounds) - 1)]
        calls += 1
        for event in events:
            yield event

    return stream_chat


_TEXT_ROUND = [TextDelta(text="done"), Done(finish_reason="stop")]


async def _run(agent, rounds, prompt="go", stream=None):
    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=stream or _stream(rounds),
        ),
    ):
        await agent.run_turn(prompt)


@pytest.mark.anyio
async def test_a_plain_reply_ends_with_a_stop_reason(tmp_path):
    session = _session(tmp_path)
    agent = Agent(session, _console(), MagicMock(return_value=True))

    await _run(agent, [_TEXT_ROUND])

    assert agent.last_stop_reason == "stop"


@pytest.mark.anyio
async def test_an_exhausted_budget_ends_with_a_budget_reason(tmp_path):
    session = _session(tmp_path)
    session.prompt_tokens = 100
    agent = Agent(session, _console(), MagicMock(return_value=True))

    with patch("kiwimatecoder.config.get_budget", return_value={"max_tokens": 10}):
        await _run(agent, [_TEXT_ROUND])

    assert agent.last_stop_reason == "budget"


@pytest.mark.anyio
async def test_a_provider_failure_ends_with_a_provider_error_reason(tmp_path):
    session = _session(tmp_path)
    agent = Agent(session, _console(), MagicMock(return_value=True))

    async def failing(*_args, **_kwargs):
        raise ProviderError("boom")
        yield Done()

    await _run(agent, [], stream=failing)

    assert agent.last_stop_reason == "provider_error"


@pytest.mark.anyio
async def test_the_tool_call_limit_ends_with_a_max_turns_reason(tmp_path):
    session = _session(tmp_path)
    agent = Agent(session, _console(), MagicMock(return_value=True), max_turns=1)
    tool_round = [
        ToolCallDelta(index=0, id="c1", name="update_todos", args_fragment='{"todos": []}'),
        Done(finish_reason="tool_calls"),
    ]

    # The limit is checked before a further tool round runs, so a second tool
    # round is what trips it.
    await _run(agent, [tool_round, tool_round])

    assert agent.last_stop_reason == "max_turns"


# --------------------------------------------------------------------------- #
# The /goal command
# --------------------------------------------------------------------------- #


def test_goal_with_no_goal_says_how_to_start_one(session):
    console = _console()

    assert dispatch("/goal", session, console) == CommandResult.CONTINUE

    assert "No goal set" in _output(console)


def test_goal_sets_an_objective_that_is_due_to_run(session):
    console = _console()

    dispatch("/goal ship the importer", session, console)

    assert session.goal is not None
    assert session.goal.objective == "ship the importer"
    assert session.goal.status == ACTIVE
    assert session.goal.due
    assert "Goal set" in _output(console)


def test_goal_replaces_an_existing_goal(session):
    dispatch("/goal first", session, _console())
    dispatch("/goal --rounds 4 second", session, _console())

    assert session.goal.objective == "second"
    assert session.goal.max_rounds == 4
    assert session.goal.rounds == 0


def test_goal_pause_during_a_round_takes_effect_after_it(session):
    dispatch("/goal work", session, _console())
    session.goal.due = False  # a round is in progress

    console = _console()
    dispatch("/goal pause", session, console)

    assert session.goal.status == PAUSED
    assert "after the current round" in _output(console)


def test_goal_pause_between_rounds_pauses_at_once(session):
    dispatch("/goal work", session, _console())

    console = _console()
    dispatch("/goal pause", session, console)

    assert session.goal.status == PAUSED
    assert "Goal paused." in _output(console)


def test_goal_pause_without_a_running_goal_is_reported(session):
    console = _console()

    dispatch("/goal pause", session, console)

    assert "No running goal to pause" in _output(console)


def test_goal_resume_restarts_a_paused_goal(session):
    dispatch("/goal work", session, _console())
    dispatch("/goal pause", session, _console())

    dispatch("/goal resume", session, _console())

    assert session.goal.status == ACTIVE
    assert session.goal.due


def test_goal_resume_refuses_an_achieved_goal(session):
    dispatch("/goal work", session, _console())
    session.goal.status = ACHIEVED

    console = _console()
    dispatch("/goal resume", session, console)

    assert session.goal.status == ACHIEVED
    assert "already achieved" in _output(console)


def test_goal_resume_without_a_goal_is_reported(session):
    console = _console()

    dispatch("/goal resume", session, console)

    assert "No goal to resume" in _output(console)


def test_goal_clear_drops_the_goal(session):
    dispatch("/goal work", session, _console())

    dispatch("/goal clear", session, _console())

    assert session.goal is None


def test_goal_double_dash_sets_a_verb_word_as_the_objective(session):
    dispatch("/goal -- clear the cache", session, _console())

    assert session.goal.objective == "clear the cache"


def test_goal_show_reports_status_progress_and_reason(session):
    dispatch("/goal --rounds 6 index the docs", session, _console())
    session.goal.rounds = 2
    session.goal.progress = "indexed half"
    pause(session.goal, "the budget is spent")
    console = _console()

    dispatch("/goal", session, console)

    output = _output(console)
    assert "index the docs" in output
    assert "paused, round 2 of 6" in output
    assert "indexed half" in output
    assert "the budget is spent" in output


def test_a_bad_goal_argument_leaves_the_goal_alone(session):
    dispatch("/goal keep me", session, _console())
    console = _console()

    dispatch("/goal --rounds 0 nope", session, console)

    assert session.goal.objective == "keep me"
    assert "--rounds takes a number" in _output(console)


def test_goal_offers_pause_resume_and_clear_as_completions(session):
    assert slash_argument_completions("goal", "p", session) == [
        ("pause", "Stop the goal after the current round.")
    ]


# --------------------------------------------------------------------------- #
# The REPL round loop
# --------------------------------------------------------------------------- #


class _FakeAgent:
    def __init__(self) -> None:
        self.last_stop_reason: str | None = None


@pytest.fixture
def repl_console(monkeypatch):
    console = _console()
    monkeypatch.setattr(repl, "console", console)
    return console


def _rounds(monkeypatch, behaviour):
    """Replace the round runner. ``behaviour(n, session)`` returns the stop reason
    for round ``n`` and may change the session, as a tool call would."""
    prompts: list[str] = []

    async def fake_turn(agent, _pt, session, line):
        prompts.append(line)
        # A loop that never stops must fail the test, not hang it.
        assert len(prompts) <= 50, "the goal loop kept running"
        agent.last_stop_reason = behaviour(len(prompts), session)
        return False

    monkeypatch.setattr(repl, "_run_turn_with_steering", fake_turn)
    return prompts


@pytest.mark.anyio
async def test_the_loop_runs_rounds_until_the_model_reports_achieved(
    session, monkeypatch, repl_console
):
    session.goal = new_goal("cover the parser", max_rounds=10)

    def behaviour(n, session):
        if n == 3:
            tools.dispatch("update_goal", {"status": "achieved", "summary": "all green"}, session)
        return "stop"

    prompts = _rounds(monkeypatch, behaviour)

    assert await repl._drive_goal(_FakeAgent(), None, session) is False

    assert len(prompts) == 3
    assert prompts[0].startswith("You are working toward a goal")
    assert "round 2 of 10" in prompts[1]
    assert session.goal.status == ACHIEVED
    assert "Goal achieved after 3 round(s)" in _output(repl_console)


@pytest.mark.anyio
async def test_the_loop_stops_at_the_round_limit(session, monkeypatch, repl_console):
    session.goal = new_goal("never ends", max_rounds=2)
    prompts = _rounds(monkeypatch, lambda n, session: "stop")

    await repl._drive_goal(_FakeAgent(), None, session)

    assert len(prompts) == 2
    assert session.goal.status == PAUSED
    assert "reached the 2-round limit" in _output(repl_console)


@pytest.mark.anyio
async def test_the_loop_stops_when_the_model_is_blocked(session, monkeypatch, repl_console):
    session.goal = new_goal("needs a key", max_rounds=5)

    def behaviour(n, session):
        tools.dispatch("update_goal", {"status": "blocked", "summary": "no API key"}, session)
        return "stop"

    prompts = _rounds(monkeypatch, behaviour)

    await repl._drive_goal(_FakeAgent(), None, session)

    assert len(prompts) == 1
    assert session.goal.status == BLOCKED
    assert "Goal blocked: no API key" in _output(repl_console)


@pytest.mark.anyio
async def test_the_loop_stops_on_a_provider_error(session, monkeypatch, repl_console):
    session.goal = new_goal("x", max_rounds=5)
    prompts = _rounds(monkeypatch, lambda n, session: "provider_error")

    await repl._drive_goal(_FakeAgent(), None, session)

    assert len(prompts) == 1
    assert session.goal.status == PAUSED
    assert "the provider returned an error" in _output(repl_console)


@pytest.mark.anyio
async def test_clearing_the_goal_during_a_round_ends_the_loop(session, monkeypatch, repl_console):
    session.goal = new_goal("x", max_rounds=5)

    def behaviour(n, session):
        session.goal = None  # what a deferred /goal clear does
        return "stop"

    prompts = _rounds(monkeypatch, behaviour)

    assert await repl._drive_goal(_FakeAgent(), None, session) is False
    assert len(prompts) == 1


@pytest.mark.anyio
async def test_a_goal_that_is_not_due_runs_nothing(session, monkeypatch, repl_console):
    session.goal = new_goal("x")
    session.goal.due = False
    prompts = _rounds(monkeypatch, lambda n, session: "stop")

    assert await repl._drive_goal(_FakeAgent(), None, session) is False
    assert prompts == []


@pytest.mark.anyio
async def test_ctrl_c_during_a_round_pauses_the_goal(session, monkeypatch, repl_console):
    """A real round, interrupted at the steering prompt, must pause the goal."""

    class _Interrupting:
        async def prompt_async(self, *_args, **_kwargs):
            raise KeyboardInterrupt

    class _HangingAgent:
        last_stop_reason: str | None = None

        async def run_turn(self, _line):
            await asyncio.Event().wait()

    session.goal = new_goal("x", max_rounds=5)
    agent = _HangingAgent()

    exit_requested = await repl._run_turn_with_steering(
        agent, _Interrupting(), session, "round"
    )

    assert exit_requested is False
    assert agent.last_stop_reason == "interrupted"

    finish_round(session.goal, agent.last_stop_reason)
    assert session.goal.status == PAUSED
    assert session.goal.outcome == "you interrupted it"


def test_an_opening_prompt_names_every_report_status():
    prompt = opening_prompt(new_goal("x"))

    for status in ("in_progress", "achieved", "blocked"):
        assert f'"{status}"' in prompt


@pytest.mark.anyio
async def test_a_goal_typed_at_the_prompt_runs_to_completion(session, monkeypatch):
    """End to end through the real REPL: /goal at the prompt, two rounds, the
    model reports achieved through update_goal, then Ctrl-D at the prompt."""
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    out = io.StringIO()
    monkeypatch.setattr(repl, "console", Console(file=out, force_terminal=False, width=200))

    async def no_update_check(_session):
        return None

    monkeypatch.setattr(repl, "_check_update_background", no_update_check)

    rounds = [
        # round 1: the model stops without reporting, so the goal goes on
        [TextDelta(text="looking"), Done(finish_reason="stop")],
        # round 2: it reports the goal achieved, then finishes the turn
        [
            ToolCallDelta(
                index=0,
                id="g1",
                name="update_goal",
                args_fragment=json.dumps({"status": "achieved", "summary": "all green"}),
            ),
            Done(finish_reason="tool_calls"),
        ],
        [TextDelta(text="done"), Done(finish_reason="stop")],
    ]

    # Ctrl-D is sent only once the goal's rounds have returned, so no steering
    # prompt is still reading the pipe when it arrives.
    finished = asyncio.Event()
    real_drive = repl._drive_goal

    async def watched_drive(agent, pt_session, sess):
        result = await real_drive(agent, pt_session, sess)
        if sess.goal is not None and sess.goal.status == ACHIEVED:
            finished.set()
        return result

    monkeypatch.setattr(repl, "_drive_goal", watched_drive)

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=DummyOutput()),
    ):
        pipe.send_text("/goal finish the task\r")

        async def end_at_the_prompt():
            # Bounded, so a broken loop fails the assertions below instead of hanging:
            # the REPL treats cancellation at the prompt like Ctrl-C and keeps going.
            try:
                await asyncio.wait_for(finished.wait(), timeout=10)
            except asyncio.TimeoutError:  # not the builtin on Python 3.10
                pass
            pipe.send_text("\x04")

        ender = asyncio.create_task(end_at_the_prompt())
        try:
            with patch(
                "kiwimatecoder.client.UnifiedClient.stream_chat", side_effect=_stream(rounds)
            ):
                await repl._run_interactive(session)
        finally:
            ender.cancel()

    assert session.goal is not None
    assert session.goal.status == ACHIEVED
    assert session.goal.outcome == "all green"
    user_turns = [m["content"] for m in session.messages if m["role"] == "user"]
    assert user_turns[0].startswith("You are working toward a goal")
    assert user_turns[1].startswith("Keep working toward the goal (round 2 of 10)")
    assert "Goal achieved after 2 round(s)" in out.getvalue()
