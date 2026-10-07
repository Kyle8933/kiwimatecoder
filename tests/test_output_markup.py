"""What the agent loop, the REPL and their helpers print is shown literally.

test_config_markup.py, test_cli_markup.py and test_slash_markup.py cover the
commands. This covers the rest of what reaches the terminal: the line printed for
every tool call, the approval prompt, errors from the provider, hooks, plugins,
MCP servers, the updater and the banner. Rich reads ``[...]`` in a string as
markup, so ``read_file app/blog/[slug]/page.tsx`` used to be shown as
``app/blog//page.tsx``, ``bash `ls [a-z]*` `` as ``ls *``, and a pytest id such as
``test_a[/tmp/x]`` raised ``MarkupError``. The approval prompt is the one that
matters most: the user answers it by reading the command, so it has to be the
command that will run.

The detector is ``StrictConsole`` (it checks every print string, table cell and
spinner label for the hostile value). Lines the agent builds with
``Text.from_markup`` bypass it, so those tests also look for the exact text in
the output.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import shlex
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console
from rich.text import Text

from kiwimatecoder import ai, config, diagnostics, hooks, plugins, repl, updater
from kiwimatecoder.agent import Agent
from kiwimatecoder.client import (
    Done,
    ProviderError,
    TextDelta,
    ToolCallDelta,
)
from kiwimatecoder.diagnostics import Check
from kiwimatecoder.mcp.manager import McpLoadResult, McpManager
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.providers import REGISTRY
from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult
from tests.test_config_markup import HOSTILE, StrictConsole, path_safe


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "config-home"
    monkeypatch.setattr(config, "CONFIG_DIR", home)
    monkeypatch.setattr(config, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", home / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for provider in REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)


def _session(tmp_path, mode=PermissionMode.AUTO) -> Session:
    return Session(
        provider_id="openrouter",
        model="test-model",
        mode=mode,
        workspace_root=tmp_path,
    )


# --------------------------------------------------------------------------- #
# Tool-call summaries (the line printed for every tool the model runs)
# --------------------------------------------------------------------------- #

# Each value in these arguments comes from the model.
SUMMARY_CALLS = [
    ("read_file", {"path": "{H}"}),
    ("write_file", {"path": "{H}"}),
    ("edit_file", {"path": "{H}"}),
    ("list_dir", {"path": "{H}"}),
    ("view_image", {"path": "{H}"}),
    ("generate_image", {"prompt": "{H}"}),
    ("generate_video", {"prompt": "{H}"}),
    ("generate_video", {"resume_job_id": "{H}"}),
    ("search", {"pattern": "{H}", "mode": "{H}"}),
    ("run_bash", {"command": "{H}"}),
    ("shell", {"command": "{H}"}),
    ("shell_jobs", {"action": "{H}", "command": "{H}"}),
    ("ask_user", {"question": "{H}"}),
    ("load_skill", {"name": "{H}"}),
    ("web_fetch", {"url": "{H}"}),
    ("web_search", {"query": "{H}"}),
    ("git", {"action": "{H}"}),
    ("forge_write", {"action": "{H}"}),
    ("recall", {"query": "{H}"}),
    ("write_clipboard", {"text": "{H}"}),
    ("http_get", {"url": "{H}"}),
    ("http_request", {"url": "{H}", "method": "{H}"}),
    ("lsp_definition", {"path": "{H}"}),
    ("lsp_references", {"path": "{H}"}),
    ("task", {"description": "{H}"}),
    ("browser", {"action": "{H}", "url": "{H}"}),
    ("browser", {"action": "click", "selector": "{H}"}),
    ("{H}", {}),  # a tool a plugin registered under that name
]


def _fill(args: dict, hostile: str) -> dict:
    return {key: value.replace("{H}", hostile) for key, value in args.items()}


@pytest.mark.parametrize("hostile", HOSTILE)
@pytest.mark.parametrize("name", [call[0] for call in SUMMARY_CALLS], ids=str)
def test_a_call_summary_keeps_the_values_the_model_passed(name, hostile, tmp_path):
    agent = Agent(_session(tmp_path), Console(quiet=True), MagicMock())

    for template_name, args in SUMMARY_CALLS:
        if template_name != name:
            continue
        for padding in ("", "y" * 100):  # a long value is cut short, never mid-escape
            summary = agent._format_call_summary(
                name.replace("{H}", hostile), _fill(args, hostile + padding)
            )
            plain = Text.from_markup(summary).plain
            assert hostile in plain, (summary, plain)


def _stream(rounds):
    """Replace the model with ``rounds`` of events, one list per request."""
    calls = 0

    async def stream_chat(*_args, **_kwargs):
        nonlocal calls
        events = rounds[min(calls, len(rounds) - 1)]
        calls += 1
        for event in events:
            yield event

    return stream_chat


def _exits_with(code: int, hostile: str) -> str:
    """A hook command that exits with ``code`` in sh and in cmd.exe, and carries the value."""
    return f'"{sys.executable}" -c "import sys; sys.exit({code})" {shlex.quote(hostile)}'


def _tool_round(name, args):
    return [
        ToolCallDelta(index=0, id="c1", name=name, args_fragment=json.dumps(args)),
        Done(finish_reason="tool_calls"),
    ]


_DONE = [TextDelta(text="done"), Done(finish_reason="stop")]


async def _run_turn(agent, rounds, prompt="go"):
    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch("kiwimatecoder.client.UnifiedClient.stream_chat", side_effect=_stream(rounds)),
    ):
        await agent.run_turn(prompt)


def _strict_agent(tmp_path, hostile, **session_args):
    console = StrictConsole(hostile)
    session = _session(tmp_path, **session_args)
    return Agent(session, console, MagicMock(return_value=True)), console


@pytest.mark.anyio
@pytest.mark.parametrize("hostile", HOSTILE)
@pytest.mark.parametrize("succeeds", [True, False], ids=["ok", "failed"])
async def test_the_line_printed_for_a_tool_call_shows_it_literally(
    hostile, succeeds, tmp_path, monkeypatch
):
    result = ToolResult(content="ok") if succeeds else ToolResult.error("nope")
    monkeypatch.setattr(FunctionTool, "execute", lambda self, args, session: result)
    agent, console = _strict_agent(tmp_path, hostile)

    for name, args in (
        ("read_file", {"path": hostile}),
        ("run_bash", {"command": f"echo {shlex.quote(hostile)}"}),
        ("search", {"pattern": hostile}),
    ):
        console.problems.clear()
        console.file.truncate(0)
        console.file.seek(0)
        await _run_turn(agent, [_tool_round(name, args), _DONE])
        output = console.file.getvalue()
        assert not console.problems, (name, console.problems)
        assert hostile in output, (name, output)
        assert ("✓" in output) if succeeds else ("(failed)" in output)


@pytest.mark.anyio
@pytest.mark.parametrize("hostile", HOSTILE)
async def test_a_dry_run_preview_shows_the_path_literally(hostile, tmp_path):
    agent, console = _strict_agent(tmp_path, hostile)
    agent.session.dry_run = True

    await _run_turn(agent, [_tool_round("write_file", {"path": hostile, "content": "x"}), _DONE])

    assert not console.problems, console.problems
    assert f"dry-run write_file {hostile}" in console.file.getvalue()


# A command that the deny rule (the hostile value, used as a regex) matches.
MATCHING_COMMAND = {
    "[/x]": "a/b",
    "[bold]x[/bold]": "bxb",
    "a[/b]c": "abc",
    "[slow]": "s",
    "C:\\[x]": "C:[x]",
}


@pytest.mark.anyio
@pytest.mark.parametrize("hostile", HOSTILE)
async def test_a_blocked_call_names_the_deny_rule_literally(hostile, tmp_path):
    agent, console = _strict_agent(tmp_path, hostile)
    agent.session.command_rules = {"allow": [], "deny": [hostile]}

    await _run_turn(
        agent, [_tool_round("run_bash", {"command": MATCHING_COMMAND[hostile]}), _DONE]
    )

    assert not console.problems, console.problems
    assert f"Blocked by command deny rule '{hostile}'." in console.file.getvalue()


@pytest.mark.anyio
@pytest.mark.parametrize("hostile", HOSTILE)
async def test_a_pre_tool_hook_block_shows_the_hook_command_literally(
    hostile, tmp_path, monkeypatch
):
    monkeypatch.setattr(FunctionTool, "execute", lambda self, args, session: ToolResult(content="ok"))
    command = _exits_with(3, hostile)
    config.add_hook("pre_tool", command)
    agent, console = _strict_agent(tmp_path, hostile)

    await _run_turn(agent, [_tool_round("read_file", {"path": "x"}), _DONE])

    output = console.file.getvalue()
    assert not console.problems, console.problems
    # the line the hook runner prints, with its status in brackets ...
    assert f"hook pre_tool [exit 3]: {command}" in output
    # ... and the reason the call was blocked
    assert f"blocked by pre_tool hook (exit code 3): {command}" in output


@pytest.mark.anyio
@pytest.mark.parametrize("hostile", HOSTILE)
async def test_a_provider_error_is_shown_literally(hostile, tmp_path):
    agent, console = _strict_agent(tmp_path, hostile)

    async def failing(*_args, **_kwargs):
        raise ProviderError(f"boom {hostile}")
        yield Done()

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy_key"),
        patch("kiwimatecoder.client.UnifiedClient.stream_chat", side_effect=failing),
    ):
        await agent.run_turn("hi")

    assert not console.problems, console.problems
    assert f"boom {hostile}" in console.file.getvalue()


@pytest.mark.anyio
@pytest.mark.parametrize("hostile", HOSTILE)
async def test_a_failover_notice_shows_provider_names_and_the_error_literally(
    hostile, tmp_path
):
    config.add_provider("hp1", hostile, "http://localhost:1/v1", "m1")
    config.add_provider("hp2", hostile, "http://localhost:1/v1", "m2")
    session = _session(tmp_path)
    session.provider_id = "hp1"
    session.model = "m1"
    session.set_active_providers(["hp1", "hp2"])
    console = StrictConsole(hostile)
    agent = Agent(session, console, MagicMock())

    class Client:
        def __init__(self, provider_id):
            self.provider_id = provider_id

        async def stream_chat(self, messages, tools, model):
            if self.provider_id == "hp1":
                raise ProviderError(f"boom {hostile}")
            yield TextDelta(text="hi")
            yield Done(finish_reason="stop")

    with patch("kiwimatecoder.agent.Agent._client", side_effect=lambda pid=None: Client(pid)):
        await agent._stream_once()

    assert not console.problems, console.problems
    assert f"{hostile} failed (boom {hostile}); trying {hostile}." in console.file.getvalue()


@pytest.mark.anyio
@pytest.mark.parametrize("hostile", HOSTILE)
async def test_the_routed_model_and_the_auto_verify_command_are_shown_literally(
    hostile, tmp_path, monkeypatch
):
    monkeypatch.setattr(FunctionTool, "execute", lambda self, args, session: ToolResult(content="ok"))
    config.set_model_routing(enabled=True, simple_model=hostile)
    agent, console = _strict_agent(tmp_path, hostile)
    agent.session.verify_command = f"pytest -k {shlex.quote(hostile)}"

    await _run_turn(
        agent,
        [_tool_round("write_file", {"path": "a.txt", "content": "x"}), _DONE],
        prompt="add a docstring",
    )

    output = console.file.getvalue()
    assert not console.problems, console.problems
    assert f"routed to {hostile}" in output
    assert f"Auto-verify: pytest -k {shlex.quote(hostile)}" in output


# --------------------------------------------------------------------------- #
# The REPL: banner, prompt, approval prompt, ask_user, attachments, lifecycle
# --------------------------------------------------------------------------- #


@pytest.fixture
def strict_repl(monkeypatch):
    """Point the REPL's module-level console at a detector, per hostile value."""

    def install(hostile: str) -> StrictConsole:
        strict = StrictConsole(hostile)
        monkeypatch.setattr(repl, "console", strict)
        return strict

    return install


@pytest.mark.parametrize("hostile", HOSTILE)
def test_the_banner_shows_the_provider_model_and_folder_literally(
    hostile, tmp_path, strict_repl
):
    config.add_provider("hp1", hostile, "http://localhost:1/v1", hostile)
    config.add_provider("hp2", hostile, "http://localhost:1/v1", hostile)
    workspace = tmp_path / f"w{path_safe(hostile.replace('/', ''))}"
    workspace.mkdir()
    session = Session(
        provider_id="hp1", model=hostile, mode=PermissionMode.ASK, workspace_root=workspace
    )
    session.set_active_providers(["hp1", "hp2"])
    console = strict_repl(hostile)

    console.print(repl._banner(session))

    output = console.file.getvalue()
    assert not console.problems, console.problems
    assert hostile in output
    assert workspace.name in output


def test_the_prompt_survives_a_model_name_with_xml_characters(tmp_path):
    # prompt_toolkit's HTML() is parsed as XML.
    session = _session(tmp_path)
    session.model = "a<b&c"

    prompt = repl._prompt_text(session)

    assert "a<b&c" in "".join(fragment[1] for fragment in prompt.__pt_formatted_text__())


CONFIRM_CASES = [
    # (summary as permissions.gate builds it, preview text)
    ("run_bash(command={cmd!r})", None),
    ("run_bash(command={cmd!r})", "ls -l"),
    ("write_file(path={cmd!r})", "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new\n"),
    ("{cmd}(path='x')", None),  # a plugin's tool, named like the value
]


@pytest.mark.parametrize("hostile", HOSTILE)
@pytest.mark.parametrize(("template", "preview"), CONFIRM_CASES)
@pytest.mark.parametrize("answer", ["y", "n", "a"])
def test_the_approval_prompt_shows_exactly_what_will_run(
    hostile, template, preview, answer, tmp_path, strict_repl
):
    console = strict_repl(hostile)
    console.input = lambda *_a, **_k: answer  # type: ignore[method-assign]
    summary = template.format(cmd=f"ls {hostile}")
    session = _session(tmp_path, PermissionMode.ASK)

    repl._make_confirm(session)(summary, preview)

    output = console.file.getvalue()
    assert not console.problems, console.problems
    assert summary in output  # the whole summary, character for character
    assert "Approve" in output


@pytest.mark.parametrize("hostile", HOSTILE)
def test_the_ask_user_panel_shows_the_question_and_options_literally(hostile, strict_repl):
    console = strict_repl(hostile)
    console.input = lambda *_a, **_k: "1"  # type: ignore[method-assign]

    chosen = repl._make_ask_user(console)(f"Use {hostile}?", [hostile, f"{hostile}2"])

    output = console.file.getvalue()
    assert chosen == hostile
    assert not console.problems, console.problems
    assert f"Use {hostile}?" in output
    assert f"1. {hostile}" in output
    assert f"2. {hostile}2" in output


@pytest.mark.parametrize("hostile", HOSTILE)
def test_image_attachment_notes_show_file_names_literally(
    hostile, tmp_path, strict_repl, monkeypatch
):
    console = strict_repl(hostile)
    session = _session(tmp_path)
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    safe = path_safe(hostile)
    names = [f"a{safe}.png", f"b{safe}.png", f"c{safe}.png"]
    for name in names:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(png)
    bad = f"bad{path_safe(hostile)}.png"
    (tmp_path / bad).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / bad).write_bytes(b"not an image")
    monkeypatch.setattr(
        repl, "get_vision", lambda: {"max_images_per_turn": 2, "max_image_bytes": 10_000}
    )

    line = " ".join(f"@{name}" for name in (*names, bad))
    repl._attach_images(line, session)

    output = console.file.getvalue()
    assert not console.problems, console.problems
    # only the file's own name is shown (what follows the last "/" of the value)
    assert f"Attached images: {Path(names[0]).name}, {Path(names[1]).name}" in output
    assert f"{Path(names[2]).name} was not attached" in output


@pytest.mark.parametrize("hostile", HOSTILE)
def test_an_unreadable_image_error_shows_the_name_literally(
    hostile, tmp_path, strict_repl, monkeypatch
):
    console = strict_repl(hostile)
    session = _session(tmp_path)
    bad = f"bad{path_safe(hostile)}.png"
    (tmp_path / bad).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / bad).write_bytes(b"not an image")
    monkeypatch.setattr(
        repl, "get_vision", lambda: {"max_images_per_turn": 2, "max_image_bytes": 10_000}
    )

    repl._attach_images(f"@{bad}", session)

    assert not console.problems, console.problems
    assert f"{(tmp_path / bad).name}" in console.file.getvalue().replace("\n", "")


@pytest.mark.parametrize("hostile", HOSTILE)
def test_deferred_commands_are_echoed_literally(hostile, tmp_path, strict_repl):
    console = strict_repl(hostile)
    session = _session(tmp_path)
    session.deferred_commands.append(f"/nonexistent{hostile}")

    asyncio.run(repl._process_deferred_commands(session))

    assert not console.problems, console.problems
    assert f"→ /nonexistent{hostile}" in console.file.getvalue()


@pytest.mark.parametrize("hostile", HOSTILE)
def test_session_hooks_report_their_command_literally(hostile, tmp_path, strict_repl):
    from kiwimatecoder import events

    command = _exits_with(2, hostile)
    config.add_hook("session_start", command)
    console = strict_repl(hostile)

    repl._run_lifecycle_hooks(events.EventBus(), "session_start", _session(tmp_path))

    output = console.file.getvalue()
    assert not console.problems, console.problems
    assert f"hook session_start [exit 2]: {command}" in output
    assert f"Hook session_start failed (exit 2): {command}" in output


@pytest.mark.parametrize("hostile", HOSTILE)
def test_a_session_hook_that_raises_is_reported_literally(hostile, tmp_path, strict_repl):
    class Bus:
        def emit(self, *_args, **_kwargs):
            raise RuntimeError(f"boom {hostile}")

    console = strict_repl(hostile)

    repl._run_lifecycle_hooks(Bus(), "session_end", _session(tmp_path))  # type: ignore[arg-type]

    assert not console.problems, console.problems
    assert f"Hook error during session_end: boom {hostile}" in console.file.getvalue()


@pytest.mark.parametrize("hostile", HOSTILE)
def test_a_plugin_that_fails_to_load_is_reported_literally(hostile, tmp_path, strict_repl):
    from kiwimatecoder import events

    stem = path_safe(hostile.replace("/", ""))
    plugin_dir = config.ensure_config_dir() / plugins.PLUGINS_DIR_NAME
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / f"{stem}.py").write_text(f"raise RuntimeError({hostile!r})\n")
    console = strict_repl(hostile)

    result = repl._load_session_plugins(_session(tmp_path), events.EventBus())

    assert result.failed, "the plugin should have failed to load"
    assert not console.problems, console.problems
    assert f"Plugin '{stem}' failed to load: import failed: {hostile}" in console.file.getvalue()


# --------------------------------------------------------------------------- #
# Hooks, plugins, MCP, diagnostics, ask, updater
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("hostile", HOSTILE)
def test_the_hook_runner_prints_its_status_in_brackets(hostile):
    # "[ok]" was read as a style tag and dropped, so the status never showed.
    console = StrictConsole(hostile)
    command = f"echo {shlex.quote(hostile)}"

    result = hooks._run_one("post_tool", command, Path.cwd(), {}, console)

    assert result.ok
    assert not console.problems, console.problems
    assert f"hook post_tool [ok]: {command}" in console.file.getvalue()


@pytest.mark.parametrize("hostile", HOSTILE)
def test_plugin_log_lines_show_the_plugin_and_message_literally(hostile):
    console = StrictConsole(hostile)

    plugins.PluginAPI(hostile, console=console).log(f"hello {hostile}")

    assert not console.problems, console.problems
    assert f"plugin {hostile}: hello {hostile}" in console.file.getvalue()


@pytest.mark.parametrize("hostile", HOSTILE)
def test_an_mcp_server_failure_is_shown_literally(hostile):
    console = StrictConsole(hostile)

    McpManager(console=console)._record_failure(
        hostile, RuntimeError(f"boom {hostile}"), McpLoadResult()
    )

    assert not console.problems, console.problems
    assert f"MCP server '{hostile}' failed: boom {hostile}" in console.file.getvalue()


@pytest.mark.parametrize("hostile", HOSTILE)
def test_diagnostics_show_check_names_and_details_literally(hostile):
    console = StrictConsole(hostile)

    diagnostics.render([Check(hostile, diagnostics.WARN, f"at {hostile}")], console)

    assert not console.problems, console.problems
    assert f"at {hostile}" in console.file.getvalue()


@pytest.mark.anyio
@pytest.mark.parametrize("hostile", HOSTILE)
async def test_ask_shows_provider_errors_and_missing_model_messages_literally(hostile):
    config.add_provider("hp1", hostile, "http://localhost:1/v1")
    provider = config.get_provider_config("hp1")
    console = StrictConsole(hostile)

    async def failing(*_args, **_kwargs):
        raise ProviderError(f"boom {hostile}")
        yield Done()

    with (
        patch.object(ai, "console", console),
        patch("kiwimatecoder.client.UnifiedClient.stream_chat", side_effect=failing),
    ):
        await ai.stream_response("hi", "key", model=None, provider=provider)
        await ai.stream_response("hi", "key", model="m", provider=provider)

    output = console.file.getvalue()
    assert not console.problems, console.problems
    assert f"boom {hostile}" in output
    assert hostile in output.split("boom")[0]  # the missing-model message names it


@pytest.mark.parametrize("hostile", HOSTILE)
def test_updater_commands_and_errors_are_shown_literally(hostile):
    console = StrictConsole(hostile)

    code = updater._run([hostile], console)  # no such program: FileNotFoundError

    output = console.file.getvalue().replace("\n", "")
    assert code == 1
    assert not console.problems, console.problems
    assert "Could not start command:" in output
    if sys.platform != "win32":  # Windows words the error differently ([WinError 2] ...)
        assert f"[Errno 2] No such file or directory: {hostile!r}" in output


def _stub_checkout(monkeypatch, root, *, branch, checkout_code, shas):
    sequence = iter(shas)
    monkeypatch.setattr(updater, "find_source_root", lambda *a, **k: root)
    monkeypatch.setattr(updater, "_has_git_remote", lambda _root: True)
    monkeypatch.setattr(updater, "_get_short_sha", lambda _root: next(sequence, "ccccccc"))
    monkeypatch.setattr(updater, "_fetch", lambda _root, _console: 0)
    monkeypatch.setattr(updater, "_get_branch", lambda _root: branch)
    monkeypatch.setattr(updater, "_commits_behind", lambda _root, _branch: 0)
    monkeypatch.setattr(updater, "_run", lambda command, console: checkout_code)


UPDATE_SCENARIOS = {
    # name: (branch is the ref?, checkout exit code, shas, text that must be shown)
    "updated": (False, 0, ["aaaaaaa", "bbbbbbb"], "Updated aaaaaaa → bbbbbbb (ref {H})."),
    "no-new-sha": (False, 0, ["aaaaaaa", None], "Updated aaaaaaa → {H} (ref {H})."),
    "checkout-failed": (False, 1, ["aaaaaaa", "bbbbbbb"], "Update failed while checking out {H}."),
    "same-commit": (False, 0, ["aaaaaaa", "aaaaaaa"], "Already on {H} (commit aaaaaaa)."),
    "already-on-branch": (True, 0, ["aaaaaaa"], "Already on {H} (commit aaaaaaa)."),
}


@pytest.mark.parametrize("hostile", HOSTILE)
@pytest.mark.parametrize("scenario", UPDATE_SCENARIOS)
def test_updating_to_a_ref_shows_the_ref_and_checkout_path_literally(
    hostile, scenario, tmp_path, monkeypatch
):
    on_branch, checkout_code, shas, expected = UPDATE_SCENARIOS[scenario]
    root = tmp_path / f"src{path_safe(hostile)}"
    root.mkdir(parents=True)
    _stub_checkout(
        monkeypatch,
        root,
        branch=hostile if on_branch else "main",
        checkout_code=checkout_code,
        shas=shas,
    )
    console = StrictConsole(hostile)

    updater.run_update(console, ref=hostile)

    output = console.file.getvalue()
    assert not console.problems, console.problems
    assert f"Target ref: {hostile}" in output
    assert f"Detected source checkout: {root}" in output
    assert expected.replace("{H}", hostile) in output
    if not on_branch:
        assert f"Checking out {hostile} from aaaaaaa" in output


@pytest.mark.parametrize("hostile", HOSTILE)
def test_updating_the_current_branch_shows_the_branch_literally(hostile, tmp_path, monkeypatch):
    # No ref: the branch is pulled, and the message names how far behind it is.
    root = tmp_path / "src"
    root.mkdir()
    _stub_checkout(monkeypatch, root, branch=hostile, checkout_code=0, shas=["aaaaaaa", "bbbbbbb"])
    monkeypatch.setattr(updater, "_commits_behind", lambda _root, _branch: 3)
    console = StrictConsole(hostile)

    updater.run_update(console)

    assert not console.problems, console.problems
    assert f"(3 commit(s) behind origin/{hostile})" in console.file.getvalue()


@pytest.mark.parametrize("hostile", HOSTILE)
@pytest.mark.parametrize("outcome", ["oserror", "exit-1"])
def test_the_fetch_step_shows_its_command_and_error_literally(
    hostile, outcome, tmp_path, monkeypatch
):
    root = tmp_path / f"src{path_safe(hostile)}"
    root.mkdir(parents=True)

    def fake_run(command, **_kwargs):
        if outcome == "oserror":
            raise OSError(f"boom {hostile}")
        return MagicMock(returncode=1)

    monkeypatch.setattr(updater.subprocess, "run", fake_run)
    console = StrictConsole(hostile)

    assert updater._fetch(root, console) == 1

    output = console.file.getvalue()
    assert not console.problems, console.problems
    assert shlex.join(["git", "-C", str(root), "fetch", "origin"]) in output.replace("\n", "")
    if outcome == "oserror":
        assert f"Could not fetch from origin: boom {hostile}" in output


@pytest.mark.parametrize("hostile", HOSTILE)
def test_a_failed_pip_update_shows_the_ref_in_its_install_command(hostile, monkeypatch):
    monkeypatch.setattr(updater, "find_source_root", lambda *a, **k: None)
    monkeypatch.setattr(updater, "_run", lambda command, console: 3)
    console = StrictConsole(hostile)

    code = updater.run_update(console, ref=hostile)

    assert code == 3
    assert not console.problems, console.problems
    assert f"@{hostile}" in console.file.getvalue().replace("\n", "")


def test_the_escaped_lines_are_unchanged_for_ordinary_values(tmp_path):
    # Benign text goes through escape() untouched, so nothing else about the
    # output moved: same words, same order.
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=200)
    agent = Agent(_session(tmp_path), console, MagicMock())

    summary = agent._format_call_summary("run_bash", {"command": "ls -la src/app"})

    assert summary == "bash [dim]`ls -la src/app`[/dim]"
    assert re.fullmatch(r"read_file \[dim\]a/b\.py\[/dim\]", agent._format_call_summary("read_file", {"path": "a/b.py"}))
