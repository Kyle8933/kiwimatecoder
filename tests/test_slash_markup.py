"""Slash commands outside /config echo typed values literally, not as markup.

test_config_markup.py covers the /config handlers. This runs the same detector
(``StrictConsole``: a Console that checks every print string and table cell at
``render_str``) over the other slash commands, with each hostile value both
typed as an argument and present in state (a file name, a todo, a touched file,
a saved session), then runs every list/show command so that state is displayed.
"""

from __future__ import annotations

import json
import os
import shlex
from types import SimpleNamespace

import pytest

from kiwimatecoder import commands, config, i18n, jobs, tools
from kiwimatecoder import session as session_module
from kiwimatecoder.checkpoints import Checkpoint
from kiwimatecoder.commands import dispatch
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.providers import REGISTRY
from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult
from tests.test_config_markup import HOSTILE, StrictConsole

# Each command with the hostile value as its argument. Left out on purpose:
# /config (its own test), /exit and /quit, /doctor (probes services), /model
# refresh and search (fetch catalogs). /jobs run and tick (they spawn processes),
# /sync and /share gist (network) are only run by the tests below that stand in
# for what they start or fetch.
TYPED = [
    "/checkpoints {V}", "/clear {V}", "/compact {V}", "/context {V}",
    "/context add {V}", "/context remove {V}", "/context remove {V}",
    "/context drop {V}", "/ctx add {V}", "/cost {V}", "/dry-run {V}",
    "/dryrun {V}", "/export {V}", "/files {V}", "/save {V}", "/load {V}",
    "/fork {V}", "/help {V}", "/index {V}", "/jobs {V}", "/jobs show {V}",
    "/jobs cancel {V}", "/lsp {V}", "/mcp {V}", "/media {V}", "/memory {V}",
    "/memory add {V}", "/memory add-user {V}", "/memory clear {V}", "/mode {V}",
    "/model {V}", "/provider {V}", "/sessions {V}", "/share import {V}",
    "/share {V}", "/sync {V}", "/templates {V}", "/todos {V}", "/tools {V}",
    "/undo {V}", "/rewind {V}", "/image {V}", "/video {V}",
    "/video resume {V}", "/nonexistent{V}", "/ {V}",
    # state that typed text alone cannot create (the file is named like the value,
    # so it is only matched by a glob, and then listed, removed, ...)
    "/context add ../{V}", "/context add /nowhere/{V}", "/context add *",
    "/context add **/*", "/context remove *", "/context add **/*",
    "/context remove ../{V}", "/context remove /nowhere/{V}",
    "/share import", "/image", "/video",
    # a saved session whose path contains the value (the home directory is named
    # like it), so the "Loaded session <path>" line echoes it
    "/load {P}",
]
# Run after the state is seeded again (/clear above empties the checkpoints, and
# the context list has been rebuilt), so every list/show command has rows to print.
SHOW = [
    "/checkpoints", "/context", "/cost", "/files", "/help", "/index", "/jobs",
    "/jobs show job1", "/jobs cancel job1", "/lsp", "/mcp", "/media", "/memory",
    "/mode", "/model", "/provider", "/sessions", "/templates", "/todos",
    "/tools", "/video", "/dry-run", "/share", "/sync", "/undo",
    "/index status", "/index build",
]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """Build an isolated home and workspace whose *paths contain* the hostile value.

    Every path the commands echo (the workspace, a saved session, an export, a
    share bundle, the index, a job log) then carries it too.
    """

    registered: list[str] = []

    def build(hostile: str):
        home = tmp_path / f"home{hostile}"
        home.mkdir(parents=True)
        monkeypatch.setattr(config, "CONFIG_DIR", home)
        monkeypatch.setattr(config, "CONFIG_FILE", home / "config.json")
        monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", home / "config")
        monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
        for provider in REGISTRY.values():
            monkeypatch.delenv(provider.key_env, raising=False)
        root = tmp_path / f"ws{hostile}"
        root.mkdir(parents=True)
        monkeypatch.chdir(root)
        i18n.set_locale(i18n.DEFAULT_LOCALE)
        # Free-form config: language servers and the media models are typed by the
        # user. One server's command is runnable (see _populate), the other's is not.
        cfg = config.load_config()
        cfg["lsp"] = {
            "enabled": True,
            "servers": {
                hostile: {"command": hostile, "extensions": [".zz"]},
                f"gone{hostile}": {"command": f"gone{hostile}", "extensions": [".yy"]},
            },
        }
        cfg["media"] = {
            "enabled": True,
            "provider": "openrouter",
            "model": hostile,
            "video_model": hostile,
        }
        config.save_config(cfg)
        # Media is enabled above, so make sure no test can reach a provider: even
        # without a key the preflight stops it, but do not depend on that.
        from kiwimatecoder import media

        def no_request(*_args, **_kwargs):
            raise media.MediaError("no network in tests")

        for name in ("generate_image", "generate_video", "resume_video"):
            monkeypatch.setattr(media, name, no_request)
        _populate(root, hostile, monkeypatch)
        # Plugins register tools with names and descriptions of their own.
        tools.register_tool(
            FunctionTool(
                name=hostile,
                description=f"{hostile}. Second sentence.",
                parameters={"type": "object", "properties": {}},
                func=lambda args, session: ToolResult(content=""),
            ),
            source="test",
        )
        registered.append(hostile)
        return root

    yield build
    for name in registered:
        tools.unregister_tool(name)
    i18n.set_locale(i18n.DEFAULT_LOCALE)


def _populate(root, hostile, monkeypatch) -> None:
    """Create the state that the list/show commands then display."""
    # Entries named like the value: a text file, a directory, a binary file and a
    # dangling symlink. A value with a "/" in it (``[/x]``) is a nested path, so
    # create the parents.
    entries = (
        (hostile, lambda path: path.write_text("hello\n")),
        (f"dir{hostile}", lambda path: path.mkdir()),
        (f"bin{hostile}", lambda path: path.write_bytes(b"\x00\x01\x02")),
        (f"link{hostile}", lambda path: path.symlink_to(root / f"gone{hostile}")),
    )
    for name, make in entries:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        make(target)
    # The text file doubles as the runnable language-server command: it is found
    # on PATH (or, for a value with a "/", relative to the working directory).
    (root / hostile).chmod(0o755)
    monkeypatch.setenv("PATH", f"{root}{os.pathsep}{os.environ['PATH']}")
    from kiwimatecoder.lsp import manager as lsp_manager

    monkeypatch.setattr(
        lsp_manager,
        "_MANAGER",
        SimpleNamespace(clients={hostile: object()}, shutdown=lambda: None),
    )
    session_module.save_session(_session(root, hostile), "ok")
    # A saved session named like the value (a "/" cannot be part of a file name).
    if "/" not in hostile:
        saved = session_module._sessions_dir() / f"{hostile}.json"
        saved.write_text(json.dumps(_session(root, hostile).to_dict()), encoding="utf-8")
    # A finished background job whose fields are all the value.
    jobs.save_job(
        jobs.JobRecord(
            id="job1",
            prompt=hostile,
            workspace=hostile,
            provider=hostile,
            model=hostile,
            mode=hostile,
            status="succeeded",
            created_at="2024-01-01T00:00:00",
            finished_at="2024-01-01T00:00:01",
            exit_code=0,
            result=hostile,
            error=hostile,
            output_path=str(jobs._output_path("job1")),
        )
    )


def _session(root, hostile) -> Session:
    session = Session(
        provider_id="openrouter",
        model=hostile,
        mode=PermissionMode.ASK,
        workspace_root=root,
    )
    _seed(session, hostile)
    return session


def _seed(session, hostile) -> None:
    """Give the session the state that /clear and other commands drop."""
    session.touched_files = [hostile, f"../{hostile}"]
    session.todos = [{"content": hostile, "status": "pending"}]
    session.checkpoints = [
        Checkpoint(id=1, label=hostile, created_at="2024-01-01T00:00:00", files={hostile: None}),
    ]
    # Pinned files as a hand-edited session might hold them: present, missing, a
    # directory, binary, and outside the workspace.
    session.context_files = [
        hostile, f"gone{hostile}", f"dir{hostile}", f"bin{hostile}", f"../{hostile}",
    ]


@pytest.mark.parametrize("hostile", HOSTILE)
def test_slash_commands_echo_typed_values_literally(hostile, setup):
    workspace = setup(hostile)
    session = _session(workspace, hostile)
    console = StrictConsole(hostile)
    crashes: list[str] = []
    problems: dict[str, str] = {}

    def run(command: str) -> None:
        console.problems.clear()
        try:
            dispatch(command, session, console)
        except Exception as exc:  # a handler crashing is what we look for
            crashes.append(f"{command.replace(hostile, '<H>')} -> {type(exc).__name__}")
        for problem in console.problems:
            problems.setdefault(command.replace(hostile, "<H>"), problem[:140])

    # Both shell-quoted (for the handlers that shlex.split their argument) and as
    # typed (most take the rest of the line as is). A backslash typed raw would be
    # eaten by shlex, so that handler would echo a different value than the one the
    # detector looks for; the quoted form keeps it.
    arguments = [shlex.quote(hostile)]
    if "\\" not in hostile:
        arguments.append(hostile)
    saved = session_module._sessions_dir() / "ok.json"
    for argument in arguments:
        for template in TYPED:
            run(template.format(V=argument, P=saved))
    _seed(session, hostile)
    for command in SHOW:
        run(command)

    assert not crashes, "commands crashed:\n" + "\n".join(crashes)
    assert not problems, "echoed without escaping:\n" + "\n".join(
        f"  {command}: {text}" for command, text in sorted(problems.items())
    )


def test_the_sweep_covers_every_command_it_does_not_exclude():
    # A command added later must be run here or knowingly left out.
    left_out = {
        "config", "exit", "quit", "doctor",
    }
    covered = {template.split()[0].lstrip("/") for template in TYPED}
    covered |= {name.lstrip("/") for name in SHOW}
    assert set(commands._COMMANDS) - left_out <= covered | {"ctx"}


@pytest.mark.parametrize(
    ("command", "hint"),
    [
        ("/memory bogus", "[list|add <text>|add-user <text>|clear project|user]"),
        ("/index bogus", "[status|build|clear]"),
        ("/sync bogus", "[status|push|pull]"),
        ("/share bogus", "/share [gist]"),
        ("/share import", "/share [gist]"),
    ],
)
def test_usage_messages_keep_their_argument_hints(command, hint, setup):
    # "[list|add <text>|...]" was parsed as a markup tag and silently dropped, so
    # the usage line read just "Usage: /memory".
    session = _session(setup("x"), "x")
    console = StrictConsole("unused")

    dispatch(command, session, console)

    assert hint in console.file.getvalue()


# Handlers report a failure with the exception text, which often quotes a path or
# a name. Make each call fail with the hostile value in the message, since the
# real failures (a full disk, a bad share bundle) cannot be provoked on demand.
def _fail(exc_type):
    def raiser(hostile):
        def raise_it(*_args, **_kwargs):
            raise exc_type(f"boom {hostile}")

        return raise_it

    return raiser


FAILURES = [
    ("/save x", commands, "save_session", _fail(OSError)),
    ("/fork x", commands, "fork_session", _fail(OSError)),
    ("/load x", commands, "load_session", _fail(ValueError)),
    ("/export x", commands, "export_session_markdown", _fail(OSError)),
]


@pytest.mark.parametrize("hostile", HOSTILE)
@pytest.mark.parametrize(
    ("command", "owner", "name", "make"), FAILURES, ids=[f[0] for f in FAILURES]
)
def test_a_reported_failure_keeps_the_error_text_literally(
    hostile, command, owner, name, make, setup, monkeypatch
):
    session = _session(setup(hostile), hostile)
    console = StrictConsole(hostile)
    monkeypatch.setattr(owner, name, make(hostile))

    dispatch(command, session, console)

    assert not console.problems, console.problems
    assert f"boom {hostile}" in console.file.getvalue()


@pytest.mark.parametrize("hostile", HOSTILE)
def test_share_sync_and_index_failures_keep_the_error_text_literally(
    hostile, setup, monkeypatch
):
    from kiwimatecoder import share, sync
    from kiwimatecoder.index import store

    session = _session(setup(hostile), hostile)
    console = StrictConsole(hostile)

    def fail(exc_type):
        def raise_it(*_args, **_kwargs):
            raise exc_type(f"boom {hostile}")

        return raise_it

    monkeypatch.setattr(share, "write_share", fail(OSError))
    monkeypatch.setattr(sync, "push", fail(sync.SyncError))
    monkeypatch.setattr(
        sync,
        "pull",
        lambda *a, **k: SimpleNamespace(
            lines=[], errors=[f"boom {hostile}"], summary=lambda: "done"
        ),
    )
    monkeypatch.setattr(store, "index_status", fail(OSError))

    for command in ("/share", "/sync push", "/sync pull", "/index status"):
        console.problems.clear()
        dispatch(command, session, console)
        assert not console.problems, (command, console.problems)

    assert console.file.getvalue().count(f"boom {hostile}") == 4


@pytest.mark.parametrize("hostile", HOSTILE)
def test_job_and_gist_reports_keep_their_text_literally(hostile, setup, monkeypatch):
    # Starting and ticking jobs spawns processes, and a gist needs the network, so
    # stand in for them; what they return or raise is what gets echoed.
    from kiwimatecoder import share

    session = _session(setup(hostile), hostile)
    console = StrictConsole(hostile)
    record = jobs.JobRecord(
        id="job2", prompt=hostile, workspace=hostile, status="running", created_at=""
    )

    def due(*, now=None, skipped=None):
        if skipped is not None:
            skipped.append((record, f"boom {hostile}"))
        return [record]

    def start(*_args, **_kwargs):
        raise jobs.JobError(f"boom {hostile}")

    monkeypatch.setattr(jobs, "run_due_jobs", due)
    monkeypatch.setattr(jobs, "start_job", start)
    monkeypatch.setattr(share, "share_to_gist", lambda *a, **k: (False, f"boom {hostile}"))
    run = [("/jobs tick", 2), ("/jobs run x", 1), ("/share gist", 1)]

    for command, shown in run:
        console.problems.clear()
        console.file.truncate(0)
        console.file.seek(0)
        dispatch(command, session, console)
        assert not console.problems, (command, console.problems)
        assert console.file.getvalue().count(hostile) >= shown, command

    monkeypatch.setattr(jobs, "start_job", lambda *a, **k: record)
    monkeypatch.setattr(share, "share_to_gist", lambda *a, **k: (True, f"https://g/{hostile}"))
    for command in ("/jobs run x", "/share gist"):
        console.problems.clear()
        dispatch(command, session, console)
        assert not console.problems, (command, console.problems)


@pytest.mark.parametrize("hostile", HOSTILE)
def test_video_progress_and_model_names_are_shown_literally(hostile, setup, monkeypatch):
    # The spinner text carries what the provider reports while a video renders.
    from kiwimatecoder import media

    session = _session(setup(hostile), hostile)
    console = StrictConsole(hostile)

    def render(_prompt, *, progress=None, **_kwargs):
        progress(media.VideoProgress("job-1", hostile, 1.0, None))
        raise media.MediaError(f"boom {hostile}")

    monkeypatch.setattr(media, "generate_video", render)

    dispatch("/video a paper boat", session, console)

    assert not console.problems, console.problems
    assert f"boom {hostile}" in console.file.getvalue()


@pytest.mark.parametrize("hostile", HOSTILE)
def test_a_hand_edited_media_provider_is_shown_literally(hostile, setup):
    # `config media provider` checks its value, but the file can be edited by hand.
    session = _session(setup(hostile), hostile)
    cfg = config.load_config()
    cfg["media"]["provider"] = hostile
    config.save_config(cfg)
    console = StrictConsole(hostile)

    for command in ("/image", "/image a fox", "/video", "/video a fox", "/media"):
        console.problems.clear()
        dispatch(command, session, console)
        assert not console.problems, (command, console.problems)


@pytest.mark.parametrize("hostile", HOSTILE)
def test_loading_a_session_that_names_a_hostile_provider(hostile, setup):
    # A saved session is a file anyone can edit; /load echoes the provider it names.
    session = _session(setup(hostile), hostile)
    data = session.to_dict()
    data["provider_id"] = hostile
    data["active_provider_ids"] = [hostile]
    saved = session_module._sessions_dir() / "edited.json"
    saved.write_text(json.dumps(data), encoding="utf-8")
    console = StrictConsole(hostile)

    dispatch(f"/load {saved}", session, console)

    assert not console.problems, console.problems
    assert f"provider={hostile}:{hostile}" in console.file.getvalue()
