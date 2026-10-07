from __future__ import annotations

import ctypes
import io
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from kiwimatecoder import config, tools
from kiwimatecoder import shell as shell_module
from kiwimatecoder.shell import (
    MAX_OUTPUT,
    ShellDisabledError,
    ShellError,
    ShellJobLimitError,
    ShellManager,
    _pid_alive,
    close_shell,
    get_shell,
)

TRUNCATION_MARKER = "\n... [output truncated]"


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    """Point config storage (and the jobs dir) at a temp directory."""
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config" / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config" / "legacy")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def close_session_shell(session):
    """Never leak a persistent shell or background job between tests."""
    yield
    close_shell(session)


@pytest.fixture
def manager(tmp_path):
    mgr = ShellManager(tmp_path)
    yield mgr
    mgr.close()


def _is_dir(printed: str, expected: Path, *, windows: bool | None = None) -> bool:
    """Whether the shell's ``pwd`` output names ``expected``.

    Git Bash prints MSYS paths (``/tmp/x`` or ``/c/Users/x``, whatever its mount table
    maps), never ``C:\\...``, so on Windows only the directory name can be compared.
    """
    path = Path(printed.strip())
    if windows is None:
        windows = os.name == "nt"
    if windows:
        return path.name == expected.name
    return path.resolve() == expected.resolve()


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


# ---------------------------------------------------------------------------
# PersistentShell
# ---------------------------------------------------------------------------


def test_cd_persists_across_calls(manager, tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()

    code, output = manager.run("pwd")
    assert code == 0
    assert _is_dir(output, tmp_path)

    code, _ = manager.run("cd sub")
    assert code == 0

    code, output = manager.run("pwd")
    assert code == 0
    assert _is_dir(output, sub)


def test_exported_environment_persists(manager):
    code, _ = manager.run("export KIWI_SHELL_TEST=persisted")
    assert code == 0

    code, output = manager.run('printf "%s" "$KIWI_SHELL_TEST"')
    assert code == 0
    assert output == "persisted"


def test_exit_codes_are_reported(manager):
    code, output = manager.run("echo ok")
    assert (code, output) == (0, "ok")

    code, output = manager.run("(exit 3)")
    assert code == 3
    assert output == ""


def test_timeout_restarts_the_shell(manager):
    started = time.monotonic()
    code, output = manager.run("sleep 30", timeout=0.5)
    elapsed = time.monotonic() - started

    assert code is None
    assert "timed out" in output
    assert elapsed < 10

    # The shell is usable again (fresh process) after a timeout.
    code, output = manager.run("echo recovered")
    assert (code, output) == (0, "recovered")


def test_output_is_truncated(manager, tmp_path):
    (tmp_path / "big.txt").write_text("x" * (MAX_OUTPUT + 5_000))

    code, output = manager.run("cat big.txt")

    assert code == 0
    assert TRUNCATION_MARKER.strip() in output
    assert len(output) <= MAX_OUTPUT + len(TRUNCATION_MARKER)


def test_disabled_persistent_shell_raises(tmp_path):
    mgr = ShellManager(tmp_path, persistent=False)
    try:
        with pytest.raises(ShellDisabledError):
            mgr.shell()
    finally:
        mgr.close()


# ---------------------------------------------------------------------------
# Background processes
# ---------------------------------------------------------------------------


def test_background_start_list_output_kill(manager):
    job_id = manager.start_background("echo background-done")
    job = manager.get_background(job_id)
    assert job is not None
    assert _wait_until(lambda: not job.alive)

    jobs = manager.list_background()
    assert [item.id for item in jobs] == [job_id]
    assert "background-done" in (manager.output_background(job_id) or "")

    assert manager.kill_background(job_id) is True
    assert manager.kill_background(job_id) is False
    assert manager.output_background(job_id) is None


def test_background_kill_stops_a_live_process(manager):
    job_id = manager.start_background("sleep 30")
    job = manager.get_background(job_id)
    assert job is not None and job.alive
    pid = job.pid

    assert manager.kill_background(job_id) is True
    assert _wait_until(lambda: not _pid_alive(pid))


def test_background_job_cap(manager):
    manager.max_jobs = 1
    first = manager.start_background("sleep 30")
    try:
        with pytest.raises(ShellJobLimitError):
            manager.start_background("sleep 30")
    finally:
        manager.kill_background(first)


def test_background_cap_from_config(tmp_path):
    config.set_shell_config(max_jobs=1)
    mgr = ShellManager(tmp_path)
    try:
        assert mgr.max_jobs == 1
        first = mgr.start_background("sleep 30")
        with pytest.raises(ShellJobLimitError):
            mgr.start_background("sleep 30")
        mgr.kill_background(first)
    finally:
        mgr.close()


def test_start_background_requires_a_command(manager):
    with pytest.raises(ShellError):
        manager.start_background("   ")


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def test_get_shell_is_cached_per_session(session):
    first = get_shell(session)
    assert get_shell(session) is first

    code, output = first.run("echo hi")
    assert code == 0
    assert output == "hi"


def test_close_shell_kills_background_jobs(session):
    manager = get_shell(session)
    job_id = manager.start_background("sleep 30")
    job = manager.get_background(job_id)
    assert job is not None
    pid = job.pid

    close_shell(session)

    assert session.shell is None
    assert _wait_until(lambda: not _pid_alive(pid))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_shell_config_defaults_and_roundtrip():
    assert config.get_shell_config() == {
        "persistent": True,
        "timeout": 120,
        "max_jobs": 8,
    }

    updated = config.set_shell_config(persistent=False, timeout=30, max_jobs=2)

    assert updated == {"persistent": False, "timeout": 30, "max_jobs": 2}
    assert config.get_shell_config() == updated


def test_shell_config_tolerates_junk_values():
    cfg = config.load_config()
    cfg["shell"] = {"persistent": "yes", "timeout": "soon", "max_jobs": -5}

    assert config.get_shell_config(cfg) == {
        "persistent": True,
        "timeout": 120,
        "max_jobs": 8,
    }


def test_shell_config_validation_errors():
    with pytest.raises(ValueError):
        config.set_shell_config(timeout=0)
    with pytest.raises(ValueError):
        config.set_shell_config(timeout=99999)
    with pytest.raises(ValueError):
        config.set_shell_config(max_jobs=0)
    with pytest.raises(ValueError):
        config.set_shell_config(persistent="yes")  # type: ignore[arg-type]


def _console():
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def test_config_shell_slash_show_and_set(session):
    from kiwimatecoder.commands import dispatch

    console = _console()
    dispatch("/config shell show", session, console)
    output = console.file.getvalue()
    assert "Persistent shell" in output
    assert "120s" in output

    console = _console()
    dispatch("/config shell timeout 45", session, console)
    assert config.get_shell_config()["timeout"] == 45

    console = _console()
    dispatch("/config shell persistent off", session, console)
    assert config.get_shell_config()["persistent"] is False

    console = _console()
    dispatch("/config shell timeout nope", session, console)
    assert "integer" in console.file.getvalue()


def test_config_shell_cli():
    from kiwimatecoder.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["config", "shell", "show"])
    assert result.exit_code == 0
    assert "Persistent shell" in result.output

    result = runner.invoke(app, ["config", "shell", "max-jobs", "3"])
    assert result.exit_code == 0
    assert config.get_shell_config()["max_jobs"] == 3

    result = runner.invoke(app, ["config", "shell", "timeout", "0"])
    assert result.exit_code == 1


def test_validate_config_flags_bad_shell_section():
    cfg = config.load_config()
    cfg["shell"] = {"persistent": "yes", "timeout": 0, "max_jobs": 0}

    errors = {
        issue["key"]
        for issue in config.validate_config(cfg)
        if issue["level"] == "error"
    }
    assert {"shell.persistent", "shell.timeout", "shell.max_jobs"} <= errors

    cfg["shell"] = ["nope"]
    assert any(
        issue["key"] == "shell" and issue["level"] == "error"
        for issue in config.validate_config(cfg)
    )

    assert config.validate_config(config.load_config()) == []


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_shell_tools_are_registered_and_approval_gated():
    shell_tool = tools.get_tool("shell")
    jobs_tool = tools.get_tool("shell_jobs")

    assert shell_tool is not None and shell_tool.needs_approval
    assert jobs_tool is not None and jobs_tool.needs_approval
    # run_bash is unchanged and still registered.
    assert tools.get_tool("run_bash") is not None


def test_shell_tool_previews(session):
    assert (
        tools.preview("shell", {"command": "cd src && ls"}, session)
        == "cd src && ls"
    )

    start = tools.preview(
        "shell_jobs", {"action": "start", "command": "pytest -q"}, session
    )
    assert "start" in start and "pytest -q" in start

    kill = tools.preview("shell_jobs", {"action": "kill", "id": "abc123"}, session)
    assert "kill" in kill and "abc123" in kill


def test_shell_tool_runs_a_command(session):
    result = tools.dispatch("shell", {"command": "echo from-tool"}, session)

    assert result.ok
    assert "from-tool" in result.content
    assert "[exit code: 0]" in result.content


def test_shell_tool_reports_nonzero_exit(session):
    result = tools.dispatch("shell", {"command": "(exit 7)"}, session)

    assert not result.ok
    assert "[exit code: 7]" in result.content


def test_shell_tool_reports_disabled(session):
    config.set_shell_config(persistent=False)
    session.shell = None

    result = tools.dispatch("shell", {"command": "pwd"}, session)

    assert not result.ok
    assert "disabled" in result.content


def test_shell_jobs_tool_actions(session):
    manager = get_shell(session)
    job_id = manager.start_background("echo job-output")
    job = manager.get_background(job_id)
    assert job is not None
    assert _wait_until(lambda: not job.alive)

    listed = tools.dispatch("shell_jobs", {"action": "list"}, session)
    assert listed.ok and job_id in listed.content

    output = tools.dispatch("shell_jobs", {"action": "output", "id": job_id}, session)
    assert output.ok and "job-output" in output.content

    killed = tools.dispatch("shell_jobs", {"action": "kill", "id": job_id}, session)
    assert killed.ok

    missing = tools.dispatch("shell_jobs", {"action": "output", "id": "ghost"}, session)
    assert not missing.ok
    assert "Unknown" in missing.content


def test_shell_jobs_tool_validation_errors(session):
    assert not tools.dispatch("shell_jobs", {}, session).ok
    assert not tools.dispatch("shell_jobs", {"action": "dance"}, session).ok
    assert not tools.dispatch("shell_jobs", {"action": "start"}, session).ok
    assert not tools.dispatch("shell_jobs", {"action": "kill"}, session).ok


def test_find_windows_bash_never_runs_a_bash_from_the_current_directory(
    monkeypatch, tmp_path
):
    for name in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)", "LOCALAPPDATA", "SystemRoot"):
        monkeypatch.delenv(name, raising=False)
    # shutil.which tries the current directory first on Windows (a hit reads ".\\bash.exe").
    monkeypatch.setattr(shell_module.shutil, "which", lambda name: ".\\bash.exe")
    assert shell_module._find_windows_bash() is None

    # A bash found on PATH (MSYS2, Cygwin, a Git outside Program Files) is still used.
    msys = str(tmp_path / "msys64" / "usr" / "bin" / "bash.exe")
    monkeypatch.setattr(shell_module.shutil, "which", lambda name: msys)
    assert shell_module._find_windows_bash() == msys


# ---------------------------------------------------------------------------
# Windows behaviour, simulated on any platform
# ---------------------------------------------------------------------------


def _fake_windows(monkeypatch):
    monkeypatch.setattr(shell_module, "_is_windows", lambda: True)


def test_shell_argv_uses_git_bash_on_windows(monkeypatch):
    _fake_windows(monkeypatch)
    monkeypatch.setattr(shell_module, "_find_windows_bash", lambda: "C:\\Git\\bin\\bash.exe")

    assert shell_module._shell_argv() == ["C:\\Git\\bin\\bash.exe"]


def test_shell_argv_fails_at_once_without_a_posix_shell_on_windows(monkeypatch, tmp_path):
    _fake_windows(monkeypatch)
    monkeypatch.setattr(shell_module, "_find_windows_bash", lambda: None)
    started: list[object] = []
    monkeypatch.setattr(
        shell_module.subprocess, "Popen", lambda *a, **k: started.append(a)
    )

    with pytest.raises(ShellError, match="Git for Windows"):
        ShellManager(tmp_path).shell()
    assert started == []


def test_find_windows_bash_prefers_git_and_skips_wsl(monkeypatch, tmp_path):
    system_root = tmp_path / "Windows"
    wsl = system_root / "System32" / "bash.exe"
    git = tmp_path / "Program Files" / "Git" / "bin" / "bash.exe"
    for path in (wsl, git):
        path.parent.mkdir(parents=True)
        path.write_text("")
    for name in ("ProgramW6432", "ProgramFiles(x86)", "LOCALAPPDATA"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SystemRoot", str(system_root))
    monkeypatch.setattr(shell_module.shutil, "which", lambda name: str(wsl))

    monkeypatch.setenv("ProgramFiles", str(tmp_path / "Program Files"))
    assert shell_module._find_windows_bash() == str(git)

    # Without Git, the System32 launcher (WSL) must not be mistaken for a usable shell.
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "nowhere"))
    assert shell_module._find_windows_bash() is None


def test_stdin_payload_is_not_crlf_translated_on_windows(monkeypatch):
    class Proc:
        def __init__(self) -> None:
            self.raw = io.BytesIO()
            # newline="\r\n" is what a text-mode pipe does to "\n" on Windows.
            self.stdin = io.TextIOWrapper(
                self.raw, encoding="utf-8", newline="\r\n", write_through=True
            )

    payload = "cd sub\n(exit 3)\n"
    monkeypatch.setattr(shell_module, "_is_windows", lambda: False)
    control = Proc()
    shell_module._write_stdin(control, payload)  # type: ignore[arg-type]
    # The text layer is what breaks commands on Windows: every command ends in "\r".
    assert control.raw.getvalue() == b"cd sub\r\n(exit 3)\r\n"

    _fake_windows(monkeypatch)
    after = Proc()
    shell_module._write_stdin(after, payload)  # type: ignore[arg-type]
    assert after.raw.getvalue() == b"cd sub\n(exit 3)\n"
    assert b"\r" not in after.raw.getvalue()


def test_send_signal_kills_the_whole_tree_on_windows(monkeypatch):
    _fake_windows(monkeypatch)
    killed: list[int] = []
    monkeypatch.setattr(shell_module, "_kill_tree_windows", killed.append)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        shell_module._send_signal(proc, shell_module.signal.SIGTERM)
        assert killed == [proc.pid]
        proc.wait(timeout=5)  # the direct child is also killed as a fallback
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_pid_alive_does_not_use_os_kill_on_windows(monkeypatch):
    _fake_windows(monkeypatch)
    seen: list[int] = []
    monkeypatch.setattr(
        shell_module, "_windows_pid_alive", lambda pid: seen.append(pid) or False
    )

    def boom(*_args):
        raise AssertionError("os.kill(pid, 0) is CTRL_C_EVENT on Windows")

    monkeypatch.setattr(shell_module.os, "kill", boom)

    assert _pid_alive(4321) is False
    assert seen == [4321]
    assert _pid_alive(0) is False


def test_kill_tree_windows_runs_taskkill_and_tolerates_failure(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(argv)
        raise FileNotFoundError("taskkill")

    monkeypatch.setattr(shell_module.subprocess, "run", fake_run)

    shell_module._kill_tree_windows(321)  # must not raise

    assert calls == [["taskkill", "/F", "/T", "/PID", "321"]]


@pytest.mark.parametrize(
    ("handle", "exit_code", "last_error", "expected"),
    [
        (0, None, 87, False),  # no such process (ERROR_INVALID_PARAMETER)
        (0, None, 5, True),  # exists, but belongs to someone else (access denied)
        (7, 259, 0, True),  # STILL_ACTIVE
        (7, 1, 0, False),  # terminated, the Popen handle keeps the object around
    ],
)
def test_windows_pid_alive_reads_the_exit_code(
    monkeypatch, handle, exit_code, last_error, expected
):
    closed: list[int] = []

    class Fn:
        restype = None
        argtypes = None

        def __init__(self, impl):
            self.impl = impl

        def __call__(self, *args):
            return self.impl(*args)

    def get_exit_code(_handle, ref):
        ref._obj.value = exit_code
        return 1

    class FakeKernel32:
        OpenProcess = Fn(lambda _access, _inherit, _pid: handle or None)
        GetExitCodeProcess = Fn(get_exit_code)
        CloseHandle = Fn(lambda h: closed.append(h) or 1)

    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: FakeKernel32, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: last_error, raising=False)

    assert shell_module._windows_pid_alive(4321) is expected
    assert closed == ([7] if handle else [])


def test_is_dir_compares_names_for_msys_paths(tmp_path):
    sub = tmp_path / "sub"
    assert _is_dir("/tmp/pytest-of-x/t0/sub\n", sub, windows=True)
    assert not _is_dir("/tmp/pytest-of-x/t0/other\n", sub, windows=True)
    assert _is_dir(str(sub) + "\n", sub, windows=False)
