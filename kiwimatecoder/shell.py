"""Persistent shell sessions and background process management.

The ``run_bash`` tool starts a fresh process for every call, so ``cd`` and
exported environment variables never survive between commands. This module keeps
one long-lived shell per session instead: commands are written to the shell's
stdin and a unique sentinel line (carrying ``$?``) is echoed after each one, so
output and exit code can be attributed to that exact command.

Background commands are intentionally separate from the persistent shell: each
one runs as a detached process with its stdout/stderr redirected to a file under
``~/.kiwimatecoder/jobs/shell/``, so it keeps running after the shell exits.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kiwimatecoder.config import ensure_config_dir, get_shell_config
from kiwimatecoder.remote import wrap_remote_command
from kiwimatecoder.sandbox import wrap_command

if TYPE_CHECKING:
    from kiwimatecoder.session import Session

MAX_OUTPUT = 30_000
KILL_GRACE_SECONDS = 5.0
_POLL_INTERVAL = 0.05
_SENTINEL_RE = re.compile(r"^__KIWI_([0-9a-f]{12})__\s+(-?\d+)\s*$")


class ShellError(RuntimeError):
    """Raised for persistent-shell and background-job failures."""


class ShellDisabledError(ShellError):
    """Raised when the persistent shell is disabled in config."""


class ShellJobLimitError(ShellError):
    """Raised when starting a background job would exceed the configured cap."""


def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [output truncated]"


def _bounded_output(text: str, truncated: bool) -> str:
    """Bound command output, adding a marker when the reader dropped content."""
    if not truncated:
        return _truncate(text)
    if len(text) > MAX_OUTPUT:
        text = text[:MAX_OUTPUT]
    return text + "\n... [output truncated]"


def _shell_argv() -> list[str]:
    if os.name == "nt":  # pragma: no cover - Windows-only branch
        return [os.environ.get("COMSPEC", "cmd.exe"), "/Q", "/K"]
    return ["/bin/sh"]


def _pid_alive(pid: int | None) -> bool:
    """Whether ``pid`` exists, tolerating permission errors as "alive"."""
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _send_signal(proc: subprocess.Popen[Any], sig: int) -> None:
    """Signal the process group when possible, falling back to the process."""
    if os.name == "nt":  # pragma: no cover - Windows-only branch
        if sig == signal.SIGTERM:
            proc.terminate()
        else:
            proc.kill()
        return
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            if sig == signal.SIGTERM:
                proc.terminate()
            else:
                proc.kill()
        except OSError:
            pass


def _terminate_process(proc: subprocess.Popen[Any], *, grace: float = KILL_GRACE_SECONDS) -> None:
    """Terminate a process (group), escalate to SIGKILL, and reap it."""
    if proc.poll() is not None:
        return
    _send_signal(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    _send_signal(proc, signal.SIGKILL)
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass


def _kill_pid(pid: int | None, *, grace: float = KILL_GRACE_SECONDS) -> None:
    """Terminate a process we do not own a ``Popen`` handle for."""
    if not _pid_alive(pid):
        return
    assert pid is not None
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(_POLL_INTERVAL)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _shell_jobs_dir() -> Path:
    path = ensure_config_dir() / "jobs" / "shell"
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


class PersistentShell:
    """A long-lived shell subprocess that keeps ``cd``/env state between calls."""

    def __init__(self, workspace: Path | str, *, timeout: float | None = None) -> None:
        self.workspace = Path(workspace)
        self.timeout = float(timeout) if timeout is not None else None
        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._cond = threading.Condition()
        self._buffer = ""
        self._truncated = False
        self._collecting = False
        self._pending_id: str | None = None
        self._exit_code: int | None = None
        self._eof = False
        self._closed = False
        self._start()

    @property
    def alive(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    def _start(self) -> None:
        with self._cond:
            self._closed = False
            self._eof = False
            self._buffer = ""
            self._truncated = False
            self._collecting = False
            self._pending_id = None
            self._exit_code = None
        try:
            shell_command = "exec /bin/sh"
            # Remote execution decides where the shell lives; a local sandbox
            # only applies when the shell is not remote.
            remote_argv, _remote_warning = wrap_remote_command(
                shell_command, workspace=self.workspace, interactive=True
            )
            if remote_argv is not None:
                argv = remote_argv
            else:
                argv = _shell_argv()
                if os.name != "nt":
                    # Keep the local shell inside the OS sandbox when enabled.
                    argv, _warning = wrap_command(
                        shell_command, workspace=self.workspace
                    )
            proc = subprocess.Popen(
                argv,
                cwd=str(self.workspace),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
        except OSError as exc:
            raise ShellError(f"Failed to start the persistent shell: {exc}") from exc
        self._proc = proc
        self._reader = threading.Thread(
            target=self._read_loop,
            args=(proc,),
            name="kiwi-shell-reader",
            daemon=True,
        )
        self._reader.start()

    def _sentinel_command(self, command_id: str) -> str:
        if os.name == "nt":  # pragma: no cover - Windows-only branch
            return f"echo __KIWI_{command_id}__ %errorlevel%"
        return f"printf '\\n__KIWI_{command_id}__ %s\\n' $?"

    def _read_loop(self, proc: subprocess.Popen[str]) -> None:
        stream = proc.stdout
        if stream is None:
            return
        try:
            while True:
                line = stream.readline()
                if not line:
                    break
                match = _SENTINEL_RE.match(line.strip())
                with self._cond:
                    if match is not None:
                        if self._collecting and match.group(1) == self._pending_id:
                            self._exit_code = int(match.group(2))
                            self._collecting = False
                        self._cond.notify_all()
                        continue
                    if self._collecting and len(self._buffer) < MAX_OUTPUT:
                        self._buffer += line
                        if len(self._buffer) > MAX_OUTPUT:
                            self._truncated = True
                    elif self._collecting:
                        self._truncated = True
                    self._cond.notify_all()
        except (OSError, ValueError):  # pragma: no cover - stream torn down
            pass
        finally:
            with self._cond:
                self._eof = True
                self._cond.notify_all()

    def _ensure_alive(self) -> None:
        if self.alive:
            return
        if self._proc is not None:
            self.close()
        self._start()

    def run(self, command: str, timeout: float | None = None) -> tuple[int | None, str]:
        """Run one command; return ``(exit_code, output)``.

        A timeout restarts the shell so a hung command can never wedge the
        session; the returned exit code is ``None`` in that case and the output
        explains what happened. Output is bounded to :data:`MAX_OUTPUT`.
        """
        if not command.strip():
            return 0, ""
        self._ensure_alive()
        proc = self._proc
        assert proc is not None and proc.stdin is not None

        command_id = uuid.uuid4().hex[:12]
        payload = f"{command}\n{self._sentinel_command(command_id)}\n"
        if timeout is not None:
            effective_timeout = float(timeout)
        elif self.timeout is not None:
            effective_timeout = self.timeout
        else:
            effective_timeout = float(get_shell_config()["timeout"])

        with self._cond:
            self._buffer = ""
            self._truncated = False
            self._collecting = True
            self._pending_id = command_id
            self._exit_code = None
        try:
            proc.stdin.write(payload)
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.close()
            return None, f"Error: the shell exited before running the command ({exc})"

        deadline = time.monotonic() + effective_timeout
        timed_out = False
        exited = False
        with self._cond:
            while self._exit_code is None:
                if self._closed or self._eof or not self.alive:
                    exited = True
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                self._cond.wait(min(remaining, _POLL_INTERVAL))
            code = self._exit_code
            output = self._buffer
            truncated = self._truncated
            self._buffer = ""
            self._truncated = False
            self._collecting = False
            self._pending_id = None
            self._exit_code = None

        if timed_out:
            self.close()
            message = (
                f"Error: command timed out after {effective_timeout:g}s; "
                "the persistent shell was restarted"
            )
            combined = "\n".join(part for part in (output.rstrip("\n"), message) if part)
            return None, _bounded_output(combined, truncated)
        if exited:
            self.close()
            message = "Error: the shell exited unexpectedly"
            combined = "\n".join(part for part in (output.rstrip("\n"), message) if part)
            return None, _bounded_output(combined, truncated)
        return code, _bounded_output(output.rstrip("\n"), truncated)

    def close(self) -> None:
        """Shut the shell down cleanly, escalating to SIGKILL if needed."""
        with self._cond:
            self._closed = True
            self._collecting = False
            self._pending_id = None
            self._exit_code = None
            self._cond.notify_all()
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        _terminate_process(proc)
        reader = self._reader
        self._reader = None
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1.0)


@dataclass
class BackgroundProcess:
    """A detached command tracked by a :class:`ShellManager`."""

    id: str
    command: str
    cwd: str
    pid: int
    output_path: Path
    started_at: float = field(default_factory=time.time)
    proc: subprocess.Popen[bytes] | None = field(default=None, repr=False)

    @property
    def alive(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        return _pid_alive(self.pid)

    @property
    def exit_code(self) -> int | None:
        if self.proc is None:
            return None
        return self.proc.poll()


class ShellManager:
    """Owns one session's persistent shell and its background processes."""

    def __init__(
        self,
        workspace: Path | str,
        *,
        persistent: bool | None = None,
        timeout: float | None = None,
        max_jobs: int | None = None,
    ) -> None:
        cfg = get_shell_config()
        self.workspace = Path(workspace)
        self.persistent = bool(cfg["persistent"]) if persistent is None else persistent
        self.timeout = float(cfg["timeout"]) if timeout is None else float(timeout)
        self.max_jobs = int(cfg["max_jobs"]) if max_jobs is None else int(max_jobs)
        self._shell: PersistentShell | None = None
        self._jobs: dict[str, BackgroundProcess] = {}
        self._lock = threading.Lock()

    def shell(self) -> PersistentShell:
        """Return the session's persistent shell, starting it on first use."""
        if not self.persistent:
            raise ShellDisabledError(
                "Persistent shell sessions are disabled "
                "(enable with `config shell persistent on`)."
            )
        with self._lock:
            if self._shell is None or not self._shell.alive:
                self._shell = PersistentShell(self.workspace, timeout=self.timeout)
            return self._shell

    def run(self, command: str, timeout: float | None = None) -> tuple[int | None, str]:
        """Convenience wrapper around :meth:`PersistentShell.run`."""
        return self.shell().run(command, timeout=timeout)

    def start_background(self, command: str, cwd: str | Path | None = None) -> str:
        """Start a detached command, redirecting output to a per-job file."""
        command = str(command).strip()
        if not command:
            raise ShellError("A background command is required.")
        workdir = Path(cwd) if cwd is not None else self.workspace
        with self._lock:
            live = sum(1 for job in self._jobs.values() if job.alive)
            if live >= self.max_jobs:
                raise ShellJobLimitError(
                    f"Background job limit reached ({self.max_jobs}); kill a job "
                    "or raise `shell.max_jobs` in config."
                )
            job_id = uuid.uuid4().hex[:12]
            output_path = _shell_jobs_dir() / f"{job_id}.log"
            try:
                output_file = output_path.open("wb")
            except OSError as exc:
                raise ShellError(f"Failed to create the job output file: {exc}") from exc
            try:
                remote_argv, _remote_warning = wrap_remote_command(
                    command, workspace=workdir
                )
                if remote_argv is not None:
                    argv = remote_argv
                    use_shell = False
                else:
                    argv, _warning = wrap_command(command, workspace=workdir)
                    use_shell = os.name == "nt"
                proc = subprocess.Popen(
                    command if use_shell else argv,
                    shell=use_shell,
                    cwd=str(workdir),
                    stdin=subprocess.DEVNULL,
                    stdout=output_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as exc:
                raise ShellError(f"Failed to start the background command: {exc}") from exc
            finally:
                output_file.close()
            self._jobs[job_id] = BackgroundProcess(
                id=job_id,
                command=command,
                cwd=str(workdir),
                pid=proc.pid,
                output_path=output_path,
                proc=proc,
            )
            return job_id

    def list_background(self) -> list[BackgroundProcess]:
        """Return tracked background jobs, oldest first."""
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.started_at)

    def get_background(self, job_id: str) -> BackgroundProcess | None:
        """Return a tracked background job, or None when the id is unknown."""
        with self._lock:
            return self._jobs.get(job_id)

    def output_background(self, job_id: str, max_bytes: int = MAX_OUTPUT) -> str | None:
        """Return a job's output (bounded), or None when the id is unknown."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return None
        try:
            with job.output_path.open("rb") as handle:
                data = handle.read(max_bytes + 1)
        except OSError:
            return ""
        truncated = len(data) > max_bytes
        if truncated:
            data = data[:max_bytes]
        text = data.decode("utf-8", errors="replace")
        if truncated:
            text += "\n... [output truncated]"
        return text

    def kill_background(self, job_id: str) -> bool:
        """Terminate a tracked background job; returns whether it existed."""
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        if job.proc is not None:
            _terminate_process(job.proc)
        else:
            _kill_pid(job.pid)
        return True

    def close(self) -> None:
        """Close the persistent shell and kill every tracked background job."""
        with self._lock:
            shell = self._shell
            self._shell = None
            jobs = list(self._jobs.values())
            self._jobs.clear()
        if shell is not None:
            shell.close()
        for job in jobs:
            if job.proc is not None:
                _terminate_process(job.proc)
            else:
                _kill_pid(job.pid)


def get_shell(session: Session) -> ShellManager:
    """Return the session's shell manager, creating it on first use."""
    manager = getattr(session, "shell", None)
    if not isinstance(manager, ShellManager):
        manager = ShellManager(session.workspace_root)
        session.shell = manager
    return manager


def close_shell(session: Session) -> None:
    """Close and detach the session's shell manager; never raises."""
    manager = getattr(session, "shell", None)
    if manager is None:
        return
    try:
        session.shell = None
    except (AttributeError, TypeError):  # pragma: no cover - defensive
        pass
    try:
        manager.close()
    except Exception:  # pragma: no cover - cleanup must not raise
        pass
