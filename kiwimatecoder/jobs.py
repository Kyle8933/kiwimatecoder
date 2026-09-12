"""Detached background and scheduled agent jobs.

A job is one headless agent run (``python -m kiwimatecoder -p ... --output-format
json``) tracked by a JSON record under ``~/.kiwimatecoder/jobs/<id>.json``. Jobs
are detached (their own session) and write combined output to
``~/.kiwimatecoder/jobs/<id>.log``, so they keep running after the process that
started them exits.

Scheduling is deliberately pull-based: :func:`schedule_job` records an interval
and :func:`run_due_jobs` starts whatever is due. Nothing runs on its own --
invoke ``run_due_jobs()`` from the CLI (``kiwimatecoder jobs tick``), a cron
entry, or a loop. There is no daemon and no platform cron integration.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from kiwimatecoder.config import ensure_config_dir, get_provider_config
from kiwimatecoder.permissions import PermissionMode

JOB_STATUSES = ("running", "succeeded", "failed", "cancelled")
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
DEFAULT_GRACE_SECONDS = 5.0
MAX_OUTPUT_BYTES = 100_000
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Live Popen handles for jobs started by this process; other processes reason
# about liveness from the recorded pid alone.
_PROCESSES: dict[str, subprocess.Popen[Any]] = {}


class JobError(RuntimeError):
    """Raised for background-job lifecycle failures."""


@dataclass
class JobRecord:
    """Persisted state for one background agent job."""

    id: str
    prompt: str
    workspace: str
    provider: str | None = None
    model: str | None = None
    mode: str = "auto-accept"
    status: str = "running"
    pid: int | None = None
    created_at: str = ""
    finished_at: str | None = None
    exit_code: int | None = None
    result: str | None = None
    error: str | None = None
    output_path: str = ""
    interval: float | None = None
    next_run_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "workspace": self.workspace,
            "provider": self.provider,
            "model": self.model,
            "mode": self.mode,
            "status": self.status,
            "pid": self.pid,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "result": self.result,
            "error": self.error,
            "output_path": self.output_path,
            "interval": self.interval,
            "next_run_at": self.next_run_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobRecord":
        """Rebuild a record, tolerating missing or mistyped fields."""
        job_id = str(data.get("id") or "").strip()
        if not job_id:
            raise ValueError("Job record has no id.")
        status = str(data.get("status") or "running").strip().lower()
        if status not in JOB_STATUSES:
            status = "failed" if data.get("finished_at") else "running"
        provider = data.get("provider")
        model = data.get("model")
        interval_value = data.get("interval")
        try:
            interval = float(interval_value) if interval_value is not None else None
        except (TypeError, ValueError):
            interval = None
        return cls(
            id=job_id,
            prompt=str(data.get("prompt") or ""),
            workspace=str(data.get("workspace") or "."),
            provider=str(provider) if provider else None,
            model=str(model) if model else None,
            mode=str(data.get("mode") or "auto-accept"),
            status=status,
            pid=_optional_int(data.get("pid")),
            created_at=str(data.get("created_at") or ""),
            finished_at=str(data["finished_at"]) if data.get("finished_at") else None,
            exit_code=_optional_int(data.get("exit_code")),
            result=str(data["result"]) if data.get("result") is not None else None,
            error=str(data["error"]) if data.get("error") is not None else None,
            output_path=str(data.get("output_path") or ""),
            interval=interval,
            next_run_at=str(data["next_run_at"]) if data.get("next_run_at") else None,
        )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _parse_iso(value: str) -> float:
    try:
        return datetime.datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return float("inf")


def _jobs_dir() -> Path:
    path = ensure_config_dir() / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def _safe_id(job_id: str) -> str | None:
    cleaned = str(job_id).strip()
    return cleaned if _JOB_ID_RE.match(cleaned) else None


def _record_path(job_id: str) -> Path:
    return _jobs_dir() / f"{job_id}.json"


def _output_path(job_id: str) -> Path:
    return _jobs_dir() / f"{job_id}.log"


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_job(record: JobRecord) -> Path:
    """Persist a job record with an atomic replace."""
    path = _record_path(record.id)
    _atomic_write(path, json.dumps(record.to_dict(), indent=2) + "\n")
    return path


def _load_record(path: Path) -> JobRecord | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return JobRecord.from_dict(data)
    except (TypeError, ValueError):
        return None


def get_job(job_id: str) -> JobRecord | None:
    """Load one job record; corrupt or missing files return None."""
    safe = _safe_id(job_id)
    if safe is None:
        return None
    path = _record_path(safe)
    if not path.is_file():
        return None
    return _load_record(path)


def list_jobs(*, refresh: bool = False) -> list[JobRecord]:
    """List job records newest first, skipping unreadable files."""
    records: list[JobRecord] = []
    try:
        paths = list(_jobs_dir().glob("*.json"))
    except OSError:
        return []
    for path in paths:
        record = _load_record(path)
        if record is not None:
            records.append(record)
    records.sort(key=lambda record: record.created_at, reverse=True)
    if refresh:
        refreshed: list[JobRecord] = []
        for record in records:
            if record.status == "running":
                record = refresh_job(record.id) or record
            refreshed.append(record)
        refreshed.sort(key=lambda record: record.created_at, reverse=True)
        return refreshed
    return records


def _pid_alive(pid: int | None) -> bool:
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


def _parse_result(output_path: str) -> dict[str, Any] | None:
    """Return the final JSON result line from a job's output, if present."""
    if not output_path:
        return None
    try:
        text = Path(output_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "success" in data:
            return data
    return None


def read_job_output(job_id: str, max_bytes: int = MAX_OUTPUT_BYTES) -> str | None:
    """Return bounded combined output for a job, or None when unknown."""
    record = get_job(job_id)
    if record is None:
        return None
    if not record.output_path:
        return ""
    try:
        with Path(record.output_path).open("rb") as handle:
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


def start_job(
    prompt: str,
    *,
    workspace: str | Path,
    provider: str | None = None,
    model: str | None = None,
    mode: str | PermissionMode = "auto-accept",
    extra_args: Sequence[str] = (),
) -> JobRecord:
    """Spawn a detached headless agent run and return its record.

    ``mode`` defaults to ``auto-accept`` so the job can act unattended; pass
    ``mode="ask"`` or ``mode="plan"`` to override (an unattended ``ask`` job
    simply has its approvals denied). Raises ``ValueError`` for an empty prompt,
    a missing workspace, or an unknown mode; ``KeyError`` for an unknown
    provider; and ``JobError`` when the process cannot be started.
    """
    cleaned_prompt = str(prompt).strip()
    if not cleaned_prompt:
        raise ValueError("Job prompt is required.")
    workspace_path = Path(workspace).expanduser()
    if not workspace_path.is_dir():
        raise ValueError(f"Workspace is not a directory: {workspace_path}")
    permission = PermissionMode.from_str(mode) if isinstance(mode, str) else mode
    resolved_mode = permission.value
    if provider is not None:
        get_provider_config(provider)  # raises UnknownProviderError (a KeyError)

    job_id = uuid.uuid4().hex[:12]
    output_path = _output_path(job_id)
    argv = [
        sys.executable,
        "-m",
        "kiwimatecoder",
        "-p",
        cleaned_prompt,
        "--output-format",
        "json",
        "--workspace",
        str(workspace_path),
        "--mode",
        resolved_mode,
    ]
    if provider:
        argv += ["--provider", provider]
    if model:
        argv += ["--model", model]
    argv += [str(arg) for arg in extra_args]

    try:
        output_file = output_path.open("wb")
    except OSError as exc:
        raise JobError(f"Failed to create the job output file: {exc}") from exc
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=output_file,
            stderr=subprocess.STDOUT,
            cwd=str(workspace_path),
            start_new_session=True,
        )
    except OSError as exc:
        raise JobError(f"Failed to start the job process: {exc}") from exc
    finally:
        output_file.close()

    record = JobRecord(
        id=job_id,
        prompt=cleaned_prompt,
        workspace=str(workspace_path),
        provider=provider,
        model=model,
        mode=resolved_mode,
        status="running",
        pid=proc.pid,
        created_at=_now_iso(),
        output_path=str(output_path),
    )
    _PROCESSES[job_id] = proc
    try:
        save_job(record)
    except OSError as exc:
        _PROCESSES.pop(job_id, None)
        raise JobError(f"Failed to save the job record: {exc}") from exc
    return record


def refresh_job(job_id: str) -> JobRecord | None:
    """Update a job's status from its process and parsed JSON result."""
    record = get_job(job_id)
    if record is None:
        return None
    if record.status in TERMINAL_STATUSES:
        return record

    proc = _PROCESSES.get(job_id)
    code: int | None = None
    if proc is not None:
        code = proc.poll()
        if code is None:
            return record
    elif _pid_alive(record.pid):
        return record
    else:
        code = record.exit_code

    _PROCESSES.pop(job_id, None)
    record.exit_code = code
    payload = _parse_result(record.output_path)
    if payload is not None:
        record.result = str(payload.get("result") or "")
        success = payload.get("success")
        if success is False or (code not in (0, None)):
            record.status = "failed"
            record.error = record.error or "The job reported failure."
        else:
            record.status = "succeeded"
    elif code == 0:
        record.status = "succeeded"
        record.result = ""
    else:
        record.status = "failed"
        record.error = (
            f"Job exited with code {code} before producing a result."
            if code is not None
            else "Job process exited without producing a result."
        )
    record.finished_at = _now_iso()
    save_job(record)
    return record


def _terminate(proc: subprocess.Popen[Any], *, grace: float = DEFAULT_GRACE_SECONDS) -> int | None:
    """SIGTERM a process, escalate to SIGKILL, and return its exit code."""
    if proc.poll() is not None:
        return proc.returncode
    proc.terminate()
    try:
        return proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    try:
        return proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        return None


def _terminate_pid(pid: int | None, *, grace: float = DEFAULT_GRACE_SECONDS) -> None:
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
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def cancel_job(job_id: str) -> JobRecord | None:
    """Terminate a running job and mark it cancelled; returns the record."""
    record = get_job(job_id)
    if record is None:
        return None
    if record.status in TERMINAL_STATUSES:
        return record
    proc = _PROCESSES.pop(job_id, None)
    if proc is not None:
        record.exit_code = _terminate(proc)
    else:
        _terminate_pid(record.pid)
    record.status = "cancelled"
    record.finished_at = _now_iso()
    save_job(record)
    return record


def clear_jobs(*, finished_only: bool = True) -> int:
    """Delete job records (and output logs); returns how many were removed.

    With ``finished_only=False`` running jobs are cancelled first so no
    untracked process is left behind.
    """
    removed = 0
    for record in list_jobs():
        if record.status not in TERMINAL_STATUSES:
            if finished_only:
                continue
            cancel_job(record.id)
        _PROCESSES.pop(record.id, None)
        try:
            _record_path(record.id).unlink(missing_ok=True)
        except OSError:
            continue
        if record.output_path:
            try:
                Path(record.output_path).unlink(missing_ok=True)
            except OSError:
                pass
        removed += 1
    return removed


def schedule_job(
    prompt: str,
    every_seconds: float | int,
    *,
    workspace: str | Path,
    provider: str | None = None,
    model: str | None = None,
    mode: str | PermissionMode = "auto-accept",
    extra_args: Sequence[str] = (),
) -> JobRecord:
    """Start a job now and record how often it should run again.

    The first run starts immediately; :func:`run_due_jobs` starts subsequent
    runs once ``next_run_at`` has passed. Raises ``ValueError`` for a
    non-positive interval.
    """
    try:
        interval = float(every_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("Schedule interval must be a number of seconds.") from exc
    if interval <= 0:
        raise ValueError("Schedule interval must be positive.")
    record = start_job(
        prompt,
        workspace=workspace,
        provider=provider,
        model=model,
        mode=mode,
        extra_args=extra_args,
    )
    record.interval = interval
    record.next_run_at = datetime.datetime.fromtimestamp(
        time.time() + interval
    ).isoformat(timespec="seconds")
    save_job(record)
    return record


def run_due_jobs(*, now: float | None = None) -> list[JobRecord]:
    """Start a new run for every scheduled job whose interval has elapsed.

    A job that is still running is skipped (runs never overlap); a finished job
    hands the schedule to the fresh run. Returns the newly started records.
    """
    current = time.time() if now is None else now
    started: list[JobRecord] = []
    for scheduled in list_jobs():
        if scheduled.interval is None or scheduled.next_run_at is None:
            continue
        if _parse_iso(scheduled.next_run_at) > current:
            continue
        refreshed = refresh_job(scheduled.id) or scheduled
        if refreshed.status == "running" or refreshed.interval is None:
            continue
        try:
            fresh = start_job(
                refreshed.prompt,
                workspace=refreshed.workspace,
                provider=refreshed.provider,
                model=refreshed.model,
                mode=refreshed.mode,
            )
        except (ValueError, KeyError, JobError):
            # A broken schedule (missing workspace, retired provider) is left
            # alone so the caller can inspect and fix or cancel the record.
            continue
        fresh.interval = refreshed.interval
        fresh.next_run_at = datetime.datetime.fromtimestamp(
            current + refreshed.interval
        ).isoformat(timespec="seconds")
        save_job(fresh)
        refreshed.interval = None
        refreshed.next_run_at = None
        save_job(refreshed)
        started.append(fresh)
    return started
