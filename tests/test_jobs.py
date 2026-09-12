from __future__ import annotations

import io
import subprocess
import sys
import time
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from kiwimatecoder import config, jobs, main
from kiwimatecoder.commands import dispatch


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    """Point config storage and job records at a temp directory."""
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config" / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config" / "legacy")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    jobs._PROCESSES.clear()
    yield
    jobs._PROCESSES.clear()


def _console():
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def _output(console) -> str:
    return console.file.getvalue()


def _record(job_id: str, workspace: Path, *, status: str = "running", **kwargs):
    output_path = config.CONFIG_DIR / "jobs" / f"{job_id}.log"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = jobs.JobRecord(
        id=job_id,
        prompt=kwargs.pop("prompt", "do the thing"),
        workspace=str(workspace),
        status=status,
        pid=kwargs.pop("pid", 999999),
        created_at=kwargs.pop("created_at", jobs._now_iso()),
        output_path=str(output_path),
        **kwargs,
    )
    jobs.save_job(record)
    return record


class FakeProcess:
    """Minimal Popen stand-in with scriptable terminate/wait behavior."""

    def __init__(self, pid: int = 4321, poll_result: int | None = None,
                 wait_timeouts: int = 0) -> None:
        self.pid = pid
        self.returncode = -15
        self._poll = poll_result
        self.calls: list[str] = []
        self.wait_timeouts = wait_timeouts

    def poll(self):
        return self._poll

    def terminate(self) -> None:
        self.calls.append("terminate")

    def kill(self) -> None:
        self.calls.append("kill")
        self._poll = -9
        self.returncode = -9

    def wait(self, timeout=None):
        if len(self.calls) <= self.wait_timeouts and "terminate" in self.calls:
            raise subprocess.TimeoutExpired(cmd="job", timeout=timeout)
        self._poll = self.returncode
        return self.returncode


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def test_record_roundtrip(tmp_path):
    record = jobs.JobRecord(
        id="abc123",
        prompt="fix the tests",
        workspace=str(tmp_path),
        provider="openrouter",
        model="test-model",
        mode="auto-accept",
        status="running",
        pid=123,
        created_at="2026-01-01T00:00:00",
        output_path=str(tmp_path / "out.log"),
    )

    jobs.save_job(record)

    assert jobs.get_job("abc123") == record


def test_records_tolerate_corruption(tmp_path):
    good = _record("goodjob", tmp_path, status="succeeded")
    jobs_dir = config.CONFIG_DIR / "jobs"
    (jobs_dir / "broken.json").write_text("{not json", encoding="utf-8")
    (jobs_dir / "partial.json").write_text('{"prompt": "no id"}', encoding="utf-8")
    (jobs_dir / "wrongtype.json").write_text("[]", encoding="utf-8")

    records = jobs.list_jobs()

    assert [record.id for record in records] == [good.id]
    assert jobs.get_job("missing") is None
    assert jobs.get_job("../escape") is None


def test_refresh_job_parses_finished_result(tmp_path, monkeypatch):
    record = _record("job1", tmp_path, status="running", pid=424242)
    Path(record.output_path).write_text(
        'progress line\n{"result": "all done", "success": true, "usage": {}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(jobs, "_pid_alive", lambda pid: False)

    refreshed = jobs.refresh_job("job1")

    assert refreshed is not None
    assert refreshed.status == "succeeded"
    assert refreshed.result == "all done"
    assert refreshed.finished_at


def test_refresh_job_marks_reported_failure(tmp_path, monkeypatch):
    record = _record("job2", tmp_path, status="running", pid=424242)
    Path(record.output_path).write_text(
        '{"result": "", "success": false}\n', encoding="utf-8"
    )
    monkeypatch.setattr(jobs, "_pid_alive", lambda pid: False)

    refreshed = jobs.refresh_job("job2")

    assert refreshed is not None
    assert refreshed.status == "failed"
    assert refreshed.error


def test_refresh_job_keeps_running_processes_running(tmp_path):
    record = _record("job3", tmp_path, status="running", pid=424242)
    proc = FakeProcess(pid=424242, poll_result=None)
    jobs._PROCESSES[record.id] = proc

    refreshed = jobs.refresh_job("job3")

    assert refreshed is not None
    assert refreshed.status == "running"


def test_refresh_job_marks_failure_without_result(tmp_path, monkeypatch):
    _record("job4", tmp_path, status="running", pid=424242)
    monkeypatch.setattr(jobs, "_pid_alive", lambda pid: False)

    refreshed = jobs.refresh_job("job4")

    assert refreshed is not None
    assert refreshed.status == "failed"
    assert "result" in (refreshed.error or "")


def test_refresh_job_ignores_finished_records(tmp_path):
    record = _record("job5", tmp_path, status="succeeded")

    assert jobs.refresh_job("job5") == record


# ---------------------------------------------------------------------------
# start_job
# ---------------------------------------------------------------------------


def test_start_job_argv_and_redirects(tmp_path, monkeypatch):
    calls = {}

    def fake_popen(argv, **kwargs):
        calls["argv"] = list(argv)
        calls["kwargs"] = kwargs
        return FakeProcess(pid=9876)

    monkeypatch.setattr(jobs.subprocess, "Popen", fake_popen)

    record = jobs.start_job(
        "fix the bug",
        workspace=tmp_path,
        provider="openrouter",
        model="test-model",
        extra_args=("--max-turns", "5"),
    )

    argv = calls["argv"]
    assert argv[0] == sys.executable
    assert argv[1:4] == ["-m", "kiwimatecoder", "-p"]
    assert argv[argv.index("-p") + 1] == "fix the bug"
    assert argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--workspace") + 1] == str(tmp_path)
    assert argv[argv.index("--mode") + 1] == "auto-accept"
    assert argv[argv.index("--provider") + 1] == "openrouter"
    assert argv[argv.index("--model") + 1] == "test-model"
    assert argv[-2:] == ["--max-turns", "5"]

    kwargs = calls["kwargs"]
    assert kwargs["stdin"] is jobs.subprocess.DEVNULL
    assert kwargs["stderr"] is jobs.subprocess.STDOUT
    assert kwargs["stdout"] is not None
    assert kwargs["start_new_session"] is True
    assert kwargs["cwd"] == str(tmp_path)

    assert record.pid == 9876
    assert record.status == "running"
    assert record.output_path.endswith(f"{record.id}.log")
    assert Path(record.output_path).parent.is_dir()
    assert jobs._PROCESSES[record.id] is not None
    assert jobs.get_job(record.id) == record


def test_start_job_validates_input(tmp_path):
    with pytest.raises(ValueError):
        jobs.start_job("   ", workspace=tmp_path)
    with pytest.raises(ValueError):
        jobs.start_job("x", workspace=tmp_path / "missing")
    with pytest.raises(ValueError):
        jobs.start_job("x", workspace=tmp_path, mode="nonsense")
    with pytest.raises(KeyError):
        jobs.start_job("x", workspace=tmp_path, provider="ghost")


# ---------------------------------------------------------------------------
# cancel / clear
# ---------------------------------------------------------------------------


def test_cancel_job_terminates_then_kills(tmp_path):
    record = _record("c1", tmp_path, status="running", pid=4242)
    proc = FakeProcess(pid=4242, poll_result=None, wait_timeouts=1)
    jobs._PROCESSES[record.id] = proc

    cancelled = jobs.cancel_job("c1")

    assert proc.calls == ["terminate", "kill"]
    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert cancelled.exit_code == -9
    assert "c1" not in jobs._PROCESSES
    assert jobs.get_job("c1").status == "cancelled"


def test_cancel_job_leaves_finished_records_alone(tmp_path):
    record = _record("c2", tmp_path, status="succeeded")

    assert jobs.cancel_job("c2") == record
    assert jobs.cancel_job("missing") is None


def test_clear_jobs_filters_by_status(tmp_path):
    finished = _record("fin", tmp_path, status="succeeded")
    running = _record("run", tmp_path, status="running", pid=4242)
    proc = FakeProcess(pid=4242, poll_result=None)
    jobs._PROCESSES[running.id] = proc
    Path(finished.output_path).write_text("done", encoding="utf-8")
    Path(running.output_path).write_text("partial", encoding="utf-8")

    removed = jobs.clear_jobs()

    assert removed == 1
    assert jobs.get_job("fin") is None
    assert not Path(finished.output_path).exists()
    assert jobs.get_job("run") is not None

    removed = jobs.clear_jobs(finished_only=False)

    assert removed == 1
    assert proc.calls == ["terminate"]
    assert jobs.get_job("run") is None


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


def test_schedule_job_and_run_due(tmp_path, monkeypatch):
    created: list[jobs.JobRecord] = []

    def fake_start(prompt, **kwargs):
        record = jobs.JobRecord(
            id=f"job{len(created)}",
            prompt=prompt,
            workspace=str(kwargs["workspace"]),
            status="running",
            created_at=jobs._now_iso(),
            output_path=str(tmp_path / f"job{len(created)}.log"),
        )
        created.append(record)
        jobs.save_job(record)
        return record

    monkeypatch.setattr(jobs, "start_job", fake_start)

    scheduled = jobs.schedule_job("watch the build", 60, workspace=tmp_path)

    assert scheduled.interval == 60
    assert scheduled.next_run_at is not None
    # Not due yet.
    assert jobs.run_due_jobs() == []

    scheduled.next_run_at = "2000-01-01T00:00:00"
    scheduled.status = "succeeded"
    jobs.save_job(scheduled)

    started = jobs.run_due_jobs(now=time.time())

    assert len(started) == 1
    assert started[0].interval == 60
    assert started[0].next_run_at is not None
    assert jobs.get_job(scheduled.id).interval is None
    assert len(created) == 2


def test_schedule_job_rejects_bad_interval(tmp_path):
    with pytest.raises(ValueError):
        jobs.schedule_job("x", 0, workspace=tmp_path)
    with pytest.raises(ValueError):
        jobs.schedule_job("x", "soon", workspace=tmp_path)  # type: ignore[arg-type]


def test_run_due_jobs_skips_broken_schedule(tmp_path, monkeypatch):
    record = _record("bad", tmp_path, status="succeeded")
    record.interval = 60
    record.next_run_at = "2000-01-01T00:00:00"
    jobs.save_job(record)

    def boom(prompt, **kwargs):
        raise ValueError("workspace is gone")

    monkeypatch.setattr(jobs, "start_job", boom)

    assert jobs.run_due_jobs() == []


# ---------------------------------------------------------------------------
# Slash command
# ---------------------------------------------------------------------------


def test_slash_jobs_list_empty(session):
    console = _console()

    dispatch("/jobs list", session, console)

    assert "No background jobs" in _output(console)


def test_slash_jobs_list_shows_running_job(session):
    record = _record("run1", session.workspace_root, status="running", pid=4242)
    jobs._PROCESSES[record.id] = FakeProcess(pid=4242, poll_result=None)
    console = _console()

    dispatch("/jobs list", session, console)

    output = _output(console)
    assert "run1" in output
    assert "running" in output


def test_slash_jobs_unknown_and_usage(session):
    console = _console()
    dispatch("/jobs show missing", session, console)
    assert "Unknown job" in _output(console)

    console = _console()
    dispatch("/jobs nonsense", session, console)
    assert "Usage:" in _output(console)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_jobs_run_cli_with_mocked_start_job(tmp_path, monkeypatch):
    record = jobs.JobRecord(
        id="cli1",
        prompt="fix the tests",
        workspace=str(tmp_path),
        status="running",
        pid=1,
        created_at=jobs._now_iso(),
        output_path=str(tmp_path / "cli1.log"),
    )
    called: dict[str, object] = {}

    def fake_start(prompt, **kwargs):
        called["prompt"] = prompt
        called["kwargs"] = kwargs
        return record

    monkeypatch.setattr(jobs, "start_job", fake_start)

    result = CliRunner().invoke(main.app, ["jobs", "run", "fix the tests"])

    assert result.exit_code == 0
    assert "cli1" in result.output
    assert called["prompt"] == "fix the tests"
    assert called["kwargs"] == {
        "workspace": Path.cwd(),
        "provider": None,
        "model": None,
        "mode": "auto-accept",
    }


def test_jobs_list_cli_empty():
    result = CliRunner().invoke(main.app, ["jobs", "list"])

    assert result.exit_code == 0
    assert "No background jobs" in result.output


def test_jobs_show_cli_unknown():
    result = CliRunner().invoke(main.app, ["jobs", "show", "ghost"])

    assert result.exit_code == 1
    assert "Unknown job" in result.output


def test_jobs_tick_cli_no_due_jobs():
    result = CliRunner().invoke(main.app, ["jobs", "tick"])

    assert result.exit_code == 0
    assert "No jobs are due" in result.output


def test_python_m_kiwimatecoder_runs():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "kiwimatecoder", "--version"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0
    assert "kiwimatecoder" in completed.stdout
