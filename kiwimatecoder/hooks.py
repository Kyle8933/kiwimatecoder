"""Run user-configured shell commands on lifecycle events.

Commands live under the ``"hooks"`` block of the config, keyed by event name
(see :data:`kiwimatecoder.config.HOOK_EVENTS`). Each command runs with
``shell=True`` in the session workspace and receives context through the
environment (``KIWI_EVENT``, ``KIWI_WORKSPACE``, and for tool events
``KIWI_TOOL_NAME`` / ``KIWI_TOOL_OK`` / ``KIWI_TOOL_ARGS``).

Hooks are deliberately best-effort: a timeout, a non-zero exit, or a broken
command is reported as a :class:`HookResult` and never raises into the caller.
``pre_tool`` hook failures are surfaced via :attr:`HookResult.blocked` so the
agent can stop the action.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from kiwimatecoder import config
from kiwimatecoder.redaction import redact

HOOK_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class HookResult:
    """Outcome of running one configured hook command."""

    name: str
    command: str
    exit_code: int
    output: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def blocked(self) -> bool:
        """Whether this result should stop the action (``pre_tool`` hooks)."""
        return not self.ok


def _workspace(session: Any) -> Path:
    root = getattr(session, "workspace_root", None)
    return Path(root) if root is not None else Path.cwd()


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _combine_output(stdout: Any, stderr: Any) -> str:
    return _as_text(stdout) + _as_text(stderr)


def _hook_environment(
    event: str,
    workspace: Path,
    tool_name: str,
    tool_args: dict[str, Any] | None,
    ok: bool | None,
    duration_ms: int | None,
) -> dict[str, str]:
    env = dict(os.environ)
    env["KIWI_EVENT"] = event
    env["KIWI_WORKSPACE"] = str(workspace)
    if tool_name:
        env["KIWI_TOOL_NAME"] = str(tool_name)
    if ok is not None:
        env["KIWI_TOOL_OK"] = "true" if ok else "false"
    if tool_args is not None:
        try:
            encoded = json.dumps(tool_args, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            encoded = str(tool_args)
        env["KIWI_TOOL_ARGS"] = redact(encoded)
    if duration_ms is not None:
        env["KIWI_TOOL_DURATION_MS"] = str(duration_ms)
    return env


def _run_one(
    event: str,
    command: str,
    workspace: Path,
    env: dict[str, str],
    console: Console | None,
) -> HookResult:
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=HOOK_TIMEOUT_SECONDS,
            env=env,
        )
        result = HookResult(
            name=event,
            command=command,
            exit_code=completed.returncode,
            output=_combine_output(completed.stdout, completed.stderr),
        )
    except subprocess.TimeoutExpired as exc:
        output = _combine_output(exc.stdout, exc.stderr)
        result = HookResult(
            name=event,
            command=command,
            exit_code=-1,
            output=output or f"timed out after {HOOK_TIMEOUT_SECONDS}s",
            timed_out=True,
        )
    except OSError as exc:
        result = HookResult(
            name=event,
            command=command,
            exit_code=-1,
            output=f"could not run hook: {exc}",
        )
    except Exception as exc:  # defensive: a hook must never break the caller
        result = HookResult(
            name=event,
            command=command,
            exit_code=-1,
            output=f"hook crashed: {exc!r}",
        )
    if console is not None:
        status = "ok" if result.ok else f"exit {result.exit_code}"
        console.print(f"[dim]hook {event} [{status}]: {redact(command)}[/dim]")
        output = result.output.strip()
        if output:
            console.print(redact(output), style="dim", markup=False, highlight=False)
    return result


def run_hooks(
    event: str,
    *,
    session: Any = None,
    console: Console | None = None,
    tool_name: str = "",
    tool_args: dict[str, Any] | None = None,
    ok: bool | None = None,
    duration_ms: int | None = None,
    cfg: dict[str, Any] | None = None,
) -> list[HookResult]:
    """Run every command configured for ``event``; never raises."""
    commands = config.get_hooks(cfg).get(event, [])
    if not commands:
        return []
    workspace = _workspace(session)
    env = _hook_environment(event, workspace, tool_name, tool_args, ok, duration_ms)
    return [_run_one(event, command, workspace, env, console) for command in commands]
