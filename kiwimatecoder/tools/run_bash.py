"""run_bash tool: execute a shell command in the workspace.

Note: unlike the read/write/list/search/edit file tools (which use
resolve_in_workspace for sandboxing), run_bash is not restricted beyond setting
cwd to the workspace root. The command string is executed via shell=True (with
user approval in ask/plan modes). Intended for git, pytest, builds, etc.

When OS-level sandboxing is enabled (``config sandbox enable on``), the command
is executed through the platform sandbox instead (macOS seatbelt or Linux
bubblewrap) and the approval preview shows the wrapped command. A missing
backend falls back to the plain shell with a warning.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
from typing import Any

from kiwimatecoder.config import get_sandbox, load_config
from kiwimatecoder.sandbox import wrap_command
from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 3600
MAX_OUTPUT = 30_000

_FALLBACK_ARGV = ["/bin/sh", "-c"]


def preview(args: dict[str, Any], session: Session) -> str:
    """Return the command text (and sandbox wrapper) for the approval prompt."""
    command = str(args.get("command", ""))
    cfg = load_config()
    settings = get_sandbox(cfg)
    if not settings["enabled"]:
        return command
    argv, warning = wrap_command(command, workspace=session.workspace_root, cfg=cfg)
    if warning:
        return f"{command}\n[sandbox] {warning}"
    mode = "network on" if settings["network"] else "network off"
    return f"{command}\n[sandbox] wrapped ({mode}): {shlex.join(argv)}"


def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT:
        return text[:MAX_OUTPUT] + "\n... [output truncated]"
    return text


def _popen(
    command: str, *, argv: list[str] | None, workspace: str
) -> subprocess.Popen[str]:
    """Start the command, through ``argv`` when sandboxed, else via the shell."""
    if argv is not None:
        return subprocess.Popen(
            argv,
            shell=False,
            cwd=workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    return subprocess.Popen(
        command,
        shell=True,
        cwd=workspace,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _run_bash(args: dict[str, Any], session: Session) -> ToolResult:
    command = str(args.get("command") or "")
    if not command:
        return ToolResult.error("'command' is required")
    try:
        timeout = int(args.get("timeout", DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT)
    except (TypeError, ValueError):
        return ToolResult.error("'timeout' must be an integer")
    timeout = max(1, min(timeout, MAX_TIMEOUT))

    argv, sandbox_warning = wrap_command(command, workspace=session.workspace_root)
    sandboxed = argv != [*_FALLBACK_ARGV, command]
    prefix: list[str] = []
    if sandbox_warning:
        prefix.append(f"[sandbox] {sandbox_warning}")

    workspace = str(session.workspace_root)
    try:
        proc = _popen(
            command, argv=argv if sandboxed else None, workspace=workspace
        )
    except OSError as exc:
        if not sandboxed:
            return ToolResult.error(f"Failed to start command: {exc}")
        # The backend exists on PATH but could not start; keep the tool usable.
        prefix.append(
            f"[sandbox] failed to start the sandboxed command ({exc}); "
            "running unsandboxed"
        )
        try:
            proc = _popen(command, argv=None, workspace=workspace)
        except OSError as exc2:
            return ToolResult.error(f"Failed to start command: {exc2}")

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # start_new_session=True put the shell (and its children) in their own
        # process group, so killpg takes down the whole tree.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return ToolResult.error(f"Command timed out after {timeout}s")

    parts = list(prefix)
    if stdout:
        parts.append(_truncate(stdout))
    if stderr:
        parts.append("[stderr]\n" + _truncate(stderr))
    parts.append(f"[exit code: {proc.returncode}]")
    return ToolResult(content="\n".join(parts), ok=proc.returncode == 0)


run_bash_tool = FunctionTool(
    name="run_bash",
    description=(
        "Run a shell command in the workspace root and return its stdout, "
        "stderr, and exit code. Use for builds, tests, git, etc."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 120, max 3600).",
            },
        },
        "required": ["command"],
    },
    func=_run_bash,
    runs=True,
)
