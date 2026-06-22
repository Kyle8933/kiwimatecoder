"""run_bash tool: execute a shell command in the workspace.

Note: unlike the read/write/list/search/edit file tools (which use
resolve_in_workspace for sandboxing), run_bash is not restricted beyond setting
cwd to the workspace root. The command string is executed via shell=True (with
user approval in ask/plan modes). Intended for git, pytest, builds, etc.
"""

from __future__ import annotations

import os
import signal
import subprocess

from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 3600
MAX_OUTPUT = 30_000


def preview(args: dict, session: Session) -> str:
    """Return the command text for the approval prompt."""
    return args.get("command", "")


def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT:
        return text[:MAX_OUTPUT] + "\n... [output truncated]"
    return text


def _run_bash(args: dict, session: Session) -> ToolResult:
    command = args.get("command")
    if not command:
        return ToolResult.error("'command' is required")
    try:
        timeout = int(args.get("timeout", DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT)
    except (TypeError, ValueError):
        return ToolResult.error("'timeout' must be an integer")
    timeout = max(1, min(timeout, MAX_TIMEOUT))

    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=str(session.workspace_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        return ToolResult.error(f"Failed to start command: {exc}")

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

    parts = []
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
