"""run_bash tool: execute a shell command in the workspace.

Note: unlike the read/write/list/search/edit file tools (which use
resolve_in_workspace for sandboxing), run_bash is not restricted beyond setting
cwd to the workspace root. The command string is executed via shell=True (with
user approval in ask/plan modes). Intended for git, pytest, builds, etc.
"""

from __future__ import annotations

import subprocess

from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult

DEFAULT_TIMEOUT = 120
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
    timeout = int(args.get("timeout", DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT)

    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(session.workspace_root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ToolResult.error(f"Command timed out after {timeout}s")

    parts = []
    if completed.stdout:
        parts.append(_truncate(completed.stdout))
    if completed.stderr:
        parts.append("[stderr]\n" + _truncate(completed.stderr))
    parts.append(f"[exit code: {completed.returncode}]")
    return ToolResult(content="\n".join(parts), ok=completed.returncode == 0)


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
                "description": "Timeout in seconds (default 120).",
            },
        },
        "required": ["command"],
    },
    func=_run_bash,
    runs=True,
)
