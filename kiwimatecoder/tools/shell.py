"""Persistent-shell tools: stateful commands and background process control.

``shell`` runs a command in the session's long-lived shell, so ``cd`` and
exported environment variables persist between calls. ``shell_jobs`` starts,
lists, reads, and kills detached background commands. Both are approval-gated.
"""

from __future__ import annotations

from typing import Any

from kiwimatecoder.shell import (
    ShellDisabledError,
    ShellError,
    ShellJobLimitError,
    get_shell,
)
from kiwimatecoder.tools.base import FunctionTool, ToolResult


def preview(args: dict[str, Any], session: Any) -> str:
    """Return the command text for the approval prompt."""
    return str(args.get("command", ""))


def jobs_preview(args: dict[str, Any], session: Any) -> str:
    """Return a short description of a background-job action for approval."""
    action = str(args.get("action") or "").strip().lower()
    if action == "start":
        return f"start: {args.get('command', '')}"
    if action in ("output", "kill"):
        return f"{action}: {args.get('id', '')}"
    return action or "list"


def _run_shell(args: dict[str, Any], session: Any) -> ToolResult:
    command = str(args.get("command") or "")
    if not command.strip():
        return ToolResult.error("'command' is required")
    try:
        exit_code, output = get_shell(session).run(command)
    except ShellDisabledError as exc:
        return ToolResult.error(str(exc))
    except ShellError as exc:
        return ToolResult.error(str(exc))

    parts = [output] if output else []
    parts.append(f"[exit code: {exit_code}]")
    return ToolResult(content="\n".join(parts), ok=exit_code == 0)


def _shell_jobs(args: dict[str, Any], session: Any) -> ToolResult:
    action = str(args.get("action") or "").strip().lower()
    if not action:
        return ToolResult.error("'action' is required (start, list, output, kill)")
    manager = get_shell(session)

    if action == "start":
        command = str(args.get("command") or "")
        if not command.strip():
            return ToolResult.error("'command' is required for action 'start'")
        try:
            job_id = manager.start_background(command)
        except ShellJobLimitError as exc:
            return ToolResult.error(str(exc))
        except ShellError as exc:
            return ToolResult.error(str(exc))
        job = manager.get_background(job_id)
        output_path = str(job.output_path) if job is not None else ""
        return ToolResult(
            content=(
                f"Started background job {job_id} (output at {output_path}). "
                "Use shell_jobs with action 'output' to read it."
            )
        )

    if action == "list":
        jobs = manager.list_background()
        if not jobs:
            return ToolResult(content="No background jobs.")
        lines = []
        for job in jobs:
            state = "running" if job.alive else f"exited ({job.exit_code})"
            lines.append(f"{job.id}  [{state}]  {job.command}")
        return ToolResult(content="\n".join(lines))

    job_id = str(args.get("id") or "").strip()
    if not job_id:
        return ToolResult.error(f"'id' is required for action '{action}'")

    if action == "output":
        text = manager.output_background(job_id)
        if text is None:
            return ToolResult.error(f"Unknown background job '{job_id}'")
        return ToolResult(content=text or "(no output yet)")

    if action == "kill":
        if not manager.kill_background(job_id):
            return ToolResult.error(f"Unknown background job '{job_id}'")
        return ToolResult(content=f"Killed background job {job_id}.")

    return ToolResult.error("Unknown action; choose start, list, output, or kill")


shell_tool = FunctionTool(
    name="shell",
    description=(
        "Run a command in a persistent shell that keeps its working directory "
        "and exported environment between calls. Prefer this over run_bash when "
        "state must persist across commands (e.g. `cd`, `export`, virtualenv "
        "activation). Returns the exit code and combined output."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
        },
        "required": ["command"],
    },
    func=_run_shell,
    runs=True,
)

shell_jobs_tool = FunctionTool(
    name="shell_jobs",
    description=(
        "Manage background shell processes. Actions: start (run a command "
        "detached; 'command' required), list, output ('id' required), kill "
        "('id' required). Background output is captured to a file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "list", "output", "kill"],
                "description": "The background-job action to perform.",
            },
            "command": {
                "type": "string",
                "description": "Command to start in the background (action 'start').",
            },
            "id": {
                "type": "string",
                "description": "Background job id (actions 'output' and 'kill').",
            },
        },
        "required": ["action"],
    },
    func=_shell_jobs,
    runs=True,
)
