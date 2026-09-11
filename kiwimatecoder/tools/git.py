"""git (read-only) and git_write (approval-gated) tools.

The read-only tool is safe to advertise in plan mode. ``git_write`` is marked
``writes=True`` so the permission gate prompts with the exact command as
preview. Checkpointing only covers file tools, so git actions rely on the
approval prompt instead.
"""

from __future__ import annotations

from typing import Any

from kiwimatecoder import git as git_module
from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult

READ_ACTIONS = ("status", "diff", "log", "show", "branches")
WRITE_ACTIONS = git_module.WRITE_ACTIONS

_LOG_LIMIT_DEFAULT = 20


def _truncate(text: str) -> str:
    if len(text) > git_module.MAX_OUTPUT:
        return text[: git_module.MAX_OUTPUT] + "\n... [output truncated]"
    return text


def _render(code: int, output: str) -> ToolResult:
    output = output.strip()
    body = output if output else "(no output)"
    content = f"{_truncate(body)}\n[exit code: {code}]"
    return ToolResult(content=content, ok=code == 0)


def _git(args: dict[str, Any], session: Session) -> ToolResult:
    action = str(args.get("action") or "status").strip().lower()
    if action not in READ_ACTIONS:
        return ToolResult.error(
            f"Unknown git action '{action}'. Choose: {', '.join(READ_ACTIONS)}."
        )
    path = args.get("path")
    path_text = str(path).strip() if path is not None else None
    ref = args.get("ref")
    ref_text = str(ref).strip() if ref is not None else None

    if action == "status":
        command = git_module.status_args()
    elif action == "diff":
        command = git_module.diff_args(
            staged=bool(args.get("staged", False)), path=path_text, ref=ref_text
        )
    elif action == "log":
        raw_limit = args.get("limit")
        try:
            limit = int(raw_limit) if raw_limit is not None else _LOG_LIMIT_DEFAULT
        except (TypeError, ValueError):
            return ToolResult.error("'limit' must be an integer")
        command = git_module.log_args(limit, path=path_text, oneline=True)
    elif action == "show":
        command = git_module.show_args(ref_text or "HEAD")
    else:
        command = git_module.branch_args()

    code, output = git_module.run_git(command, session)
    return _render(code, output)


def _git_write(args: dict[str, Any], session: Session) -> ToolResult:
    action = str(args.get("action") or "").strip().lower()
    if action not in WRITE_ACTIONS:
        return ToolResult.error(
            f"Unknown git_write action '{action}'. Choose: {', '.join(WRITE_ACTIONS)}."
        )
    try:
        command = git_module.build_write_args(args)
    except ValueError as exc:
        return ToolResult.error(str(exc))

    code, output = git_module.run_git(command, session, timeout=60)
    return _render(code, output)


git_tool = FunctionTool(
    name="git",
    description=(
        "Inspect the repository (read-only): status, diff (staged or not, "
        "optionally against a ref), log (1-100 commits), show a ref, or list "
        "branches. Use it before staging or committing to check what changed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(READ_ACTIONS),
                "description": "Which read-only git command to run.",
            },
            "path": {
                "type": "string",
                "description": "Limit diff/log to this path.",
            },
            "ref": {
                "type": "string",
                "description": "Commit/branch for diff or show (defaults to HEAD).",
            },
            "staged": {
                "type": "boolean",
                "description": "Diff the staged changes instead of the working tree.",
            },
            "limit": {
                "type": "integer",
                "description": "Number of log entries (1-100, default 20).",
            },
        },
        "required": ["action"],
    },
    func=_git,
    writes=False,
    runs=False,
)

git_write_tool = FunctionTool(
    name="git_write",
    description=(
        "Make a git change: stage paths, unstage paths, commit with a message, "
        "or create and switch to a new branch. Requires user approval. Push, "
        "reset, and clean are intentionally unavailable."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(WRITE_ACTIONS),
                "description": "Which git mutation to perform.",
            },
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Paths to stage or unstage.",
            },
            "message": {
                "type": "string",
                "description": "Commit message (required for commit).",
            },
            "branch": {
                "type": "string",
                "description": "New branch name (required for checkout_branch).",
            },
        },
        "required": ["action"],
    },
    func=_git_write,
    writes=True,
)

preview = git_module.preview
