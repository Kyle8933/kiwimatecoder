"""forge (read-only) and forge_write (approval-gated) tools.

Both delegate to the official forge CLI (``gh``/``glab``) so the user's
existing login is used. Only list/view and create actions are exposed; merge,
close, and delete are intentionally not implemented.
"""

from __future__ import annotations

import shlex
from typing import Any

from kiwimatecoder import forge as forge_module
from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult

READ_ACTIONS = forge_module.READ_ACTIONS
WRITE_ACTIONS = forge_module.WRITE_ACTIONS


def _truncate(text: str) -> str:
    if len(text) > forge_module.MAX_OUTPUT:
        return text[: forge_module.MAX_OUTPUT] + "\n... [output truncated]"
    return text


def _render(cli: str, action: str, code: int, output: str) -> ToolResult:
    output = output.strip()
    body = output if output else "(no output)"
    content = f"{cli} {action}:\n{_truncate(body)}\n[exit code: {code}]"
    return ToolResult(content=content, ok=code == 0)


def _resolve(session: Session) -> tuple[str | None, str | None, ToolResult | None]:
    """Detect the forge and CLI, or return a clear error result."""
    remote = forge_module.current_remote_url(session)
    if not remote:
        return (
            None,
            None,
            ToolResult.error(
                "No git remote 'origin' found in this workspace. Add one with "
                "`git remote add origin <url>` and try again."
            ),
        )
    kind = forge_module.detect_forge(remote)
    if kind is None:
        return (
            None,
            None,
            ToolResult.error(
                f"Remote '{remote}' is not a supported forge host. Only "
                "github.com and gitlab.com remotes are supported."
            ),
        )
    cli = forge_module.forge_cli(kind)
    if not cli or not forge_module.cli_available(cli):
        return (
            None,
            None,
            ToolResult.error(
                f"The '{cli or kind}' CLI is required for {kind} but was not "
                "found on PATH. Install and authenticate it, then try again."
            ),
        )
    return kind, cli, None


def _number(args: dict[str, Any]) -> tuple[int | None, ToolResult | None]:
    raw = args.get("number")
    if raw is None:
        return None, ToolResult.error("'number' must be an integer")
    try:
        number = int(raw)
    except (TypeError, ValueError):
        return None, ToolResult.error("'number' must be an integer")
    if number < 1:
        return None, ToolResult.error("'number' must be a positive integer")
    return number, None


def _forge(args: dict[str, Any], session: Session) -> ToolResult:
    kind, cli, error = _resolve(session)
    if error is not None:
        return error
    assert kind is not None and cli is not None

    action = str(args.get("action") or "").strip().lower()
    if action not in READ_ACTIONS:
        return ToolResult.error(
            f"Unknown forge action '{action}'. Choose: {', '.join(READ_ACTIONS)}."
        )

    command: list[str]
    if action == "pr_list":
        command = forge_module.pr_list_args(kind)
    elif action == "issue_list":
        command = forge_module.issue_list_args()
    elif action == "pr_view":
        number, number_error = _number(args)
        if number_error is not None:
            return number_error
        command = forge_module.pr_view_args(number if number is not None else 0, forge=kind)
    else:  # issue_view
        number, number_error = _number(args)
        if number_error is not None:
            return number_error
        command = forge_module.issue_view_args(number if number is not None else 0)

    code, output = forge_module.run_forge(cli, command, session)
    return _render(cli, action, code, output)


def _forge_write(args: dict[str, Any], session: Session) -> ToolResult:
    kind, cli, error = _resolve(session)
    if error is not None:
        return error
    assert kind is not None and cli is not None

    try:
        command = forge_module.build_write_args(kind, args)
    except ValueError as exc:
        return ToolResult.error(str(exc))

    code, output = forge_module.run_forge(cli, command, session)
    action = str(args.get("action") or "").strip().lower()
    return _render(cli, action, code, output)


def preview(args: dict[str, Any], session: Session) -> str:
    """Return the exact CLI command line for the approval prompt."""
    kind, cli, error = _resolve(session)
    if error is not None:
        return error.content
    assert kind is not None and cli is not None
    try:
        command = forge_module.build_write_args(kind, args)
    except ValueError as exc:
        return str(exc)
    return f"{cli} " + " ".join(shlex.quote(part) for part in command)


forge_tool = FunctionTool(
    name="forge",
    description=(
        "Read GitHub/GitLab data through the gh/glab CLI: list or view pull "
        "requests (merge requests on GitLab) and issues. The forge is detected "
        "from the origin remote. Requires the matching CLI to be installed and "
        "authenticated."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(READ_ACTIONS),
                "description": "Which forge query to run.",
            },
            "number": {
                "type": "integer",
                "description": "PR/MR or issue number for the view actions.",
            },
        },
        "required": ["action"],
    },
    func=_forge,
    writes=False,
    runs=False,
)

forge_write_tool = FunctionTool(
    name="forge_write",
    description=(
        "Create a pull request (merge request on GitLab) or an issue using "
        "gh/glab. Requires user approval. Merge, close, and delete are "
        "intentionally unavailable."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(WRITE_ACTIONS),
                "description": "Which forge object to create.",
            },
            "title": {
                "type": "string",
                "description": "Title for the PR/MR or issue.",
            },
            "body": {
                "type": "string",
                "description": "Body text for the PR/MR or issue.",
            },
            "base": {
                "type": "string",
                "description": "Base/target branch for a PR/MR.",
            },
            "draft": {
                "type": "boolean",
                "description": "Open a PR/MR as a draft.",
            },
        },
        "required": ["action", "title"],
    },
    func=_forge_write,
    writes=True,
)
