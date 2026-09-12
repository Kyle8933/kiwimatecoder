"""remember (approval-gated) and recall (read-only) persistent-memory tools."""

from __future__ import annotations

from typing import Any

from kiwimatecoder import config
from kiwimatecoder import memory as memory_module
from kiwimatecoder.redaction import redact
from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult


def _remember(args: dict[str, Any], session: Session) -> ToolResult:
    content = str(args.get("content") or "").strip()
    if not content:
        return ToolResult.error("'content' is required")
    scope = str(args.get("scope") or "project").strip().lower()
    try:
        path = memory_module.append_memory(scope, session.workspace_root, content)
    except ValueError as exc:
        return ToolResult.error(str(exc))
    return ToolResult(content=f"Remembered ({scope}): {path}\n- {content}")


def _recall(args: dict[str, Any], session: Session) -> ToolResult:
    query = str(args.get("query") or "").strip()
    settings = config.get_memory()
    cap = int(settings["max_bytes"])
    sections: list[str] = []
    for scope in memory_module.SCOPES:
        text = memory_module.read_memory(scope, session.workspace_root, cap)
        if not text:
            continue
        if query:
            text = "\n".join(
                line
                for line in text.splitlines()
                if query.lower() in line.lower()
            )
        if not text.strip():
            continue
        path = memory_module.memory_path(scope, session.workspace_root)
        sections.append(f"{scope} memory ({path}):\n{text}")
    if not sections:
        if query:
            return ToolResult(content=f"No memory entries match {query!r}.")
        return ToolResult(content="No memory stored yet.")
    body = "\n\n".join(sections)
    if len(body) > cap:
        body = body[:cap] + "\n[truncated]"
    return ToolResult(content=body)


def preview(args: dict[str, Any], session: Session) -> str:
    """Show the target file and bullet for the approval prompt."""
    scope = str(args.get("scope") or "project").strip().lower()
    try:
        path = memory_module.memory_path(scope, session.workspace_root)
    except ValueError as exc:
        return str(exc)
    content = str(args.get("content") or "").strip()
    if not content:
        return f"Append to memory {path}: (empty; will be rejected)"
    return f"Append to memory {path}:\n- {redact(content)}"


remember_tool = FunctionTool(
    name="remember",
    description=(
        "Persist a durable fact about the user or this project so it survives "
        "future sessions. Use scope='project' for workspace-specific facts and "
        "scope='user' for preferences that apply everywhere. Requires approval; "
        "never store secrets, credentials, or API keys."
    ),
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact to remember, as a short statement.",
            },
            "scope": {
                "type": "string",
                "enum": list(memory_module.SCOPES),
                "description": (
                    "Where to store it: project (this workspace) or user "
                    "(all workspaces). Defaults to project."
                ),
            },
        },
        "required": ["content"],
    },
    func=_remember,
    writes=True,
)

recall_tool = FunctionTool(
    name="recall",
    description=(
        "Read persistent memory for the project and user. Pass a query to "
        "filter entries by case-insensitive substring. Read-only, so it is "
        "available in plan mode."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional case-insensitive filter.",
            },
        },
    },
    func=_recall,
)
