"""update_todos tool: keep an explicit task list for multi-step work."""

from __future__ import annotations

from typing import Any

from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult

VALID_STATUSES = ("pending", "in_progress", "completed")
_MARKERS = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


def _update_todos(args: dict[str, Any], session: Session) -> ToolResult:
    raw = args.get("todos")
    if not isinstance(raw, list):
        return ToolResult.error("'todos' must be a list of {content, status} objects")

    cleaned: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            return ToolResult.error(
                "each todo must be an object with 'content' and 'status'"
            )
        content = str(item.get("content") or "").strip()
        status = str(item.get("status") or "pending").strip().lower()
        if not content:
            return ToolResult.error("todo 'content' is required")
        if status not in VALID_STATUSES:
            return ToolResult.error(
                f"Unknown todo status '{status}'. "
                f"Choose: {', '.join(VALID_STATUSES)}."
            )
        cleaned.append({"content": content, "status": status})

    session.todos = cleaned
    if not cleaned:
        return ToolResult(content="Task list cleared.")

    lines = [f"{_MARKERS[t['status']]} {t['content']}" for t in cleaned]
    done = sum(1 for todo in cleaned if todo["status"] == "completed")
    return ToolResult(content="\n".join(lines) + f"\n\n{done}/{len(cleaned)} complete.")


update_todos_tool = FunctionTool(
    name="update_todos",
    description=(
        "Replace the task list with the given items. Use it for multi-step work: "
        "keep exactly one task 'in_progress' at a time and mark tasks "
        "'completed' as soon as they are done. Passing an empty list clears it."
    ),
    parameters={
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "The full task list, in order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Short imperative description of the task.",
                        },
                        "status": {
                            "type": "string",
                            "enum": list(VALID_STATUSES),
                        },
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["todos"],
    },
    func=_update_todos,
)
