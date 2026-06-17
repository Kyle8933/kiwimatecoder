"""edit_file tool: replace a unique string in an existing file."""

from __future__ import annotations

import difflib

from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult
from kiwimatecoder.tools.paths import PathError, display_path, resolve_in_workspace


class EditError(ValueError):
    """Raised when an edit cannot be applied unambiguously."""


def compute_edit(text: str, old: str, new: str, replace_all: bool) -> str:
    """Apply the edit to ``text`` or raise :class:`EditError`."""
    if old == new:
        raise EditError("old_string and new_string are identical")
    count = text.count(old)
    if count == 0:
        raise EditError("old_string not found in file")
    if count > 1 and not replace_all:
        raise EditError(
            f"old_string is not unique ({count} matches); add surrounding "
            "context to disambiguate or set replace_all=true"
        )
    if replace_all:
        return text.replace(old, new)
    return text.replace(old, new, 1)


def _diff(old_text: str, new_text: str, rel: str) -> str:
    return "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=rel,
            tofile=rel,
            n=3,
        )
    )


def preview(args: dict, session: Session) -> str:
    """Return a unified diff for the approval prompt."""
    path = args.get("path", "")
    try:
        resolved = resolve_in_workspace(path, session.workspace_root)
    except PathError as exc:
        return str(exc)
    if not resolved.exists():
        return f"(file not found: {path})"
    rel = display_path(resolved, session.workspace_root)
    old_text = resolved.read_text(errors="replace")
    try:
        new_text = compute_edit(
            old_text,
            args.get("old_string", ""),
            args.get("new_string", ""),
            bool(args.get("replace_all")),
        )
    except EditError as exc:
        return f"(cannot apply edit: {exc})"
    return _diff(old_text, new_text, rel) or "(no changes)"


def _edit_file(args: dict, session: Session) -> ToolResult:
    path = args.get("path")
    if not path:
        return ToolResult.error("'path' is required")
    try:
        resolved = resolve_in_workspace(path, session.workspace_root)
    except PathError as exc:
        return ToolResult.error(str(exc))
    if not resolved.exists():
        return ToolResult.error(f"File not found: {path}")

    old_text = resolved.read_text(errors="replace")
    try:
        new_text = compute_edit(
            old_text,
            args.get("old_string", ""),
            args.get("new_string", ""),
            bool(args.get("replace_all")),
        )
    except EditError as exc:
        return ToolResult.error(str(exc))

    resolved.write_text(new_text)
    session.record_touched(display_path(resolved, session.workspace_root))
    return ToolResult(content=f"Edited {path}.")


edit_file_tool = FunctionTool(
    name="edit_file",
    description=(
        "Replace an exact string in an existing file. old_string must match "
        "uniquely unless replace_all is true. Include enough surrounding context "
        "to make the match unambiguous."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path relative to the workspace root.",
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to find and replace.",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence instead of requiring uniqueness.",
            },
        },
        "required": ["path", "old_string", "new_string"],
    },
    func=_edit_file,
    writes=True,
)
