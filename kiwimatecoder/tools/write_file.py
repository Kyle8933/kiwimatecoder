"""write_file tool: create or overwrite a file in the workspace."""

from __future__ import annotations

import difflib

from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult
from kiwimatecoder.tools.paths import (
    PathError,
    atomic_write_text,
    display_path,
    resolve_in_workspace,
)


def preview(args: dict, session: Session) -> str:
    """Return a human-readable preview (diff) for the approval prompt."""
    path = args.get("path", "")
    content = args.get("content", "")
    try:
        resolved = resolve_in_workspace(path, session.workspace_root)
    except PathError as exc:
        return str(exc)
    rel = display_path(resolved, session.workspace_root)
    if resolved.exists():
        old = resolved.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        new = content.splitlines(keepends=True)
        diff = "".join(
            difflib.unified_diff(old, new, fromfile=rel, tofile=rel, n=3)
        )
        return diff or f"(no changes to {rel})"
    return f"+++ create {rel}\n" + "\n".join(
        f"+{line}" for line in content.splitlines()
    )


def _write_file(args: dict, session: Session) -> ToolResult:
    path = args.get("path")
    if not path:
        return ToolResult.error("'path' is required")
    content = args.get("content", "")
    try:
        resolved = resolve_in_workspace(path, session.workspace_root)
    except PathError as exc:
        return ToolResult.error(str(exc))

    existed = resolved.exists()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write_text(resolved, content)
    except OSError as exc:
        return ToolResult.error(f"Could not write file: {exc}")
    session.record_touched(display_path(resolved, session.workspace_root))
    verb = "Updated" if existed else "Created"
    return ToolResult(content=f"{verb} {path} ({len(content)} bytes).")


write_file_tool = FunctionTool(
    name="write_file",
    description=(
        "Create a new file or overwrite an existing one with the given content. "
        "Parent directories are created automatically."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path relative to the workspace root.",
            },
            "content": {
                "type": "string",
                "description": "Full content to write to the file.",
            },
        },
        "required": ["path", "content"],
    },
    func=_write_file,
    writes=True,
)
