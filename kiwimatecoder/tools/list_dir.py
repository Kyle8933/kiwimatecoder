"""list_dir tool: list the entries of a directory in the workspace."""

from __future__ import annotations

from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult
from kiwimatecoder.tools.paths import PathError, resolve_in_workspace

MAX_ENTRIES = 500
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache"}


def _list_dir(args: dict, session: Session) -> ToolResult:
    path = args.get("path", ".") or "."
    try:
        resolved = resolve_in_workspace(path, session.workspace_root)
    except PathError as exc:
        return ToolResult.error(str(exc))

    if not resolved.exists():
        return ToolResult.error(f"Directory not found: {path}")
    if not resolved.is_dir():
        return ToolResult.error(f"'{path}' is not a directory, use read_file")

    entries = []
    for child in sorted(resolved.iterdir(), key=lambda p: (p.is_file(), p.name)):
        if child.name in SKIP_DIRS:
            continue
        suffix = "/" if child.is_dir() else ""
        entries.append(child.name + suffix)
        if len(entries) >= MAX_ENTRIES:
            entries.append(f"... [truncated at {MAX_ENTRIES} entries]")
            break

    if not entries:
        return ToolResult(content="[empty directory]")
    return ToolResult(content="\n".join(entries))


list_dir_tool = FunctionTool(
    name="list_dir",
    description="List the files and subdirectories of a directory in the workspace.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path relative to the workspace root. "
                "Defaults to the workspace root.",
            },
        },
    },
    func=_list_dir,
)
