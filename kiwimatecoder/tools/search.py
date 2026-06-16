"""search tool: grep-style content search and glob-style filename search."""

from __future__ import annotations

import re
from pathlib import Path

from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult
from kiwimatecoder.tools.paths import PathError, display_path, resolve_in_workspace

MAX_MATCHES = 200
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache"}


def _iter_files(root: Path):
    for p in root.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_file():
            yield p


def _glob_search(root: Path, pattern: str, session: Session) -> list[str]:
    results = []
    for p in sorted(root.rglob(pattern)):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        results.append(display_path(p, session.workspace_root))
        if len(results) >= MAX_MATCHES:
            break
    return results


def _grep_search(
    root: Path, pattern: str, glob: str | None, session: Session
) -> list[str]:
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid regex: {exc}") from exc

    results: list[str] = []
    files = root.rglob(glob) if glob else _iter_files(root)
    for p in sorted(files):
        if any(part in SKIP_DIRS for part in p.parts) or not p.is_file():
            continue
        try:
            data = p.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:1024]:
            continue
        rel = display_path(p, session.workspace_root)
        for lineno, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            if regex.search(line):
                results.append(f"{rel}:{lineno}: {line.strip()}")
                if len(results) >= MAX_MATCHES:
                    return results
    return results


def _search(args: dict, session: Session) -> ToolResult:
    pattern = args.get("pattern")
    if not pattern:
        return ToolResult.error("'pattern' is required")
    mode = args.get("mode", "grep")
    path = args.get("path", ".") or "."
    try:
        root = resolve_in_workspace(path, session.workspace_root)
    except PathError as exc:
        return ToolResult.error(str(exc))
    if not root.is_dir():
        return ToolResult.error(f"'{path}' is not a directory")

    try:
        if mode == "glob":
            results = _glob_search(root, pattern, session)
        else:
            results = _grep_search(root, pattern, args.get("glob"), session)
    except ValueError as exc:
        return ToolResult.error(str(exc))

    if not results:
        return ToolResult(content="No matches found.")
    header = ""
    if len(results) >= MAX_MATCHES:
        header = f"[showing first {MAX_MATCHES} matches]\n"
    return ToolResult(content=header + "\n".join(results))


search_tool = FunctionTool(
    name="search",
    description=(
        "Search the workspace. mode='grep' finds a regex in file contents "
        "(optionally filtered by a glob); mode='glob' finds files by name "
        "pattern (e.g. '**/*.py')."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex (grep mode) or filename glob (glob mode).",
            },
            "mode": {
                "type": "string",
                "enum": ["grep", "glob"],
                "description": "Search mode. Defaults to grep.",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in. Defaults to workspace root.",
            },
            "glob": {
                "type": "string",
                "description": "Optional filename glob to restrict grep (e.g. '**/*.py').",
            },
        },
        "required": ["pattern"],
    },
    func=_search,
)
