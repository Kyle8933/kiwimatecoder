"""search tool: grep-style content search and glob-style filename search."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult
from kiwimatecoder.tools.paths import (
    PathError,
    display_path,
    get_workspace_ignore,
    resolve_in_workspace,
)

MAX_MATCHES = 200


def _iter_files(root: Path, session: Session, glob_pattern: str | None = None):
    ignore = get_workspace_ignore(session.workspace_root)
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        dir_p = Path(dirpath)
        dirnames[:] = [
            d for d in dirnames
            if not ignore.is_ignored(dir_p / d, is_dir=True)
        ]
        dirnames.sort()
        for f in sorted(filenames):
            file_p = dir_p / f
            if ignore.is_ignored(file_p, is_dir=False):
                continue
            if glob_pattern:
                rel = display_path(file_p, session.workspace_root)
                if not fnmatch.fnmatch(f, glob_pattern) and not fnmatch.fnmatch(
                    rel, glob_pattern
                ):
                    continue
            yield file_p


def _glob_search(root: Path, pattern: str, session: Session) -> list[str]:
    ignore = get_workspace_ignore(session.workspace_root)
    results: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        dir_p = Path(dirpath)
        dirnames[:] = [
            d for d in dirnames
            if not ignore.is_ignored(dir_p / d, is_dir=True)
        ]
        dirnames.sort()
        for name in sorted(filenames + dirnames):
            p = dir_p / name
            is_dir = name in dirnames
            if ignore.is_ignored(p, is_dir=is_dir):
                continue
            rel = display_path(p, session.workspace_root)
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel, pattern):
                results.append(rel)
                if len(results) >= MAX_MATCHES:
                    return results
    return results


def _grep_search(
    root: Path, pattern: str, glob: str | None, session: Session
) -> list[str]:
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid regex: {exc}") from exc

    results: list[str] = []
    for p in _iter_files(root, session, glob):
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
