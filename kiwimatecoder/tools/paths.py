"""Workspace path resolution and sandboxing.

Every file-touching tool resolves its path through :func:`resolve_in_workspace`,
which rejects paths that escape the workspace root (via ``..`` or symlinks).
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

DEFAULT_SKIP_DIRS: set[str] = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".eggs",
}


class WorkspaceIgnore:
    """Evaluates whether paths should be ignored based on default skip dirs and .gitignore."""

    workspace_root: Path
    rules: list[tuple[bool, str, bool]]

    def __init__(
        self,
        workspace_root: Path,
        rules: list[tuple[bool, str, bool]] | None = None,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.rules = (
            rules if rules is not None else self._load_rules()
        )

    def _load_rules(self) -> list[tuple[bool, str, bool]]:
        gitignore_path = self.workspace_root / ".gitignore"
        if not gitignore_path.is_file():
            return []
        rules: list[tuple[bool, str, bool]] = []
        try:
            content = gitignore_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            is_neg = line.startswith("!")
            if is_neg:
                line = line[1:].strip()
            if not line:
                continue
            dir_only = line.endswith("/")
            if dir_only:
                line = line[:-1]
            if line.startswith("/"):
                line = line[1:]
            rules.append((is_neg, line, dir_only))
        return rules

    def is_ignored(self, path: Path, is_dir: bool = False) -> bool:
        """Check if path should be ignored."""
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path

        parts = resolved.parts
        if any(part in DEFAULT_SKIP_DIRS for part in parts):
            return True

        if not self.rules:
            return False

        try:
            rel = resolved.relative_to(self.workspace_root).as_posix()
        except ValueError:
            rel = path.as_posix()

        ignored = False
        name = resolved.name
        for is_neg, pat, dir_only in self.rules:
            matches = False
            if dir_only:
                if is_dir:
                    if "/" in pat:
                        matches = fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(
                            rel, f"{pat}/*"
                        )
                    else:
                        matches = (
                            fnmatch.fnmatch(name, pat)
                            or fnmatch.fnmatch(rel, f"**/{pat}")
                            or fnmatch.fnmatch(rel, f"{pat}/*")
                        )
                else:
                    if "/" in pat:
                        matches = fnmatch.fnmatch(
                            rel, f"{pat}/*"
                        ) or fnmatch.fnmatch(rel, f"**/{pat}/*")
                    else:
                        matches = (
                            fnmatch.fnmatch(rel, f"{pat}/*")
                            or fnmatch.fnmatch(rel, f"**/{pat}/*")
                            or (pat in resolved.parts)
                        )
            else:
                if "/" in pat:
                    matches = fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(
                        rel, f"{pat}/*"
                    )
                else:
                    matches = (
                        fnmatch.fnmatch(name, pat)
                        or fnmatch.fnmatch(rel, f"**/{pat}")
                        or fnmatch.fnmatch(rel, f"{pat}/*")
                    )

            if matches:
                ignored = not is_neg

        return ignored


_IGNORE_CACHE: dict[str, tuple[float, WorkspaceIgnore]] = {}


def get_workspace_ignore(workspace_root: Path) -> WorkspaceIgnore:
    """Return a cached WorkspaceIgnore instance for the given root."""
    root = workspace_root.resolve()
    key = str(root)
    gi_path = root / ".gitignore"
    try:
        mtime = gi_path.stat().st_mtime if gi_path.exists() else 0.0
    except OSError:
        mtime = 0.0
    cached = _IGNORE_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    ignore = WorkspaceIgnore(root)
    _IGNORE_CACHE[key] = (mtime, ignore)
    return ignore


class PathError(ValueError):
    """Raised when a path escapes the workspace root."""


def atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically as UTF-8.

    Writes to a sibling temp file then ``os.replace``s it into place, so a
    crash mid-write cannot truncate the destination. Parent directories must
    already exist.
    """
    tmp = path.with_name(path.name + ".kiwi.tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def resolve_in_workspace(path: str, workspace_root: Path) -> Path:
    """Resolve ``path`` against ``workspace_root`` and ensure it stays inside.

    Symlinks are resolved before the containment check so they cannot be used
    to escape the sandbox.
    """
    root = workspace_root.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise PathError(
            f"Path '{path}' is outside the workspace root ({root}). "
            + "Access is restricted to the current project directory."
        )
    return resolved


def display_path(path: Path, workspace_root: Path) -> str:
    """Return a path relative to the workspace root for display, if possible."""
    root = workspace_root.resolve()
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)
