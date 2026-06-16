"""Workspace path resolution and sandboxing.

Every file-touching tool resolves its path through :func:`resolve_in_workspace`,
which rejects paths that escape the workspace root (via ``..`` or symlinks).
"""

from __future__ import annotations

from pathlib import Path


class PathError(ValueError):
    """Raised when a path escapes the workspace root."""


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
            "Access is restricted to the current project directory."
        )
    return resolved


def display_path(path: Path, workspace_root: Path) -> str:
    """Return a path relative to the workspace root for display, if possible."""
    root = workspace_root.resolve()
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)
