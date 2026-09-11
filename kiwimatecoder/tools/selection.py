"""Apply a hunk-level approval selection to file-changing tool arguments."""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from typing import Any

from kiwimatecoder import hunks
from kiwimatecoder.session import Session
from kiwimatecoder.tools.edit_file import compute_new_content
from kiwimatecoder.tools.paths import PathError, display_path, resolve_in_workspace


def _unified_diff(old_text: str, new_text: str, rel: str) -> str:
    return "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=rel,
            tofile=rel,
            n=3,
        )
    )


def _selection_in_range(selected: Sequence[int], count: int) -> bool:
    return all(1 <= index <= count for index in selected)


def _select_write(
    args: dict[str, Any], session: Session, selected: Sequence[int]
) -> dict[str, Any] | None:
    path = str(args.get("path") or "")
    if not path:
        return None
    content = str(args.get("content", "") or "")
    try:
        resolved = resolve_in_workspace(path, session.workspace_root)
    except PathError:
        return None
    if resolved.exists():
        old_text = resolved.read_text(encoding="utf-8", errors="replace")
    else:
        old_text = ""
    rel = display_path(resolved, session.workspace_root)
    diff = _unified_diff(old_text, content, rel)
    if not diff:
        return {**args}
    parsed = hunks.split_hunks(diff)
    if not parsed or not _selection_in_range(selected, len(parsed)):
        return None
    new_content = hunks.apply_selected_hunks(old_text, diff, selected)
    if new_content == old_text:
        return None
    return {**args, "content": new_content}


def _select_edit(
    args: dict[str, Any], session: Session, selected: Sequence[int]
) -> dict[str, Any] | None:
    path = str(args.get("path") or "")
    if not path:
        return None
    try:
        resolved = resolve_in_workspace(path, session.workspace_root)
    except PathError:
        return None
    if not resolved.exists():
        return None
    old_text = resolved.read_text(encoding="utf-8", errors="replace")
    new_text = compute_new_content(old_text, args)
    if new_text is None:
        return None
    rel = display_path(resolved, session.workspace_root)
    diff = _unified_diff(old_text, new_text, rel)
    if not diff:
        return {**args}
    parsed = hunks.split_hunks(diff)
    if not parsed or not _selection_in_range(selected, len(parsed)):
        return None
    new_content = hunks.apply_selected_hunks(old_text, diff, selected)
    if new_content == old_text:
        return None
    return {
        **args,
        "old_string": old_text,
        "new_string": new_content,
        "replace_all": False,
    }


def select_hunks(
    name: str,
    args: dict[str, Any],
    session: Session,
    selected_hunks: Sequence[int],
) -> dict[str, Any] | None:
    """Return ``args`` rewritten to apply only ``selected_hunks``.

    Returns ``None`` when the selection cannot be applied; callers should treat
    that as a denial. Tools without hunk support also return ``None``.
    """
    if name == "write_file":
        return _select_write(args, session, selected_hunks)
    if name == "edit_file":
        return _select_edit(args, session, selected_hunks)
    return None