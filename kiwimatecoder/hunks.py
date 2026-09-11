"""Pure helpers for parsing unified diffs and applying hunk selections.

The approval UI can offer per-hunk review of a diff; this module turns a diff
into :class:`Hunk` records and rebuilds file content from only the selected
hunks. It performs no I/O and never touches the filesystem.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

# The body of a hunk is built from these prefix characters; ``\`` marks a GNU
# "no newline at end of file" line.
_BODY_PREFIXES = (" ", "+", "-", "\\")


@dataclass(frozen=True)
class Hunk:
    """One ``@@`` hunk of a unified diff."""

    index: int
    header: str
    lines: tuple[str, ...]
    old_start: int
    old_count: int
    new_start: int
    new_count: int


def split_hunks(diff_text: str) -> list[Hunk]:
    """Parse ``diff_text`` into 1-based-indexed hunks.

    Returns an empty list when the text carries no parseable unified-diff
    hunks (for example the ``+++ create`` preview used for new files).
    """
    hunks: list[Hunk] = []
    body: list[str] = []
    current: tuple[str, int, int, int, int] | None = None

    def flush() -> None:
        nonlocal current, body
        if current is None:
            return
        header, old_start, old_count, new_start, new_count = current
        hunks.append(
            Hunk(
                index=len(hunks) + 1,
                header=header,
                lines=tuple(body),
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
            )
        )
        body = []

    for line in diff_text.splitlines(keepends=True):
        match = _HEADER_RE.match(line)
        if match:
            flush()
            current = (
                line.rstrip("\n"),
                int(match.group(1)),
                int(match.group(2) or 1),
                int(match.group(3)),
                int(match.group(4) or 1),
            )
            continue
        if current is not None and line[:1] in _BODY_PREFIXES:
            body.append(line)
    flush()
    return hunks


def _parse_body(
    body: Sequence[str],
    old_lines: Sequence[str],
    old_index: int,
    old_count: int,
) -> list[tuple[str, str]] | None:
    """Split physical body lines into ``(prefix, content)`` pairs.

    ``difflib`` does not always terminate a diff line with a newline, so a
    physical line can glue several logical segments together (for example
    ``-old+new`` at end of file). Context and deletion segments are anchored
    against the original lines, which disambiguates glued additions.
    """
    out: list[tuple[str, str]] = []

    def walk(line_no: int, position: int, old_i: int) -> int | None:
        if line_no == len(body):
            return old_i
        raw = body[line_no]
        if position == len(raw):
            return walk(line_no + 1, 0, old_i)
        char = raw[position]
        if char == "\n":
            return walk(line_no, position + 1, old_i)
        if char in (" ", "-"):
            if old_i >= len(old_lines):
                return None
            expected = old_lines[old_i]
            end = position + 1 + len(expected)
            if raw[position + 1 : end] != expected:
                return None
            out.append((char, expected))
            result = walk(line_no, end, old_i + 1)
            if result is not None:
                return result
            out.pop()
            return None
        if char == "+":
            candidates = [len(raw)]
            for candidate in range(position + 1, len(raw)):
                if raw[candidate] not in (" ", "-") or old_i >= len(old_lines):
                    continue
                expected = old_lines[old_i]
                end = candidate + 1 + len(expected)
                if raw[candidate + 1 : end] == expected:
                    candidates.append(candidate)
            for candidate in sorted(set(candidates), reverse=True):
                out.append(("+", raw[position + 1 : candidate]))
                result = walk(line_no, candidate, old_i)
                if result is not None:
                    return result
                out.pop()
            return None
        if char == "\\":
            if out:
                prefix, content = out[-1]
                out[-1] = (prefix, content.rstrip("\n"))
                result = walk(line_no, len(raw), old_i)
                if result is not None:
                    return result
                out[-1] = (prefix, content)
                return None
            return walk(line_no, len(raw), old_i)
        return None

    consumed = walk(0, 0, old_index)
    if consumed is None or consumed != old_index + old_count:
        return None
    return out


def _naive_segments(body: Sequence[str]) -> list[tuple[str, str]]:
    """Best-effort parse for diffs whose old lines are not consistent."""
    segments: list[tuple[str, str]] = []
    for line in body:
        if line.startswith("\\"):
            continue
        if line[:1] in (" ", "+", "-"):
            segments.append((line[:1], line[1:]))
    return segments


def _creation_target(diff_text: str) -> str | None:
    """Rebuild the content targeted by a ``+++ create`` preview, if any."""
    created = False
    lines: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+++ create "):
            created = True
            continue
        if created and line.startswith("+"):
            lines.append(line[1:])
    if not created:
        return None
    return "\n".join(lines)


def apply_selected_hunks(
    old_text: str,
    diff_text: str,
    selected: Sequence[int] | None,
) -> str:
    """Return ``old_text`` with only the selected 1-based hunks applied.

    ``selected=None`` applies every hunk. When the diff has no parseable hunks
    (such as a ``+++ create`` preview) any non-empty selection (or ``None``)
    yields the target content and an empty selection yields ``old_text``.
    """
    old_lines = old_text.splitlines(keepends=True)
    parsed = split_hunks(diff_text)
    if not parsed:
        if selected is None or len(selected) > 0:
            target = _creation_target(diff_text)
            if target is not None:
                return target
        return old_text

    chosen = None if selected is None else set(selected)
    result: list[str] = []
    cursor = 0
    for hunk in parsed:
        start = max(cursor, hunk.old_start - 1)
        if start > cursor:
            result.extend(old_lines[cursor:start])
        cursor = start
        selected_hunk = chosen is None or hunk.index in chosen
        if selected_hunk:
            segments = _parse_body(hunk.lines, old_lines, cursor, hunk.old_count)
            if segments is None:
                segments = _naive_segments(hunk.lines)
            result.extend(
                content for prefix, content in segments if prefix in (" ", "+")
            )
        else:
            result.extend(old_lines[cursor : cursor + hunk.old_count])
        cursor += hunk.old_count
    result.extend(old_lines[cursor:])
    return "".join(result)


HunkSelection = tuple[int, ...] | Literal["all", "none"]


def parse_hunk_selection(answer: str, count: int) -> HunkSelection | None:
    """Parse a user's hunk selection.

    Accepts ``all``/``none``, comma- or space-separated indices (``1,3``) and
    inclusive ranges (``1-2``). Returns ``None`` for an invalid answer.
    """
    text = answer.strip().lower()
    if not text:
        return None
    if text in ("all", "a", "*"):
        return "all"
    if text in ("none", "n"):
        return "none"

    selected: set[int] = set()
    for part in re.split(r"[,\s]+", text):
        if not part:
            continue
        if "-" in part:
            pieces = part.split("-")
            if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
                return None
            low, high = int(pieces[0]), int(pieces[1])
            if low > high:
                return None
            candidates: Sequence[int] = range(low, high + 1)
        elif part.isdigit():
            candidates = (int(part),)
        else:
            return None
        for index in candidates:
            if not 1 <= index <= count:
                return None
            selected.add(index)
    if not selected:
        return None
    return tuple(sorted(selected))