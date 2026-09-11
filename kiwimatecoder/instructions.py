"""Project instruction files (``AGENTS.md`` and friends).

Instruction files are project-owned guidance that the agent should follow:
build commands, style rules, testing conventions. They are loaded from the
workspace root and injected into the system prompt with a size budget so a
pathological file cannot crowd out the conversation.
"""

from __future__ import annotations

from pathlib import Path

# Checked in order; the first matching file for each name wins.
INSTRUCTION_FILES: tuple[str, ...] = (
    "AGENTS.md",
    "CLAUDE.md",
    ".kiwimatecoder/AGENTS.md",
    ".kiwimatecoder/instructions.md",
)

MAX_INSTRUCTION_FILE_BYTES = 32 * 1024
MAX_INSTRUCTION_TOTAL_BYTES = 48 * 1024


def discover_instruction_files(workspace_root: Path) -> list[Path]:
    """Return existing instruction files under ``workspace_root``, in order."""
    found: list[Path] = []
    for relative in INSTRUCTION_FILES:
        candidate = workspace_root / relative
        if candidate.is_file():
            found.append(candidate)
    return found


def load_instructions(
    workspace_root: Path,
    *,
    file_budget: int = MAX_INSTRUCTION_FILE_BYTES,
    total_budget: int = MAX_INSTRUCTION_TOTAL_BYTES,
) -> list[tuple[str, str]]:
    """Load instruction files as ``(relative_path, text)`` pairs.

    Returns an empty list when there are none. Missing/unreadable/binary files
    are skipped. Total and per-file byte budgets are enforced.
    """
    loaded: list[tuple[str, str]] = []
    remaining = max(0, total_budget)
    for path in discover_instruction_files(workspace_root):
        if remaining <= 0:
            break
        limit = min(file_budget, remaining)
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:1024]:
            continue
        clipped = raw[:limit]
        text = clipped.decode("utf-8", "replace").strip()
        if not text:
            continue
        truncated = len(raw) > len(clipped)
        if truncated:
            text += f"\n\n[truncated: showing {len(clipped)} of {len(raw)} bytes]"
        loaded.append((path.relative_to(workspace_root).as_posix(), text))
        remaining -= len(clipped)
    return loaded


def instructions_section(workspace_root: Path) -> str:
    """Render the project-instructions block for the system prompt."""
    loaded = load_instructions(workspace_root)
    if not loaded:
        return ""
    blocks = [
        "Project instructions follow. These come from the repository and should "
        "be followed for work in this workspace:"
    ]
    for relative, text in loaded:
        blocks.append(
            f'<project_instructions path="{relative}">\n{text}\n</project_instructions>'
        )
    return "\n\n" + "\n\n".join(blocks)
