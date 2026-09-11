"""On-demand Agent Skills: markdown bundles the model loads when relevant.

A skill is a directory containing ``SKILL.md`` under
``~/.kiwimatecoder/skills/`` or ``<workspace>/.kiwimatecoder/skills/`` (the
workspace wins name collisions). The system prompt advertises only each
skill's name and one-line description; the full body is fetched through the
``load_skill`` tool when a task matches, so unused skills cost no tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kiwimatecoder import config

WORKSPACE_SKILLS_DIR = Path(".kiwimatecoder") / "skills"
USER_SKILLS_DIR_NAME = "skills"
SKILL_FILE_NAME = "SKILL.md"
MAX_SKILL_BYTES = 32 * 1024
MAX_LISTING_BYTES = 2048
MAX_DESCRIPTION_CHARS = 200


@dataclass(frozen=True)
class Skill:
    """One discoverable skill on disk."""

    name: str
    path: Path
    description: str


def _skill_dirs(workspace_root: Path) -> list[Path]:
    """Workspace first so it wins name collisions with user skills."""
    dirs = [workspace_root / WORKSPACE_SKILLS_DIR]
    try:
        dirs.append(config.ensure_config_dir() / USER_SKILLS_DIR_NAME)
    except OSError:
        # An unwritable home directory just means no user-level skills.
        pass
    return dirs


def _read_raw(path: Path) -> bytes | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:1024]:
        return None
    return raw


def _description(text: str, fallback: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("#").strip()
        if line:
            return line[:MAX_DESCRIPTION_CHARS]
    return fallback


def discover_skills(workspace_root: Path) -> list[Skill]:
    """Return every valid skill sorted by name; workspace shadows user."""
    found: dict[str, Skill] = {}
    for directory in _skill_dirs(workspace_root):
        if not directory.is_dir():
            continue
        for entry in sorted(directory.iterdir()):
            # A bare SKILL.md directly in the skills directory is not a skill;
            # skills must live inside their own named directory.
            if not entry.is_dir():
                continue
            path = entry / SKILL_FILE_NAME
            if not path.is_file():
                continue
            name = entry.name.strip()
            if not name or name in found:
                continue
            raw = _read_raw(path)
            if raw is None:
                continue
            text = raw.decode("utf-8", "replace")
            found[name] = Skill(
                name=name, path=path, description=_description(text, name)
            )
    return [found[name] for name in sorted(found)]


def load_skill(name: str, workspace_root: Path) -> str | None:
    """Return a skill's markdown body, capped and truncation-noted, or None."""
    key = str(name).strip()
    if not key:
        return None
    skill = next(
        (item for item in discover_skills(workspace_root) if item.name == key), None
    )
    if skill is None:
        return None
    raw = _read_raw(skill.path)
    if raw is None:
        return None
    clipped = raw[:MAX_SKILL_BYTES]
    text = clipped.decode("utf-8", "replace")
    if len(raw) > len(clipped):
        text += f"\n\n[truncated: showing {len(clipped)} of {len(raw)} bytes]"
    return text


def skills_section(workspace_root: Path) -> str:
    """Render the compact "available skills" block for the system prompt.

    Only names and descriptions are listed; the full body is intentionally
    left out so unused skills do not consume context. The listing is capped
    at :data:`MAX_LISTING_BYTES` with a note when skills are omitted.
    """
    skills = discover_skills(workspace_root)
    if not skills:
        return ""
    lines = [
        "Available skills — call load_skill(name) when a task matches one; "
        "their full instructions are not loaded until you do:"
    ]
    budget = MAX_LISTING_BYTES
    appended = 0
    omitted = 0
    for skill in skills:
        line = f"- {skill.name}: {skill.description}"
        cost = len(line.encode("utf-8")) + 1
        if cost > budget:
            omitted = len(skills) - appended
            break
        lines.append(line)
        budget -= cost
        appended += 1
    if omitted:
        lines.append(f"[{omitted} more skill(s) omitted]")
    return "\n\n" + "\n".join(lines)
