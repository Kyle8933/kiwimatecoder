"""Custom prompt templates runnable as REPL slash commands.

Users drop Markdown files named after the command into
``~/.kiwimatecoder/commands/`` or ``<workspace>/.kiwimatecoder/commands/``.
The file body is the prompt; ``$ARGUMENTS`` is replaced with whatever the user
typed after the command name. A workspace template wins over a user-level
template with the same name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from kiwimatecoder import config

WORKSPACE_COMMANDS_DIR = Path(".kiwimatecoder") / "commands"
USER_COMMANDS_DIR_NAME = "commands"
MAX_TEMPLATE_BYTES = 16 * 1024
MAX_NAME_LENGTH = 40
ARGUMENTS_TOKEN = "$ARGUMENTS"

_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")


@dataclass(frozen=True)
class TemplateInfo:
    """One discovered prompt template."""

    name: str
    path: Path
    description: str
    body: str


def _valid_name(name: str) -> bool:
    return bool(name) and len(name) <= MAX_NAME_LENGTH and bool(_NAME_PATTERN.match(name))


def _first_line(text: str, fallback: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("#").strip()
        if line:
            return line
    return fallback


def _load_template(path: Path) -> TemplateInfo | None:
    name = path.stem.strip().lower()
    if not _valid_name(name):
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:1024]:
        return None
    body = raw[:MAX_TEMPLATE_BYTES].decode("utf-8", "replace")
    return TemplateInfo(
        name=name,
        path=path,
        description=_first_line(body, name),
        body=body,
    )


def _template_dirs(workspace_root: Path) -> list[Path]:
    """Workspace first so it wins name collisions with user templates."""
    dirs = [workspace_root / WORKSPACE_COMMANDS_DIR]
    try:
        dirs.append(config.ensure_config_dir() / USER_COMMANDS_DIR_NAME)
    except OSError:
        # An unwritable home directory just means no user-level templates.
        pass
    return dirs


def discover_templates(workspace_root: Path) -> dict[str, TemplateInfo]:
    """Return every valid template keyed by command name, sorted by name."""
    found: dict[str, TemplateInfo] = {}
    for directory in _template_dirs(workspace_root):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.md")):
            template = _load_template(path)
            if template is None:
                continue
            found.setdefault(template.name, template)
    return dict(sorted(found.items()))


def find_template(name: str, workspace_root: Path) -> TemplateInfo | None:
    """Look up a template by command name; None when unknown or invalid."""
    key = str(name).strip().lower()
    if not key:
        return None
    return discover_templates(workspace_root).get(key)


def render_template(body: str, arguments: str) -> str:
    """Substitute ``$ARGUMENTS``; append the arguments when the body lacks it."""
    text = arguments.strip()
    if ARGUMENTS_TOKEN in body:
        return body.replace(ARGUMENTS_TOKEN, text)
    if text:
        return f"{body}\n\nArguments: {text}"
    return body
