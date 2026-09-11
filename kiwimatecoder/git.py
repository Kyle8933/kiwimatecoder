"""Git helpers: pure argument builders and a subprocess runner.

Only safe, non-destructive actions are exposed. ``push``, ``reset``, and
``clean`` are deliberately absent: mutating the remote or discarding work is
not something an agent should do through this surface.

Argument builders return argv lists so callers pass values as separate
arguments. That removes any shell-quoting surface: the runner never uses
``shell=True``.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from typing import Any

DEFAULT_TIMEOUT = 30
MAX_OUTPUT = 30_000
LOG_LIMIT_MIN = 1
LOG_LIMIT_MAX = 100
WRITE_ACTIONS = ("stage", "unstage", "commit", "checkout_branch")

_BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


def status_args(*, short: bool = True) -> list[str]:
    """``git status`` argv, short format by default."""
    return ["status", "--short"] if short else ["status"]


def diff_args(
    *, staged: bool = False, path: str | None = None, ref: str | None = None
) -> list[str]:
    """``git diff`` argv for unstaged/staged changes, optionally against ``ref``."""
    args = ["diff"]
    if staged:
        args.append("--staged")
    if ref:
        args.append(str(ref))
    if path:
        args.extend(["--", str(path)])
    return args


def log_args(
    limit: int = 20, *, path: str | None = None, oneline: bool = True
) -> list[str]:
    """``git log`` argv with a clamped ``limit`` (1-100)."""
    try:
        count = int(limit)
    except (TypeError, ValueError):
        count = 20
    count = max(LOG_LIMIT_MIN, min(count, LOG_LIMIT_MAX))
    args = ["log", f"-{count}", "--no-color"]
    if oneline:
        args.append("--oneline")
    if path:
        args.extend(["--", str(path)])
    return args


def show_args(ref: str = "HEAD") -> list[str]:
    """``git show`` argv for one commit."""
    return ["show", "--no-color", str(ref or "HEAD")]


def branch_args() -> list[str]:
    """``git branch`` argv listing local branches."""
    return ["branch", "--list", "--no-color"]


def valid_branch(name: str) -> bool:
    """Whether ``name`` is a safe, plausible branch name.

    Mirrors the common ``git check-ref-format --branch`` rules that matter
    here without shelling out: no leading dash, no ``..``/``@{``/``//``, and
    no trailing ``/``, ``.``, or ``.lock``.
    """
    cleaned = str(name).strip()
    if not cleaned or cleaned.startswith("-"):
        return False
    if not _BRANCH_RE.match(cleaned):
        return False
    if ".." in cleaned or "@{" in cleaned or "//" in cleaned:
        return False
    return not cleaned.endswith(("/", ".", ".lock"))


def _clean_paths(raw: Any) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("'paths' must be a list of paths.")
    paths: list[str] = []
    for item in raw:
        text = str(item).strip()
        if not text:
            raise ValueError("'paths' entries must be non-empty.")
        paths.append(text)
    return paths


def build_write_args(args: dict[str, Any]) -> list[str]:
    """Build the argv for a ``git_write`` call, raising ``ValueError`` when invalid."""
    action = str(args.get("action") or "").strip().lower()
    if action in {"stage", "unstage"}:
        paths = _clean_paths(args.get("paths"))
        if not paths:
            raise ValueError(f"'paths' is required for {action}.")
        if action == "stage":
            return ["add", "--", *paths]
        return ["restore", "--staged", "--", *paths]
    if action == "commit":
        message = str(args.get("message") or "").strip()
        if not message:
            raise ValueError("A non-empty 'message' is required for commit.")
        return ["commit", "-m", message]
    if action == "checkout_branch":
        branch = str(args.get("branch") or "").strip()
        if not valid_branch(branch):
            raise ValueError(f"Invalid branch name '{branch}'.")
        return ["checkout", "-b", branch]
    raise ValueError(f"Unknown git_write action '{action}'.")


def run_git(
    args: list[str], session: Any, timeout: int = DEFAULT_TIMEOUT
) -> tuple[int, str]:
    """Run ``git`` with ``args`` in the session workspace.

    Returns ``(returncode, output)`` where output is stdout plus stderr. Never
    raises: a missing binary and a timeout become synthetic exit codes so the
    tool can report them as text.
    """
    command = ["git", *(str(arg) for arg in args)]
    try:
        proc = subprocess.run(
            command,
            cwd=str(session.workspace_root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return 127, "git is not installed or not on PATH."
    except subprocess.TimeoutExpired:
        return 124, f"git timed out after {timeout}s."
    output = proc.stdout or ""
    if proc.stderr:
        output = f"{output}\n{proc.stderr}" if output else proc.stderr
    return proc.returncode, output


def preview(args: dict[str, Any], session: Any) -> str:
    """Return the exact ``git ...`` command line for the approval prompt."""
    del session
    try:
        command = build_write_args(args)
    except ValueError as exc:
        return str(exc)
    return "git " + " ".join(shlex.quote(part) for part in command)
