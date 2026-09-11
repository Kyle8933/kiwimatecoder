"""Forge integration through official CLIs (no API tokens in-process).

GitHub work goes through ``gh`` and GitLab through ``glab``; the user's CLI
login provides credentials, so no token is read or stored by KiwiMateCoder.
Only create/list/view actions are exposed — merge, close, and delete are
deliberately absent.

The argument builders are pure so they can be unit-tested, and they know the
CLI differences: GitLab calls pull requests "merge requests" and uses
``--description``/``--target-branch`` instead of ``--body``/``--base``.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any
from urllib.parse import urlsplit

MAX_OUTPUT = 30_000
DEFAULT_TIMEOUT = 60
FORGE_CLIS = {"github": "gh", "gitlab": "glab"}
READ_ACTIONS = ("pr_list", "pr_view", "issue_list", "issue_view")
WRITE_ACTIONS = ("pr_create", "issue_create")


def _host_from_remote(remote_url: str) -> str:
    raw = str(remote_url).strip()
    if not raw:
        return ""
    if "://" in raw:
        return (urlsplit(raw).hostname or "").lower().rstrip(".")
    if "@" in raw and ":" in raw:
        return raw.split("@", 1)[1].split(":", 1)[0].strip().lower().rstrip(".")
    return raw.split("/", 1)[0].strip().lower().rstrip(".")


def detect_forge(remote_url: str | None) -> str | None:
    """Return ``"github"``/``"gitlab"`` for a known host, else ``None``.

    Recognizes the public hosts (``github.com``, ``gitlab.com``) over https,
    ssh, and scp-style remotes. Self-hosted instances are not guessed.
    """
    host = _host_from_remote(remote_url or "")
    if host in {"github.com", "www.github.com"}:
        return "github"
    if host in {"gitlab.com", "www.gitlab.com"}:
        return "gitlab"
    return None


def forge_cli(forge: str | None) -> str | None:
    """The CLI binary for a forge, or ``None`` when unknown."""
    return FORGE_CLIS.get(str(forge or "").lower())


def cli_available(command: str) -> bool:
    """Whether ``command`` is on ``PATH``."""
    return shutil.which(command) is not None


def _pr_verb(forge: str) -> str:
    return "mr" if str(forge).lower() == "gitlab" else "pr"


def _body_flag(forge: str) -> str:
    return "--description" if str(forge).lower() == "gitlab" else "--body"


def _base_flag(forge: str) -> str:
    return "--target-branch" if str(forge).lower() == "gitlab" else "--base"


def pr_list_args(forge: str = "github") -> list[str]:
    return [_pr_verb(forge), "list"]


def pr_view_args(number: int, forge: str = "github") -> list[str]:
    return [_pr_verb(forge), "view", str(number)]


def pr_create_args(
    title: str,
    body: str = "",
    base: str | None = None,
    draft: bool = False,
    forge: str = "github",
) -> list[str]:
    args = [
        _pr_verb(forge),
        "create",
        "--title",
        str(title),
        _body_flag(forge),
        str(body),
    ]
    if base:
        args.extend([_base_flag(forge), str(base)])
    if draft:
        args.append("--draft")
    return args


def issue_list_args() -> list[str]:
    return ["issue", "list"]


def issue_view_args(number: int) -> list[str]:
    return ["issue", "view", str(number)]


def issue_create_args(title: str, body: str = "", forge: str = "github") -> list[str]:
    return [
        "issue",
        "create",
        "--title",
        str(title),
        _body_flag(forge),
        str(body),
    ]


def build_write_args(forge: str, args: dict[str, Any]) -> list[str]:
    """Build the argv for a ``forge_write`` call, raising ``ValueError``."""
    action = str(args.get("action") or "").strip().lower()
    title = str(args.get("title") or "").strip()
    body = str(args.get("body") or "")
    base = str(args["base"]).strip() if args.get("base") else None
    if action == "pr_create":
        if not title:
            raise ValueError("A non-empty 'title' is required for pr_create.")
        return pr_create_args(title, body, base, bool(args.get("draft")), forge=forge)
    if action == "issue_create":
        if not title:
            raise ValueError("A non-empty 'title' is required for issue_create.")
        return issue_create_args(title, body, forge=forge)
    raise ValueError(f"Unknown forge_write action '{action}'.")


def current_remote_url(session: Any) -> str | None:
    """Return the ``origin`` remote URL for the session workspace, if any."""
    from kiwimatecoder import git as git_module

    code, output = git_module.run_git(["remote", "get-url", "origin"], session)
    if code != 0:
        return None
    text = output.strip()
    return text.splitlines()[0].strip() if text else None


def run_forge(
    cli: str, args: list[str], session: Any, timeout: int = DEFAULT_TIMEOUT
) -> tuple[int, str]:
    """Run a forge CLI command in the workspace; never raises, never a shell."""
    command = [cli, *(str(arg) for arg in args)]
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
        return 127, f"{cli} is not installed or not on PATH."
    except subprocess.TimeoutExpired:
        return 124, f"{cli} timed out after {timeout}s."
    output = proc.stdout or ""
    if proc.stderr:
        output = f"{output}\n{proc.stderr}" if output else proc.stderr
    return proc.returncode, output
