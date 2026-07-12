"""Self-update helper for the KiwiMateCoder CLI."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

from rich.console import Console

PACKAGE_NAME = "kiwimatecoder"
GITHUB_REPO_URL = "https://github.com/Kyle8933/kiwimatecoder.git"


def build_update_command() -> list[str]:
    """Return the fallback update command for packaged (non-Git) installs.

    KiwiMateCoder is not published to PyPI, so the fallback installs from the
    GitHub repository instead of a PyPI package name. ``--force-reinstall`` is
    required because the package version string is often unchanged between
    commits, so ``--upgrade`` alone can be a no-op ("Requirement already
    satisfied").
    """
    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        f"git+{GITHUB_REPO_URL}",
    ]


def build_git_pull_command(source_root: Path) -> list[str]:
    """Return the command used to fetch source updates for a Git checkout."""
    return ["git", "-C", str(source_root), "pull", "--ff-only"]


def build_source_install_command(source_root: Path) -> list[str]:
    """Return the command used to reinstall KiwiMateCoder from a source checkout."""
    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "-e",
        str(source_root),
    ]


def find_source_root(package_file: Path | None = None) -> Path | None:
    """Find the project root for editable/local source installs."""
    current = (package_file or Path(__file__)).resolve()
    for candidate in (current.parent, *current.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / PACKAGE_NAME).is_dir()
        ):
            return candidate
    return None


def _has_git_remote(source_root: Path) -> bool:
    probe = subprocess.run(
        ["git", "-C", str(source_root), "remote", "get-url", "origin"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return probe.returncode == 0


def _git(args: list[str], source_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(source_root), *args],
        capture_output=True,
        text=True,
    )


def _get_short_sha(source_root: Path) -> str | None:
    """Return the short SHA of HEAD, or None if it cannot be determined."""
    cp = _git(["rev-parse", "--short", "HEAD"], source_root)
    if cp.returncode != 0:
        return None
    return cp.stdout.strip() or None


def _get_branch(source_root: Path) -> str | None:
    """Return the current branch name, or None when detached/unknown."""
    cp = _git(["rev-parse", "--abbrev-ref", "HEAD"], source_root)
    if cp.returncode != 0:
        return None
    branch = cp.stdout.strip()
    return None if branch in ("", "HEAD") else branch


def _commits_behind(source_root: Path, branch: str | None) -> int | None:
    """Return how many commits HEAD is behind origin/<branch>, or None if unknown."""
    if not branch:
        return None
    cp = _git(["rev-list", "--count", f"HEAD..origin/{branch}"], source_root)
    if cp.returncode != 0:
        return None
    try:
        return int(cp.stdout.strip())
    except ValueError:
        return None


def _fetch(source_root: Path, console: Console) -> int:
    """Best-effort `git fetch origin`. Failures are non-fatal."""
    command = ["git", "-C", str(source_root), "fetch", "origin"]
    console.print(f"[dim]{shlex.join(command)}[/dim]")
    try:
        completed = subprocess.run(command, text=True)
    except OSError as exc:
        console.print(f"[yellow]Could not fetch from origin: {exc}[/yellow]")
        return 1
    if completed.returncode != 0:
        console.print(
            "[yellow]git fetch failed; continuing with a direct pull.[/yellow]"
        )
    return completed.returncode


def _run(command: list[str], console: Console) -> int:
    console.print(f"[dim]{shlex.join(command)}[/dim]")
    try:
        completed = subprocess.run(command, text=True)
    except OSError as exc:
        console.print(f"[red]Could not start command: {exc}[/red]")
        return 1
    return completed.returncode


def run_update(console: Console | None = None) -> int:
    """Update KiwiMateCoder in the current Python environment."""
    console = console or Console()
    console.print("[cyan]Updating KiwiMateCoder...[/cyan]")

    source_root = find_source_root()
    if source_root is not None and _has_git_remote(source_root):
        console.print(f"[dim]Detected source checkout: {source_root}[/dim]")

        old_sha = _get_short_sha(source_root)
        _fetch(source_root, console)
        branch = _get_branch(source_root)
        behind = _commits_behind(source_root, branch)

        if behind == 0:
            console.print(
                f"[green]Already on the latest version "
                f"(commit {old_sha or 'unknown'}).[/green]"
            )
            return 0

        if old_sha and behind and branch:
            console.print(
                f"[cyan]Updating from {old_sha} "
                f"({behind} commit(s) behind origin/{branch})…[/cyan]"
            )
        else:
            console.print(
                f"[cyan]Updating KiwiMateCoder from {old_sha or 'unknown'}…[/cyan]"
            )

        pull_code = _run(build_git_pull_command(source_root), console)
        if pull_code != 0:
            console.print(
                "[red]Update failed while pulling from Git.[/red] "
                "[yellow]Commit/stash local changes or resolve Git errors, "
                "then try again.[/yellow]"
            )
            return pull_code

        new_sha = _get_short_sha(source_root)
        if new_sha == old_sha:
            console.print(
                f"[green]Already on the latest version "
                f"(commit {old_sha or 'unknown'}).[/green]"
            )
            return 0

        if old_sha and new_sha:
            console.print(f"[dim]Updated {old_sha} → {new_sha}.[/dim]")

        code = _run(build_source_install_command(source_root), console)
    else:
        code = _run(build_update_command(), console)
        if code != 0:
            console.print(
                f"[red]Update failed with exit code {code}.[/red] "
                "[yellow]KiwiMateCoder is not on PyPI; install from a source "
                f"checkout or run: pip install --upgrade --force-reinstall "
                f"git+{GITHUB_REPO_URL}[/yellow]"
            )
            return code

    if code == 0:
        console.print("[green]KiwiMateCoder update complete.[/green]")
    else:
        console.print(f"[red]Update failed with exit code {code}.[/red]")
    return code
