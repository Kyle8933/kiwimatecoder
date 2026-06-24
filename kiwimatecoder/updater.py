"""Self-update helper for the KiwiMateCoder CLI."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

from rich.console import Console

PACKAGE_NAME = "kiwimatecoder"


def build_update_command() -> list[str]:
    """Return the fallback PyPI-style update command."""
    return [sys.executable, "-m", "pip", "install", "--upgrade", PACKAGE_NAME]


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
        pull_code = _run(build_git_pull_command(source_root), console)
        if pull_code != 0:
            console.print(
                "[red]Update failed while pulling from Git.[/red] "
                "[yellow]Commit/stash local changes or resolve Git errors, then try again.[/yellow]"
            )
            return pull_code
        code = _run(build_source_install_command(source_root), console)
    else:
        code = _run(build_update_command(), console)

    if code == 0:
        console.print("[green]KiwiMateCoder update complete.[/green]")
    else:
        console.print(f"[red]Update failed with exit code {code}.[/red]")
    return code
