"""Self-update helper for the KiwiMateCoder CLI."""

from __future__ import annotations

import shlex
import subprocess
import sys

from rich.console import Console


def build_update_command() -> list[str]:
    """Return the command used to update the installed package."""
    return [sys.executable, "-m", "pip", "install", "--upgrade", "kiwimatecoder"]


def run_update(console: Console | None = None) -> int:
    """Update KiwiMateCoder using the current Python environment's pip."""
    console = console or Console()
    command = build_update_command()
    console.print("[cyan]Updating KiwiMateCoder...[/cyan]")
    console.print(f"[dim]{shlex.join(command)}[/dim]")

    try:
        code = subprocess.call(command)
    except OSError as exc:
        console.print(f"[red]Could not start updater: {exc}[/red]")
        return 1

    if code == 0:
        console.print("[green]KiwiMateCoder is up to date.[/green]")
    else:
        console.print(f"[red]Update failed with exit code {code}.[/red]")
    return code
