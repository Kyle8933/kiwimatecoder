"""Interactive REPL: the main loop launched by the bare ``kiwimatecoder`` command."""

from __future__ import annotations

import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from kiwimatecoder.agent import Agent
from kiwimatecoder.commands import CommandResult, dispatch
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.session import Session

console = Console()


def _banner(session: Session) -> Panel:
    return Panel(
        "[bold green]KiwiMateCoder[/bold green] — agentic coding assistant\n"
        "[dim]Type your request, or /help for commands. Ctrl-C cancels, Ctrl-D exits.[/dim]",
        subtitle=f"{session.provider_id}:{session.model} · {session.mode.value}",
    )


def _prompt_text(session: Session) -> HTML:
    return HTML(
        f"<ansigreen>kiwi</ansigreen> "
        f"<ansiblue>({session.provider_id}:{session.model} · {session.mode.value})</ansiblue> "
        f"<ansicyan>›</ansicyan> "
    )


def _make_confirm(session: Session):
    """Build the approval callback used by the permission gate."""

    def confirm(summary: str, preview_text: str | None) -> bool:
        console.print()
        if preview_text:
            lexer = "diff" if preview_text.lstrip().startswith(("---", "+++", "@@", "+", "-")) else "bash"
            console.print(
                Panel(
                    Syntax(preview_text, lexer, theme="ansi_dark", word_wrap=True),
                    title=f"[bold]Approve: {summary}[/bold]",
                    border_style="yellow",
                )
            )
        else:
            console.print(f"[yellow]Approve: {summary}[/yellow]")

        try:
            answer = console.input(
                "[bold]Allow?[/bold] [y]es / [n]o / [a]lways this tool: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("[yellow]Denied.[/yellow]")
            return False

        if answer in ("a", "always"):
            tool_name = summary.split("(", 1)[0]
            session.allow_always(tool_name)
            return True
        return answer in ("y", "yes")

    return confirm


def run(session: Session) -> None:
    """Run the interactive loop until the user exits."""
    console.print(_banner(session))
    confirm = _make_confirm(session)
    agent = Agent(session, console, confirm)
    pt_session: PromptSession = PromptSession(history=InMemoryHistory())

    while True:
        try:
            line = pt_session.prompt(_prompt_text(session))
        except KeyboardInterrupt:
            # Ctrl-C at the prompt: clear the line, keep going.
            continue
        except EOFError:
            # Ctrl-D: exit.
            console.print("[dim]Goodbye![/dim]")
            break

        line = line.strip()
        if not line:
            continue

        if line.startswith("/"):
            if dispatch(line, session, console) == CommandResult.EXIT:
                break
            continue

        try:
            asyncio.run(agent.run_turn(line))
        except KeyboardInterrupt:
            # Ctrl-C during a turn: cancel and return to the prompt.
            console.print("\n[yellow]Interrupted.[/yellow]")
