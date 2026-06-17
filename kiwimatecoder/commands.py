"""Slash command handlers for the REPL."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from kiwimatecoder import tools
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.providers import get_provider, list_providers
from kiwimatecoder.session import Session


class CommandResult:
    """Sentinel results a command can return."""

    CONTINUE = "continue"
    EXIT = "exit"


def dispatch(line: str, session: Session, console: Console) -> str:
    """Run a slash command. Returns CommandResult.CONTINUE or .EXIT."""
    parts = line[1:].strip().split(maxsplit=1)
    name = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    handler = _COMMANDS.get(name)
    if handler is None:
        console.print(f"[yellow]Unknown command '/{name}'. Try /help.[/yellow]")
        return CommandResult.CONTINUE
    return handler(arg, session, console)


def _help(arg: str, session: Session, console: Console) -> str:
    table = Table(title="KiwiMateCoder commands", show_header=True)
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    for cmd, desc in _HELP.items():
        table.add_row(cmd, desc)
    console.print(table)
    return CommandResult.CONTINUE


def _exit(arg: str, session: Session, console: Console) -> str:
    console.print("[dim]Goodbye![/dim]")
    return CommandResult.EXIT


def _clear(arg: str, session: Session, console: Console) -> str:
    session.reset_history()
    console.print("[dim]Conversation cleared.[/dim]")
    return CommandResult.CONTINUE


def _model(arg: str, session: Session, console: Console) -> str:
    if not arg:
        console.print(f"Current model: [cyan]{session.model}[/cyan]")
        return CommandResult.CONTINUE
    session.model = arg
    console.print(f"Model set to [cyan]{arg}[/cyan].")
    return CommandResult.CONTINUE


def _provider(arg: str, session: Session, console: Console) -> str:
    if not arg:
        table = Table(title="Providers", show_header=True)
        table.add_column("id", style="cyan")
        table.add_column("name")
        table.add_column("default model")
        for p in list_providers():
            marker = " (active)" if p.id == session.provider_id else ""
            table.add_row(p.id + marker, p.name, p.default_model)
        console.print(table)
        return CommandResult.CONTINUE
    try:
        get_provider(arg)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return CommandResult.CONTINUE
    session.set_provider(arg)
    console.print(
        f"Provider set to [cyan]{session.provider_id}[/cyan] "
        f"(model: [cyan]{session.model}[/cyan])."
    )
    return CommandResult.CONTINUE


def _mode(arg: str, session: Session, console: Console) -> str:
    if not arg:
        console.print(f"Current mode: [cyan]{session.mode.value}[/cyan]")
        return CommandResult.CONTINUE
    try:
        session.mode = PermissionMode.from_str(arg)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return CommandResult.CONTINUE
    console.print(f"Mode set to [cyan]{session.mode.value}[/cyan].")
    return CommandResult.CONTINUE


def _tools(arg: str, session: Session, console: Console) -> str:
    table = Table(title="Tools", show_header=True)
    table.add_column("name", style="cyan")
    table.add_column("approval")
    table.add_column("description")
    for tool in tools.TOOLS.values():
        approval = "needs approval" if tool.needs_approval else "auto"
        table.add_row(tool.name, approval, tool.description.split(".")[0])
    console.print(table)
    return CommandResult.CONTINUE


def _files(arg: str, session: Session, console: Console) -> str:
    if not session.touched_files:
        console.print("[dim]No files changed this session.[/dim]")
        return CommandResult.CONTINUE
    console.print("[bold]Files changed this session:[/bold]")
    for path in session.touched_files:
        console.print(f"  • {path}")
    return CommandResult.CONTINUE


def _cost(arg: str, session: Session, console: Console) -> str:
    console.print(
        f"Tokens this session — prompt: [cyan]{session.prompt_tokens}[/cyan], "
        f"completion: [cyan]{session.completion_tokens}[/cyan], "
        f"total: [cyan]{session.prompt_tokens + session.completion_tokens}[/cyan]"
    )
    return CommandResult.CONTINUE


_COMMANDS = {
    "help": _help,
    "exit": _exit,
    "quit": _exit,
    "clear": _clear,
    "model": _model,
    "provider": _provider,
    "mode": _mode,
    "tools": _tools,
    "files": _files,
    "cost": _cost,
}

_HELP = {
    "/help": "Show this help.",
    "/exit, /quit": "Leave the session.",
    "/clear": "Clear the conversation history.",
    "/model [name]": "Show or set the model.",
    "/provider [id]": "List providers or switch the active one.",
    "/mode [ask|auto-accept|plan]": "Show or set the permission mode.",
    "/tools": "List available tools.",
    "/files": "List files changed this session.",
    "/cost": "Show token usage for this session.",
}
