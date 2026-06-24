"""Slash command handlers for the REPL."""

from __future__ import annotations

import shlex
from glob import has_magic
from pathlib import Path

from rich.console import Console
from rich.table import Table

from kiwimatecoder import tools
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.providers import get_provider, list_providers
from kiwimatecoder.session import Session
from kiwimatecoder.tools.paths import PathError, display_path, resolve_in_workspace


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


def _looks_binary(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return b"\x00" in f.read(1024)
    except OSError:
        return False


def _context_ref(path: str, session: Session) -> str:
    resolved = resolve_in_workspace(path, session.workspace_root)
    return display_path(resolved, session.workspace_root)


def _expand_context_input(
    raw_path: str, session: Session
) -> tuple[list[Path], list[str]]:
    if not has_magic(raw_path):
        try:
            return [resolve_in_workspace(raw_path, session.workspace_root)], []
        except PathError as exc:
            return [], [str(exc)]

    if Path(raw_path).is_absolute():
        return [], [f"Glob patterns must be relative to the workspace: {raw_path}"]

    root = session.workspace_root.resolve()
    errors: list[str] = []
    matches: list[Path] = []
    for candidate in sorted(root.glob(raw_path)):
        try:
            matches.append(resolve_in_workspace(str(candidate), session.workspace_root))
        except PathError as exc:
            errors.append(str(exc))
    if not matches and not errors:
        errors.append(f"No files matched: {raw_path}")
    return matches, errors


def _add_context(paths: list[str], session: Session, console: Console) -> None:
    if not paths:
        console.print("[yellow]Usage: /context add <path-or-glob> [...][/yellow]")
        return

    added = 0
    skipped = 0
    for raw_path in paths:
        matches, errors = _expand_context_input(raw_path, session)
        for error in errors:
            skipped += 1
            console.print(f"[yellow]Skipped {raw_path}: {error}[/yellow]")

        for resolved in matches:
            rel = display_path(resolved, session.workspace_root)
            if not resolved.exists():
                skipped += 1
                console.print(f"[yellow]Skipped {rel}: file not found[/yellow]")
                continue
            if resolved.is_dir():
                skipped += 1
                console.print(f"[yellow]Skipped {rel}: is a directory[/yellow]")
                continue
            if _looks_binary(resolved):
                skipped += 1
                console.print(f"[yellow]Skipped {rel}: appears to be binary[/yellow]")
                continue
            if session.add_context_file(rel):
                added += 1
                console.print(f"[green]Added context:[/green] {rel}")
            else:
                skipped += 1
                console.print(f"[dim]Already in context:[/dim] {rel}")

    if added or skipped:
        console.print(
            f"[dim]Context files: {len(session.context_files)} "
            f"({added} added, {skipped} skipped).[/dim]"
        )


def _remove_context(paths: list[str], session: Session, console: Console) -> None:
    if not paths:
        console.print("[yellow]Usage: /context remove <path> [...][/yellow]")
        return

    removed = 0
    for raw_path in paths:
        try:
            rel = _context_ref(raw_path, session)
        except PathError as exc:
            console.print(f"[yellow]Skipped {raw_path}: {exc}[/yellow]")
            continue
        if session.remove_context_file(rel):
            removed += 1
            console.print(f"[green]Removed context:[/green] {rel}")
        else:
            console.print(f"[dim]Not in context:[/dim] {rel}")

    console.print(
        f"[dim]Context files: {len(session.context_files)} "
        f"({removed} removed).[/dim]"
    )


def _show_context(session: Session, console: Console) -> None:
    if not session.context_files:
        console.print("[dim]No pinned context files.[/dim]")
        return

    table = Table(title="Pinned context", show_header=True)
    table.add_column("path", style="cyan")
    table.add_column("status")
    for path in session.context_files:
        try:
            resolved = resolve_in_workspace(path, session.workspace_root)
        except PathError as exc:
            table.add_row(path, f"[red]{exc}[/red]")
            continue
        if not resolved.exists():
            table.add_row(path, "[yellow]missing[/yellow]")
        elif resolved.is_dir():
            table.add_row(path, "[yellow]directory[/yellow]")
        elif _looks_binary(resolved):
            table.add_row(path, "[yellow]binary[/yellow]")
        else:
            table.add_row(path, f"{resolved.stat().st_size} bytes")
    console.print(table)


def _context(arg: str, session: Session, console: Console) -> str:
    try:
        parts = shlex.split(arg)
    except ValueError as exc:
        console.print(f"[red]Could not parse command: {exc}[/red]")
        return CommandResult.CONTINUE

    if not parts or parts[0] in {"list", "ls"}:
        _show_context(session, console)
        return CommandResult.CONTINUE

    action = parts[0].lower()
    rest = parts[1:]
    if action in {"add", "pin"}:
        _add_context(rest, session, console)
    elif action in {"remove", "rm", "drop", "delete"}:
        _remove_context(rest, session, console)
    elif action == "clear":
        count = session.clear_context_files()
        console.print(f"[green]Cleared {count} context file(s).[/green]")
    else:
        _add_context(parts, session, console)
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
    "context": _context,
    "ctx": _context,
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
    "/context [list|add|remove|clear]": (
        "Manage pinned files included with each turn."
    ),
    "/cost": "Show token usage for this session.",
}

_COMMAND_DESCRIPTIONS = {
    "help": "Show available commands.",
    "exit": "Leave the session.",
    "quit": "Leave the session.",
    "clear": "Clear the conversation history.",
    "model": "Show or set the model.",
    "provider": "List providers or switch the active one.",
    "mode": "Show or set the permission mode.",
    "tools": "List available tools.",
    "files": "List files changed this session.",
    "context": "Manage pinned files included with each turn.",
    "ctx": "Alias for /context.",
    "cost": "Show token usage for this session.",
}

_CONTEXT_ACTION_DESCRIPTIONS = {
    "list": "Show pinned context files.",
    "add": "Pin one or more files or globs.",
    "remove": "Unpin one or more files.",
    "clear": "Remove all pinned context files.",
}

_MODE_DESCRIPTIONS = {
    "ask": "Approve writes and shell commands.",
    "auto-accept": "Run writes and commands without prompting.",
    "plan": "Read-only planning mode.",
}


def slash_command_completions(prefix: str = "") -> list[tuple[str, str]]:
    """Return slash command completions matching ``prefix`` without the slash."""
    normalized = prefix.lower()
    return [
        (f"/{name}", description)
        for name, description in _COMMAND_DESCRIPTIONS.items()
        if name.startswith(normalized)
    ]


def slash_argument_completions(command: str, prefix: str = "") -> list[tuple[str, str]]:
    """Return first-argument completions for slash commands that have them."""
    normalized = prefix.lower()
    command = command.lower()
    if command in {"context", "ctx"}:
        choices = _CONTEXT_ACTION_DESCRIPTIONS
    elif command == "mode":
        choices = _MODE_DESCRIPTIONS
    elif command == "provider":
        choices = {p.id: p.name for p in list_providers()}
    else:
        return []
    return [
        (value, description)
        for value, description in choices.items()
        if value.startswith(normalized)
    ]
