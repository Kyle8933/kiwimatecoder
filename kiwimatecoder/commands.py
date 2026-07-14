"""Slash command handlers for the REPL."""

from __future__ import annotations

import shlex
from collections.abc import Callable
from dataclasses import dataclass
from glob import has_magic
from pathlib import Path

from rich.console import Console
from rich.table import Table

from kiwimatecoder import tools
from kiwimatecoder.config import (
    add_provider,
    get_key,
    get_model_filter,
    get_provider_config,
    list_provider_configs,
    list_visible_models,
    remove_key,
    remove_provider,
    set_key,
    set_model_filter,
    set_selected_model,
    set_selected_provider,
)
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.providers import DEFAULT_PROVIDER_ID, REGISTRY
from kiwimatecoder.session import Session
from kiwimatecoder.tools.paths import PathError, display_path, resolve_in_workspace


class CommandResult:
    """Sentinel results a command can return."""

    CONTINUE = "continue"
    EXIT = "exit"


@dataclass(frozen=True)
class CommandOption:
    """One value offered by an interactive slash-command selector."""

    value: str
    label: str


@dataclass(frozen=True)
class SelectionPrompt:
    """Terminal-agnostic description of an interactive selection prompt."""

    title: str
    text: str
    options: tuple[CommandOption, ...]
    selected: str | None = None
    empty_message: str = "No choices are available."


CommandSelector = Callable[[SelectionPrompt], str | None]


def dispatch(
    line: str,
    session: Session,
    console: Console,
    selector: CommandSelector | None = None,
) -> str:
    """Run a slash command. Returns CommandResult.CONTINUE or .EXIT."""
    parts = line[1:].strip().split(maxsplit=1)
    name = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    handler = _COMMANDS.get(name)
    if handler is None:
        console.print(f"[yellow]Unknown command '/{name}'. Try /help.[/yellow]")
        return CommandResult.CONTINUE

    if not arg and selector is not None:
        prompt = _selection_prompt(name, session)
        if prompt is not None:
            if not prompt.options:
                console.print(f"[yellow]{prompt.empty_message}[/yellow]")
                return CommandResult.CONTINUE
            selected = selector(prompt)
            if selected is None:
                return CommandResult.CONTINUE
            if selected not in {option.value for option in prompt.options}:
                console.print("[red]The selector returned an invalid choice.[/red]")
                return CommandResult.CONTINUE
            arg = selected
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
        for p in list_provider_configs():
            marker = " (active)" if p.id == session.provider_id else ""
            table.add_row(p.id + marker, p.name, p.default_model)
        console.print(table)
        return CommandResult.CONTINUE
    try:
        get_provider_config(arg)
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


def _config_help(console: Console) -> None:
    table = Table(title="/config commands", show_header=True)
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    rows = [
        ("/config", "Show active provider, model, key, and model filter."),
        ("/config providers", "List built-in and custom providers."),
        (
            "/config provider add <id> <name> <base_url> <default_model> [key_env]",
            "Add an OpenAI-compatible custom provider. Quote names with spaces.",
        ),
        ("/config provider remove <id>", "Remove a custom provider."),
        ("/config provider use <id>", "Persist and switch to a provider."),
        ("/config key set <provider> <key>", "Save an API key."),
        ("/config key remove <provider>", "Remove a stored API key."),
        ("/config key list", "Show which providers have keys configured."),
        ("/config model set <model>", "Persist the default model."),
        ("/config model reset", "Use the provider default model."),
        (
            "/config models allow <model> [...]",
            "Only show these models for the active provider.",
        ),
        (
            "/config models deny <model> [...]",
            "Hide these models for the active provider.",
        ),
        ("/config models clear", "Clear model visibility for the active provider."),
    ]
    for command, description in rows:
        table.add_row(command, description)
    console.print(table)


def _config_show(session: Session, console: Console) -> None:
    provider = session.provider
    key = get_key(provider.id)
    model_filter = get_model_filter(provider.id)
    console.print(
        f"Provider: [cyan]{provider.id}[/cyan] ({provider.name})\n"
        f"Model: [cyan]{session.model}[/cyan]\n"
        f"Key: {'[green]configured[/green]' if key else '[yellow]missing[/yellow]'} "
        f"({provider.key_env})\n"
        f"Model visibility: [cyan]{model_filter['mode']}[/cyan]"
    )
    if model_filter["models"]:
        console.print("Models: " + ", ".join(model_filter["models"]))


def _config_providers(
    action_parts: list[str], session: Session, console: Console
) -> None:
    action = action_parts[0].lower() if action_parts else "list"
    rest = action_parts[1:]

    if action in {"list", "ls"}:
        table = Table(title="Providers", show_header=True)
        table.add_column("id", style="cyan")
        table.add_column("type")
        table.add_column("name")
        table.add_column("default model")
        table.add_column("base URL")
        for provider in list_provider_configs():
            marker = " (active)" if provider.id == session.provider_id else ""
            kind = "built-in" if provider.id in REGISTRY else "custom"
            table.add_row(
                provider.id + marker,
                kind,
                provider.name,
                provider.default_model,
                provider.base_url,
            )
        console.print(table)
        return

    if action in {"add", "create"}:
        if len(rest) < 4:
            console.print(
                "[yellow]Usage: /config provider add <id> <name> "
                "<base_url> <default_model> [key_env][/yellow]"
            )
            return
        provider_id, name, base_url, default_model = rest[:4]
        key_env = rest[4] if len(rest) > 4 else None
        try:
            provider = add_provider(provider_id, name, base_url, default_model, key_env)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(
            f"[green]Added provider[/green] [cyan]{provider.id}[/cyan] "
            f"({provider.name})."
        )
        return

    if action in {"remove", "rm", "delete"}:
        if not rest:
            console.print("[yellow]Usage: /config provider remove <id>[/yellow]")
            return
        provider_id = rest[0]
        was_active = provider_id == session.provider_id
        try:
            remove_provider(provider_id)
        except (KeyError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            return
        if was_active:
            session.set_provider(DEFAULT_PROVIDER_ID)
        console.print(f"[green]Removed provider[/green] [cyan]{provider_id}[/cyan].")
        return

    if action in {"use", "select", "set"}:
        if not rest:
            console.print("[yellow]Usage: /config provider use <id>[/yellow]")
            return
        provider_id = rest[0]
        try:
            set_selected_provider(provider_id)
            session.set_provider(provider_id)
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(
            f"Provider set to [cyan]{session.provider_id}[/cyan] "
            f"(model: [cyan]{session.model}[/cyan])."
        )
        return

    console.print("[yellow]Unknown provider config action. Try /config help.[/yellow]")


def _config_keys(action_parts: list[str], console: Console) -> None:
    action = action_parts[0].lower() if action_parts else "list"
    rest = action_parts[1:]

    if action in {"list", "ls"}:
        table = Table(title="API keys", show_header=True)
        table.add_column("provider", style="cyan")
        table.add_column("env var")
        table.add_column("status")
        for provider in list_provider_configs():
            key = get_key(provider.id)
            status = (
                f"[green]configured (...{key[-4:]})[/green]"
                if key
                else "[dim]missing[/dim]"
            )
            table.add_row(provider.id, provider.key_env, status)
        console.print(table)
        return

    if action in {"set", "save", "add"}:
        if len(rest) < 2:
            console.print("[yellow]Usage: /config key set <provider> <key>[/yellow]")
            return
        provider_id, key = rest[0], rest[1]
        try:
            set_key(provider_id, key)
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(f"[green]Saved API key for[/green] [cyan]{provider_id}[/cyan].")
        return

    if action in {"remove", "rm", "delete", "clear"}:
        if not rest:
            console.print("[yellow]Usage: /config key remove <provider>[/yellow]")
            return
        provider_id = rest[0]
        try:
            existed = remove_key(provider_id)
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        if existed:
            console.print(
                f"[green]Removed stored API key for[/green] [cyan]{provider_id}[/cyan]."
            )
        else:
            console.print(f"[dim]No stored API key for {provider_id}.[/dim]")
        return

    console.print("[yellow]Unknown key config action. Try /config help.[/yellow]")


def _config_model(action_parts: list[str], session: Session, console: Console) -> None:
    action = action_parts[0].lower() if action_parts else "show"
    rest = action_parts[1:]

    if action == "show":
        console.print(f"Current model: [cyan]{session.model}[/cyan]")
        return
    if action == "set":
        if not rest:
            console.print("[yellow]Usage: /config model set <model>[/yellow]")
            return
        model = rest[0]
        set_selected_model(model)
        session.model = model
        console.print(f"[green]Default model set to[/green] [cyan]{model}[/cyan].")
        return
    if action in {"reset", "clear"}:
        set_selected_model(None)
        session.model = session.provider.default_model
        console.print(
            f"[green]Default model reset.[/green] "
            f"Using [cyan]{session.model}[/cyan]."
        )
        return
    console.print("[yellow]Unknown model config action. Try /config help.[/yellow]")


def _config_models(action_parts: list[str], session: Session, console: Console) -> None:
    action = action_parts[0].lower() if action_parts else "show"
    models = action_parts[1:]
    provider_id = session.provider_id

    if action in {"show", "list", "ls"}:
        model_filter = get_model_filter(provider_id)
        visible = list_visible_models(provider_id)
        console.print(
            f"Model visibility for [cyan]{provider_id}[/cyan]: "
            f"[cyan]{model_filter['mode']}[/cyan]"
        )
        if model_filter["models"]:
            console.print("Configured list: " + ", ".join(model_filter["models"]))
        console.print("Shown in completions: " + (", ".join(visible) or "[none]"))
        return

    if action in {"allow", "only"}:
        if not models:
            console.print("[yellow]Usage: /config models allow <model> [...][/yellow]")
            return
        try:
            set_model_filter(provider_id, "allow", models)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(
            f"[green]Only showing these models for {provider_id}:[/green] "
            + ", ".join(models)
        )
        return

    if action in {"deny", "hide", "block"}:
        if not models:
            console.print("[yellow]Usage: /config models deny <model> [...][/yellow]")
            return
        try:
            set_model_filter(provider_id, "deny", models)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(
            f"[green]Hiding these models for {provider_id}:[/green] "
            + ", ".join(models)
        )
        return

    if action in {"clear", "reset", "all"}:
        set_model_filter(provider_id, "all", [])
        console.print(f"[green]Cleared model visibility for {provider_id}.[/green]")
        return

    console.print("[yellow]Unknown models config action. Try /config help.[/yellow]")


def _config(arg: str, session: Session, console: Console) -> str:
    try:
        parts = shlex.split(arg)
    except ValueError as exc:
        console.print(f"[red]Could not parse command: {exc}[/red]")
        return CommandResult.CONTINUE

    if not parts or parts[0] in {"show", "status"}:
        _config_show(session, console)
        return CommandResult.CONTINUE

    section = parts[0].lower()
    rest = parts[1:]
    if section in {"help", "?"}:
        _config_help(console)
    elif section in {"providers", "provider"}:
        _config_providers(rest, session, console)
    elif section in {"keys", "key", "api-key", "api-keys"}:
        _config_keys(rest, console)
    elif section == "use":
        _config_providers(["use", *rest], session, console)
    elif section == "model":
        _config_model(rest, session, console)
    elif section == "models":
        _config_models(rest, session, console)
    else:
        console.print("[yellow]Unknown config command. Try /config help.[/yellow]")
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
    "config": _config,
    "cost": _cost,
}

_HELP = {
    "/help": "Show this help.",
    "/exit, /quit": "Leave the session.",
    "/clear": "Clear the conversation history.",
    "/model [name]": "Choose a visible model, or set one by name.",
    "/provider [id]": "Choose a provider, or switch by id.",
    "/mode [ask|auto-accept|plan]": "Choose or directly set the permission mode.",
    "/tools": "List available tools.",
    "/files": "List files changed this session.",
    "/context [list|add|remove|clear]": (
        "Manage pinned files included with each turn."
    ),
    "/config": (
        "Show or change providers, API keys, model defaults, and model filters."
    ),
    "/cost": "Show token usage for this session.",
}

_COMMAND_DESCRIPTIONS = {
    "help": "Show available commands.",
    "exit": "Leave the session.",
    "quit": "Leave the session.",
    "clear": "Clear the conversation history.",
    "model": "Choose a visible model, or set one by name.",
    "provider": "Choose a provider, or switch by id.",
    "mode": "Choose or directly set the permission mode.",
    "tools": "List available tools.",
    "files": "List files changed this session.",
    "context": "Manage pinned files included with each turn.",
    "ctx": "Alias for /context.",
    "config": (
        "Show or change providers, API keys, model defaults, and model filters."
    ),
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

_CONFIG_ACTION_DESCRIPTIONS = {
    "show": "Show active config.",
    "help": "Show /config usage.",
    "providers": "List or manage providers.",
    "provider": "List or manage providers.",
    "key": "Save, remove, or list API keys.",
    "keys": "Save, remove, or list API keys.",
    "use": "Persist and switch provider.",
    "model": "Set or reset the default model.",
    "models": "Manage model allow/deny filters.",
}


def _selection_prompt(name: str, session: Session) -> SelectionPrompt | None:
    """Build the selector shown for choice-based commands without arguments."""
    if name == "model":
        models = list_visible_models(session.provider_id)
        return SelectionPrompt(
            title="Select model",
            text=(
                f"Choose a model from {session.provider.name} "
                f"({session.provider_id}).\nCurrent model: {session.model}"
            ),
            options=tuple(CommandOption(model, model) for model in models),
            selected=session.model if session.model in models else None,
            empty_message=(
                f"No models are visible for {session.provider_id}. "
                "Use /config models clear or /model <name>."
            ),
        )

    if name == "provider":
        providers = list_provider_configs()
        return SelectionPrompt(
            title="Select provider",
            text="Choose the provider to use for this session.",
            options=tuple(
                CommandOption(
                    provider.id,
                    f"{provider.name} — {provider.default_model}",
                )
                for provider in providers
            ),
            selected=session.provider_id,
        )

    if name == "mode":
        return SelectionPrompt(
            title="Select permission mode",
            text="Choose how KiwiMateCoder may use tools in this session.",
            options=tuple(
                CommandOption(value, f"{value} — {description}")
                for value, description in _MODE_DESCRIPTIONS.items()
            ),
            selected=session.mode.value,
        )

    return None


def slash_command_completions(prefix: str = "") -> list[tuple[str, str]]:
    """Return slash command completions matching ``prefix`` without the slash."""
    normalized = prefix.lower()
    return [
        (f"/{name}", description)
        for name, description in _COMMAND_DESCRIPTIONS.items()
        if name.startswith(normalized)
    ]


def slash_argument_completions(
    command: str, prefix: str = "", session: Session | None = None
) -> list[tuple[str, str]]:
    """Return first-argument completions for slash commands that have them."""
    normalized = prefix.lower()
    command = command.lower()
    if command in {"context", "ctx"}:
        choices = _CONTEXT_ACTION_DESCRIPTIONS
    elif command == "mode":
        choices = _MODE_DESCRIPTIONS
    elif command == "model" and session is not None:
        choices = {
            model: f"{session.provider_id} model"
            for model in list_visible_models(session.provider_id)
        }
    elif command == "provider":
        choices = {p.id: p.name for p in list_provider_configs()}
    elif command == "config":
        choices = _CONFIG_ACTION_DESCRIPTIONS
    else:
        return []
    return [
        (value, description)
        for value, description in choices.items()
        if value.startswith(normalized)
    ]
