"""Slash command handlers for the REPL."""

from __future__ import annotations

import datetime
import shlex
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from glob import has_magic
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from kiwimatecoder import tools
from kiwimatecoder import memory as memory_module
from kiwimatecoder.catalog import ModelCatalog, summarize_ids
from kiwimatecoder.config import (
    add_command_rule,
    add_provider,
    apply_model_filter,
    apply_profile,
    clear_always_allowed_tools,
    clear_budget,
    clear_command_rules,
    describe_key,
    get_acp,
    get_always_allowed_tools,
    get_budget,
    get_browser,
    get_command_rules,
    get_default_mode,
    get_lsp,
    get_mcp_servers,
    get_memory,
    get_model_catalog,
    get_model_filter,
    get_network,
    get_profile,
    get_profiles,
    get_prompt_cache,
    get_provider_config,
    get_remote,
    get_sampling,
    get_sandbox,
    get_shell_config,
    get_subagents,
    get_ui,
    get_vision,
    get_web,
    list_provider_configs,
    list_visible_models,
    project_config_path,
    remove_always_allowed_tool,
    remove_command_rule,
    remove_key,
    remove_profile,
    remove_provider,
    reset_default_mode,
    reset_sampling,
    save_profile,
    search_model_catalog,
    set_active_providers,
    set_acp,
    set_budget,
    set_browser,
    set_default_mode,
    set_key,
    set_lsp,
    set_model_filter,
    set_network,
    set_output_style,
    set_prompt_cache,
    set_remote,
    set_sampling,
    set_sandbox,
    set_selected_model,
    set_selected_provider,
    set_shell_config,
    set_subagents,
    set_system_prompt,
    set_trusted_workspace,
    set_ui,
    set_verify_command,
    set_vision,
    set_web,
    update_provider,
)
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.providers import DEFAULT_PROVIDER_ID, REGISTRY, ProviderConfig
from kiwimatecoder.session import (
    Session,
    apply_session_profile,
    export_session_markdown,
    fork_session,
    list_saved_sessions,
    load_session,
    save_session,
)
from kiwimatecoder.templates import discover_templates
from kiwimatecoder.tools.paths import PathError, display_path, resolve_in_workspace

if TYPE_CHECKING:
    from kiwimatecoder.lsp import LspManager
    from kiwimatecoder.mcp import McpManager


class CommandResult:
    """Sentinel results a command can return."""

    CONTINUE: str = "continue"
    EXIT: str = "exit"


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


@dataclass(frozen=True)
class MultiSelectionPrompt:
    """Terminal-agnostic description of an interactive checklist prompt."""

    title: str
    text: str
    options: tuple[CommandOption, ...]
    selected: tuple[str, ...] = ()
    empty_message: str = "No choices are available."


CommandSelector = Callable[[SelectionPrompt], str | None]
CommandMultiSelector = Callable[[MultiSelectionPrompt], list[str] | None]


def dispatch(
    line: str,
    session: Session,
    console: Console,
    selector: CommandSelector | None = None,
    prompt_input: Callable[[str], str] | None = None,
    multi_selector: CommandMultiSelector | None = None,
) -> str:
    """Run a slash command. Returns CommandResult.CONTINUE or .EXIT."""
    parts = line[1:].strip().split(maxsplit=1)
    name = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    handler = _COMMANDS.get(name)
    if handler is None:
        console.print(f"[yellow]Unknown command '/{name}'. Try /help.[/yellow]")
        return CommandResult.CONTINUE

    if name == "config" and not arg and selector is not None:
        return _config_interact(session, console, selector, prompt_input)

    if name == "provider" and not arg and multi_selector is not None:
        checklist = _multi_selection_prompt(session)
        if checklist is not None:
            if not checklist.options:
                console.print(f"[yellow]{checklist.empty_message}[/yellow]")
                return CommandResult.CONTINUE
            chosen = multi_selector(checklist)
            if chosen is None:
                return CommandResult.CONTINUE
            valid = {option.value for option in checklist.options}
            if not chosen or any(pid not in valid for pid in chosen):
                console.print("[red]The selector returned an invalid choice.[/red]")
                return CommandResult.CONTINUE
            _apply_provider_checklist(session, console, chosen)
            return CommandResult.CONTINUE

    if not arg and selector is not None:
        prompt = _selection_prompt(name, session, console)
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
    if name == "model":
        # /model search <term> needs the selector to offer a filtered picker.
        return _model(arg, session, console, selector)
    return handler(arg, session, console)


def _help(arg: str, session: Session, console: Console) -> str:
    for group, entries in _HELP_GROUPS:
        table = Table(title=group, show_header=True, expand=False)
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description")
        for cmd, desc in entries:
            # Escaped: rich reads argument hints like [name|refresh] as markup
            # tags and would drop them from the table.
            table.add_row(escape(cmd), desc)
        console.print(table)
    _print_custom_commands(session, console)
    console.print(
        "[dim]Tip: no API key yet? From the shell run "
        "`kiwimatecoder setup`.[/dim]"
    )
    return CommandResult.CONTINUE


def _print_custom_commands(session: Session, console: Console) -> None:
    """Append the discovered prompt templates as a help group."""
    templates = discover_templates(session.workspace_root)
    if not templates:
        return
    table = Table(title="Custom commands", show_header=True, expand=False)
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description")
    for name, template in templates.items():
        table.add_row(escape(f"/{name}"), escape(template.description))
    console.print(table)


def _templates(arg: str, session: Session, console: Console) -> str:
    templates = discover_templates(session.workspace_root)
    if not templates:
        console.print(
            "[dim]No custom command templates. Add .md files to "
            ".kiwimatecoder/commands/ or ~/.kiwimatecoder/commands/.[/dim]"
        )
        return CommandResult.CONTINUE
    table = Table(title="Custom command templates", show_header=True)
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description")
    table.add_column("Path", style="dim", overflow="fold")
    for name, template in templates.items():
        table.add_row(
            escape(f"/{name}"),
            escape(template.description),
            escape(str(template.path)),
        )
    console.print(table)
    return CommandResult.CONTINUE


def _exit(arg: str, session: Session, console: Console) -> str:
    console.print("[dim]Goodbye![/dim]")
    return CommandResult.EXIT


def _clear(arg: str, session: Session, console: Console) -> str:
    session.reset_history()
    console.print("[dim]Conversation cleared.[/dim]")
    return CommandResult.CONTINUE


_MODEL_REFRESH_WORDS = {"refresh", "--refresh", "-r", "update"}
_MODEL_LIST_WORDS = {"list", "--list", "ls"}
_MODEL_SEARCH_WORDS = {"search", "find", "--search"}


def _format_age(seconds: float) -> str:
    """Render a cache age as a short human string."""
    if seconds < 90:
        return "just now"
    minutes = seconds / 60
    if minutes < 90:
        return f"{int(minutes)}m ago"
    hours = minutes / 60
    if hours < 48:
        return f"{int(hours)}h ago"
    return f"{int(hours / 24)}d ago"


def _catalog_status(catalog: ModelCatalog) -> str:
    """Describe where a catalog came from, for display."""
    if catalog.source == "curated":
        return "built-in list"
    age = (
        _format_age(max(0.0, time.time() - catalog.fetched_at))
        if catalog.fetched_at
        else "unknown age"
    )
    return f"live from provider, {'refreshed' if catalog.source == 'live' else 'cached'} {age}"


def _report_catalog(
    catalog: ModelCatalog, session: Session, console: Console, *, verbose: bool
) -> None:
    """Print what a catalog refresh changed.

    Kept quiet on the selector path (``verbose=False``) — only real news is
    worth interrupting a model pick: models that appeared, models the provider
    retired, and a current model that no longer exists.
    """
    if catalog.source == "live":
        if verbose:
            console.print(
                f"[green]Refreshed {session.provider.name} models[/green] — "
                f"{len(catalog.models)} offered."
            )
        if catalog.added:
            console.print(
                f"[green]New ({len(catalog.added)}):[/green] "
                f"{summarize_ids(catalog.added)}"
            )
        if catalog.removed:
            console.print(
                f"[yellow]Deprecated, removed ({len(catalog.removed)}):[/yellow] "
                f"{summarize_ids(catalog.removed)}"
            )
        if verbose and not catalog.added and not catalog.removed:
            console.print("[dim]No changes since the last check.[/dim]")
        if session.model not in catalog.models:
            console.print(
                f"[yellow]Current model[/yellow] [cyan]{session.model}[/cyan] "
                f"[yellow]is no longer offered by {session.provider.name}.[/yellow] "
                "Pick another with /model, or run /config model reset."
            )
        return

    if catalog.error:
        console.print(
            f"[dim]Could not refresh models ({catalog.error}); "
            f"using the {'cached' if catalog.source == 'cache' else 'built-in'} "
            "list.[/dim]"
        )
    elif verbose:
        console.print(f"[dim]Using the {_catalog_status(catalog)}.[/dim]")


def _model_catalog_for(
    session: Session, console: Console, *, force: bool = False, verbose: bool = False
) -> ModelCatalog:
    """Resolve the catalog for the active provider and report what changed."""
    with console.status(f"Checking {session.provider.name} for new models…"):
        catalog = get_model_catalog(
            session.provider_id, refresh=True, force=force, keep=(session.model,)
        )
    _report_catalog(catalog, session, console, verbose=verbose)
    return catalog


def _print_model_catalog(session: Session, console: Console, catalog: ModelCatalog) -> None:
    visible = apply_model_filter(session.provider_id, catalog.models)
    table = Table(
        title=f"{session.provider.name} models ({_catalog_status(catalog)})",
        show_header=True,
    )
    table.add_column("model", style="cyan")
    table.add_column("")
    for model in visible:
        table.add_row(model, "active" if model == session.model else "")
    console.print(table)
    if not visible:
        console.print(
            f"[yellow]No models are visible for {session.provider_id}.[/yellow] "
            "Use /config models clear or /model <name>."
        )


def _print_model_search(
    session: Session, console: Console, models: Sequence[str], query: str
) -> None:
    table = Table(
        title=f"{len(models)} match(es) for '{query}' "
        f"({_catalog_status(get_model_catalog(session.provider_id))})",
        show_header=True,
    )
    table.add_column("model", style="cyan")
    table.add_column("")
    for model in models:
        table.add_row(model, "active" if model == session.model else "")
    console.print(table)


def _model_search(
    session: Session,
    console: Console,
    query: str,
    selector: CommandSelector | None,
) -> None:
    """Search the full catalog and offer the matches for selection."""
    with console.status(f"Searching {session.provider.name} models for '{query}'…"):
        matches = search_model_catalog(
            session.provider_id, query, refresh=True, keep=(session.model,)
        )
    if not matches:
        console.print(
            f"[yellow]No models matching '{query}' "
            f"for {session.provider.name} ({session.provider_id}).[/yellow]"
        )
        return

    if selector is not None:
        prompt = SelectionPrompt(
            title=f"Search: {query}",
            text=(
                f"{len(matches)} match(es) from {session.provider.name} "
                f"({session.provider_id}).\nCurrent model: {session.model}"
            ),
            options=tuple(CommandOption(model, model) for model in matches),
            selected=session.model if session.model in matches else None,
        )
        selected = _run_selector(selector, prompt)
        if selected is None:
            return
        _apply_model(session, selected, console)
        return

    _print_model_search(session, console, matches, query)


def _model(arg: str, session: Session, console: Console, selector: CommandSelector | None = None) -> str:
    if not arg:
        console.print(f"Current model: [cyan]{session.model}[/cyan]")
        return CommandResult.CONTINUE

    parts = arg.strip().split(maxsplit=1)
    action = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""
    if action in _MODEL_REFRESH_WORDS:
        catalog = _model_catalog_for(session, console, force=True, verbose=True)
        _print_model_catalog(session, console, catalog)
        return CommandResult.CONTINUE
    if action in _MODEL_LIST_WORDS:
        catalog = _model_catalog_for(session, console)
        _print_model_catalog(session, console, catalog)
        return CommandResult.CONTINUE
    if action in _MODEL_SEARCH_WORDS:
        if not rest:
            console.print("[yellow]Usage: /model search <term>[/yellow]")
            return CommandResult.CONTINUE
        _model_search(session, console, rest, selector)
        return CommandResult.CONTINUE

    _apply_model(session, arg, console)
    return CommandResult.CONTINUE


def _apply_model(session: Session, model: str, console: Console) -> None:
    """Switch the session model and remember it as the default for next time."""
    session.model = model
    set_selected_model(model)
    console.print(f"Model set to [cyan]{model}[/cyan].")


def _provider(arg: str, session: Session, console: Console) -> str:
    if not arg:
        active = {p.id for p in session.active_providers}
        table = Table(title="Providers", show_header=True)
        table.add_column("id", style="cyan")
        table.add_column("name")
        table.add_column("default model")
        for p in list_provider_configs():
            if p.id in active:
                marker = " (primary)" if p.id == session.provider_id else " (active)"
            else:
                marker = ""
            table.add_row(p.id + marker, p.name, p.default_model or "(from server)")
        console.print(table)
        return CommandResult.CONTINUE
    try:
        get_provider_config(arg)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return CommandResult.CONTINUE
    session.set_active_providers(set_active_providers([arg]))
    console.print(
        f"Provider set to [cyan]{session.provider_id}[/cyan] "
        f"(model: [cyan]{session.model}[/cyan])."
    )
    return CommandResult.CONTINUE


def _apply_provider_checklist(
    session: Session, console: Console, provider_ids: list[str]
) -> None:
    """Persist and apply a checked list of active providers."""
    try:
        ids = set_active_providers(provider_ids)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        return
    session.set_active_providers(ids)
    primary = ids[0]
    fallbacks = ids[1:]
    summary = (
        f"Active providers: [cyan]{', '.join(ids)}[/cyan] "
        f"(primary: [cyan]{primary}[/cyan], model: [cyan]{session.model}[/cyan])."
    )
    if fallbacks:
        summary += f"\n[dim]Fallbacks in order: {', '.join(fallbacks)}[/dim]"
    console.print(summary)


def _provider_default_label(provider: ProviderConfig) -> str:
    """Provider label fallback for locals with no static default model."""
    if provider.default_model:
        return provider.default_model
    return "key required (local)" if provider.requires_key else "no key needed (local)"


def _multi_selection_prompt(session: Session) -> MultiSelectionPrompt | None:
    """Build the checklist shown for a bare ``/provider``."""
    providers = list_provider_configs()
    by_id = {provider.id: provider for provider in providers}
    roster = [
        pid
        for pid in (session.active_provider_ids or [session.provider_id])
        if pid in by_id
    ]
    ordered = [by_id[pid] for pid in roster]
    seen = set(roster)
    ordered.extend(provider for provider in providers if provider.id not in seen)

    def _label(provider: ProviderConfig) -> str:
        default = _provider_default_label(provider)
        if roster and provider.id == roster[0]:
            role = " (primary)"
        elif provider.id in roster:
            role = f" (fallback {roster.index(provider.id)})"
        else:
            role = ""
        return f"{provider.name} — {default}{role}"

    return MultiSelectionPrompt(
        title="Select active providers",
        text=(
            "Check every provider you want on the failover roster. The first "
            "checked provider is the primary; later checks are fallbacks "
            "tried in that order if the primary fails."
        ),
        options=tuple(
            CommandOption(provider.id, _label(provider)) for provider in ordered
        ),
        selected=tuple(roster),
    )


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


def _dry_run(arg: str, session: Session, console: Console) -> str:
    token = arg.strip().lower()
    if token in {"", "status", "show"}:
        state = "on" if session.dry_run else "off"
        console.print(f"Dry-run is [cyan]{state}[/cyan].")
        return CommandResult.CONTINUE
    if token in {"on", "true", "yes"}:
        session.dry_run = True
    elif token in {"off", "false", "no"}:
        session.dry_run = False
    elif token == "toggle":
        session.dry_run = not session.dry_run
    else:
        console.print("[yellow]Usage: /dry-run [on|off|toggle][/yellow]")
        return CommandResult.CONTINUE
    console.print(
        f"[green]Dry-run {'on' if session.dry_run else 'off'}.[/green]"
        + (
            " Mutating tools will show their preview without running."
            if session.dry_run
            else ""
        )
    )
    return CommandResult.CONTINUE


def _tools(arg: str, session: Session, console: Console) -> str:
    table = Table(title="Available Tools", show_header=True)
    table.add_column("Tool", style="cyan bold", no_wrap=True)
    table.add_column("Permission", no_wrap=True)
    table.add_column("Description")
    for tool in tools.TOOLS.values():
        if tool.needs_approval:
            perm = (
                "[green]always allowed[/green]"
                if session.is_always_allowed(tool.name)
                else "[yellow]needs approval[/yellow]"
            )
        else:
            perm = "[dim green]read-only[/dim green]"
        desc = tool.description.split(". ")[0] + "."
        table.add_row(tool.name, perm, desc)
    console.print(table)
    return CommandResult.CONTINUE


def _files(arg: str, session: Session, console: Console) -> str:
    if not session.touched_files:
        console.print("[dim]No files changed this session.[/dim]")
        return CommandResult.CONTINUE
    table = Table(title="Files Changed This Session", show_header=True)
    table.add_column("File Path", style="cyan")
    table.add_column("Status")
    table.add_column("Size", justify="right")
    for rel in session.touched_files:
        p = session.workspace_root / rel
        if p.exists() and p.is_file():
            size_bytes = p.stat().st_size
            size_str = (
                f"{size_bytes} B"
                if size_bytes < 1024
                else f"{size_bytes / 1024:.1f} KB"
            )
            status = "[green]exists[/green]"
        elif p.exists() and p.is_dir():
            size_str = "-"
            status = "[blue]directory[/blue]"
        else:
            size_str = "-"
            status = "[red]deleted/missing[/red]"
        table.add_row(rel, status, size_str)
    console.print(table)
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


def _mcp(arg: str, session: Session, console: Console) -> str:
    from kiwimatecoder import mcp

    parts = arg.strip().split(maxsplit=1)
    action = parts[0].lower() if parts and parts[0] else "list"
    manager = mcp.get_manager()
    if action in {"list", "ls", "status", "show"}:
        _mcp_show(manager, console)
        return CommandResult.CONTINUE
    if action in {"reload", "refresh"}:
        if manager is None:
            manager = mcp.McpManager(console=console)
            mcp.set_manager(manager)
        result = manager.reload()
        console.print(
            f"[green]MCP reloaded:[/green] {len(result.servers)} server(s) "
            f"connected, {len(result.tools)} tool(s) registered."
        )
        _mcp_show(manager, console)
        return CommandResult.CONTINUE
    console.print("[yellow]Usage: /mcp [list|reload][/yellow]")
    return CommandResult.CONTINUE


def _mcp_show(manager: "McpManager | None", console: Console) -> None:
    servers = get_mcp_servers()
    if not servers:
        console.print(
            "[dim]No MCP servers configured. Add one with "
            "`kiwimatecoder config mcp add <name> --command ...` or the "
            "mcp_servers config key.[/dim]"
        )
        return
    table = Table(title="MCP servers", show_header=True)
    table.add_column("server", style="cyan")
    table.add_column("transport")
    table.add_column("status")
    table.add_column("tools", justify="right")
    table.add_column("resources", justify="right")
    for name, spec in servers.items():
        transport = "stdio" if spec.get("command") else "http"
        failure = manager.failure_for(name) if manager is not None else None
        if spec.get("disabled"):
            status = "[dim]disabled[/dim]"
        elif manager is not None and manager.is_connected(name):
            status = "[green]connected[/green]"
        elif failure:
            status = f"[red]failed[/red] [dim]{escape(failure)}[/dim]"
        else:
            status = "[yellow]not connected[/yellow]"
        registered = manager.tools_for(name) if manager is not None else []
        resources = manager.resource_count(name) if manager is not None else None
        table.add_row(
            name,
            transport,
            status,
            str(len(registered)) if manager is not None else "-",
            "-" if resources is None else str(resources),
        )
    console.print(table)

    tool_rows: list[tuple[str, str, str]] = []
    for server in servers:
        names = manager.tools_for(server) if manager is not None else []
        for tool_name in names:
            tool = tools.get_tool(tool_name)
            if tool is None:
                continue
            permission = (
                "[yellow]needs approval[/yellow]"
                if tool.needs_approval
                else "[dim green]read-only[/dim green]"
            )
            summary = tool.description.split(". ")[0] + "."
            tool_rows.append((tool_name, permission, escape(summary)))
    if tool_rows:
        tool_table = Table(title="MCP tools", show_header=True)
        tool_table.add_column("tool", style="cyan")
        tool_table.add_column("permission")
        tool_table.add_column("description")
        for row in tool_rows:
            tool_table.add_row(*row)
        console.print(tool_table)


def _lsp(arg: str, session: Session, console: Console) -> str:
    from kiwimatecoder import lsp

    parts = arg.strip().split(maxsplit=1)
    action = parts[0].lower() if parts and parts[0] else "status"
    manager = lsp.get_manager()
    if action in {"status", "show", "list", "ls"}:
        _lsp_show(manager, console)
        return CommandResult.CONTINUE
    if action in {"on", "enable", "enabled"}:
        settings = set_lsp(enabled=True)
        console.print(
            "[green]LSP on:[/green] servers start on first use "
            "(diagnostics after edits: "
            f"{'on' if settings['diagnostics_after_edits'] else 'off'})."
        )
        _lsp_show(manager, console)
        return CommandResult.CONTINUE
    if action in {"off", "disable", "disabled"}:
        set_lsp(enabled=False)
        if manager is not None:
            manager.shutdown()
        console.print("[green]LSP off.[/green] Running language servers stopped.")
        return CommandResult.CONTINUE
    if action in {"restart", "reload"}:
        if manager is None:
            console.print("[dim]No language servers have started yet.[/dim]")
            return CommandResult.CONTINUE
        manager.shutdown()
        console.print(
            "[green]LSP restarted:[/green] running servers stopped and "
            "failures cleared; they start again on first use."
        )
        return CommandResult.CONTINUE
    console.print("[yellow]Usage: /lsp [status|on|off|restart][/yellow]")
    return CommandResult.CONTINUE


def _lsp_show(manager: "LspManager | None", console: Console) -> None:
    from kiwimatecoder import lsp

    settings = get_lsp()
    state = "on" if settings["enabled"] else "off"
    console.print(
        f"LSP: [cyan]{state}[/cyan] "
        f"(timeout {settings['timeout']:g}s, diagnostics after edits "
        f"[cyan]{'on' if settings['diagnostics_after_edits'] else 'off'}[/cyan])"
    )
    servers = lsp.effective_servers()
    installed = lsp.available_servers()
    if installed:
        console.print(
            "Available servers: "
            + ", ".join(
                f"[cyan]{name}[/cyan] ({spec.command})"
                for name, spec in sorted(installed.items())
            )
        )
    else:
        console.print(
            "[dim]No language server commands found on PATH (install e.g. "
            "pyright-langserver, typescript-language-server, gopls, "
            "rust-analyzer, or clangd).[/dim]"
        )
    missing = sorted(set(servers) - set(installed))
    if missing:
        console.print(
            "[dim]Not installed: "
            + ", ".join(f"{name} ({servers[name].command})" for name in missing)
            + "[/dim]"
        )
    running = sorted(manager.clients) if manager is not None else []
    if running:
        console.print(
            "Running clients: " + ", ".join(f"[cyan]{name}[/cyan]" for name in running)
        )
    else:
        console.print("[dim]Running clients: none[/dim]")


def _config_help(console: Console) -> None:
    table = Table(title="/config commands", show_header=True)
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    rows = [
        ("/config", "Show active providers, model, key, and model filter."),
        ("/config providers", "List built-in and custom providers."),
        (
            "/config provider add <id> <name> <base_url> <default_model> [key_env] "
            "[key_header=...] [key_prefix=...] [api_version=...]",
            "Add an OpenAI-compatible custom provider. Quote names with spaces. "
            "Use key_header=api-key key_prefix= api_version=<date> for Azure.",
        ),
        ("/config provider remove <id>", "Remove a custom provider."),
        ("/config provider use <id>", "Persist and switch to a provider."),
        (
            "/config provider edit <id> name=... base_url=... default_model=... "
            "[key_env=...] [compat=...] [key_header=...] [key_prefix=...] "
            "[api_version=...]",
            "Update fields of a custom provider.",
        ),
        ("/config key set <provider> <key>", "Save an API key."),
        ("/config key remove <provider>", "Remove a stored API key."),
        ("/config key list", "Show which providers have keys configured."),
        ("/config model set <model>", "Persist the default model."),
        ("/config model reset", "Use the provider default model."),
        ("/config mode <set|reset>", "Set or reset the default permission mode."),
        (
            "/config models allow <model> [...]",
            "Only show these models for the active provider.",
        ),
        (
            "/config models deny <model> [...]",
            "Hide these models for the active provider.",
        ),
        ("/config models clear", "Clear model visibility for the active provider."),
        (
            "/config models refresh",
            "Fetch the provider's live model list, dropping deprecated ids.",
        ),
        (
            "/config permissions [list|remove <tool>|clear]",
            "Manage tools you approved with 'always'.",
        ),
        (
            "/config commands [list|allow <regex>|deny <regex>|"
            "remove <kind> <regex>|clear]",
            "Auto-approve or hard-block run_bash commands by regex.",
        ),
        (
            "/config trust [on|off]",
            "Allow or forbid read-only access outside the workspace root.",
        ),
        (
            "/config verify [set <command>|clear]",
            "Run a command automatically after successful file edits.",
        ),
        (
            "/config budget [show|tokens <n>|cost <usd>|clear]",
            "Set or clear session token/cost limits.",
        ),
        (
            "/config subagents [show|enable on|off|max-steps <n>]",
            "Configure the subagent task tool and its step limit.",
        ),
        (
            "/config browser [show|enable on|off|headless on|off|timeout <ms>]",
            "Configure optional Playwright browser automation.",
        ),
        (
            "/config shell [show|persistent on|off|timeout <s>|max-jobs <n>]",
            "Configure the persistent shell and background job cap.",
        ),
        (
            "/config sandbox [show|enable on|off|network on|off|"
            "add-path <path>|remove-path <path>|clear-paths]",
            "Sandbox shell commands with seatbelt (macOS) or bubblewrap (Linux).",
        ),
        (
            "/config remote [show|enable on|off|host <h>|user <u>|port <n>|"
            "identity <path>|workspace <path>|devcontainer <auto|off|name>]",
            "Run shell commands over SSH or in a devcontainer/Docker container.",
        ),
        (
            "/config acp [show|timeout <s>]",
            "Permission timeout for the ACP server used by editors.",
        ),
        (
            "/config sampling [show|set key=value ...|reset]",
            "Get or set temperature, top_p, max_tokens, reasoning_effort.",
        ),
        (
            "/config web [show|max-chars <n>|timeout <s>|allow-local <on|off>]",
            "Set web fetch/search limits and local-address access.",
        ),
        (
            "/config network [show|proxy <url|clear>|ca <path|clear>|"
            "offline <on|off>]",
            "Set the proxy, custom CA bundle, and offline mode.",
        ),
        (
            "/config vision [show|max-bytes <n>|max-images <n>]",
            "Set image attachment size and per-turn count limits.",
        ),
        (
            "/config style [set <default|concise|explanatory|code>]",
            "Show or set the output style.",
        ),
        (
            "/config prompt [set <text>|clear]",
            "Show, set, or clear a custom system-prompt addition.",
        ),
        (
            "/config profile [list|show <name>|save <name>|use <name>|remove <name>]",
            "Save or apply named configuration presets.",
        ),
        (
            "/config ui [show|color <auto|always|never>|"
            "output <normal|compact|verbose>|ascii <on|off>|"
            "theme <default|ocean|magenta|mono>]",
            "Set theme, color, output verbosity, and ASCII mode.",
        ),
        (
            "/config cache [on|off]",
            "Cache the system prompt and tools on native Anthropic providers.",
        ),
        ("/doctor", "Run environment, config, and provider diagnostics."),
    ]
    for command, description in rows:
        table.add_row(escape(command), description)
    console.print(table)


def _config_show(session: Session, console: Console) -> None:
    provider = session.provider
    model_filter = get_model_filter(provider.id)
    active = [item.id for item in session.active_providers]
    active_line = ", ".join(
        f"[cyan]{pid}[/cyan]" + (" (primary)" if pid == active[0] else "")
        for pid in active
    )
    console.print(
        f"Active providers: {active_line}\n"
        f"Model: [cyan]{session.model}[/cyan]\n"
        f"Key: [cyan]{describe_key(provider.id)}[/cyan] ({provider.key_env})\n"
        f"Model visibility: [cyan]{model_filter['mode']}[/cyan]"
    )
    if model_filter["models"]:
        console.print("Models: " + ", ".join(model_filter["models"]))
    sampling = get_sampling()
    sampling_line = (
        ", ".join(f"{key}={value}" for key, value in sampling.items())
        or "provider defaults"
    )
    console.print(
        f"Output style: [cyan]{session.output_style}[/cyan]\n"
        f"Sampling: [cyan]{sampling_line}[/cyan]\n"
        f"Custom system prompt: [cyan]{'set' if session.custom_system_prompt else 'none'}[/cyan]\n"
        f"Always-allowed tools: [cyan]{', '.join(sorted(session.always_allowed)) or 'none'}[/cyan]\n"
        f"Trusted workspace: [cyan]{'on' if session.trusted_workspace else 'off'}[/cyan]"
    )
    ui_config = get_ui()
    console.print(
        f"UI: [cyan]{ui_config['theme']}[/cyan] theme, "
        f"[cyan]{ui_config['color']}[/cyan] color, "
        f"[cyan]{ui_config['output_mode']}[/cyan] output, "
        f"ascii [cyan]{'on' if ui_config['ascii'] else 'off'}[/cyan]"
    )
    network_config = get_network()
    console.print(
        "Network: proxy "
        f"[cyan]{network_config['proxy'] or 'none'}[/cyan], "
        f"CA [cyan]{network_config['ca_bundle'] or 'system'}[/cyan], "
        f"offline [cyan]{'on' if network_config['offline'] else 'off'}[/cyan]"
    )
    remote_config = get_remote()
    console.print(
        "Remote: "
        f"[cyan]{'on' if remote_config['enabled'] else 'off'}[/cyan] "
        f"(host [cyan]{escape(_remote_summary(remote_config))}[/cyan], "
        f"devcontainer [cyan]{escape(remote_config['devcontainer'])}[/cyan])"
    )
    project_path = project_config_path()
    if project_path is not None:
        console.print(f"Project config: [cyan]{project_path}[/cyan] (overrides global)")
    command_rules = get_command_rules()
    if command_rules["allow"] or command_rules["deny"]:
        console.print(
            "Command rules: "
            + f"[cyan]{len(command_rules['deny'])} deny[/cyan], "
            + f"[cyan]{len(command_rules['allow'])} allow[/cyan]"
        )


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
        table.add_column("auth")
        table.add_column("base URL")
        active = {item.id for item in session.active_providers}
        for provider in list_provider_configs():
            if provider.id in active:
                marker = " (primary)" if provider.id == session.provider_id else " (active)"
            else:
                marker = ""
            if provider.is_local:
                kind = "local"
            else:
                kind = "built-in" if provider.id in REGISTRY else "custom"
            auth = provider.key_header
            if provider.key_prefix:
                auth += f": {provider.key_prefix.strip()}"
            table.add_row(
                provider.id + marker,
                kind,
                provider.name,
                provider.default_model or "(from server)",
                auth,
                provider.base_url,
            )
        console.print(table)
        return

    provider_add_fields = {"key_header", "key_prefix", "api_version"}

    if action in {"add", "create"}:
        if len(rest) < 4:
            console.print(
                "[yellow]Usage: /config provider add <id> <name> "
                "<base_url> <default_model> [key_env] [key_header=...] "
                "[key_prefix=...] [api_version=...][/yellow]"
            )
            return
        provider_id, name, base_url, default_model = rest[:4]
        extras = rest[4:]
        key_env: str | None = None
        if extras and "=" not in extras[0]:
            key_env = extras.pop(0)
        options: dict[str, str] = {}
        for pair in extras:
            if "=" not in pair:
                console.print(
                    f"[yellow]Expected field=value, got '{pair}'. "
                    "Known fields: key_header, key_prefix, api_version.[/yellow]"
                )
                return
            field, _, value = pair.partition("=")
            field = field.strip()
            if field not in provider_add_fields:
                console.print(
                    f"[yellow]Unknown provider field '{field}'. "
                    "Known fields: key_header, key_prefix, api_version.[/yellow]"
                )
                return
            options[field] = value
        try:
            provider = add_provider(
                provider_id,
                name,
                base_url,
                default_model,
                key_env,
                **options,
            )
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
        roster = session.active_provider_ids or [session.provider_id]
        was_in_roster = provider_id in roster
        try:
            remove_provider(provider_id)
        except (KeyError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            return
        if was_in_roster:
            remaining = [pid for pid in roster if pid != provider_id]
            session.set_active_providers(remaining or [DEFAULT_PROVIDER_ID])
        console.print(f"[green]Removed provider[/green] [cyan]{provider_id}[/cyan].")
        return

    if action in {"use", "select", "set"}:
        if not rest:
            console.print("[yellow]Usage: /config provider use <id>[/yellow]")
            return
        provider_id = rest[0]
        try:
            set_selected_provider(provider_id)
            session.set_active_providers([provider_id])
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(
            f"Provider set to [cyan]{session.provider_id}[/cyan] "
            f"(model: [cyan]{session.model}[/cyan])."
        )
        return

    if action in {"edit", "update"}:
        if len(rest) < 2:
            console.print(
                "[yellow]Usage: /config provider edit <id> name=... "
                "base_url=... default_model=... key_env=... compat=... "
                "key_header=... key_prefix=... api_version=...[/yellow]"
            )
            return
        provider_id = rest[0]
        known_fields = {
            "name",
            "base_url",
            "default_model",
            "key_env",
            "compat",
            "key_header",
            "key_prefix",
            "api_version",
        }
        kwargs: dict[str, str] = {}
        for pair in rest[1:]:
            if "=" not in pair:
                console.print(
                    f"[yellow]Expected field=value, got '{pair}'. "
                    "Known fields: name, base_url, default_model, key_env, "
                    "compat, key_header, key_prefix, api_version.[/yellow]"
                )
                return
            field, _, value = pair.partition("=")
            field = field.strip()
            if field not in known_fields:
                console.print(
                    f"[yellow]Unknown provider field '{field}'. "
                    "Known fields: name, base_url, default_model, key_env, "
                    "compat, key_header, key_prefix, api_version.[/yellow]"
                )
                return
            kwargs[field] = value
        try:
            provider = update_provider(provider_id, **kwargs)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(
            f"[green]Updated provider[/green] [cyan]{provider.id}[/cyan]."
        )
        return

    console.print("[yellow]Unknown provider config action. Try /config help.[/yellow]")


def _config_keys(
    action_parts: list[str],
    console: Console,
    selector: CommandSelector | None = None,
    prompt_input: Callable[[str], str] | None = None,
) -> None:
    action = action_parts[0].lower() if action_parts else "list"
    rest = action_parts[1:]

    if action in {"list", "ls"}:
        table = Table(title="API keys", show_header=True)
        table.add_column("provider", style="cyan")
        table.add_column("env var")
        table.add_column("status")
        for provider in list_provider_configs():
            table.add_row(provider.id, provider.key_env, describe_key(provider.id))
        console.print(table)
        console.print(
            "[dim]Change a key with /config key set <provider> <key> or from "
            "the /config menu.[/dim]"
        )
        return

    if action in {"set", "save", "add"}:
        if len(rest) < 2:
            if rest and selector is not None:
                _config_key_enter(rest[0], console, selector, prompt_input)
                return
            console.print("[yellow]Usage: /config key set <provider> <key>[/yellow]")
            return
        provider_id, key = rest[0], rest[1]
        try:
            warning = set_key(provider_id, key)
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(
            f"[green]Saved API key for[/green] [cyan]{provider_id}[/cyan] — "
            f"now {describe_key(provider_id)}."
        )
        if warning:
            console.print(f"[yellow]{warning}[/yellow]")
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

    if action in {"edit", "change"}:
        if not rest:
            console.print(
                "[yellow]Usage: /config key edit <provider>[/yellow]"
            )
            return
        if selector is None:
            console.print(
                "[yellow]Key editing needs the interactive menu. "
                "Run /config and pick Keys.[/yellow]"
            )
            return
        _config_key_enter(rest[0], console, selector, prompt_input)
        return

    console.print("[yellow]Unknown key config action. Try /config help.[/yellow]")


def _config_key_enter(
    provider_id: str,
    console: Console,
    selector: CommandSelector,
    prompt_input: Callable[[str], str] | None,
) -> None:
    """Interactive 'change this key' flow: set or remove, then a text entry."""
    try:
        get_provider_config(provider_id)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return

    action_prompt = SelectionPrompt(
        title=f"Change key for {provider_id}",
        text=(
            f"Current key: {describe_key(provider_id)}.\n"
            "Set a new one, or remove the stored key."
        ),
        options=(
            CommandOption("set", "Enter a new API key"),
            CommandOption("remove", "Remove the stored API key"),
        ),
    )
    selected = _run_selector(selector, action_prompt)
    if selected is None:
        return

    if selected == "remove":
        existed = remove_key(provider_id)
        if existed:
            console.print(
                f"[green]Removed stored API key for[/green] [cyan]{provider_id}[/cyan]."
            )
        else:
            console.print(f"[dim]No stored API key for {provider_id}.[/dim]")
        return

    if prompt_input is None:
        prompt_input = console.input
    console.print("[bold]Enter the new API key:[/bold]")
    try:
        new_key = prompt_input("key> ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("[yellow]Cancelled.[/yellow]")
        return
    if not new_key:
        console.print("[yellow]No key entered; nothing changed.[/yellow]")
        return
    warning = set_key(provider_id, new_key)
    console.print(
        f"[green]Saved API key for[/green] [cyan]{provider_id}[/cyan] — "
        f"now {describe_key(provider_id)}."
    )
    if warning:
        console.print(f"[yellow]{warning}[/yellow]")


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


def _config_mode(action_parts: list[str], session: Session, console: Console) -> None:
    action = action_parts[0].lower() if action_parts else "show"
    rest = action_parts[1:]

    if action in {"show", "status"}:
        console.print(f"Default mode: [cyan]{get_default_mode()}[/cyan]")
        return
    if action in {"set", "save"}:
        if not rest:
            console.print(
                "[yellow]Usage: /config mode set <ask|auto-accept|plan>[/yellow]"
            )
            return
        try:
            effective = set_default_mode(rest[0])
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(
            f"[green]Default mode set to[/green] [cyan]{effective}[/cyan]."
            f"[dim] (affects new sessions)[/dim]"
        )
        return
    if action in {"reset", "clear"}:
        effective = reset_default_mode()
        console.print(
            f"[green]Default mode reset to[/green] [cyan]{effective}[/cyan]."
        )
        return
    console.print("[yellow]Unknown mode config action. Try /config help.[/yellow]")


def _config_models(action_parts: list[str], session: Session, console: Console) -> None:
    action = action_parts[0].lower() if action_parts else "show"
    models = action_parts[1:]
    provider_id = session.provider_id

    if action in {"show", "list", "ls"}:
        model_filter = get_model_filter(provider_id)
        catalog = get_model_catalog(provider_id)
        visible = apply_model_filter(provider_id, catalog.models)
        console.print(
            f"Model visibility for [cyan]{provider_id}[/cyan]: "
            f"[cyan]{model_filter['mode']}[/cyan]\n"
            f"Catalog: [cyan]{_catalog_status(catalog)}[/cyan] "
            f"({len(catalog.models)} models)"
        )
        if model_filter["models"]:
            console.print("Configured list: " + ", ".join(model_filter["models"]))
        console.print("Shown in completions: " + (", ".join(visible) or "[none]"))
        return

    if action in {"refresh", "update", "fetch"}:
        catalog = _model_catalog_for(session, console, force=True, verbose=True)
        _print_model_catalog(session, console, catalog)
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


def _config_permissions(
    action_parts: list[str], session: Session, console: Console
) -> None:
    action = action_parts[0].lower() if action_parts else "list"
    rest = action_parts[1:]

    if action in {"list", "ls", "show"}:
        allowed = sorted(session.always_allowed)
        if not allowed:
            console.print(
                "[dim]No persisted tool approvals. Answer 'always' at an "
                "approval prompt to add one.[/dim]"
            )
            return
        table = Table(title="Always-allowed tools", show_header=True)
        table.add_column("Tool", style="cyan")
        for name in allowed:
            table.add_row(name)
        console.print(table)
        return

    if action in {"remove", "rm", "delete", "unallow"}:
        if not rest:
            console.print("[yellow]Usage: /config permissions remove <tool>[/yellow]")
            return
        name = rest[0]
        removed = remove_always_allowed_tool(name)
        session.always_allowed.discard(name)
        if removed:
            console.print(f"[green]Removed persisted approval for {name}.[/green]")
        else:
            console.print(f"[dim]No persisted approval for {name}.[/dim]")
        return

    if action in {"clear", "reset"}:
        count = clear_always_allowed_tools()
        session.always_allowed.clear()
        console.print(f"[green]Cleared {count} persisted tool approval(s).[/green]")
        return

    console.print(
        "[yellow]Usage: /config permissions [list|remove <tool>|clear][/yellow]"
    )


def _config_trust(action_parts: list[str], session: Session, console: Console) -> None:
    action = action_parts[0].lower() if action_parts else "show"

    if action in {"on", "true", "enable", "enabled"}:
        session.trusted_workspace = True
        set_trusted_workspace(True)
        console.print(
            "[yellow]Trusted workspace on:[/yellow] read-only tools may now "
            "read paths outside the workspace root. Writes stay sandboxed."
        )
        return

    if action in {"off", "false", "disable", "disabled"}:
        session.trusted_workspace = False
        set_trusted_workspace(False)
        console.print("[green]Trusted workspace off: reads are sandboxed.[/green]")
        return

    if action in {"show", "status", "list"}:
        state = "on" if session.trusted_workspace else "off"
        console.print(f"Trusted workspace: [cyan]{state}[/cyan]")
        return

    console.print("[yellow]Usage: /config trust [on|off][/yellow]")


def _config_verify(action_parts: list[str], session: Session, console: Console) -> None:
    action = action_parts[0].lower() if action_parts else "show"
    rest = action_parts[1:]

    if action in {"set", "update"}:
        command = " ".join(rest).strip()
        if not command:
            console.print("[yellow]Usage: /config verify set <command>[/yellow]")
            return
        set_verify_command(command)
        session.verify_command = command
        console.print(
            f"[green]Auto-verify command set:[/green] {command}\n"
            "[dim]Run after successful file edits, with output fed back to the model.[/dim]"
        )
        return

    if action in {"clear", "reset", "off", "none"}:
        set_verify_command("")
        session.verify_command = ""
        console.print("[green]Auto-verify disabled.[/green]")
        return

    if session.verify_command:
        console.print(f"Auto-verify: [cyan]{session.verify_command}[/cyan]")
    else:
        console.print("[dim]Auto-verify is off. Set it with /config verify set <command>.[/dim]")


def _config_budget(action_parts: list[str], session: Session, console: Console) -> None:
    action = action_parts[0].lower() if action_parts else "show"
    rest = action_parts[1:]

    def _limit(raw: str | None) -> str | None:
        if raw is None or raw.strip().lower() in {"clear", "none", "off", ""}:
            return None
        return raw.strip()

    if action in {"show", "list", "status"}:
        budget = get_budget()
        if not budget:
            console.print("[dim]No budget limits set.[/dim]")
            return
        console.print(
            "Budget: "
            + ", ".join(f"[cyan]{key}[/cyan]={value}" for key, value in budget.items())
        )
        return

    if action in {"tokens", "token"}:
        try:
            set_budget(max_tokens=_limit(rest[0] if rest else None))
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        budget = get_budget()
        console.print(
            "[green]Budget updated:[/green] "
            + (", ".join(f"{key}={value}" for key, value in budget.items()) or "none")
        )
        return

    if action in {"cost", "usd", "max-cost"}:
        try:
            set_budget(max_cost_usd=_limit(rest[0] if rest else None))
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        budget = get_budget()
        console.print(
            "[green]Budget updated:[/green] "
            + (", ".join(f"{key}={value}" for key, value in budget.items()) or "none")
        )
        return

    if action in {"clear", "reset"}:
        clear_budget()
        console.print("[green]Budget limits cleared.[/green]")
        return

    console.print(
        "[yellow]Usage: /config budget [show|tokens <n>|cost <usd>|clear][/yellow]"
    )


def _config_subagents(action_parts: list[str], console: Console) -> None:
    action = action_parts[0].lower() if action_parts else "show"
    rest = action_parts[1:]

    if action in {"show", "list", "status"}:
        settings = get_subagents()
        state = "on" if settings["enabled"] else "off"
        model = settings["model"] or "(session model)"
        console.print(
            f"Subagents: [cyan]{state}[/cyan] "
            f"(max steps {settings['max_steps']}, model {model})"
        )
        return

    if action in {"enable", "enabled", "on", "off"}:
        token = action if action in {"on", "off"} else (rest[0].lower() if rest else "")
        if token not in {"on", "off"}:
            console.print("[yellow]Usage: /config subagents enable <on|off>[/yellow]")
            return
        settings = set_subagents(enabled=token == "on")
        console.print(
            f"[green]Subagents {'on' if settings['enabled'] else 'off'}.[/green]"
        )
        return

    if action in {"max-steps", "steps"}:
        if not rest:
            console.print(
                f"Subagent max steps: [cyan]{get_subagents()['max_steps']}[/cyan]"
            )
            return
        try:
            settings = set_subagents(max_steps=rest[0])
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(f"[green]Subagent max steps:[/green] {settings['max_steps']}")
        return

    if action == "model":
        if not rest:
            current = get_subagents()["model"] or "(session model)"
            console.print(f"Subagent model: [cyan]{current}[/cyan]")
            return
        value = "" if rest[0].strip().lower() in {"clear", "none", "off"} else rest[0]
        settings = set_subagents(model=value)
        console.print(
            f"[green]Subagent model:[/green] {settings['model'] or '(session model)'}"
        )
        return

    console.print(
        "[yellow]Usage: /config subagents "
        "[show|enable on|off|max-steps <n>][/yellow]"
    )


def _reset_browser_driver() -> None:
    """Drop the live Playwright page so the next call picks up new settings."""
    try:
        from kiwimatecoder import browser

        browser.reset_driver()
    except Exception:  # a stopped browser must never break config commands
        pass


def _config_browser(action_parts: list[str], console: Console) -> None:
    action = action_parts[0].lower() if action_parts else "show"
    rest = action_parts[1:]

    if action in {"show", "list", "ls", "status"}:
        settings = get_browser()
        console.print(
            f"Browser automation: "
            f"[cyan]{'on' if settings['enabled'] else 'off'}[/cyan]\n"
            f"Headless: [cyan]{'on' if settings['headless'] else 'off'}[/cyan]\n"
            f"Timeout: [cyan]{settings['timeout_ms']}ms[/cyan]"
        )
        return

    if action in {"enable", "enabled", "on", "off"}:
        token = action if action in {"on", "off"} else (rest[0].lower() if rest else "")
        if token not in {"on", "off"}:
            console.print("[yellow]Usage: /config browser enable <on|off>[/yellow]")
            return
        settings = set_browser(enabled=token == "on")
        _reset_browser_driver()
        console.print(
            f"[green]Browser automation "
            f"{'on' if settings['enabled'] else 'off'}.[/green]"
        )
        return

    if action == "headless":
        token = rest[0].lower() if rest else ""
        if token not in {"on", "off"}:
            console.print("[yellow]Usage: /config browser headless <on|off>[/yellow]")
            return
        settings = set_browser(headless=token == "on")
        _reset_browser_driver()
        console.print(
            f"[green]Browser headless:[/green] "
            f"{'on' if settings['headless'] else 'off'}"
        )
        return

    if action in {"timeout", "time-out"}:
        if not rest:
            console.print(
                f"Browser timeout: [cyan]{get_browser()['timeout_ms']}ms[/cyan]"
            )
            return
        try:
            settings = set_browser(timeout_ms=rest[0])
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        _reset_browser_driver()
        console.print(f"[green]Browser timeout:[/green] {settings['timeout_ms']}ms")
        return

    console.print(
        "[yellow]Usage: /config browser "
        "[show|enable on|off|headless on|off|timeout <ms>][/yellow]"
    )


def _config_shell(action_parts: list[str], console: Console) -> None:
    action = action_parts[0].lower() if action_parts else "show"
    rest = action_parts[1:]

    if action in {"show", "list", "ls", "status"}:
        settings = get_shell_config()
        console.print(
            f"Persistent shell: "
            f"[cyan]{'on' if settings['persistent'] else 'off'}[/cyan]\n"
            f"Command timeout: [cyan]{settings['timeout']}s[/cyan]\n"
            f"Background job cap: [cyan]{settings['max_jobs']}[/cyan]"
        )
        return

    if action in {"persistent", "enable", "enabled", "on", "off"}:
        token = (
            action
            if action in {"on", "off"}
            else (rest[0].lower() if rest else "")
        )
        if token not in {"on", "off"}:
            console.print("[yellow]Usage: /config shell persistent <on|off>[/yellow]")
            return
        settings = set_shell_config(persistent=token == "on")
        console.print(
            f"[green]Persistent shell "
            f"{'on' if settings['persistent'] else 'off'}.[/green]"
        )
        return

    if action in {"timeout", "time-out"}:
        if not rest:
            console.print(
                f"Shell command timeout: [cyan]{get_shell_config()['timeout']}s[/cyan]"
            )
            return
        try:
            settings = set_shell_config(timeout=rest[0])
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(f"[green]Shell timeout:[/green] {settings['timeout']}s")
        return

    if action in {"max-jobs", "jobs", "max"}:
        if not rest:
            console.print(
                "Background job cap: "
                f"[cyan]{get_shell_config()['max_jobs']}[/cyan]"
            )
            return
        try:
            settings = set_shell_config(max_jobs=rest[0])
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(f"[green]Background job cap:[/green] {settings['max_jobs']}")
        return

    console.print(
        "[yellow]Usage: /config shell "
        "[show|persistent on|off|timeout <s>|max-jobs <n>][/yellow]"
    )


def _config_sandbox(action_parts: list[str], console: Console) -> None:
    action = action_parts[0].lower() if action_parts else "show"
    rest = action_parts[1:]

    if action in {"show", "list", "ls", "status"}:
        settings = get_sandbox()
        writable = ", ".join(settings["extra_writable"]) or "(none)"
        console.print(
            f"Sandbox: [cyan]{'on' if settings['enabled'] else 'off'}[/cyan]\n"
            f"Network: [cyan]{'on' if settings['network'] else 'off'}[/cyan]\n"
            f"Extra writable paths: [cyan]{escape(writable)}[/cyan]"
        )
        return

    if action in {"enable", "enabled"}:
        token = rest[0].lower() if rest else ""
        if token not in {"on", "off"}:
            console.print("[yellow]Usage: /config sandbox enable <on|off>[/yellow]")
            return
        settings = set_sandbox(enabled=token == "on")
        console.print(
            f"[green]Sandbox {'on' if settings['enabled'] else 'off'}.[/green]"
        )
        return

    if action in {"network", "net"}:
        token = rest[0].lower() if rest else ""
        if token not in {"on", "off"}:
            console.print("[yellow]Usage: /config sandbox network <on|off>[/yellow]")
            return
        settings = set_sandbox(network=token == "on")
        console.print(
            f"[green]Sandbox network "
            f"{'on' if settings['network'] else 'off'}.[/green]"
        )
        return

    if action in {"add-path", "add"}:
        if not rest:
            console.print("[yellow]Usage: /config sandbox add-path <path>[/yellow]")
            return
        paths = list(get_sandbox()["extra_writable"])
        if rest[0] not in paths:
            paths.append(rest[0])
        try:
            set_sandbox(extra_writable=paths)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(f"[green]Writable in sandbox:[/green] {rest[0]}")
        return

    if action in {"remove-path", "remove"}:
        if not rest:
            console.print("[yellow]Usage: /config sandbox remove-path <path>[/yellow]")
            return
        current = get_sandbox()["extra_writable"]
        paths = [item for item in current if item != rest[0]]
        if len(paths) == len(current):
            console.print(f"[dim]No such writable path: {rest[0]}[/dim]")
            return
        set_sandbox(extra_writable=paths)
        console.print(f"[green]Removed writable path:[/green] {rest[0]}")
        return

    if action in {"clear-paths", "clear"}:
        set_sandbox(extra_writable=[])
        console.print("[green]Sandbox extra writable paths cleared.[/green]")
        return

    console.print(
        "[yellow]Usage: /config sandbox "
        "[show|enable on|off|network on|off|add-path <path>|"
        "remove-path <path>|clear-paths][/yellow]"
    )


def _remote_summary(settings: dict[str, Any]) -> str:
    host = str(settings.get("host") or "")
    if host:
        user = str(settings.get("user") or "")
        address = f"{user}@{host}" if user else host
        port = int(settings.get("port") or 22)
        if port != 22:
            address += f":{port}"
    else:
        address = "(none)"
    return address


def _config_remote(action_parts: list[str], console: Console) -> None:
    action = action_parts[0].lower() if action_parts else "show"
    rest = action_parts[1:]

    if action in {"show", "list", "ls", "status"}:
        settings = get_remote()
        console.print(
            f"Remote: [cyan]{'on' if settings['enabled'] else 'off'}[/cyan]\n"
            f"Host: [cyan]{escape(_remote_summary(settings))}[/cyan]\n"
            f"Identity: [cyan]{escape(settings['identity'] or '(default)')}[/cyan]\n"
            f"Workspace: [cyan]{escape(settings['workspace'] or '(remote default)')}[/cyan]\n"
            f"Devcontainer: [cyan]{escape(settings['devcontainer'])}[/cyan]"
        )
        return

    if action in {"enable", "enabled"}:
        token = rest[0].lower() if rest else ""
        if token not in {"on", "off"}:
            console.print("[yellow]Usage: /config remote enable <on|off>[/yellow]")
            return
        try:
            settings = set_remote(enabled=token == "on")
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(f"[green]Remote {'on' if settings['enabled'] else 'off'}.[/green]")
        return

    fields = {
        "host": "host",
        "user": "user",
        "port": "port",
        "identity": "identity",
        "workspace": "workspace",
        "devcontainer": "devcontainer",
    }
    field = fields.get(action)
    if field is not None:
        if not rest:
            current = get_remote()[field]
            shown = str(current) if current not in ("", None) else "(none)"
            console.print(f"Remote {field}: [cyan]{escape(shown)}[/cyan]")
            return
        try:
            updates: dict[str, Any] = {field: rest[0]}
            settings = set_remote(**updates)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        updated = settings[field]
        shown = str(updated) if updated not in ("", None) else "(none)"
        console.print(f"[green]Remote {field}:[/green] [cyan]{escape(shown)}[/cyan]")
        return

    console.print(
        "[yellow]Usage: /config remote [show|enable on|off|host <host>|"
        "user <user>|port <n>|identity <path>|workspace <path>|"
        "devcontainer <auto|off|name>][/yellow]"
    )


def _config_acp(action_parts: list[str], console: Console) -> None:
    action = action_parts[0].lower() if action_parts else "show"
    rest = action_parts[1:]
    if action in {"show", "status"}:
        settings = get_acp()
        console.print(
            f"ACP permission timeout: [cyan]{settings['permission_timeout']}s[/cyan]"
        )
        return
    if action in {"timeout", "permission-timeout"}:
        if not rest:
            console.print("[yellow]Usage: /config acp timeout <seconds>[/yellow]")
            return
        try:
            settings = set_acp(permission_timeout=rest[0])
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(
            f"[green]ACP permission timeout: "
            f"{settings['permission_timeout']}s[/green]"
        )
        return
    console.print("[yellow]Usage: /config acp [show|timeout <s>][/yellow]")


def _config_commands(  # noqa: C901 - small parser, mirrors the other config sections
    action_parts: list[str], session: Session, console: Console
) -> None:
    action = action_parts[0].lower() if action_parts else "list"
    rest = action_parts[1:]

    if action in {"list", "ls", "show"}:
        rules = get_command_rules()
        session.command_rules = rules
        if not rules["allow"] and not rules["deny"]:
            console.print("[dim]No command rules set.[/dim]")
            return
        table = Table(title="Command rules (run_bash)", show_header=True)
        table.add_column("Kind", style="cyan")
        table.add_column("Pattern")
        for kind in ("deny", "allow"):
            for pattern in rules[kind]:
                table.add_row(kind, pattern)
        console.print(table)
        return

    if action in {"allow", "deny"}:
        if not rest:
            console.print(
                f"[yellow]Usage: /config commands {action} <regex>[/yellow]"
            )
            return
        pattern = " ".join(rest)
        try:
            rules = add_command_rule(action, pattern)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        session.command_rules = rules
        console.print(f"[green]Added {action} rule:[/green] {pattern}")
        return

    if action in {"remove", "rm", "delete"}:
        if len(rest) < 2:
            console.print(
                "[yellow]Usage: /config commands remove <allow|deny> <regex>[/yellow]"
            )
            return
        kind, pattern = rest[0].lower(), " ".join(rest[1:])
        try:
            existed = remove_command_rule(kind, pattern)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        session.command_rules = get_command_rules()
        if existed:
            console.print(f"[green]Removed {kind} rule:[/green] {pattern}")
        else:
            console.print(f"[dim]No such {kind} rule: {pattern}[/dim]")
        return

    if action in {"clear", "reset"}:
        session.command_rules = clear_command_rules()
        console.print("[green]Cleared all command rules.[/green]")
        return

    console.print(
        "[yellow]Usage: /config commands [list|allow <regex>|deny <regex>|"
        "remove <allow|deny> <regex>|clear][/yellow]"
    )


def _config_sampling(action_parts: list[str], console: Console) -> None:
    action = action_parts[0].lower() if action_parts else "show"
    rest = action_parts[1:]

    if action in {"show", "list", "ls"}:
        sampling = get_sampling()
        if not sampling:
            console.print("[dim]Sampling: provider defaults (nothing set).[/dim]")
            return
        console.print(
            "Sampling: "
            + ", ".join(f"[cyan]{key}[/cyan]={value}" for key, value in sampling.items())
        )
        return

    if action in {"set", "update"}:
        if not rest:
            console.print(
                "[yellow]Usage: /config sampling set temperature=0.2 top_p=0.9 "
                "max_tokens=4096 reasoning_effort=medium[/yellow]"
            )
            return
        updates: dict[str, str] = {}
        for item in rest:
            if "=" not in item:
                console.print(f"[red]Expected key=value, got '{item}'.[/red]")
                return
            key, value = item.split("=", 1)
            updates[key.strip()] = value.strip()
        try:
            effective = set_sampling(updates)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(
            "[green]Sampling set:[/green] "
            + ", ".join(f"{key}={value}" for key, value in effective.items())
        )
        return

    if action in {"reset", "clear"}:
        reset_sampling()
        console.print("[green]Sampling reset to provider defaults.[/green]")
        return

    console.print(
        "[yellow]Usage: /config sampling [show|set key=value ...|reset][/yellow]"
    )


def _config_web(action_parts: list[str], console: Console) -> None:
    action = action_parts[0].lower() if action_parts else "show"
    rest = action_parts[1:]

    if action in {"show", "list", "ls", "status"}:
        settings = get_web()
        console.print(
            f"Web max chars: [cyan]{settings['max_chars']}[/cyan]\n"
            f"Web timeout: [cyan]{settings['timeout']:g}s[/cyan]\n"
            f"Allow local addresses: "
            f"[cyan]{'on' if settings['allow_local'] else 'off'}[/cyan]\n"
            f"Search provider: [cyan]{settings['search_provider']}[/cyan]"
        )
        return

    if action in {"max-chars", "max_chars", "chars"}:
        if not rest:
            console.print("[yellow]Usage: /config web max-chars <n>[/yellow]")
            return
        try:
            settings = set_web(max_chars=rest[0])
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(
            f"[green]Web max chars:[/green] {settings['max_chars']}"
        )
        return

    if action in {"timeout", "time-out"}:
        if not rest:
            console.print("[yellow]Usage: /config web timeout <seconds>[/yellow]")
            return
        try:
            settings = set_web(timeout=rest[0])
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(f"[green]Web timeout:[/green] {settings['timeout']:g}s")
        return

    if action in {"allow-local", "allow_local", "local"}:
        if not rest:
            console.print("[yellow]Usage: /config web allow-local <on|off>[/yellow]")
            return
        token = rest[0].strip().lower()
        if token not in {"on", "off", "true", "false", "yes", "no"}:
            console.print("[red]Expected 'on' or 'off'.[/red]")
            return
        enabled = token in {"on", "true", "yes"}
        settings = set_web(allow_local=enabled)
        console.print(
            f"[green]Allow local addresses:[/green] "
            f"{'on' if settings['allow_local'] else 'off'}"
        )
        return

    console.print(
        "[yellow]Usage: /config web [show|max-chars <n>|timeout <s>|"
        "allow-local <on|off>][/yellow]"
    )


_NETWORK_CLEAR_TOKENS = {"clear", "none", "off", "-", "default"}


def _config_network(action_parts: list[str], console: Console) -> None:
    action = action_parts[0].lower() if action_parts else "show"
    rest = action_parts[1:]

    if action in {"show", "list", "ls", "status"}:
        settings = get_network()
        console.print(
            f"Proxy: [cyan]{settings['proxy'] or 'none'}[/cyan]\n"
            f"CA bundle: [cyan]{settings['ca_bundle'] or 'system default'}[/cyan]\n"
            f"Offline mode: [cyan]{'on' if settings['offline'] else 'off'}[/cyan]"
        )
        return

    if action in {"proxy", "http-proxy"}:
        if not rest:
            console.print("[yellow]Usage: /config network proxy <url|clear>[/yellow]")
            return
        value = "" if rest[0].strip().lower() in _NETWORK_CLEAR_TOKENS else rest[0]
        try:
            settings = set_network(proxy=value)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(f"[green]Proxy:[/green] {settings['proxy'] or 'none'}")
        return

    if action in {"ca", "ca-bundle", "ca_bundle"}:
        if not rest:
            console.print(
                "[yellow]Usage: /config network ca <path|clear>[/yellow]"
            )
            return
        value = "" if rest[0].strip().lower() in _NETWORK_CLEAR_TOKENS else rest[0]
        try:
            settings = set_network(ca_bundle=value)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(
            f"[green]CA bundle:[/green] "
            f"{settings['ca_bundle'] or 'system default'}"
        )
        return

    if action in {"offline", "air-gap", "airgap"}:
        if not rest:
            current = "on" if get_network()["offline"] else "off"
            console.print(f"Offline mode: [cyan]{current}[/cyan]")
            return
        token = rest[0].strip().lower()
        if token in {"on", "true", "yes"}:
            set_network(offline=True)
            console.print(
                "[yellow]Offline mode on:[/yellow] cloud requests and web tools "
                "are blocked; local providers still work."
            )
        elif token in {"off", "false", "no"}:
            set_network(offline=False)
            console.print("[green]Offline mode off.[/green]")
        else:
            console.print("[red]Expected 'on' or 'off'.[/red]")
        return

    console.print(
        "[yellow]Usage: /config network [show|proxy <url|clear>|ca <path|clear>|"
        "offline <on|off>][/yellow]"
    )


def _config_vision(action_parts: list[str], console: Console) -> None:
    action = action_parts[0].lower() if action_parts else "show"
    rest = action_parts[1:]

    if action in {"show", "list", "ls", "status"}:
        settings = get_vision()
        console.print(
            f"Vision max image bytes: [cyan]{settings['max_image_bytes']:,}[/cyan]\n"
            f"Vision max images per turn: "
            f"[cyan]{settings['max_images_per_turn']}[/cyan]"
        )
        return

    if action in {"max-bytes", "max_bytes", "bytes"}:
        if not rest:
            console.print("[yellow]Usage: /config vision max-bytes <n>[/yellow]")
            return
        try:
            settings = set_vision(max_image_bytes=rest[0])
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(
            f"[green]Vision max image bytes:[/green] "
            f"{settings['max_image_bytes']:,}"
        )
        return

    if action in {"max-images", "max_images", "images"}:
        if not rest:
            console.print("[yellow]Usage: /config vision max-images <n>[/yellow]")
            return
        try:
            settings = set_vision(max_images_per_turn=rest[0])
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(
            f"[green]Vision max images per turn:[/green] "
            f"{settings['max_images_per_turn']}"
        )
        return

    console.print(
        "[yellow]Usage: /config vision [show|max-bytes <n>|max-images <n>][/yellow]"
    )


_UI_USAGE = (
    "/config ui [show|color <auto|always|never>|"
    "output <normal|compact|verbose>|ascii <on|off>|"
    "theme <default|ocean|magenta|mono>]"
)


def _config_ui(action_parts: list[str], console: Console) -> None:
    action = action_parts[0].lower() if action_parts else "show"
    rest = action_parts[1:]

    if action in {"show", "list", "status"}:
        current = get_ui()
        console.print(
            "UI: "
            f"theme=[cyan]{current['theme']}[/cyan], "
            f"color=[cyan]{current['color']}[/cyan], "
            f"output=[cyan]{current['output_mode']}[/cyan], "
            f"ascii=[cyan]{'on' if current['ascii'] else 'off'}[/cyan]"
        )
        return

    if action in {"color", "output", "ascii", "theme"}:
        if not rest:
            console.print(f"[yellow]Usage: /config ui {action} <value>[/yellow]")
            return
        value = rest[0]
        try:
            if action == "color":
                set_ui(color=value)
            elif action == "output":
                set_ui(output_mode=value)
            elif action == "theme":
                set_ui(theme=value)
            else:
                token = value.strip().lower()
                if token in {"on", "true", "yes", "enable", "enabled"}:
                    set_ui(ascii=True)
                elif token in {"off", "false", "no", "disable", "disabled"}:
                    set_ui(ascii=False)
                else:
                    console.print("[yellow]Usage: /config ui ascii <on|off>[/yellow]")
                    return
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        current = get_ui()
        console.print(
            "[green]UI updated:[/green] "
            f"theme=[cyan]{current['theme']}[/cyan], "
            f"color=[cyan]{current['color']}[/cyan], "
            f"output=[cyan]{current['output_mode']}[/cyan], "
            f"ascii=[cyan]{'on' if current['ascii'] else 'off'}[/cyan]"
        )
        console.print(
            "[dim]Restart the session for color, theme, ascii, and output "
            "mode changes to apply.[/dim]"
        )
        return

    console.print(f"[yellow]Usage: {_UI_USAGE}[/yellow]")


def _config_style(action_parts: list[str], session: Session, console: Console) -> None:
    from kiwimatecoder.config import OUTPUT_STYLES

    action = action_parts[0].lower() if action_parts else "show"
    rest = action_parts[1:]

    if action in {"set", "use"}:
        if not rest:
            console.print(
                "[yellow]Usage: /config style set <"
                + "|".join(OUTPUT_STYLES)
                + ">[/yellow]"
            )
            return
        try:
            style = set_output_style(rest[0])
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        session.output_style = style
        console.print(f"[green]Output style set to {style}.[/green]")
        return

    console.print(
        "Output style: "
        + f"[cyan]{session.output_style}[/cyan] "
        + f"(available: {', '.join(OUTPUT_STYLES)})"
    )


def _config_prompt(action_parts: list[str], session: Session, console: Console) -> None:
    action = action_parts[0].lower() if action_parts else "show"
    rest = action_parts[1:]

    if action in {"set", "update"}:
        text = " ".join(rest).strip()
        if not text:
            console.print("[yellow]Usage: /config prompt set <text>[/yellow]")
            return
        set_system_prompt(text)
        session.custom_system_prompt = text
        console.print("[green]Custom system prompt saved.[/green]")
        return

    if action in {"clear", "reset"}:
        set_system_prompt(None)
        session.custom_system_prompt = None
        console.print("[green]Custom system prompt cleared.[/green]")
        return

    if session.custom_system_prompt:
        console.print("[bold]Custom system prompt:[/bold]")
        console.print(session.custom_system_prompt)
    else:
        console.print("[dim]No custom system prompt set.[/dim]")


def _config_cache(action_parts: list[str], console: Console) -> None:
    action = action_parts[0].lower() if action_parts else "show"

    if action in {"on", "true", "enable", "enabled"}:
        set_prompt_cache(True)
        console.print(
            "[green]Prompt caching on:[/green] native Anthropic requests will "
            "mark the system prompt and tool definitions as cache breakpoints."
        )
        return

    if action in {"off", "false", "disable", "disabled"}:
        set_prompt_cache(False)
        console.print("[green]Prompt caching off.[/green]")
        return

    if action in {"show", "status", "list"}:
        state = "on" if get_prompt_cache() else "off"
        console.print(f"Prompt caching: [cyan]{state}[/cyan]")
        console.print(
            "[dim]Applies to native Anthropic providers; OpenAI-compatible "
            "providers cache automatically.[/dim]"
        )
        return

    console.print("[yellow]Usage: /config cache [on|off][/yellow]")


def _config_profile(
    action_parts: list[str], session: Session, console: Console
) -> None:
    action = action_parts[0].lower() if action_parts else "list"
    rest = action_parts[1:]

    if action in {"list", "ls"}:
        profiles = get_profiles()
        if not profiles:
            console.print(
                "[dim]No profiles saved. Save one with "
                "/config profile save <name>.[/dim]"
            )
            return
        table = Table(title="Profiles", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Provider")
        table.add_column("Model")
        table.add_column("Mode")
        for name in sorted(profiles):
            values = profiles[name]
            table.add_row(
                name,
                str(values.get("provider") or ""),
                str(values.get("model") or "(provider default)"),
                str(values.get("mode") or ""),
            )
        console.print(table)
        return

    if action == "show":
        if not rest:
            console.print("[yellow]Usage: /config profile show <name>[/yellow]")
            return
        profile = get_profile(rest[0])
        if profile is None:
            console.print(f"[red]Unknown profile '{rest[0]}'.[/red]")
            return
        console.print(f"[bold]{rest[0]}[/bold]")
        console.print_json(data=profile)
        return

    if action in {"save", "add", "capture"}:
        if not rest:
            console.print("[yellow]Usage: /config profile save <name>[/yellow]")
            return
        try:
            profile = save_profile(rest[0])
        except (ValueError, KeyError) as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(
            f"[green]Saved profile [cyan]{rest[0]}[/cyan][/green] "
            f"({len(profile)} setting(s))."
        )
        return

    if action in {"use", "apply", "load"}:
        if not rest:
            console.print("[yellow]Usage: /config profile use <name>[/yellow]")
            return
        name = rest[0]
        try:
            profile = apply_profile(name)
        except (ValueError, KeyError) as exc:
            console.print(f"[red]{exc}[/red]")
            return
        apply_session_profile(session, profile)
        console.print(
            f"[green]Applied profile [cyan]{name}[/cyan][/green] — "
            f"provider: [cyan]{session.provider_id}[/cyan], "
            f"model: [cyan]{session.model}[/cyan], "
            f"mode: [cyan]{session.mode.value}[/cyan]."
        )
        return

    if action in {"remove", "rm", "delete"}:
        if not rest:
            console.print("[yellow]Usage: /config profile remove <name>[/yellow]")
            return
        if remove_profile(rest[0]):
            console.print(f"[green]Removed profile {rest[0]}.[/green]")
        else:
            console.print(f"[dim]No profile named {rest[0]}.[/dim]")
        return

    console.print(
        "[yellow]Usage: /config profile "
        "[list|show <name>|save <name>|use <name>|remove <name>][/yellow]"
    )


def _doctor(arg: str, session: Session, console: Console) -> str:
    from kiwimatecoder import diagnostics

    diagnostics.render(diagnostics.run_checks(session), console)
    return CommandResult.CONTINUE


def _config(arg: str, session: Session, console: Console,
            selector: CommandSelector | None = None,
            prompt_input: Callable[[str], str] | None = None) -> str:
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
        _config_keys(rest, console, selector, prompt_input)
    elif section == "use":
        _config_providers(["use", *rest], session, console)
    elif section == "model":
        _config_model(rest, session, console)
    elif section == "models":
        _config_models(rest, session, console)
    elif section == "mode":
        _config_mode(rest, session, console)
    elif section in {"permissions", "perm"}:
        _config_permissions(rest, session, console)
    elif section in {"commands", "command-rules", "rules"}:
        _config_commands(rest, session, console)
    elif section in {"trust", "trusted", "trusted-workspace"}:
        _config_trust(rest, session, console)
    elif section == "verify":
        _config_verify(rest, session, console)
    elif section == "budget":
        _config_budget(rest, session, console)
    elif section in {"subagents", "subagent"}:
        _config_subagents(rest, console)
    elif section == "browser":
        _config_browser(rest, console)
    elif section == "shell":
        _config_shell(rest, console)
    elif section == "sandbox":
        _config_sandbox(rest, console)
    elif section == "remote":
        _config_remote(rest, console)
    elif section == "acp":
        _config_acp(rest, console)
    elif section == "sampling":
        _config_sampling(rest, console)
    elif section == "web":
        _config_web(rest, console)
    elif section == "network":
        _config_network(rest, console)
    elif section == "vision":
        _config_vision(rest, console)
    elif section == "ui":
        _config_ui(rest, console)
    elif section == "style":
        _config_style(rest, session, console)
    elif section == "prompt":
        _config_prompt(rest, session, console)
    elif section in {"profile", "profiles"}:
        _config_profile(rest, session, console)
    elif section in {"cache", "prompt-cache"}:
        _config_cache(rest, console)
    else:
        console.print("[yellow]Unknown config command. Try /config help.[/yellow]")
    return CommandResult.CONTINUE


def _run_selector(selector: CommandSelector | None, prompt: SelectionPrompt) -> str | None:
    """Ask the interactive selector for one option, validating the reply."""
    if selector is None:
        return None
    selected = selector(prompt)
    if selected is None:
        return None
    if selected not in {option.value for option in prompt.options}:
        return None
    return selected


def _config_interact(
    session: Session,
    console: Console,
    selector: CommandSelector,
    prompt_input: Callable[[str], str] | None,
) -> str:
    """Staged interactive configuration: section menu, then deeper steps."""
    section_prompt = SelectionPrompt(
        title="Configure KiwiMateCoder",
        text=(
            "Pick a setting to view or change. Keys lets you set or remove an "
            "API key interactively."
        ),
        options=(
            CommandOption("show", "Show active providers, model, key, filter"),
            CommandOption("providers", "List providers (add/remove/use/edit)"),
            CommandOption("keys", "Set or remove an API key"),
            CommandOption("model", "Set or reset the default model"),
            CommandOption("models", "Manage model visibility / refresh catalog"),
            CommandOption("mode", "Set the default permission mode"),
            CommandOption("permissions", "Manage tools approved with 'always'"),
            CommandOption("commands", "Manage shell command allow/deny rules"),
            CommandOption("trust", "Allow reads outside the workspace root"),
            CommandOption("sampling", "Set temperature/top_p/max_tokens"),
            CommandOption("browser", "Optional Playwright browser automation"),
            CommandOption("shell", "Persistent shell and background job settings"),
            CommandOption("remote", "SSH/devcontainer command execution"),
            CommandOption("acp", "ACP editor-integration permission timeout"),
            CommandOption("web", "Web fetch/search limits and local access"),
            CommandOption("network", "Proxy, custom CA bundle, and offline mode"),
            CommandOption("vision", "Image attachment size and count limits"),
            CommandOption("style", "Show or set the output style"),
            CommandOption("ui", "Theme, color, output mode, and ASCII mode"),
            CommandOption("prompt", "Show, set, or clear a custom system prompt"),
            CommandOption("profile", "Save or apply configuration profiles"),
            CommandOption("cache", "Toggle Anthropic prompt caching"),
            CommandOption("help", "Show all /config commands"),
        ),
    )
    section = _run_selector(selector, section_prompt)
    if section is None:
        return CommandResult.CONTINUE

    if section == "show":
        _config_show(session, console)
        return CommandResult.CONTINUE
    if section == "keys":
        provider_prompt = SelectionPrompt(
            title="Choose a provider",
            text="Which provider's API key do you want to change?",
            options=tuple(
                CommandOption(provider.id, provider.name)
                for provider in list_provider_configs()
            ),
            selected=session.provider_id,
        )
        provider_id = _run_selector(selector, provider_prompt)
        if provider_id is None:
            return CommandResult.CONTINUE
        _config_key_enter(provider_id, console, selector, prompt_input)
        return CommandResult.CONTINUE
    if section in {"providers", "provider"}:
        _config_providers([], session, console)
        return CommandResult.CONTINUE
    if section in {
        "model",
        "models",
        "mode",
        "help",
        "permissions",
        "sampling",
        "browser",
        "shell",
        "remote",
        "acp",
        "web",
        "network",
        "vision",
        "style",
        "ui",
        "prompt",
        "profile",
        "cache",
        "commands",
        "trust",
    }:
        _config(section, session, console, selector, prompt_input)
        return CommandResult.CONTINUE
    return CommandResult.CONTINUE


def _cost(arg: str, session: Session, console: Console) -> str:
    from kiwimatecoder.pricing import estimate_cost, find_price

    total = session.prompt_tokens + session.completion_tokens
    is_local = session.provider.is_local
    price = find_price(session.model, session.provider_id, is_local=is_local)
    cost = estimate_cost(
        session.prompt_tokens,
        session.completion_tokens,
        session.model,
        session.provider_id,
        is_local=is_local,
    )
    table = Table(title="Session Token & Cost Usage", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Model", f"{session.provider_id}:{session.model}")
    table.add_row("Prompt Tokens", f"{session.prompt_tokens:,}")
    table.add_row("Completion Tokens", f"{session.completion_tokens:,}")
    table.add_row("Total Tokens", f"[bold]{total:,}[/bold]")
    if is_local:
        table.add_row("Estimated Cost (USD)", "~$0.0000 [dim](local provider)[/dim]")
    elif price is None:
        table.add_row("Estimated Cost (USD)", "[dim]unknown model — no price entry[/dim]")
    else:
        cost_text = f"~${cost:.4f}" if cost is not None else "—"
        rates = f"${price.input_per_mtok:g} in / ${price.output_per_mtok:g} out per Mtok"
        table.add_row("Estimated Cost (USD)", f"{cost_text} [dim]({rates})[/dim]")
    table.add_row("Conversation History Messages", f"{len(session.messages)}")
    window = max(1, session.context_window)
    used = session.estimated_history_tokens
    table.add_row(
        "Context Gauge",
        f"~{used * 100 // window}% of {window:,} tokens",
    )
    table.add_row("Estimated Context Tokens", f"~{used:,}")
    console.print(table)
    return CommandResult.CONTINUE


def _save(arg: str, session: Session, console: Console) -> str:
    try:
        dest = save_session(session, arg.strip() or None)
        console.print(
            f"[green]Session saved to [bold]{dest.name}[/bold] "
            f"({len(session.messages)} messages, {session.total_tokens:,} tokens).[/green]"
        )
    except Exception as exc:
        console.print(f"[red]Failed to save session: {exc}[/red]")
    return CommandResult.CONTINUE


def _load(arg: str, session: Session, console: Console) -> str:
    if not arg.strip():
        return _sessions("", session, console)
    try:
        loaded = load_session(arg.strip(), workspace_root=session.workspace_root)
        session.messages = loaded.messages
        session.provider_id = loaded.provider_id
        session.model = loaded.model
        session.mode = loaded.mode
        session.prompt_tokens = loaded.prompt_tokens
        session.completion_tokens = loaded.completion_tokens
        session.touched_files = loaded.touched_files
        session.context_files = loaded.context_files
        session.active_provider_ids = loaded.active_provider_ids
        session.models = loaded.models
        session.always_allowed = set(loaded.always_allowed)
        session.always_allowed.update(get_always_allowed_tools())
        session.output_style = loaded.output_style
        session.custom_system_prompt = loaded.custom_system_prompt
        session.command_rules = loaded.command_rules or get_command_rules()
        session.dry_run = loaded.dry_run
        session.todos = list(loaded.todos)
        session.trusted_workspace = loaded.trusted_workspace
        session.verify_command = loaded.verify_command
        session.compact_at_tokens = loaded.compact_at_tokens
        session.context_window = loaded.context_window
        console.print(
            f"[green]Loaded session [bold]{arg.strip()}[/bold]: "
            f"{len(session.messages)} messages, provider={session.provider_id}:{session.model}[/green]"
        )
    except Exception as exc:
        console.print(f"[red]Failed to load session: {exc}[/red]")
    return CommandResult.CONTINUE


def _sessions(arg: str, session: Session, console: Console) -> str:
    saved = list_saved_sessions()
    if not saved:
        console.print("[dim]No saved sessions found in ~/.kiwimatecoder/sessions/[/dim]")
        return CommandResult.CONTINUE
    table = Table(title="Saved Sessions", show_header=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Provider / Model")
    table.add_column("Messages", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Saved At", style="dim")
    for s in saved:
        prov_model = f"{s['provider']}:{s['model']}" if s["model"] else s["provider"]
        table.add_row(
            s["name"],
            prov_model,
            str(s["messages"]),
            f"{s['tokens']:,}",
            str(s["saved_at"]).split(".")[0].replace("T", " "),
        )
    console.print(table)
    console.print("[dim]Use /load <name> to resume a session.[/dim]")
    return CommandResult.CONTINUE


def _fork(arg: str, session: Session, console: Console) -> str:
    try:
        path = fork_session(session, arg.strip() or None)
    except Exception as exc:
        console.print(f"[red]Failed to fork session: {exc}[/red]")
        return CommandResult.CONTINUE
    console.print(
        f"[green]Forked session to [bold]{path.name}[/bold][/green] "
        f"({len(session.messages)} messages)."
    )
    return CommandResult.CONTINUE


def _export(arg: str, session: Session, console: Console) -> str:
    if arg.strip():
        candidate = Path(arg.strip()).expanduser()
        destination = (
            candidate
            if candidate.is_absolute()
            else session.workspace_root / candidate
        )
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = session.workspace_root / f"session_{stamp}.md"
    try:
        destination.write_text(
            export_session_markdown(session), encoding="utf-8"
        )
    except OSError as exc:
        console.print(f"[red]Failed to export session: {exc}[/red]")
        return CommandResult.CONTINUE
    console.print(f"[green]Exported session to [bold]{destination}[/bold].[/green]")
    return CommandResult.CONTINUE


def _undo(arg: str, session: Session, console: Console) -> str:
    if not session.checkpoints:
        console.print(
            "[dim]No checkpoints yet. File edits are checkpointed as they run.[/dim]"
        )
        return CommandResult.CONTINUE
    count = 1
    token = arg.strip()
    if token:
        try:
            count = int(token)
        except ValueError:
            console.print("[yellow]Usage: /undo [count][/yellow]")
            return CommandResult.CONTINUE
    restored = session.undo_checkpoints(count)
    if not restored:
        console.print("[dim]Nothing to undo.[/dim]")
        return CommandResult.CONTINUE
    for item in restored:
        console.print(f"[green]Undid[/green] {item.label} [dim](#{item.id})[/dim]")
    paths = sorted({path for item in restored for path in item.paths})
    if paths:
        console.print("[dim]Restored:[/dim] " + ", ".join(paths))
    return CommandResult.CONTINUE


def _checkpoints(arg: str, session: Session, console: Console) -> str:
    if not session.checkpoints:
        console.print("[dim]No checkpoints captured this session.[/dim]")
        return CommandResult.CONTINUE
    table = Table(title="Checkpoints", show_header=True)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Action")
    table.add_column("Files")
    table.add_column("Captured", style="dim")
    for item in session.checkpoints:
        table.add_row(
            str(item.id),
            item.label,
            ", ".join(item.paths) or "-",
            item.created_at.split(".")[0].replace("T", " "),
        )
    console.print(table)
    console.print("[dim]Use /undo [count] to restore.[/dim]")
    return CommandResult.CONTINUE


def _todos(arg: str, session: Session, console: Console) -> str:
    if not session.todos:
        console.print(
            "[dim]No task list yet. The agent creates one for multi-step work.[/dim]"
        )
        return CommandResult.CONTINUE
    labels = {
        "pending": "[dim]pending[/dim]",
        "in_progress": "[yellow]in progress[/yellow]",
        "completed": "[green]done[/green]",
    }
    table = Table(title="Task list", show_header=True)
    table.add_column("Status")
    table.add_column("Task")
    for todo in session.todos:
        status = str(todo.get("status") or "pending")
        table.add_row(labels.get(status, status), str(todo.get("content") or ""))
    console.print(table)
    return CommandResult.CONTINUE


_MEMORY_USAGE = "/memory [list|add <text>|add-user <text>|clear project|user]"


def _memory(arg: str, session: Session, console: Console) -> str:
    parts = arg.strip().split(maxsplit=1)
    action = parts[0].lower() if parts else "list"
    rest = parts[1].strip() if len(parts) > 1 else ""

    if action in {"list", "show", "ls"}:
        settings = get_memory()
        state = "on" if settings["enabled"] else "off"
        console.print(
            f"Memory: [cyan]{state}[/cyan] "
            f"(prompt budget {settings['max_bytes']} bytes)"
        )
        for scope in memory_module.SCOPES:
            path = memory_module.memory_path(scope, session.workspace_root)
            text = memory_module.read_memory(
                scope, session.workspace_root, settings["max_bytes"]
            )
            console.print(f"\n[bold]{scope}[/bold] [dim]{path}[/dim]")
            if text:
                console.print(text, markup=False, highlight=False)
            else:
                console.print("[dim](empty)[/dim]")
        return CommandResult.CONTINUE

    if action in {"add", "remember"}:
        if not rest:
            console.print("[yellow]Usage: /memory add <text>[/yellow]")
            return CommandResult.CONTINUE
        return _memory_append("project", rest, session, console)

    if action in {"add-user", "user-add", "add_user"}:
        if not rest:
            console.print("[yellow]Usage: /memory add-user <text>[/yellow]")
            return CommandResult.CONTINUE
        return _memory_append("user", rest, session, console)

    if action in {"clear", "remove", "reset"}:
        scope = rest.lower() or "project"
        if scope not in memory_module.SCOPES:
            console.print(
                f"[red]Unknown memory scope '{scope}'. Choose: project, user.[/red]"
            )
            return CommandResult.CONTINUE
        if memory_module.clear_memory(scope, session.workspace_root):
            console.print(f"[green]Cleared {scope} memory.[/green]")
        else:
            console.print(f"[dim]No {scope} memory to clear.[/dim]")
        return CommandResult.CONTINUE

    console.print(f"[yellow]Usage: {_MEMORY_USAGE}[/yellow]")
    return CommandResult.CONTINUE


def _memory_append(
    scope: str, text: str, session: Session, console: Console
) -> str:
    try:
        path = memory_module.append_memory(scope, session.workspace_root, text)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return CommandResult.CONTINUE
    console.print(f"[green]Saved to {scope} memory:[/green] {path}")
    return CommandResult.CONTINUE


_INDEX_USAGE = "/index [status|build|clear]"


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _index(arg: str, session: Session, console: Console) -> str:
    from kiwimatecoder import config as config_module
    from kiwimatecoder.index import builder
    from kiwimatecoder.index.store import clear_store, index_status

    action = arg.strip().lower() or "status"
    try:
        if action in {"status", "show", "list", "ls"}:
            status = index_status(session.workspace_root)
            state = "on" if status.enabled else "off"
            console.print(
                f"Codebase index: [cyan]{state}[/cyan]\n"
                f"Files indexed: [cyan]{status.files}[/cyan] "
                f"([yellow]{status.stale}[/yellow] stale)\n"
                f"Terms: [cyan]{status.terms}[/cyan]\n"
                f"Store: [cyan]{_format_bytes(status.size_bytes)}[/cyan] "
                f"[dim]{status.path}[/dim]\n"
                f"Embeddings: [cyan]{'on' if status.embeddings else 'off'}[/cyan]"
            )
            return CommandResult.CONTINUE
        if action in {"build", "update", "refresh"}:
            if not config_module.get_index()["enabled"]:
                console.print(
                    "[yellow]Codebase indexing is disabled. Set "
                    '"index": {"enabled": true} in the config to enable it.[/yellow]'
                )
                return CommandResult.CONTINUE
            stats = builder.build_index(session.workspace_root)
            console.print(
                f"[green]Indexed {stats.files} file(s):[/green] "
                f"{stats.added} added, {stats.updated} updated, "
                f"{stats.removed} removed, {stats.unchanged} unchanged "
                f"([cyan]{stats.terms}[/cyan] terms)."
            )
            return CommandResult.CONTINUE
        if action in {"clear", "remove", "delete", "reset"}:
            if clear_store(session.workspace_root):
                console.print("[green]Codebase index cleared.[/green]")
            else:
                console.print("[dim]No codebase index to clear.[/dim]")
            return CommandResult.CONTINUE
    except OSError as exc:
        console.print(f"[red]Codebase index unavailable: {exc}[/red]")
        return CommandResult.CONTINUE
    console.print(f"[yellow]Usage: {_INDEX_USAGE}[/yellow]")
    return CommandResult.CONTINUE


def _compact(arg: str, session: Session, console: Console) -> str:
    target: int | None = None
    token = arg.strip()
    if token:
        try:
            target = int(token)
        except ValueError:
            console.print("[yellow]Usage: /compact [token_budget][/yellow]")
            return CommandResult.CONTINUE

    result = session.compact(target)
    window = max(1, session.context_window)
    if result["before"] <= result["after"] and result["after"] <= result["budget"]:
        console.print(
            f"[dim]History already within budget "
            f"(~{result['after']:,} / {result['budget']:,} tokens).[/dim]"
        )
    else:
        console.print(
            f"[green]Compacted history:[/green] ~{result['before']:,} → "
            f"~{result['after']:,} tokens (budget {result['budget']:,})."
        )
    console.print(
        f"[dim]Context gauge: ~{result['after'] * 100 // window}% of "
        f"{window:,} window.[/dim]"
    )
    return CommandResult.CONTINUE


def _jobs(arg: str, session: Session, console: Console) -> str:
    """List, inspect, cancel, tick, or start detached agent jobs."""
    from kiwimatecoder import jobs as jobs_module

    try:
        parts = shlex.split(arg)
    except ValueError as exc:
        console.print(f"[red]Could not parse command: {exc}[/red]")
        return CommandResult.CONTINUE

    action = parts[0].lower() if parts else "list"
    rest = parts[1:]

    if action in {"list", "ls"}:
        records = jobs_module.list_jobs(refresh=True)
        if not records:
            console.print("[dim]No background jobs.[/dim]")
            return CommandResult.CONTINUE
        table = Table(title="Background jobs", show_header=True)
        table.add_column("id", style="cyan", no_wrap=True)
        table.add_column("status")
        table.add_column("created", no_wrap=True)
        table.add_column("prompt", overflow="fold")
        for job in records:
            style = {
                "running": "yellow",
                "succeeded": "green",
                "failed": "red",
                "cancelled": "dim",
            }.get(job.status, "white")
            table.add_row(
                job.id,
                f"[{style}]{job.status}[/{style}]",
                job.created_at,
                escape(_shorten(job.prompt, 60)),
            )
        console.print(table)
        return CommandResult.CONTINUE

    if action == "show":
        if not rest:
            console.print("[yellow]Usage: /jobs show <id>[/yellow]")
            return CommandResult.CONTINUE
        record = jobs_module.refresh_job(rest[0]) or jobs_module.get_job(rest[0])
        if record is None:
            console.print(f"[red]Unknown job '{rest[0]}'.[/red]")
            return CommandResult.CONTINUE
        console.print(f"ID: [cyan]{record.id}[/cyan]")
        console.print(f"Status: [bold]{record.status}[/bold]")
        console.print(f"Prompt: {escape(record.prompt)}")
        console.print(f"Workspace: {escape(record.workspace)}")
        console.print(f"Mode: {record.mode}")
        if record.finished_at:
            console.print(f"Finished: {record.finished_at}")
        if record.exit_code is not None:
            console.print(f"Exit code: {record.exit_code}")
        if record.error:
            console.print(f"[red]Error: {escape(record.error)}[/red]")
        if record.result:
            console.print(Panel(escape(record.result), title="Result"))
        console.print(f"[dim]Output: {record.output_path}[/dim]")
        return CommandResult.CONTINUE

    if action == "cancel":
        if not rest:
            console.print("[yellow]Usage: /jobs cancel <id>[/yellow]")
            return CommandResult.CONTINUE
        record = jobs_module.cancel_job(rest[0])
        if record is None:
            console.print(f"[red]Unknown job '{rest[0]}'.[/red]")
            return CommandResult.CONTINUE
        console.print(
            f"[green]Job[/green] [cyan]{record.id}[/cyan] is now "
            f"[bold]{record.status}[/bold]."
        )
        return CommandResult.CONTINUE

    if action == "tick":
        started = jobs_module.run_due_jobs()
        if not started:
            console.print("[dim]No jobs are due.[/dim]")
            return CommandResult.CONTINUE
        for job in started:
            console.print(
                f"[green]Started[/green] [cyan]{job.id}[/cyan] "
                f"([dim]{escape(_shorten(job.prompt, 60))}[/dim])"
            )
        return CommandResult.CONTINUE

    if action == "run":
        prompt = " ".join(rest).strip()
        if not prompt:
            console.print("[yellow]Usage: /jobs run <prompt>[/yellow]")
            return CommandResult.CONTINUE
        try:
            record = jobs_module.start_job(
                prompt,
                workspace=session.workspace_root,
                provider=session.provider_id,
                model=session.model,
            )
        except (ValueError, KeyError, jobs_module.JobError) as exc:
            console.print(f"[red]{exc}[/red]")
            return CommandResult.CONTINUE
        console.print(
            f"[green]Started job[/green] [cyan]{record.id}[/cyan] "
            f"([dim]{escape(_shorten(record.prompt, 60))}[/dim]). "
            f"Use /jobs show {record.id} to track it."
        )
        return CommandResult.CONTINUE

    console.print(
        "[yellow]Usage: /jobs [list|show <id>|cancel <id>|tick|run <prompt>][/yellow]"
    )
    return CommandResult.CONTINUE


def _shorten(text: str, limit: int) -> str:
    cleaned = " ".join(str(text).split())
    return cleaned if len(cleaned) <= limit else f"{cleaned[: limit - 3]}..."


_COMMANDS: dict[str, Callable[[str, Session, Console], str]] = {
    "help": _help,
    "exit": _exit,
    "quit": _exit,
    "clear": _clear,
    "model": _model,
    "provider": _provider,
    "mode": _mode,
    "dry-run": _dry_run,
    "dryrun": _dry_run,
    "tools": _tools,
    "files": _files,
    "context": _context,
    "ctx": _context,
    "config": _config,
    "cost": _cost,
    "doctor": _doctor,
    "save": _save,
    "load": _load,
    "sessions": _sessions,
    "fork": _fork,
    "export": _export,
    "undo": _undo,
    "rewind": _undo,
    "checkpoints": _checkpoints,
    "todos": _todos,
    "memory": _memory,
    "compact": _compact,
    "index": _index,
    "templates": _templates,
    "mcp": _mcp,
    "lsp": _lsp,
    "jobs": _jobs,
}


def has_command(name: str) -> bool:
    """Whether ``name`` (with or without a leading slash) is registered."""
    return name.strip().lstrip("/").lower() in _COMMANDS


_BUILTIN_COMMANDS = frozenset(_COMMANDS)


def register_command(
    name: str,
    handler: Callable[[str, Session, Console], str],
    description: str = "",
) -> str:
    """Register a slash command dynamically and return its normalized name.

    Raises ``ValueError`` when the name is empty or already registered;
    built-in commands are protected so extensions cannot shadow core behavior.
    """
    key = name.strip().lstrip("/").lower()
    if not key:
        raise ValueError("Command name must be non-empty.")
    if key in _COMMANDS:
        raise ValueError(f"Command '/{key}' is already registered.")
    _COMMANDS[key] = handler
    _COMMAND_DESCRIPTIONS[key] = description or f"Custom command /{key}."
    return key


def unregister_command(name: str) -> bool:
    """Remove a dynamically registered command; built-ins are protected."""
    key = name.strip().lstrip("/").lower()
    if key not in _COMMANDS or key in _BUILTIN_COMMANDS:
        return False
    del _COMMANDS[key]
    _COMMAND_DESCRIPTIONS.pop(key, None)
    return True


_HELP_GROUPS = [
    (
        "Session",
        [
            ("/help", "Show this help."),
            ("/exit, /quit", "Leave the session."),
            ("/clear", "Clear the conversation history."),
            ("/cost", "Show token usage for this session."),
            ("/doctor", "Run environment, config, and provider diagnostics."),
            ("/files", "List files changed this session."),
            ("/tools", "List available tools."),
            ("/save [name]", "Save the active session state."),
            ("/load <name>", "Load a previously saved session."),
            ("/sessions", "List all saved sessions."),
            ("/fork [name]", "Save an independent copy of this session."),
            ("/export [path]", "Export the conversation as Markdown."),
            ("/undo [count]", "Restore files changed by recent tool actions."),
            ("/checkpoints", "List captured file checkpoints."),
            ("/todos", "Show the agent's task list."),
            (
                "/memory [list|add|add-user|clear]",
                "Show or edit persistent project and user memory.",
            ),
            (
                "/index [status|build|clear]",
                "Show, refresh, or delete the codebase index used by semantic "
                "search.",
            ),
            ("/compact [budget]", "Trim older history to fit a token budget."),
            ("/templates", "List custom prompt templates."),
            (
                "/jobs [list|show <id>|cancel <id>|tick|run <prompt>]",
                "List or manage detached background agent jobs (tick starts "
                "scheduled runs).",
            ),
        ],
    ),
    (
        "Model & provider",
        [
            (
                "/model [name|refresh|list|search <term>]",
                "Choose a model (the list is refreshed from the provider), "
                "set one by name, refresh the list, or search the full "
                "catalog by name. The choice is remembered for the next session.",
            ),
            (
                "/provider [id]",
                "Choose a failover roster (checklist), or replace it with one provider by id.",
            ),
            (
                "/mode [ask|auto-accept|plan]",
                "Choose or directly set the permission mode.",
            ),
            (
                "/dry-run [on|off|toggle]",
                "Preview mutating actions without running them.",
            ),
        ],
    ),
    (
        "Context",
        [
            (
                "/context [list|add|remove|clear]",
                "Manage pinned files included with each turn.",
            ),
        ],
    ),
    (
        "Configuration",
        [
            (
                "/config",
                "Show or change providers, API keys, model defaults, model "
                "filters, themes, output modes, and accessibility settings.",
            ),
            ("/config help", "List every /config command."),
            (
                "/mcp [list|reload]",
                "List configured MCP servers and their tools, or reconnect "
                "every server and re-register its tools.",
            ),
            (
                "/lsp [status|on|off|restart]",
                "Show or toggle language-server diagnostics and navigation "
                "(definitions, references).",
            ),
        ],
    ),
]

_COMMAND_DESCRIPTIONS = {
    "help": "Show available commands.",
    "exit": "Leave the session.",
    "quit": "Leave the session.",
    "clear": "Clear the conversation history.",
    "model": "Choose a model, set one by name, refresh the list, or search.",
    "provider": "Choose a failover roster (checklist), or replace it with one provider by id.",
    "mode": "Choose or directly set the permission mode.",
    "dry-run": "Preview mutating actions without running them.",
    "dryrun": "Alias for /dry-run.",
    "tools": "List available tools.",
    "files": "List files changed this session.",
    "context": "Manage pinned files included with each turn.",
    "ctx": "Alias for /context.",
    "config": (
        "Show or change providers, API keys, model defaults, and model filters."
    ),
    "cost": "Show token usage for this session.",
    "doctor": "Run environment, config, and provider diagnostics.",
    "save": "Save the current session to disk.",
    "load": "Load a saved session from disk.",
    "sessions": "List all saved sessions.",
    "fork": "Save an independent copy of this session.",
    "export": "Export the conversation as Markdown.",
    "undo": "Restore files changed by recent tool actions.",
    "rewind": "Alias for /undo.",
    "checkpoints": "List captured file checkpoints.",
    "todos": "Show the agent's task list.",
    "memory": "Show or edit persistent project and user memory.",
    "compact": "Trim older history to fit a token budget.",
    "index": "Show, refresh, or delete the codebase index used by semantic search.",
    "templates": "List custom prompt templates.",
    "mcp": "List MCP servers and tools, or reconnect and re-register them.",
    "lsp": "Show or toggle language-server diagnostics and navigation.",
    "jobs": "List or manage background and scheduled agent jobs.",
}

_CONTEXT_ACTION_DESCRIPTIONS = {
    "list": "Show pinned context files.",
    "add": "Pin one or more files or globs.",
    "remove": "Unpin one or more files.",
    "clear": "Remove all pinned context files.",
}

_MODEL_ACTION_DESCRIPTIONS = {
    "refresh": "Fetch the provider's newest models now.",
    "list": "Show the models currently offered.",
    "search": "Find models by name in the full catalog.",
}

_MODE_DESCRIPTIONS = {
    "ask": "Approve writes and shell commands.",
    "auto-accept": "Run writes and commands without prompting.",
    "plan": "Read-only planning mode.",
}

_MEMORY_ACTION_DESCRIPTIONS = {
    "list": "Show project and user memory.",
    "add": "Append a fact to project memory.",
    "add-user": "Append a fact to user memory.",
    "clear": "Delete project or user memory.",
}

_INDEX_ACTION_DESCRIPTIONS = {
    "status": "Show indexed files, terms, size, and stale count.",
    "build": "Refresh the index incrementally.",
    "clear": "Delete the index for this workspace.",
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
    "models": "Refresh the model catalog or manage allow/deny filters.",
    "mode": "Set or reset the default permission mode.",
    "permissions": "List or remove tools approved with 'always'.",
    "commands": "Manage run_bash allow/deny regex rules.",
    "trust": "Allow or forbid reads outside the workspace root.",
    "verify": "Set the command run automatically after edits.",
    "budget": "Set or clear token/cost budget limits.",
    "subagents": "Configure the subagent task tool and its step limit.",
    "browser": "Configure optional Playwright browser automation.",
    "shell": "Configure the persistent shell and background job cap.",
    "remote": "Configure SSH/devcontainer command execution.",
    "sampling": "Show, set, or reset sampling parameters.",
    "web": "Set web fetch/search limits and local-address access.",
    "network": "Set proxy, CA bundle, and offline mode.",
    "vision": "Set image attachment size and count limits.",
    "style": "Show or set the output style.",
    "prompt": "Show, set, or clear a custom system prompt.",
    "profile": "Save, apply, or remove named configuration presets.",
    "profiles": "Save, apply, or remove named configuration presets.",
    "ui": "Set theme, color, output verbosity, and ASCII mode.",
    "cache": "Toggle prompt caching for native Anthropic providers.",
}

_MCP_ACTION_DESCRIPTIONS = {
    "list": "Show configured servers, status, and registered tools.",
    "reload": "Reconnect every server and re-register its tools.",
}

_LSP_ACTION_DESCRIPTIONS = {
    "status": "Show LSP settings, available servers, and running clients.",
    "on": "Enable language-server diagnostics.",
    "off": "Disable LSP and stop running servers.",
    "restart": "Stop running servers and clear failures.",
}

_JOBS_ACTION_DESCRIPTIONS = {
    "list": "Show tracked background jobs.",
    "show": "Show one job's record and result.",
    "cancel": "Terminate a running job.",
    "tick": "Start scheduled jobs whose interval has elapsed.",
    "run": "Start a detached job in this workspace.",
}


def _selection_prompt(
    name: str, session: Session, console: Console | None = None
) -> SelectionPrompt | None:
    """Build the selector shown for choice-based commands without arguments.

    Opening ``/model`` is the moment the active provider is actually in use, so
    that branch refreshes the catalog from the provider (subject to the cache
    TTL) before offering it. ``console`` is optional so callers that only want
    the prompt can skip the reporting.
    """
    if name == "model":
        quiet = console or Console(quiet=True)
        catalog = _model_catalog_for(session, quiet)
        models = apply_model_filter(session.provider_id, catalog.models)
        return SelectionPrompt(
            title="Select model",
            text=(
                f"Choose a model from {session.provider.name} "
                f"({session.provider_id}) — {_catalog_status(catalog)}."
                f"\nCurrent model: {session.model}"
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
                    f"{provider.name} — {_provider_default_label(provider)}",
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
    elif command == "memory":
        choices = _MEMORY_ACTION_DESCRIPTIONS
    elif command == "index":
        choices = _INDEX_ACTION_DESCRIPTIONS
    elif command == "model" and session is not None:
        # Completion runs on every keystroke, so this reads the cached catalog
        # only — refreshing from the provider happens in /model itself.
        choices = {
            **_MODEL_ACTION_DESCRIPTIONS,
            **{
                model: f"{session.provider_id} model"
                for model in list_visible_models(session.provider_id)
            },
        }
    elif command == "provider":
        choices = {p.id: p.name for p in list_provider_configs()}
    elif command == "config":
        choices = _CONFIG_ACTION_DESCRIPTIONS
    elif command == "mcp":
        choices = _MCP_ACTION_DESCRIPTIONS
    elif command == "lsp":
        choices = _LSP_ACTION_DESCRIPTIONS
    elif command == "jobs":
        choices = _JOBS_ACTION_DESCRIPTIONS
    elif command == "load":
        choices = {
            s["name"]: f"{s['provider']}:{s['model']} ({s['messages']} msgs)"
            for s in list_saved_sessions()
        }
    else:
        return []
    return [
        (value, description)
        for value, description in choices.items()
        if value.startswith(normalized)
    ]
