from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kiwimatecoder import __version__
from kiwimatecoder.ai import stream_response
from kiwimatecoder.catalog import probe, summarize_ids
from kiwimatecoder.config import (
    add_command_rule,
    add_provider,
    apply_model_filter,
    clear_always_allowed_tools,
    clear_budget,
    clear_command_rules,
    describe_key,
    get_active_provider_ids,
    get_always_allowed_tools,
    get_budget,
    get_command_rules,
    get_compact_at_tokens,
    get_context_window,
    get_default_mode,
    get_key,
    get_model_catalog,
    get_model_filter,
    get_output_style,
    get_provider_config,
    get_sampling,
    get_selected_provider_id,
    get_system_prompt,
    get_trusted_workspace,
    get_verify_command,
    list_provider_configs,
    load_config,
    project_config_path,
    remove_always_allowed_tool,
    remove_command_rule,
    remove_key,
    remove_provider,
    reset_default_mode,
    reset_sampling,
    resolve_default_model,
    set_budget,
    set_default_mode,
    set_key,
    set_model_filter,
    set_output_style,
    set_sampling,
    set_selected_model,
    set_selected_provider,
    set_system_prompt,
    set_trusted_workspace,
    set_verify_command,
    update_provider,
)
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.providers import ProviderConfig
from kiwimatecoder.session import Session
from kiwimatecoder.updater import run_update

app = typer.Typer(
    help="KiwiMateCoder - agentic AI coding assistant CLI",
    invoke_without_command=True,
    no_args_is_help=False,
)
console = Console()
config_app = typer.Typer(
    help="Manage KiwiMateCoder configuration", invoke_without_command=True
)
app.add_typer(config_app, name="config")


@config_app.callback()
def config_main(ctx: typer.Context) -> None:
    """Show a quick orientation when no config subcommand is given."""
    if ctx.invoked_subcommand is not None:
        return
    console.print("[bold]Manage KiwiMateCoder configuration[/bold]")
    console.print("  [cyan]config show[/cyan]               Show current settings")
    console.print("  [cyan]config key set <provider> <key>[/cyan]  Save an API key")
    console.print("  [cyan]config provider use <id>[/cyan]     Set the default provider")
    console.print("  [cyan]config model set <id>[/cyan]       Set the default model")
    console.print("  [cyan]config mode set <ask|auto-accept|plan>[/cyan]  Set default mode")
    console.print("  [cyan]config models show[/cyan]         List the models offered")
    console.print("Run [cyan]config <section> --help[/cyan] for details.")


def _resolve_provider(provider_id: str) -> None:
    """Validate a provider id, exiting with a red message when unknown."""
    try:
        get_provider_config(provider_id)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


key_app = typer.Typer(help="Save, remove, or list API keys.")
config_app.add_typer(key_app, name="key")

provider_app = typer.Typer(help="List or manage providers.")
config_app.add_typer(provider_app, name="provider")

model_app = typer.Typer(help="Set or reset the default model.")
config_app.add_typer(model_app, name="model")

mode_app = typer.Typer(help="Set or reset the default permission mode.")
config_app.add_typer(mode_app, name="mode")

models_app = typer.Typer(help="Manage model visibility and the model catalog.")
config_app.add_typer(models_app, name="models")


# --- canonical `config key ...` ---------------------------------------------


@key_app.command("set")
def key_set(
    provider: Annotated[str, typer.Argument(help="Provider id")],
    key: Annotated[str, typer.Argument(help="Your API key")],
) -> None:
    """Save an API key for a provider."""
    try:
        warning = set_key(provider, key)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]✓ API key saved for[/green] [cyan]{provider}[/cyan] "
        + f"— {describe_key(provider)}."
    )
    if warning:
        console.print(f"[yellow]{warning}[/yellow]")


@key_app.command("remove")
def key_remove(provider: Annotated[str, typer.Argument(help="Provider id")]) -> None:
    """Remove a stored API key for a provider."""
    try:
        existed = remove_key(provider)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    if existed:
        console.print(
            f"[green]✓ Removed stored API key for[/green] [cyan]{provider}[/cyan]."
        )
    else:
        console.print(f"[dim]No stored API key for {provider}.[/dim]")


@key_app.command("list")
def key_list() -> None:
    """List providers and the status of their API keys."""
    table = Table(title="API keys", show_header=True)
    table.add_column("provider", style="cyan")
    table.add_column("env var")
    table.add_column("status")
    for provider in list_provider_configs():
        table.add_row(provider.id, provider.key_env, describe_key(provider.id))
    console.print(table)


# --- canonical `config provider ...` ----------------------------------------


@provider_app.command("use")
def provider_use(provider: Annotated[str, typer.Argument(help="Provider id")]) -> None:
    """Persist and switch to a provider as the default."""
    try:
        set_selected_provider(provider)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓ Default provider set to [cyan]{provider}[/cyan].[/green]")


@provider_app.command("list")
def provider_list() -> None:
    """List all built-in and custom providers."""
    table = Table(title="Providers", show_header=True)
    table.add_column("id", style="cyan")
    table.add_column("name")
    table.add_column("default model")
    for provider in list_provider_configs():
        table.add_row(
            provider.id, provider.name, provider.default_model or "(from server)"
        )
    console.print(table)


@provider_app.command("add")
def provider_add(
    provider_id: Annotated[str, typer.Argument(help="Unique id (no spaces)")],
    name: Annotated[str, typer.Argument(help="Display name")],
    base_url: Annotated[str, typer.Argument(help="Base URL including /v1")],
    default_model: Annotated[str, typer.Argument(help="Default model id")],
    key_env: Annotated[
        str | None,
        typer.Option(
            "--key-env",
            help="API key environment variable (default: <ID>_API_KEY)",
        ),
    ] = None,
) -> None:
    """Add an OpenAI-compatible custom provider."""
    try:
        provider = add_provider(provider_id, name, base_url, default_model, key_env)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]✓ Added provider[/green] [cyan]{provider.id}[/cyan] ({provider.name})."
    )


@provider_app.command("remove")
def provider_remove(provider: Annotated[str, typer.Argument(help="Provider id")]) -> None:
    """Remove a custom provider."""
    try:
        remove_provider(provider)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓ Removed provider {provider}.[/green]")


@provider_app.command("edit")
def provider_edit(
    provider: Annotated[str, typer.Argument(help="Provider id")],
    name: Annotated[str | None, typer.Option("--name", help="New display name")] = None,
    base_url: Annotated[str | None, typer.Option("--base-url", help="New base URL")] = None,
    default_model: Annotated[
        str | None, typer.Option("--default-model", help="New default model")
    ] = None,
    key_env: Annotated[
        str | None, typer.Option("--key-env", help="New key environment variable")
    ] = None,
    compat: Annotated[
        str | None, typer.Option("--compat", help="'openai' or 'anthropic'")
    ] = None,
) -> None:
    """Update fields of a custom provider."""
    try:
        update_provider(
            provider,
            name=name,
            base_url=base_url,
            default_model=default_model,
            key_env=key_env,
            compat=compat,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓ Updated provider {provider}.[/green]")


# --- canonical `config model ...` -------------------------------------------


@model_app.command("set")
def model_set(model: Annotated[str, typer.Argument(help="Model id")]) -> None:
    """Set the default model (overrides the provider default)."""
    set_selected_model(model)
    console.print(f"[green]✓ Default model set to [cyan]{model}[/cyan].[/green]")


@model_app.command("reset")
def model_reset() -> None:
    """Use the provider's default model again."""
    set_selected_model(None)
    console.print("[green]✓ Default model reset (using provider default).[/green]")


# --- canonical `config mode ...` --------------------------------------------


@mode_app.command("set")
def mode_set(
    mode: Annotated[
        str,
        typer.Argument(help="Permission mode: ask, auto-accept, or plan"),
    ]
) -> None:
    """Set the default permission mode for new sessions."""
    try:
        effective = set_default_mode(mode)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓ Default mode set to [cyan]{effective}[/cyan].[/green]")


@mode_app.command("reset")
def mode_reset() -> None:
    """Reset the default permission mode to 'ask'."""
    effective = reset_default_mode()
    console.print(f"[green]✓ Default mode reset to [cyan]{effective}[/cyan].[/green]")


# --- canonical `config models ...` ------------------------------------------


@models_app.command("show")
def models_show(
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider", "-p", help="Provider id (default: configured provider)"
        ),
    ] = None,
) -> None:
    """List the models offered for a provider, newest first."""
    _print_models(provider, refresh=False)


@models_app.command("refresh")
def models_refresh(
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider", "-p", help="Provider id (default: configured provider)"
        ),
    ] = None,
) -> None:
    """Fetch the provider's live model list, dropping deprecated ids."""
    _print_models(provider, refresh=True)


@models_app.command("allow")
def models_allow(
    models: Annotated[list[str], typer.Argument(help="Model ids to show")],
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider", "-p", help="Provider id (default: configured provider)"
        ),
    ] = None,
) -> None:
    """Only show these models for a provider."""
    _set_filter(provider, "allow", models)


@models_app.command("deny")
def models_deny(
    models: Annotated[list[str], typer.Argument(help="Model ids to hide")],
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider", "-p", help="Provider id (default: configured provider)"
        ),
    ] = None,
) -> None:
    """Hide these models for a provider."""
    _set_filter(provider, "deny", models)


@models_app.command("clear")
def models_clear(
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider", "-p", help="Provider id (default: configured provider)"
        ),
    ] = None,
) -> None:
    """Clear model visibility for a provider."""
    pid = provider or get_selected_provider_id()
    _resolve_provider(pid)
    set_model_filter(pid, "all", [])
    console.print(f"[green]✓ Cleared model visibility for {pid}.[/green]")


# --- canonical `config permissions ...` -------------------------------------

permissions_app = typer.Typer(help="Manage persisted 'always allow' tool approvals.")
config_app.add_typer(permissions_app, name="permissions")


@permissions_app.command("list")
def permissions_list() -> None:
    """List tools approved with 'always'."""
    names = get_always_allowed_tools()
    if not names:
        console.print(
            "[dim]No persisted tool approvals. Answer 'always' at an approval "
            "prompt to add one.[/dim]"
        )
        return
    table = Table(title="Always-allowed tools", show_header=True)
    table.add_column("Tool", style="cyan")
    for name in names:
        table.add_row(name)
    console.print(table)


@permissions_app.command("remove")
def permissions_remove(
    tool: Annotated[str, typer.Argument(help="Tool name")],
) -> None:
    """Remove a persisted tool approval."""
    if remove_always_allowed_tool(tool):
        console.print(f"[green]✓ Removed persisted approval for {tool}.[/green]")
    else:
        console.print(f"[dim]No persisted approval for {tool}.[/dim]")


@permissions_app.command("clear")
def permissions_clear() -> None:
    """Remove every persisted tool approval."""
    count = clear_always_allowed_tools()
    console.print(f"[green]✓ Cleared {count} persisted tool approval(s).[/green]")


# --- canonical `config commands ...` ----------------------------------------

commands_app = typer.Typer(
    help="Auto-approve or hard-block run_bash commands by regex."
)
config_app.add_typer(commands_app, name="commands")


@commands_app.command("list")
def commands_list() -> None:
    """List the run_bash allow/deny rules."""
    rules = get_command_rules()
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


def _add_rule(kind: str, pattern: str) -> None:
    try:
        rules = add_command_rule(kind, pattern)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓ Added {kind} rule:[/green] {rules[kind][-1]}")


@commands_app.command("allow")
def commands_allow(
    pattern: Annotated[str, typer.Argument(help="Regex that auto-approves a command")],
) -> None:
    """Auto-approve matching run_bash commands."""
    _add_rule("allow", pattern)


@commands_app.command("deny")
def commands_deny(
    pattern: Annotated[str, typer.Argument(help="Regex that blocks a command")],
) -> None:
    """Hard-block matching run_bash commands."""
    _add_rule("deny", pattern)


@commands_app.command("remove")
def commands_remove(
    kind: Annotated[str, typer.Argument(help="'allow' or 'deny'")],
    pattern: Annotated[str, typer.Argument(help="Regex to remove")],
) -> None:
    """Remove one command rule."""
    try:
        existed = remove_command_rule(kind, pattern)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    if existed:
        console.print(f"[green]✓ Removed {kind} rule:[/green] {pattern}")
    else:
        console.print(f"[dim]No such {kind} rule: {pattern}[/dim]")


@commands_app.command("clear")
def commands_clear() -> None:
    """Remove every command rule."""
    clear_command_rules()
    console.print("[green]✓ Cleared all command rules.[/green]")


@config_app.command("trusted-workspace")
def trusted_workspace_cmd(
    state: Annotated[
        str | None,
        typer.Argument(help="'on' or 'off' (omit to show the current value)"),
    ] = None,
) -> None:
    """Allow or forbid read-only access outside the workspace root."""
    if state is None:
        current = "on" if get_trusted_workspace() else "off"
        console.print(f"Trusted workspace: [cyan]{current}[/cyan]")
        return
    token = state.strip().lower()
    if token in {"on", "true", "enable", "enabled"}:
        set_trusted_workspace(True)
        console.print(
            "[yellow]✓ Trusted workspace on:[/yellow] read-only tools may read "
            "outside the workspace root; writes stay sandboxed."
        )
    elif token in {"off", "false", "disable", "disabled"}:
        set_trusted_workspace(False)
        console.print("[green]✓ Trusted workspace off.[/green]")
    else:
        console.print("[red]Expected 'on' or 'off'.[/red]")
        raise typer.Exit(1)


@config_app.command("verify")
def verify_cmd(
    action: Annotated[str, typer.Argument(help="show, set, or clear")] = "show",
    command: Annotated[
        str | None, typer.Argument(help="Command to run when action is 'set'")
    ] = None,
) -> None:
    """Show, set, or clear the command run automatically after edits."""
    if action in {"show", "status"}:
        current = get_verify_command()
        if current:
            console.print(f"Auto-verify: [cyan]{current}[/cyan]")
        else:
            console.print("[dim]Auto-verify is off.[/dim]")
        return
    if action in {"set", "update"}:
        if not command:
            console.print("[red]Usage: config verify set <command>[/red]")
            raise typer.Exit(1)
        set_verify_command(command)
        console.print(f"[green]✓ Auto-verify set to:[/green] {command}")
        return
    if action in {"clear", "reset", "off"}:
        set_verify_command("")
        console.print("[green]✓ Auto-verify disabled.[/green]")
        return
    console.print("[red]Expected 'show', 'set', or 'clear'.[/red]")
    raise typer.Exit(1)


@config_app.command("budget")
def budget_cmd(
    action: Annotated[
        str, typer.Argument(help="show, tokens, cost, or clear")
    ] = "show",
    value: Annotated[
        str | None, typer.Argument(help="Token count or USD amount")
    ] = None,
) -> None:
    """Set or clear session token/cost budget limits."""
    if action in {"show", "status"}:
        budget = get_budget()
        if not budget:
            console.print("[dim]No budget limits set.[/dim]")
        else:
            console.print(
                "Budget: "
                + ", ".join(f"[cyan]{key}[/cyan]={val}" for key, val in budget.items())
            )
        return
    if action in {"clear", "reset"}:
        clear_budget()
        console.print("[green]✓ Budget limits cleared.[/green]")
        return
    if action in {"tokens", "token"}:
        limit = None if value in (None, "clear", "none", "off") else value
        try:
            set_budget(max_tokens=limit)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        console.print(f"[green]✓ Token budget:[/green] {get_budget().get('max_tokens')}")
        return
    if action in {"cost", "usd"}:
        limit = None if value in (None, "clear", "none", "off") else value
        try:
            set_budget(max_cost_usd=limit)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        console.print(
            f"[green]✓ Cost budget:[/green] ${get_budget().get('max_cost_usd')}"
        )
        return
    console.print("[red]Expected 'show', 'tokens', 'cost', or 'clear'.[/red]")
    raise typer.Exit(1)


# --- canonical `config sampling ...` ----------------------------------------

sampling_app = typer.Typer(help="Get or set sampling parameters.")
config_app.add_typer(sampling_app, name="sampling")


def _print_sampling(sampling: dict[str, object]) -> None:
    if not sampling:
        console.print("[dim]Sampling: provider defaults (nothing set).[/dim]")
        return
    console.print(
        "Sampling: " + ", ".join(f"[cyan]{key}[/cyan]={value}" for key, value in sampling.items())
    )


@sampling_app.command("show")
def sampling_show() -> None:
    """Show the configured sampling parameters."""
    _print_sampling(get_sampling())


@sampling_app.command("set")
def sampling_set(
    values: Annotated[
        list[str],
        typer.Argument(help="key=value pairs, e.g. temperature=0.2 max_tokens=4096"),
    ],
) -> None:
    """Set sampling parameters (temperature, top_p, max_tokens, reasoning_effort)."""
    updates: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            console.print(f"[red]Expected key=value, got '{item}'.[/red]")
            raise typer.Exit(1)
        key, value = item.split("=", 1)
        updates[key.strip()] = value.strip()
    try:
        effective = set_sampling(updates)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓ Sampling set:[/green] {effective}")


@sampling_app.command("reset")
def sampling_reset() -> None:
    """Reset sampling parameters to provider defaults."""
    reset_sampling()
    console.print("[green]✓ Sampling reset to provider defaults.[/green]")


# --- canonical `config style ...` and `config prompt ...` -------------------

style_app = typer.Typer(help="Show or set the output style.")
config_app.add_typer(style_app, name="style")


@style_app.command("show")
def style_show() -> None:
    """Show the current output style."""
    console.print(f"Output style: [cyan]{get_output_style()}[/cyan]")


@style_app.command("set")
def style_set(
    style: Annotated[str, typer.Argument(help="default, concise, explanatory, or code")],
) -> None:
    """Set the output style."""
    try:
        effective = set_output_style(style)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓ Output style set to [cyan]{effective}[/cyan].[/green]")


prompt_app = typer.Typer(help="Show, set, or clear a custom system prompt.")
config_app.add_typer(prompt_app, name="prompt")


@prompt_app.command("show")
def prompt_show() -> None:
    """Show the custom system-prompt addition."""
    text = get_system_prompt()
    if text:
        console.print(text)
    else:
        console.print("[dim]No custom system prompt set.[/dim]")


@prompt_app.command("set")
def prompt_set(
    text: Annotated[list[str], typer.Argument(help="Prompt text (quote it)")],
) -> None:
    """Set a custom system-prompt addition."""
    joined = " ".join(text).strip()
    if not joined:
        console.print("[red]Prompt text is required.[/red]")
        raise typer.Exit(1)
    set_system_prompt(joined)
    console.print("[green]✓ Custom system prompt saved.[/green]")


@prompt_app.command("clear")
def prompt_clear() -> None:
    """Clear the custom system-prompt addition."""
    set_system_prompt(None)
    console.print("[green]✓ Custom system prompt cleared.[/green]")


# --- `config show` ----------------------------------------------------------


@config_app.command("show")
def config_show() -> None:
    """Show the active providers, model, key, and model filter."""
    cfg = load_config()
    provider_id = get_selected_provider_id(cfg)
    provider = get_provider_config(provider_id, cfg)
    active_ids = get_active_provider_ids(cfg)
    active_line = ", ".join(
        f"{pid}" + (" (primary)" if pid == active_ids[0] else "") for pid in active_ids
    )
    console.print(
        f"Provider: [cyan]{provider.id}[/cyan] ({provider.name})\n"
        + f"Active providers: [cyan]{active_line}[/cyan]\n"
        + f"Model: [cyan]{cfg.get('selected_model') or provider.default_model or '(from server)'}[/cyan]\n"
        + f"Mode: [cyan]{get_default_mode(cfg)}[/cyan]\n"
        + f"Key: [cyan]{describe_key(provider_id)}[/cyan] ({provider.key_env})\n"
        + f"Model visibility: [cyan]{get_model_filter(provider_id)['mode']}[/cyan]"
    )
    sampling = get_sampling(cfg)
    sampling_line = (
        ", ".join(f"{key}={value}" for key, value in sampling.items())
        or "provider defaults"
    )
    console.print(
        f"Output style: [cyan]{get_output_style(cfg)}[/cyan] "
        + f"(custom prompt: {'set' if get_system_prompt(cfg) else 'none'})\n"
        + f"Sampling: [cyan]{sampling_line}[/cyan]\n"
        + "Always-allowed tools: "
        + f"[cyan]{', '.join(get_always_allowed_tools(cfg)) or 'none'}[/cyan]\n"
        + f"Trusted workspace: [cyan]{'on' if get_trusted_workspace(cfg) else 'off'}[/cyan]\n"
        + f"Auto-verify: [cyan]{get_verify_command(cfg) or 'off'}[/cyan]\n"
        + "Budget: "
        + (
            f"[cyan]{get_budget(cfg)}[/cyan]"
            if get_budget(cfg)
            else "[cyan]none[/cyan]"
        )
    )
    project_path = project_config_path()
    if project_path is not None:
        console.print(f"Project config: [cyan]{project_path}[/cyan] (overrides global)")


def _print_models(provider: str | None, *, refresh: bool) -> None:
    cfg = load_config()
    provider_id = provider or get_selected_provider_id(cfg)
    try:
        provider_cfg = get_provider_config(provider_id, cfg)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    catalog = get_model_catalog(provider_id, force=refresh, cfg=cfg)
    if catalog.error:
        console.print(f"[yellow]Could not refresh models: {catalog.error}[/yellow]")
    if catalog.added:
        console.print(f"[green]New:[/green] {summarize_ids(catalog.added)}")
    if catalog.removed:
        console.print(
            f"[yellow]Deprecated, removed:[/yellow] {summarize_ids(catalog.removed)}"
        )

    source = {
        "live": "live from provider",
        "cache": "cached",
        "curated": "built-in list",
    }[catalog.source]
    console.print(f"[cyan]{provider_cfg.name}[/cyan] models ({source}):")
    for model in apply_model_filter(provider_id, catalog.models):
        console.print(f"  {model}")


def _set_filter(provider: str | None, mode: str, models: list[str]) -> None:
    pid = provider or get_selected_provider_id()
    _resolve_provider(pid)
    try:
        set_model_filter(pid, mode, models)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    verb = "showing" if mode == "allow" else "hiding"
    console.print(
        f"[green]✓ Now {verb} these models for {pid}:[/green] "
        + ", ".join(models)
    )


# --- legacy aliases (kept for backward compatibility) -----------------------


@config_app.command("set-key", hidden=True, deprecated=True)
def set_key_cmd(
    key: Annotated[str, typer.Argument(help="Your API key")],
    provider: Annotated[
        str,
        typer.Option("--provider", "-p", help="Provider id this key belongs to"),
    ] = "openrouter",
) -> None:
    """Deprecated: use `config key set <provider> <key>`."""
    key_set(provider, key)


@config_app.command("set-provider", hidden=True, deprecated=True)
def set_provider_cmd(
    provider: Annotated[str, typer.Argument(help="Provider id")]
) -> None:
    """Deprecated: use `config provider use <id>`."""
    provider_use(provider)


@config_app.command("set-model", hidden=True, deprecated=True)
def set_model_cmd(model: Annotated[str, typer.Argument(help="Model id")]) -> None:
    """Deprecated: use `config model set <id>`."""
    model_set(model)


@config_app.command("check", hidden=True, deprecated=True)
def check() -> None:
    """Deprecated: use `config show`."""
    config_show()


@config_app.command("list", hidden=True, deprecated=True)
def list_cmd() -> None:
    """Deprecated: use `config provider list`."""
    provider_list()


@app.callback()
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show the installed KiwiMateCoder version and exit.",
            is_eager=True,
        ),
    ] = False,
    update: Annotated[
        bool,
        typer.Option(
            "-update",
            "--update",
            help="Update KiwiMateCoder in the current Python environment.",
        ),
    ] = False,
    resume: Annotated[
        str | None,
        typer.Option(
            "-resume",
            "--resume",
            help="Resume a saved session by name or path ('last' = autosave).",
        ),
    ] = None,
    continue_: Annotated[
        bool,
        typer.Option(
            "--continue",
            "-c",
            help="Continue the most recent session (same as --resume last).",
        ),
    ] = False,
) -> None:
    """Launch the interactive session when run with no subcommand."""
    if version:
        console.print(f"kiwimatecoder {__version__}")
        raise typer.Exit(0)

    if update:
        raise typer.Exit(run_update(console))

    if ctx.invoked_subcommand is not None:
        return

    from kiwimatecoder import repl

    session: Session | None = None
    if resume or continue_:
        from kiwimatecoder.session import load_session

        target = resume or "last"
        try:
            session = load_session(target, workspace_root=Path.cwd())
            console.print(
                f"[bold green]Resumed session '{target}'[/bold green] "
                + f"([dim]{len(session.messages)} messages, {session.total_tokens:,} tokens[/dim])"
            )
        except Exception as exc:
            if resume:
                console.print(f"[red]Could not resume session '{resume}': {exc}[/red]")
                raise typer.Exit(1)
            console.print(
                f"[dim]No previous session to continue ({exc}). Starting fresh.[/dim]"
            )

    if session is None:
        cfg = load_config()
        provider_id = get_selected_provider_id(cfg)
        provider = get_provider_config(provider_id, cfg)
        model = str(cfg.get("selected_model") or "") or resolve_default_model(provider)

        try:
            mode = PermissionMode.from_str(str(cfg.get("default_mode", "ask")))
        except ValueError:
            mode = PermissionMode.ASK

        if not get_key(provider_id) and provider.needs_key:
            console.print(
                Panel(
                    f"[yellow]No API key set for {provider.name}.[/yellow]\n"
                    + "Run [cyan]kiwimatecoder setup[/cyan] to choose a provider and "
                    + f"enter a key, or export [cyan]{provider.key_env}[/cyan].",
                    title="Quick start",
                )
            )
            if _stdin_is_tty() and _prompt_yes_no("Run setup now?"):
                _run_setup(provider_id, key=None)

        session = Session(
            provider_id=provider_id,
            model=model,
            mode=mode,
            workspace_root=Path.cwd(),
            active_provider_ids=get_active_provider_ids(cfg),
            always_allowed=set(get_always_allowed_tools(cfg)),
            output_style=get_output_style(cfg),
            custom_system_prompt=get_system_prompt(cfg),
            command_rules=get_command_rules(cfg),
            trusted_workspace=get_trusted_workspace(cfg),
            verify_command=get_verify_command(cfg),
            compact_at_tokens=get_compact_at_tokens(cfg),
            context_window=get_context_window(cfg),
        )

    # Persisted approvals are user preferences, so a resumed session picks up
    # anything granted since its last save. Command rules come from config too.
    session.always_allowed.update(get_always_allowed_tools())
    session.command_rules = get_command_rules()
    session.trusted_workspace = get_trusted_workspace()
    session.verify_command = get_verify_command()
    session.compact_at_tokens = get_compact_at_tokens()
    session.context_window = get_context_window()

    repl.run(session)


def _stdin_is_tty() -> bool:
    """Whether interactive prompts can be safely shown on stdin."""
    try:
        return bool(sys.stdin.isatty())
    except (AttributeError, OSError):
        return False


def _prompt_yes_no(question: str) -> bool:
    """Ask a yes/no question on the console; false on cancel/EOF/unknown."""
    try:
        answer = console.input(f"{question} [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return False
    return answer in ("y", "yes")


def _interactive_select_provider(
    providers: Sequence[ProviderConfig],
    selected: str,
    local_status: dict[str, bool] | None = None,
) -> str | None:
    """Keyboard-driven provider picker, matching the REPL's selector style."""
    from prompt_toolkit.shortcuts import choice

    def _label(p: ProviderConfig) -> str:
        if p.is_local:
            status = (local_status or {}).get(p.id)
            if status is None:
                note = "key required" if p.requires_key else "no key needed"
            elif status:
                note = "running"
            else:
                note = "not detected"
            return f"{p.name} — {note} ({p.base_url})"
        return f"{p.name} — {p.default_model}"

    try:
        return choice(
            message="Choose a provider to configure",
            options=[(p.id, _label(p)) for p in providers],
            default=selected,
            show_frame=True,
            bottom_toolbar="↑/↓ move • Enter select • Ctrl-C cancel",
        )
    except (EOFError, KeyboardInterrupt):
        return None


def _interactive_api_key() -> str | None:
    """Prompt for an API key, returning None on cancel/EOF."""
    try:
        console.print("[bold]Enter the API key for the provider:[/bold]")
        return console.input("key> ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return None


def _run_setup(provider_id: str, key: str | None) -> bool:
    """Store an API key for a provider and switch to it (the setup wizard body).

    Returns True on success and False when the provider is unknown or the key
    entry was cancelled/empty. Callers decide how to treat a failure.
    """
    try:
        provider = get_provider_config(provider_id)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return False
    if key is None and provider.is_local and not provider.requires_key:
        # Keyless local servers (Ollama, LM Studio) — just select the provider
        # and let the session model resolve from whatever the server has loaded.
        # Key-enforcing locals (Unsloth) fall through to the normal key prompt.
        set_selected_provider(provider_id)
        set_selected_model(None)  # don't carry a stale model across providers
        console.print(
            f"[green]✓ {provider.name} needs no API key[/green] — "
            + f"models are read from the server at {provider.base_url}."
        )
        if not probe(provider):
            console.print(
                f"[yellow]No server answered at {provider.base_url} — "
                + "start it before chatting.[/yellow]"
            )
        console.print("Ready to go. Run [cyan]kiwimatecoder[/cyan] to start a session.")
        return True
    if key is None:
        key = _interactive_api_key()
        if key is None:
            console.print("[yellow]Setup cancelled; nothing changed.[/yellow]")
            return False
        if not key:
            console.print("[red]No key entered; nothing changed.[/red]")
            return False
    try:
        warning = set_key(provider_id, key)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return False
    set_selected_provider(provider_id)
    console.print(
        f"[green]✓ API key saved for {provider_id}[/green] "
        + f"— {describe_key(provider_id)}."
    )
    if warning:
        console.print(f"[yellow]{warning}[/yellow]")
    console.print("Ready to go. Run [cyan]kiwimatecoder[/cyan] to start a session.")
    return True


@app.command("setup")
def setup(
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider", "-p", help="Provider id (default: configured provider)"
        ),
    ] = None,
    key: Annotated[
        str | None,
        typer.Option("--key", "-k", help="API key (skips the interactive prompt)"),
    ] = None,
) -> None:
    """Choose a provider and save its API key — the quick-start guide."""
    cfg = load_config()
    provider_id = provider or get_selected_provider_id(cfg)
    if provider is None and key is None and _stdin_is_tty():
        providers = list_provider_configs(cfg)
        # Detect which local servers are actually up; localhost refuses fast,
        # so this costs ~nothing when they are not running. Providers that
        # enforce auth get their key when one is stored, so the probe sees a
        # true 200 instead of their 401.
        local_status = {
            p.id: probe(p, api_key=get_key(p.id)) for p in providers if p.is_local
        }
        chosen = _interactive_select_provider(providers, provider_id, local_status)
        if chosen is None:
            console.print("[yellow]Setup cancelled.[/yellow]")
            raise typer.Exit(1)
        provider_id = chosen
    if not _run_setup(provider_id, key):
        raise typer.Exit(1)


@app.command()
def ask(
    prompt: Annotated[str, typer.Argument(help="Your coding question")],
    file: Annotated[
        Path | None, typer.Option("--file", "-f", help="Path to a code file to include")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Override the default model")
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider", "-p", help="Provider id (default: configured provider)"
        ),
    ] = None,
) -> None:
    """Ask KiwiMateCoder a one-shot coding question."""
    cfg = load_config()
    provider_id = provider or get_selected_provider_id(cfg)
    try:
        provider_cfg = get_provider_config(provider_id, cfg)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    api_key = get_key(provider_id)
    if not api_key and provider_cfg.needs_key:
        console.print(
            f"[red]No API key for {provider_cfg.name}. "
            + f"Run: kiwimatecoder setup --provider {provider_id}[/red]"
        )
        raise typer.Exit(1)

    full_prompt = prompt
    if file:
        if not file.exists():
            console.print(f"[red]File not found: {file}[/red]")
            raise typer.Exit(1)
        full_prompt = f"{prompt}\n\n```\n{file.read_text()}\n```"

    asyncio.run(
        stream_response(
            full_prompt,
            api_key or "",
            model=model
            or str(cfg.get("selected_model") or "")
            or resolve_default_model(provider_cfg),
            provider=provider_cfg,
        )
    )
    console.print()


@app.command("doctor")
def doctor_cmd() -> None:
    """Run environment, config, provider, and workspace diagnostics."""
    from kiwimatecoder import diagnostics

    cfg = load_config()
    provider_id = get_selected_provider_id(cfg)
    provider = get_provider_config(provider_id, cfg)
    model = str(cfg.get("selected_model") or "") or resolve_default_model(provider)
    try:
        mode = PermissionMode.from_str(str(cfg.get("default_mode", "ask")))
    except ValueError:
        mode = PermissionMode.ASK
    session = Session(
        provider_id=provider_id,
        model=model,
        mode=mode,
        workspace_root=Path.cwd(),
        active_provider_ids=get_active_provider_ids(cfg),
        always_allowed=set(get_always_allowed_tools(cfg)),
        output_style=get_output_style(cfg),
        custom_system_prompt=get_system_prompt(cfg),
    )
    diagnostics.render(diagnostics.run_checks(session), console)


@app.command("update")
def update_cmd() -> None:
    """Update KiwiMateCoder in the current Python environment."""
    raise typer.Exit(run_update(console))


@app.command("version")
def version_cmd() -> None:
    """Show the installed KiwiMateCoder version."""
    console.print(f"kiwimatecoder {__version__}")