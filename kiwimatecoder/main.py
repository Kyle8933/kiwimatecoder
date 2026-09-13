from __future__ import annotations

import asyncio
import json
import shlex
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from kiwimatecoder import __version__
from kiwimatecoder.ai import stream_response
from kiwimatecoder.catalog import probe, summarize_ids
from kiwimatecoder.config import (
    add_command_rule,
    add_provider,
    apply_model_filter,
    apply_profile,
    clear_always_allowed_tools,
    clear_budget,
    clear_command_rules,
    describe_key,
    get_active_provider_ids,
    get_acp,
    get_always_allowed_tools,
    get_browser,
    get_budget,
    get_command_rules,
    get_compact_at_tokens,
    get_context_window,
    get_default_mode,
    get_index,
    get_key,
    get_lsp,
    get_mcp_servers,
    get_media,
    get_memory,
    get_model_catalog,
    get_model_filter,
    get_network,
    get_output_style,
    get_profile,
    get_profiles,
    get_prompt_cache,
    get_provider_config,
    get_remote,
    get_sampling,
    get_sandbox,
    get_selected_provider_id,
    get_shell_config,
    get_subagents,
    get_system_prompt,
    get_telemetry,
    get_trusted_workspace,
    get_ui,
    get_verify_command,
    get_web,
    list_provider_configs,
    load_config,
    project_config_path,
    remove_always_allowed_tool,
    remove_command_rule,
    remove_key,
    remove_mcp_server,
    remove_profile,
    remove_provider,
    rename_profile,
    reset_default_mode,
    reset_sampling,
    resolve_default_model,
    save_profile,
    set_acp,
    set_budget,
    set_browser,
    set_default_mode,
    set_index,
    set_key,
    set_lsp,
    set_mcp_server,
    set_media,
    set_memory,
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
    set_sync,
    set_system_prompt,
    set_telemetry,
    set_trusted_workspace,
    set_ui,
    set_verify_command,
    set_web,
    update_provider,
    validate_config,
)
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.providers import ProviderConfig
from kiwimatecoder.session import Session, apply_session_profile
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
    console.print("  [cyan]config ui show[/cyan]             Theme, color, output, ASCII")
    console.print("  [cyan]config web show[/cyan]            Web fetch/search limits")
    console.print("  [cyan]config network show[/cyan]        Proxy, CA bundle, offline mode")
    console.print("  [cyan]config index show[/cyan]          Codebase index status")
    console.print("  [cyan]config sandbox show[/cyan]        OS-level command sandbox")
    console.print("  [cyan]config remote show[/cyan]         SSH/devcontainer command execution")
    console.print("  [cyan]config acp show[/cyan]            ACP editor-integration timeout")
    console.print("  [cyan]config media show[/cyan]          Opt-in image generation")
    console.print("  [cyan]config telemetry show[/cyan]      Opt-in local telemetry and crash reports")
    console.print("Run [cyan]config <section> --help[/cyan] for details.")


def _resolve_provider(provider_id: str) -> None:
    """Validate a provider id, exiting with a red message when unknown."""
    try:
        get_provider_config(provider_id)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


def _check() -> str:
    """Return the active success glyph, escaped for Rich markup."""
    from rich.markup import escape

    from kiwimatecoder.ui import glyph

    return escape(glyph("check"))


def _print_ui(current: dict[str, Any]) -> None:
    """Print the current UI preferences as a short summary."""
    console.print(
        f"Theme: [cyan]{current['theme']}[/cyan]\n"
        f"Color: [cyan]{current['color']}[/cyan]\n"
        f"Output mode: [cyan]{current['output_mode']}[/cyan]\n"
        f"ASCII: [cyan]{'on' if current['ascii'] else 'off'}[/cyan]"
    )


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

media_app = typer.Typer(help="Configure opt-in image generation.")
config_app.add_typer(media_app, name="media")

telemetry_app = typer.Typer(
    help="Configure opt-in local telemetry, debug logging, and crash reports."
)
config_app.add_typer(telemetry_app, name="telemetry")

jobs_app = typer.Typer(help="Manage background and scheduled agent jobs.")
app.add_typer(jobs_app, name="jobs")

sync_app = typer.Typer(
    help="Sync saved sessions across machines through a shared folder (opt-in)."
)
app.add_typer(sync_app, name="sync")

eval_app = typer.Typer(
    help="Run the eval harness: deterministic prompt/tool regression checks."
)
app.add_typer(eval_app, name="eval")


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
        f"[green]{_check()} API key saved for[/green] [cyan]{provider}[/cyan] "
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
            f"[green]{_check()} Removed stored API key for[/green] [cyan]{provider}[/cyan]."
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
    console.print(f"[green]{_check()} Default provider set to [cyan]{provider}[/cyan].[/green]")


@provider_app.command("list")
def provider_list() -> None:
    """List all built-in and custom providers."""
    table = Table(title="Providers", show_header=True)
    table.add_column("id", style="cyan")
    table.add_column("name")
    table.add_column("default model")
    table.add_column("auth")
    for provider in list_provider_configs():
        auth = provider.key_header
        if provider.key_prefix:
            auth += f": {provider.key_prefix.strip()}"
        table.add_row(
            provider.id,
            provider.name,
            provider.default_model or "(from server)",
            auth,
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
    key_header: Annotated[
        str | None,
        typer.Option(
            "--key-header",
            help="Auth header name (default: Authorization)",
        ),
    ] = None,
    key_prefix: Annotated[
        str | None,
        typer.Option(
            "--key-prefix",
            help="Prefix before the key (default: 'Bearer '; use '' for none)",
        ),
    ] = None,
    api_version: Annotated[
        str | None,
        typer.Option(
            "--api-version",
            help="Azure-style ?api-version= value (default: none)",
        ),
    ] = None,
) -> None:
    """Add an OpenAI-compatible custom provider."""
    try:
        provider = add_provider(
            provider_id,
            name,
            base_url,
            default_model,
            key_env,
            key_header=key_header,
            key_prefix=key_prefix,
            api_version=api_version,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Added provider[/green] [cyan]{provider.id}[/cyan] ({provider.name})."
    )


@provider_app.command("remove")
def provider_remove(provider: Annotated[str, typer.Argument(help="Provider id")]) -> None:
    """Remove a custom provider."""
    try:
        remove_provider(provider)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]{_check()} Removed provider {provider}.[/green]")


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
    key_header: Annotated[
        str | None,
        typer.Option("--key-header", help="New auth header name"),
    ] = None,
    key_prefix: Annotated[
        str | None,
        typer.Option(
            "--key-prefix",
            help="New prefix before the key ('' clears the prefix)",
        ),
    ] = None,
    api_version: Annotated[
        str | None,
        typer.Option(
            "--api-version",
            help="New ?api-version= value ('' clears it)",
        ),
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
            key_header=key_header,
            key_prefix=key_prefix,
            api_version=api_version,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]{_check()} Updated provider {provider}.[/green]")


# --- canonical `config model ...` -------------------------------------------


@model_app.command("set")
def model_set(model: Annotated[str, typer.Argument(help="Model id")]) -> None:
    """Set the default model (overrides the provider default)."""
    set_selected_model(model)
    console.print(f"[green]{_check()} Default model set to [cyan]{model}[/cyan].[/green]")


@model_app.command("reset")
def model_reset() -> None:
    """Use the provider's default model again."""
    set_selected_model(None)
    console.print(f"[green]{_check()} Default model reset (using provider default).[/green]")


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
    console.print(f"[green]{_check()} Default mode set to [cyan]{effective}[/cyan].[/green]")


@mode_app.command("reset")
def mode_reset() -> None:
    """Reset the default permission mode to 'ask'."""
    effective = reset_default_mode()
    console.print(f"[green]{_check()} Default mode reset to [cyan]{effective}[/cyan].[/green]")


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
    console.print(f"[green]{_check()} Cleared model visibility for {pid}.[/green]")


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
        console.print(f"[green]{_check()} Removed persisted approval for {tool}.[/green]")
    else:
        console.print(f"[dim]No persisted approval for {tool}.[/dim]")


@permissions_app.command("clear")
def permissions_clear() -> None:
    """Remove every persisted tool approval."""
    count = clear_always_allowed_tools()
    console.print(f"[green]{_check()} Cleared {count} persisted tool approval(s).[/green]")


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
    console.print(f"[green]{_check()} Added {kind} rule:[/green] {rules[kind][-1]}")


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
        console.print(f"[green]{_check()} Removed {kind} rule:[/green] {pattern}")
    else:
        console.print(f"[dim]No such {kind} rule: {pattern}[/dim]")


@commands_app.command("clear")
def commands_clear() -> None:
    """Remove every command rule."""
    clear_command_rules()
    console.print(f"[green]{_check()} Cleared all command rules.[/green]")


# --- canonical `config mcp ...` ---------------------------------------------

mcp_app = typer.Typer(help="Manage MCP (Model Context Protocol) servers.")
config_app.add_typer(mcp_app, name="mcp")


@mcp_app.command("list")
def mcp_list() -> None:
    """List configured MCP servers."""
    servers = get_mcp_servers()
    if not servers:
        console.print("[dim]No MCP servers configured.[/dim]")
        return
    table = Table(title="MCP servers", show_header=True)
    table.add_column("server", style="cyan")
    table.add_column("transport")
    table.add_column("target")
    table.add_column("status")
    for name, spec in servers.items():
        if spec.get("command"):
            transport = "stdio"
            target = " ".join(
                [str(spec["command"]), *[str(arg) for arg in spec.get("args") or []]]
            )
        else:
            transport = "http"
            target = str(spec.get("url") or "")
        status = "disabled" if spec.get("disabled") else "enabled"
        table.add_row(name, transport, target, status)
    console.print(table)


def _parse_header(raw: str) -> tuple[str, str]:
    name, _, value = raw.partition(":")
    name = name.strip()
    value = value.strip()
    if not name or not value:
        raise ValueError(f"Invalid header '{raw}': expected 'Name: value'.")
    return name, value


@mcp_app.command("add")
def mcp_add(
    name: Annotated[str, typer.Argument(help="Server name ([a-z0-9][a-z0-9_-]*)")],
    command: Annotated[
        str | None, typer.Option("--command", help="stdio command to run")
    ] = None,
    args: Annotated[
        str | None,
        typer.Option("--args", help="stdio arguments (quote the whole value)"),
    ] = None,
    url: Annotated[
        str | None, typer.Option("--url", help="HTTP MCP endpoint")
    ] = None,
    header: Annotated[
        list[str] | None,
        typer.Option("--header", "-H", help="HTTP header 'Name: value' (repeatable)"),
    ] = None,
) -> None:
    """Add or replace an MCP server (set exactly one of --command or --url)."""
    spec: dict[str, Any] = {}
    if command:
        spec["command"] = command
    if args:
        spec["args"] = shlex.split(args)
    if url:
        spec["url"] = url
    if header:
        headers: dict[str, str] = {}
        try:
            for raw in header:
                key, value = _parse_header(raw)
                headers[key] = value
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        spec["headers"] = headers
    try:
        servers = set_mcp_server(name, spec)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} MCP server [cyan]{name}[/cyan] saved "
        f"({len(servers)} total).[/green]"
    )


@mcp_app.command("remove")
def mcp_remove(name: Annotated[str, typer.Argument(help="Server name")]) -> None:
    """Remove an MCP server."""
    if remove_mcp_server(name):
        console.print(f"[green]{_check()} Removed MCP server {name}.[/green]")
    else:
        console.print(f"[dim]No MCP server named {name}.[/dim]")


# --- canonical `config lsp ...` ---------------------------------------------

lsp_app = typer.Typer(help="Language server diagnostics settings.")
config_app.add_typer(lsp_app, name="lsp")


def _lsp_state(value: str) -> bool:
    """Parse an on/off argument, raising ``ValueError`` otherwise."""
    token = value.strip().lower()
    if token in {"on", "true", "yes", "enable", "enabled"}:
        return True
    if token in {"off", "false", "no", "disable", "disabled"}:
        return False
    raise ValueError("Expected 'on' or 'off'.")


@lsp_app.command("show")
def lsp_show() -> None:
    """Show language server settings and which presets are installed."""
    from kiwimatecoder import lsp as lsp_module

    settings = get_lsp()
    state = "on" if settings["enabled"] else "off"
    console.print(
        f"LSP: [cyan]{state}[/cyan] "
        f"(timeout {settings['timeout']:g}s, diagnostics after edits "
        f"[cyan]{'on' if settings['diagnostics_after_edits'] else 'off'}[/cyan])"
    )
    installed = lsp_module.available_servers()
    if installed:
        console.print(
            "Available servers: "
            + ", ".join(
                f"[cyan]{name}[/cyan] ({spec.command})"
                for name, spec in sorted(installed.items())
            )
        )
    else:
        console.print("[dim]No language server commands found on PATH.[/dim]")
    overrides = settings["servers"]
    if overrides:
        console.print(
            "Overrides: "
            + ", ".join(f"[cyan]{name}[/cyan]" for name in sorted(overrides))
        )


@lsp_app.command("enable")
def lsp_enable(
    state: Annotated[str, typer.Argument(help="'on' or 'off'")] = "on",
) -> None:
    """Enable or disable language-server diagnostics."""
    try:
        enabled = _lsp_state(state)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    settings = set_lsp(enabled=enabled)
    label = "on" if settings["enabled"] else "off"
    console.print(
        f"[green]{_check()} LSP {label}.[/green] Servers start on first use."
    )


@lsp_app.command("after-edits")
def lsp_after_edits(
    state: Annotated[str, typer.Argument(help="'on' or 'off'")] = "on",
) -> None:
    """Toggle diagnostics appended after successful file edits."""
    try:
        enabled = _lsp_state(state)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    settings = set_lsp(diagnostics_after_edits=enabled)
    label = "on" if settings["diagnostics_after_edits"] else "off"
    console.print(
        f"[green]{_check()} LSP diagnostics after edits {label}.[/green]"
    )


# --- canonical `config subagents ...` ---------------------------------------

subagents_app = typer.Typer(help="Subagent (task tool) settings.")
config_app.add_typer(subagents_app, name="subagents")


def _subagents_line() -> str:
    settings = get_subagents()
    state = "on" if settings["enabled"] else "off"
    model = settings["model"] or "(session model)"
    return (
        f"Subagents: [cyan]{state}[/cyan] "
        f"(max steps {settings['max_steps']}, model {model})"
    )


@subagents_app.command("show")
def subagents_show() -> None:
    """Show whether subagents are enabled and their step/model limits."""
    console.print(_subagents_line())


@subagents_app.command("enable")
def subagents_enable(
    state: Annotated[str, typer.Argument(help="'on' or 'off'")] = "on",
) -> None:
    """Enable or disable the task tool."""
    token = state.strip().lower()
    if token in {"on", "true", "enable", "enabled"}:
        enabled = True
    elif token in {"off", "false", "disable", "disabled"}:
        enabled = False
    else:
        console.print("[red]Expected 'on' or 'off'.[/red]")
        raise typer.Exit(1)
    set_subagents(enabled=enabled)
    console.print(
        f"[green]{_check()} Subagents {'on' if enabled else 'off'}.[/green]"
    )


@subagents_app.command("max-steps")
def subagents_max_steps(
    value: Annotated[
        str | None,
        typer.Argument(help="Step limit 1-100 (omit to show the current value)"),
    ] = None,
) -> None:
    """Set the maximum tool batches a subagent may run."""
    if value is None:
        console.print(
            f"Subagent max steps: [cyan]{get_subagents()['max_steps']}[/cyan]"
        )
        return
    try:
        settings = set_subagents(max_steps=value)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Subagent max steps:[/green] {settings['max_steps']}"
    )


@subagents_app.command("model")
def subagents_model(
    value: Annotated[
        str | None,
        typer.Argument(help="Model override (omit to show; 'clear' to reset)"),
    ] = None,
) -> None:
    """Set the model subagents use (empty means the session model)."""
    if value is None:
        current = get_subagents()["model"] or "(session model)"
        console.print(f"Subagent model: [cyan]{current}[/cyan]")
        return
    if value.strip().lower() in {"clear", "none", "off"}:
        value = ""
    settings = set_subagents(model=value)
    console.print(
        f"[green]{_check()} Subagent model:[/green] "
        f"{settings['model'] or '(session model)'}"
    )


# --- canonical `config browser ...` -----------------------------------------

browser_app = typer.Typer(help="Optional Playwright browser automation settings.")
config_app.add_typer(browser_app, name="browser")


def _print_browser(settings: dict[str, Any]) -> None:
    console.print(
        f"Browser automation: "
        f"[cyan]{'on' if settings['enabled'] else 'off'}[/cyan]\n"
        f"Headless: [cyan]{'on' if settings['headless'] else 'off'}[/cyan]\n"
        f"Timeout: [cyan]{settings['timeout_ms']}ms[/cyan]"
    )


@browser_app.command("show")
def browser_show() -> None:
    """Show browser automation settings."""
    _print_browser(get_browser())


@browser_app.command("enable")
def browser_enable(
    state: Annotated[str, typer.Argument(help="'on' or 'off'")] = "on",
) -> None:
    """Enable or disable the browser tool."""
    token = state.strip().lower()
    if token in {"on", "true", "enable", "enabled"}:
        enabled = True
    elif token in {"off", "false", "disable", "disabled"}:
        enabled = False
    else:
        console.print("[red]Expected 'on' or 'off'.[/red]")
        raise typer.Exit(1)
    set_browser(enabled=enabled)
    console.print(
        f"[green]{_check()} Browser automation "
        f"{'on' if enabled else 'off'}.[/green]"
    )


@browser_app.command("headless")
def browser_headless(
    state: Annotated[
        str | None,
        typer.Argument(help="'on' or 'off' (omit to show the current value)"),
    ] = None,
) -> None:
    """Run Chromium headless (on) or with a visible window (off)."""
    if state is None:
        current = "on" if get_browser()["headless"] else "off"
        console.print(f"Browser headless: [cyan]{current}[/cyan]")
        return
    token = state.strip().lower()
    if token in {"on", "true", "enable", "enabled"}:
        enabled = True
    elif token in {"off", "false", "disable", "disabled"}:
        enabled = False
    else:
        console.print("[red]Expected 'on' or 'off'.[/red]")
        raise typer.Exit(1)
    settings = set_browser(headless=enabled)
    console.print(
        f"[green]{_check()} Browser headless:[/green] "
        f"{'on' if settings['headless'] else 'off'}"
    )


@browser_app.command("timeout")
def browser_timeout(
    value: Annotated[
        str | None,
        typer.Argument(help="Timeout in ms 1000-120000 (omit to show)"),
    ] = None,
) -> None:
    """Set the default action timeout in milliseconds."""
    if value is None:
        console.print(f"Browser timeout: [cyan]{get_browser()['timeout_ms']}ms[/cyan]")
        return
    try:
        settings = set_browser(timeout_ms=value)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]{_check()} Browser timeout:[/green] {settings['timeout_ms']}ms")


# --- canonical `config shell ...` -------------------------------------------


shell_app = typer.Typer(help="Persistent shell and background job settings.")
config_app.add_typer(shell_app, name="shell")


def _print_shell(settings: dict[str, Any]) -> None:
    console.print(
        f"Persistent shell: "
        f"[cyan]{'on' if settings['persistent'] else 'off'}[/cyan]\n"
        f"Command timeout: [cyan]{settings['timeout']}s[/cyan]\n"
        f"Background job cap: [cyan]{settings['max_jobs']}[/cyan]"
    )


def _parse_on_off(value: str) -> bool:
    token = value.strip().lower()
    if token in {"on", "true", "enable", "enabled"}:
        return True
    if token in {"off", "false", "disable", "disabled"}:
        return False
    console.print("[red]Expected 'on' or 'off'.[/red]")
    raise typer.Exit(1)


@shell_app.command("show")
def shell_show() -> None:
    """Show persistent-shell settings."""
    _print_shell(get_shell_config())


@shell_app.command("persistent")
def shell_persistent(
    state: Annotated[
        str | None,
        typer.Argument(help="'on' or 'off' (omit to show the current value)"),
    ] = None,
) -> None:
    """Enable or disable the stateful `shell` tool."""
    if state is None:
        current = "on" if get_shell_config()["persistent"] else "off"
        console.print(f"Persistent shell: [cyan]{current}[/cyan]")
        return
    settings = set_shell_config(persistent=_parse_on_off(state))
    console.print(
        f"[green]{_check()} Persistent shell:[/green] "
        f"{'on' if settings['persistent'] else 'off'}"
    )


@shell_app.command("timeout")
def shell_timeout(
    value: Annotated[
        str | None,
        typer.Argument(help="Timeout in seconds 1-3600 (omit to show)"),
    ] = None,
) -> None:
    """Set how long one `shell` command may run before it is killed."""
    if value is None:
        console.print(f"Shell timeout: [cyan]{get_shell_config()['timeout']}s[/cyan]")
        return
    try:
        settings = set_shell_config(timeout=value)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]{_check()} Shell timeout:[/green] {settings['timeout']}s")


@shell_app.command("max-jobs")
def shell_max_jobs(
    value: Annotated[
        str | None,
        typer.Argument(help="Concurrent background jobs 1-100 (omit to show)"),
    ] = None,
) -> None:
    """Set the maximum number of concurrent background jobs."""
    if value is None:
        console.print(
            f"Background job cap: [cyan]{get_shell_config()['max_jobs']}[/cyan]"
        )
        return
    try:
        settings = set_shell_config(max_jobs=value)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]{_check()} Background job cap:[/green] {settings['max_jobs']}")


# --- canonical `config sandbox ...` -----------------------------------------


sandbox_app = typer.Typer(help="Optional OS-level command sandboxing.")
config_app.add_typer(sandbox_app, name="sandbox")


def _print_sandbox(settings: dict[str, Any]) -> None:
    writable = ", ".join(settings["extra_writable"]) or "(none)"
    console.print(
        f"Sandbox: [cyan]{'on' if settings['enabled'] else 'off'}[/cyan]\n"
        f"Network: [cyan]{'on' if settings['network'] else 'off'}[/cyan]\n"
        f"Extra writable paths: [cyan]{escape(writable)}[/cyan]"
    )


@sandbox_app.command("show")
def sandbox_show() -> None:
    """Show sandbox settings and the detected backend."""
    from kiwimatecoder import sandbox as sandbox_module

    _print_sandbox(get_sandbox())
    available = sandbox_module.sandbox_available()
    if available is None:
        console.print("[dim]No sandbox backend detected on this platform.[/dim]")
    else:
        backend, path = available
        console.print(f"Backend: [cyan]{backend}[/cyan] ({path})")


@sandbox_app.command("enable")
def sandbox_enable(
    state: Annotated[
        str | None,
        typer.Argument(help="'on' or 'off' (omit to show the current value)"),
    ] = None,
) -> None:
    """Enable or disable OS-level sandboxing for shell commands."""
    if state is None:
        current = "on" if get_sandbox()["enabled"] else "off"
        console.print(f"Sandbox: [cyan]{current}[/cyan]")
        return
    settings = set_sandbox(enabled=_parse_on_off(state))
    console.print(
        f"[green]{_check()} Sandbox:[/green] "
        f"{'on' if settings['enabled'] else 'off'}"
    )


@sandbox_app.command("network")
def sandbox_network(
    state: Annotated[
        str | None,
        typer.Argument(help="'on' or 'off' (omit to show the current value)"),
    ] = None,
) -> None:
    """Allow or block network access inside the sandbox."""
    if state is None:
        current = "on" if get_sandbox()["network"] else "off"
        console.print(f"Sandbox network: [cyan]{current}[/cyan]")
        return
    settings = set_sandbox(network=_parse_on_off(state))
    console.print(
        f"[green]{_check()} Sandbox network:[/green] "
        f"{'on' if settings['network'] else 'off'}"
    )


@sandbox_app.command("add-path")
def sandbox_add_path(
    path: Annotated[
        str, typer.Argument(help="Path to make writable inside the sandbox")
    ],
) -> None:
    """Add an extra writable path to the sandbox profile."""
    paths = list(get_sandbox()["extra_writable"])
    if path not in paths:
        paths.append(path)
    try:
        set_sandbox(extra_writable=paths)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]{_check()} Writable in sandbox:[/green] {path}")


@sandbox_app.command("remove-path")
def sandbox_remove_path(
    path: Annotated[str, typer.Argument(help="Path to remove")],
) -> None:
    """Remove an extra writable path from the sandbox profile."""
    current = get_sandbox()["extra_writable"]
    paths = [item for item in current if item != path]
    if len(paths) == len(current):
        console.print(f"[dim]No such writable path: {path}[/dim]")
        return
    set_sandbox(extra_writable=paths)
    console.print(f"[green]{_check()} Removed writable path:[/green] {path}")


@sandbox_app.command("clear-paths")
def sandbox_clear_paths() -> None:
    """Remove every extra writable path."""
    set_sandbox(extra_writable=[])
    console.print(f"[green]{_check()} Sandbox extra writable paths cleared.[/green]")


# --- canonical `config remote ...` ------------------------------------------


remote_app = typer.Typer(help="Run shell commands over SSH or in a devcontainer.")
config_app.add_typer(remote_app, name="remote")


def _remote_address(settings: dict[str, Any]) -> str:
    host = str(settings["host"] or "")
    if not host:
        return "(none)"
    address = f"{settings['user']}@{host}" if settings["user"] else host
    if settings["port"] != 22:
        address += f":{settings['port']}"
    return address


def _print_remote(settings: dict[str, Any]) -> None:
    console.print(
        f"Remote: [cyan]{'on' if settings['enabled'] else 'off'}[/cyan]\n"
        f"Host: [cyan]{escape(_remote_address(settings))}[/cyan]\n"
        f"Identity: [cyan]{escape(settings['identity'] or '(default)')}[/cyan]\n"
        f"Workspace: [cyan]{escape(settings['workspace'] or '(remote default)')}[/cyan]\n"
        f"Devcontainer: [cyan]{escape(settings['devcontainer'])}[/cyan]"
    )


@remote_app.command("show")
def remote_config_show() -> None:
    """Show remote settings and any devcontainer detected in this directory."""
    from kiwimatecoder import remote as remote_module

    _print_remote(get_remote())
    info = remote_module.detect_devcontainer(Path.cwd())
    if info is not None:
        console.print(
            f"Detected devcontainer: [cyan]{escape(info.name)}[/cyan] "
            f"(workspace folder [cyan]{escape(info.workspace_folder)}[/cyan])"
        )


@remote_app.command("enable")
def remote_config_enable(
    state: Annotated[
        str | None,
        typer.Argument(help="'on' or 'off' (omit to show the current value)"),
    ] = None,
) -> None:
    """Enable or disable remote/devcontainer command execution."""
    if state is None:
        current = "on" if get_remote()["enabled"] else "off"
        console.print(f"Remote: [cyan]{current}[/cyan]")
        return
    try:
        settings = set_remote(enabled=_parse_on_off(state))
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Remote:[/green] "
        f"{'on' if settings['enabled'] else 'off'}"
    )


@remote_app.command("host")
def remote_config_host(
    value: Annotated[
        str | None, typer.Argument(help="SSH host (omit to show the current value)")
    ] = None,
) -> None:
    """Set the SSH host remote commands run on."""
    if value is None:
        console.print(
            f"Remote host: [cyan]{escape(get_remote()['host'] or '(none)')}[/cyan]"
        )
        return
    try:
        settings = set_remote(host=value)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Remote host:[/green] "
        f"{escape(settings['host'] or '(none)')}"
    )


@remote_app.command("user")
def remote_config_user(
    value: Annotated[
        str | None, typer.Argument(help="SSH user (omit to show the current value)")
    ] = None,
) -> None:
    """Set the SSH user remote commands run as."""
    if value is None:
        console.print(
            f"Remote user: [cyan]{escape(get_remote()['user'] or '(none)')}[/cyan]"
        )
        return
    try:
        settings = set_remote(user=value)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Remote user:[/green] "
        f"{escape(settings['user'] or '(none)')}"
    )


@remote_app.command("port")
def remote_config_port(
    value: Annotated[
        str | None, typer.Argument(help="SSH port 1-65535 (omit to show)")
    ] = None,
) -> None:
    """Set the SSH port (default 22)."""
    if value is None:
        console.print(f"Remote port: [cyan]{get_remote()['port']}[/cyan]")
        return
    try:
        settings = set_remote(port=value)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]{_check()} Remote port:[/green] {settings['port']}")


@remote_app.command("identity")
def remote_config_identity(
    value: Annotated[
        str | None,
        typer.Argument(help="SSH identity file (omit to show the current value)"),
    ] = None,
) -> None:
    """Set the SSH private key used for remote commands."""
    if value is None:
        console.print(
            f"Remote identity: [cyan]{escape(get_remote()['identity'] or '(default)')}[/cyan]"
        )
        return
    try:
        settings = set_remote(identity=value)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Remote identity:[/green] "
        f"{escape(settings['identity'] or '(default)')}"
    )


@remote_app.command("workspace")
def remote_config_workspace(
    value: Annotated[
        str | None,
        typer.Argument(help="Remote working directory (omit to show the current value)"),
    ] = None,
) -> None:
    """Set the directory remote commands run in."""
    if value is None:
        console.print(
            "Remote workspace: "
            f"[cyan]{escape(get_remote()['workspace'] or '(remote default)')}[/cyan]"
        )
        return
    try:
        settings = set_remote(workspace=value)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Remote workspace:[/green] "
        f"{escape(settings['workspace'] or '(remote default)')}"
    )


@remote_app.command("devcontainer")
def remote_config_devcontainer(
    value: Annotated[
        str | None,
        typer.Argument(
            help="'auto', 'off', or a container name (omit to show the current value)"
        ),
    ] = None,
) -> None:
    """Choose how a detected devcontainer is used for command execution."""
    if value is None:
        console.print(f"Devcontainer: [cyan]{escape(get_remote()['devcontainer'])}[/cyan]")
        return
    try:
        settings = set_remote(devcontainer=value)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Devcontainer:[/green] {escape(settings['devcontainer'])}"
    )


# --- canonical `config acp ...` ---------------------------------------------


acp_config_app = typer.Typer(help="Agent Client Protocol (editor integration) settings.")
config_app.add_typer(acp_config_app, name="acp")


@acp_config_app.command("show")
def acp_config_show() -> None:
    """Show ACP server settings."""
    settings = get_acp()
    console.print(
        f"ACP permission timeout: [cyan]{settings['permission_timeout']}s[/cyan]"
    )


@acp_config_app.command("timeout")
def acp_config_timeout(
    seconds: Annotated[
        int | None,
        typer.Argument(help="Seconds to wait for the editor's approval (omit to show)"),
    ] = None,
) -> None:
    """Set how long the editor has to answer a permission request."""
    if seconds is None:
        console.print(
            f"ACP permission timeout: [cyan]{get_acp()['permission_timeout']}s[/cyan]"
        )
        return
    try:
        settings = set_acp(permission_timeout=seconds)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} ACP permission timeout:[/green] "
        f"{settings['permission_timeout']}s"
    )


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
            f"[yellow]{_check()} Trusted workspace on:[/yellow] read-only tools may read "
            "outside the workspace root; writes stay sandboxed."
        )
    elif token in {"off", "false", "disable", "disabled"}:
        set_trusted_workspace(False)
        console.print(f"[green]{_check()} Trusted workspace off.[/green]")
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
        console.print(f"[green]{_check()} Auto-verify set to:[/green] {command}")
        return
    if action in {"clear", "reset", "off"}:
        set_verify_command("")
        console.print(f"[green]{_check()} Auto-verify disabled.[/green]")
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
        console.print(f"[green]{_check()} Budget limits cleared.[/green]")
        return
    if action in {"tokens", "token"}:
        limit = None if value in (None, "clear", "none", "off") else value
        try:
            set_budget(max_tokens=limit)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        console.print(f"[green]{_check()} Token budget:[/green] {get_budget().get('max_tokens')}")
        return
    if action in {"cost", "usd"}:
        limit = None if value in (None, "clear", "none", "off") else value
        try:
            set_budget(max_cost_usd=limit)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        console.print(
            f"[green]{_check()} Cost budget:[/green] ${get_budget().get('max_cost_usd')}"
        )
        return
    console.print("[red]Expected 'show', 'tokens', 'cost', or 'clear'.[/red]")
    raise typer.Exit(1)


@config_app.command("cache")
def cache_cmd(
    state: Annotated[
        str | None,
        typer.Argument(help="'on' or 'off' (omit to show the current value)"),
    ] = None,
) -> None:
    """Toggle prompt caching for native Anthropic providers."""
    if state is None:
        current = "on" if get_prompt_cache() else "off"
        console.print(f"Prompt caching: [cyan]{current}[/cyan]")
        return
    token = state.strip().lower()
    if token in {"on", "true", "enable", "enabled"}:
        set_prompt_cache(True)
        console.print(
            f"[green]{_check()} Prompt caching on:[/green] native Anthropic requests mark "
            "the system prompt and tool definitions as cache breakpoints."
        )
    elif token in {"off", "false", "disable", "disabled"}:
        set_prompt_cache(False)
        console.print(f"[green]{_check()} Prompt caching off.[/green]")
    else:
        console.print("[red]Expected 'on' or 'off'.[/red]")
        raise typer.Exit(1)


# --- canonical `config web ...` ---------------------------------------------

web_app = typer.Typer(help="Web fetch/search settings.")
config_app.add_typer(web_app, name="web")


def _print_web(settings: dict[str, object]) -> None:
    console.print(
        f"Web max chars: [cyan]{settings['max_chars']}[/cyan]\n"
        f"Web timeout: [cyan]{settings['timeout']}s[/cyan]\n"
        f"Allow local addresses: "
        f"[cyan]{'on' if settings['allow_local'] else 'off'}[/cyan]\n"
        f"Search provider: [cyan]{settings['search_provider']}[/cyan]"
    )


@web_app.command("show")
def web_show() -> None:
    """Show web fetch/search limits and local-address access."""
    _print_web(get_web())


@web_app.command("max-chars")
def web_max_chars(
    value: Annotated[
        str | None,
        typer.Argument(help="Maximum characters to return (omit to show the current value)"),
    ] = None,
) -> None:
    """Set the maximum characters web_fetch returns."""
    if value is None:
        console.print(f"Web max chars: [cyan]{get_web()['max_chars']}[/cyan]")
        return
    try:
        settings = set_web(max_chars=value)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]{_check()} Web max chars:[/green] {settings['max_chars']}")


@web_app.command("timeout")
def web_timeout(
    value: Annotated[
        str | None,
        typer.Argument(help="Timeout in seconds (omit to show the current value)"),
    ] = None,
) -> None:
    """Set the web fetch/search timeout in seconds."""
    if value is None:
        console.print(f"Web timeout: [cyan]{get_web()['timeout']:g}s[/cyan]")
        return
    try:
        settings = set_web(timeout=value)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]{_check()} Web timeout:[/green] {settings['timeout']:g}s")


@web_app.command("allow-local")
def web_allow_local(
    state: Annotated[
        str | None,
        typer.Argument(help="'on' or 'off' (omit to show the current value)"),
    ] = None,
) -> None:
    """Allow web_fetch to reach local/private addresses (default off)."""
    if state is None:
        current = "on" if get_web()["allow_local"] else "off"
        console.print(f"Allow local addresses: [cyan]{current}[/cyan]")
        return
    token = state.strip().lower()
    if token in {"on", "true", "enable", "enabled"}:
        set_web(allow_local=True)
        console.print(
            f"[yellow]{_check()} Allow local addresses on:[/yellow] web_fetch may "
            "reach localhost and private network hosts."
        )
    elif token in {"off", "false", "disable", "disabled"}:
        set_web(allow_local=False)
        console.print(f"[green]{_check()} Allow local addresses off.[/green]")
    else:
        console.print("[red]Expected 'on' or 'off'.[/red]")
        raise typer.Exit(1)


# --- canonical `config network ...` -----------------------------------------

network_app = typer.Typer(help="Proxy, custom CA, and offline mode.")
config_app.add_typer(network_app, name="network")

_NETWORK_CLEAR_TOKENS = {"clear", "none", "off", "-", "default"}


def _is_network_clear(value: str) -> bool:
    return value.strip().lower() in _NETWORK_CLEAR_TOKENS


def _print_network(settings: dict[str, Any]) -> None:
    console.print(
        f"Proxy: [cyan]{settings['proxy'] or 'none'}[/cyan]\n"
        f"CA bundle: [cyan]{settings['ca_bundle'] or 'system default'}[/cyan]\n"
        f"Offline mode: [cyan]{'on' if settings['offline'] else 'off'}[/cyan]"
    )


@network_app.command("show")
def network_show() -> None:
    """Show proxy, custom CA bundle, and offline-mode settings."""
    _print_network(get_network())


@network_app.command("proxy")
def network_proxy(
    value: Annotated[
        str | None,
        typer.Argument(help="Proxy URL, 'clear', or omit to show the current value"),
    ] = None,
) -> None:
    """Set or clear the HTTP(S) proxy used for every request."""
    if value is None:
        console.print(f"Proxy: [cyan]{get_network()['proxy'] or 'none'}[/cyan]")
        return
    cleaned = "" if _is_network_clear(value) else value
    try:
        settings = set_network(proxy=cleaned)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]{_check()} Proxy:[/green] {settings['proxy'] or 'none'}")


@network_app.command("ca")
def network_ca(
    value: Annotated[
        str | None,
        typer.Argument(
            help="CA bundle path, 'clear', or omit to show the current value"
        ),
    ] = None,
) -> None:
    """Set or clear the custom CA bundle used for TLS verification."""
    if value is None:
        console.print(
            f"CA bundle: [cyan]{get_network()['ca_bundle'] or 'system default'}[/cyan]"
        )
        return
    cleaned = "" if _is_network_clear(value) else value
    try:
        settings = set_network(ca_bundle=cleaned)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} CA bundle:[/green] "
        f"{settings['ca_bundle'] or 'system default'}"
    )


@network_app.command("offline")
def network_offline(
    state: Annotated[
        str | None,
        typer.Argument(help="'on' or 'off' (omit to show the current value)"),
    ] = None,
) -> None:
    """Block cloud requests; local providers keep working."""
    if state is None:
        current = "on" if get_network()["offline"] else "off"
        console.print(f"Offline mode: [cyan]{current}[/cyan]")
        return
    token = state.strip().lower()
    if token in {"on", "true", "enable", "enabled", "yes"}:
        set_network(offline=True)
        console.print(
            f"[yellow]{_check()} Offline mode on:[/yellow] cloud requests and web "
            "tools are blocked; local providers still work."
        )
    elif token in {"off", "false", "disable", "disabled", "no"}:
        set_network(offline=False)
        console.print(f"[green]{_check()} Offline mode off.[/green]")
    else:
        console.print("[red]Expected 'on' or 'off'.[/red]")
        raise typer.Exit(1)


# --- canonical `config index ...` -------------------------------------------

index_app = typer.Typer(help="Codebase index and semantic search settings.")
config_app.add_typer(index_app, name="index")


@index_app.command("show")
def index_show() -> None:
    """Show the index status for the current directory."""
    from kiwimatecoder.index.store import index_status

    settings = get_index()
    status = index_status(Path.cwd())
    embeddings = settings["embeddings"]
    if embeddings["provider"] and embeddings["model"]:
        embed_line = f"[cyan]on[/cyan] ({embeddings['provider']}:{embeddings['model']})"
    else:
        embed_line = "[cyan]off[/cyan]"
    console.print(
        f"Index: [cyan]{'on' if status.enabled else 'off'}[/cyan]\n"
        f"Files indexed: [cyan]{status.files}[/cyan] "
        f"([yellow]{status.stale}[/yellow] stale)\n"
        f"Terms: [cyan]{status.terms}[/cyan]\n"
        f"Store: [cyan]{status.path}[/cyan] ({status.size_bytes} bytes)\n"
        f"Embeddings: {embed_line}"
    )


@index_app.command("embed-provider")
def index_embed_provider(
    provider: Annotated[
        str,
        typer.Argument(help="Provider id ('none' or empty clears it)"),
    ] = "",
) -> None:
    """Set the OpenAI-compatible provider used for embeddings."""
    cleaned = provider.strip()
    if cleaned.lower() in {"", "none", "off", "-", "clear"}:
        cleaned = ""
    try:
        settings = set_index(embed_provider=cleaned)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Embedding provider:[/green] "
        f"[cyan]{settings['embeddings']['provider'] or 'off'}[/cyan]"
    )


@index_app.command("embed-model")
def index_embed_model(
    model: Annotated[str, typer.Argument(help="Embedding model id")],
) -> None:
    """Set the embedding model id (empty clears it)."""
    cleaned = model.strip()
    if cleaned.lower() in {"none", "off", "-", "clear"}:
        cleaned = ""
    settings = set_index(embed_model=cleaned)
    console.print(
        f"[green]{_check()} Embedding model:[/green] "
        f"[cyan]{settings['embeddings']['model'] or 'off'}[/cyan]"
    )


@index_app.command("clear")
def index_clear() -> None:
    """Delete the index for the current directory."""
    from kiwimatecoder.index.store import clear_store

    if clear_store(Path.cwd()):
        console.print(f"[green]{_check()} Codebase index cleared.[/green]")
    else:
        console.print("[dim]No codebase index to clear.[/dim]")


# --- canonical `config memory ...` ------------------------------------------

memory_app = typer.Typer(help="Persistent project and user memory settings.")
config_app.add_typer(memory_app, name="memory")


def _print_memory(settings: dict[str, object]) -> None:
    console.print(
        f"Memory: [cyan]{'on' if settings['enabled'] else 'off'}[/cyan]\n"
        f"Max bytes in the prompt: [cyan]{settings['max_bytes']}[/cyan]"
    )


@memory_app.command("show")
def memory_show() -> None:
    """Show whether memory is enabled and the prompt byte budget."""
    _print_memory(get_memory())


@memory_app.command("max-bytes")
def memory_max_bytes(
    value: Annotated[
        str | None,
        typer.Argument(help="Prompt byte budget (omit to show the current value)"),
    ] = None,
) -> None:
    """Set the maximum memory bytes included in the system prompt."""
    if value is None:
        console.print(
            f"Max memory bytes: [cyan]{get_memory()['max_bytes']}[/cyan]"
        )
        return
    try:
        settings = set_memory(max_bytes=value)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Max memory bytes:[/green] {settings['max_bytes']}"
    )


@memory_app.command("enable")
def memory_enable(
    state: Annotated[
        str | None,
        typer.Argument(help="'on' or 'off' (omit to show the current value)"),
    ] = None,
) -> None:
    """Enable or disable persistent memory in the system prompt."""
    if state is None:
        console.print(
            f"Memory: [cyan]{'on' if get_memory()['enabled'] else 'off'}[/cyan]"
        )
        return
    token = state.strip().lower()
    if token in {"on", "true", "enable", "enabled"}:
        set_memory(enabled=True)
        console.print(f"[green]{_check()} Memory on.[/green]")
    elif token in {"off", "false", "disable", "disabled"}:
        set_memory(enabled=False)
        console.print(f"[green]{_check()} Memory off.[/green]")
    else:
        console.print("[red]Expected 'on' or 'off'.[/red]")
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
    console.print(f"[green]{_check()} Sampling set:[/green] {effective}")


@sampling_app.command("reset")
def sampling_reset() -> None:
    """Reset sampling parameters to provider defaults."""
    reset_sampling()
    console.print(f"[green]{_check()} Sampling reset to provider defaults.[/green]")


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
    console.print(f"[green]{_check()} Output style set to [cyan]{effective}[/cyan].[/green]")


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
    console.print(f"[green]{_check()} Custom system prompt saved.[/green]")


@prompt_app.command("clear")
def prompt_clear() -> None:
    """Clear the custom system-prompt addition."""
    set_system_prompt(None)
    console.print(f"[green]{_check()} Custom system prompt cleared.[/green]")


# --- canonical `config ui ...` ----------------------------------------------

ui_app = typer.Typer(
    help="Themes, output modes, color control, and accessibility settings."
)
config_app.add_typer(ui_app, name="ui")


@ui_app.command("show")
def ui_show() -> None:
    """Show the theme, color, output, and ASCII settings."""
    _print_ui(get_ui())


@ui_app.command("color")
def ui_color(
    value: Annotated[str, typer.Argument(help="auto, always, or never")],
) -> None:
    """Control color output: auto, always, or never."""
    try:
        current = set_ui(color=value)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Color mode set to [cyan]{current['color']}[/cyan].[/green]"
    )


@ui_app.command("output")
def ui_output(
    value: Annotated[str, typer.Argument(help="normal, compact, or verbose")],
) -> None:
    """Set how much the agent prints as tools run."""
    try:
        current = set_ui(output_mode=value)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Output mode set to "
        f"[cyan]{current['output_mode']}[/cyan].[/green]"
    )


@ui_app.command("ascii")
def ui_ascii(
    state: Annotated[str, typer.Argument(help="'on' or 'off'")],
) -> None:
    """Enable or disable ASCII mode for screen readers and plain terminals."""
    token = state.strip().lower()
    if token in {"on", "true", "yes", "enable", "enabled"}:
        enabled = True
    elif token in {"off", "false", "no", "disable", "disabled"}:
        enabled = False
    else:
        console.print("[red]Expected 'on' or 'off'.[/red]")
        raise typer.Exit(1)
    set_ui(ascii=enabled)
    console.print(
        f"[green]{_check()} ASCII mode "
        f"{'on' if enabled else 'off'}.[/green]"
    )


@ui_app.command("theme")
def ui_theme(
    value: Annotated[str, typer.Argument(help="default, ocean, magenta, or mono")],
) -> None:
    """Set the visual theme used for accents and the banner."""
    try:
        current = set_ui(theme=value)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Theme set to [cyan]{current['theme']}[/cyan].[/green]"
    )


# --- canonical `config profile ...` -----------------------------------------

profile_app = typer.Typer(help="Save, apply, or remove named configuration profiles.")
config_app.add_typer(profile_app, name="profile")


@profile_app.command("list")
def profile_list() -> None:
    """List saved configuration profiles."""
    profiles = get_profiles()
    if not profiles:
        console.print(
            "[dim]No profiles saved. Capture one with "
            "`config profile save <name>`.[/dim]"
        )
        return
    table = Table(title="Profiles", show_header=True)
    table.add_column("name", style="cyan")
    table.add_column("provider")
    table.add_column("model")
    table.add_column("mode")
    for name in sorted(profiles):
        values = profiles[name]
        table.add_row(
            name,
            str(values.get("provider") or ""),
            str(values.get("model") or "(provider default)"),
            str(values.get("mode") or ""),
        )
    console.print(table)


@profile_app.command("show")
def profile_show(name: Annotated[str, typer.Argument(help="Profile name")]) -> None:
    """Show one profile's settings."""
    profile = get_profile(name)
    if profile is None:
        console.print(f"[red]Unknown profile '{name}'.[/red]")
        raise typer.Exit(1)
    console.print(f"[bold]{name}[/bold]")
    console.print_json(data=profile)


@profile_app.command("save")
def profile_save_cmd(
    name: Annotated[str, typer.Argument(help="Profile name")],
) -> None:
    """Capture the current effective settings under ``name``."""
    try:
        profile = save_profile(name)
    except (ValueError, KeyError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Saved profile [cyan]{name}[/cyan][/green] "
        f"({len(profile)} setting(s))."
    )


@profile_app.command("use")
def profile_use(name: Annotated[str, typer.Argument(help="Profile name")]) -> None:
    """Apply a profile's settings to the global config."""
    try:
        apply_profile(name)
    except (ValueError, KeyError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    cfg = load_config()
    provider_id = get_selected_provider_id(cfg)
    console.print(
        f"[green]{_check()} Applied profile [cyan]{name}[/cyan][/green] — "
        f"provider: [cyan]{provider_id}[/cyan], "
        f"model: [cyan]{cfg.get('selected_model') or '(provider default)'}[/cyan], "
        f"mode: [cyan]{get_default_mode(cfg)}[/cyan]."
    )


@profile_app.command("remove")
def profile_remove(name: Annotated[str, typer.Argument(help="Profile name")]) -> None:
    """Remove a saved profile."""
    if remove_profile(name):
        console.print(f"[green]{_check()} Removed profile {name}.[/green]")
    else:
        console.print(f"[dim]No profile named {name}.[/dim]")


@profile_app.command("rename")
def profile_rename(
    old: Annotated[str, typer.Argument(help="Current name")],
    new: Annotated[str, typer.Argument(help="New name")],
) -> None:
    """Rename a saved profile."""
    try:
        renamed = rename_profile(old, new)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    if renamed:
        console.print(f"[green]{_check()} Renamed profile {old} to {new}.[/green]")
    else:
        console.print(f"[dim]No profile named {old}.[/dim]")


@config_app.command("validate")
def validate_cmd() -> None:
    """Validate the stored configuration; exit 1 when errors are found."""
    issues = validate_config()
    if not issues:
        console.print(f"[green]{_check()} Configuration is valid.[/green]")
        return
    table = Table(title="Configuration issues", show_header=True)
    table.add_column("level", style="cyan")
    table.add_column("key")
    table.add_column("message")
    for issue in issues:
        color = "red" if issue["level"] == "error" else "yellow"
        table.add_row(
            f"[{color}]{issue['level']}[/{color}]", issue["key"], issue["message"]
        )
    console.print(table)
    errors = sum(1 for issue in issues if issue["level"] == "error")
    warnings = len(issues) - errors
    console.print(f"[dim]{errors} error(s), {warnings} warning(s).[/dim]")
    if errors:
        raise typer.Exit(1)


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
    subagents = get_subagents(cfg)
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
        + "\nPrompt caching: "
        + f"[cyan]{'on' if get_prompt_cache(cfg) else 'off'}[/cyan]"
        + "\nSubagents: "
        + f"[cyan]{'on' if subagents['enabled'] else 'off'}[/cyan] "
        + f"(max steps {subagents['max_steps']})"
    )
    ui_config = get_ui(cfg)
    console.print(
        "UI: "
        + f"[cyan]{ui_config['theme']}[/cyan] theme, "
        + f"[cyan]{ui_config['color']}[/cyan] color, "
        + f"[cyan]{ui_config['output_mode']}[/cyan] output, "
        + f"ascii [cyan]{'on' if ui_config['ascii'] else 'off'}[/cyan]"
    )
    network_config = get_network(cfg)
    console.print(
        "Network: proxy "
        + f"[cyan]{network_config['proxy'] or 'none'}[/cyan], "
        + f"CA [cyan]{network_config['ca_bundle'] or 'system'}[/cyan], "
        + f"offline [cyan]{'on' if network_config['offline'] else 'off'}[/cyan]"
    )
    remote_config = get_remote(cfg)
    console.print(
        "Remote: "
        + f"[cyan]{'on' if remote_config['enabled'] else 'off'}[/cyan] "
        + f"(host [cyan]{escape(_remote_address(remote_config))}[/cyan], "
        + f"devcontainer [cyan]{escape(remote_config['devcontainer'])}[/cyan])"
    )
    media_config = get_media(cfg)
    console.print(
        "Media: "
        + f"[cyan]{'on' if media_config['enabled'] else 'off'}[/cyan] "
        + f"(provider [cyan]{media_config['provider']}[/cyan], "
        + f"model [cyan]{media_config['model']}[/cyan], "
        + f"size [cyan]{media_config['size']}[/cyan], "
        + f"output [cyan]{media_config['output_dir']}[/cyan])"
    )
    telemetry_config = get_telemetry(cfg)
    console.print(
        "Telemetry: "
        + f"[cyan]{'on' if telemetry_config['enabled'] else 'off'}[/cyan] "
        + f"(level [cyan]{telemetry_config['level']}[/cyan])"
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
        f"[green]{_check()} Now {verb} these models for {pid}:[/green] "
        + ", ".join(models)
    )


# --- `config media ...` -----------------------------------------------------


def _print_media(settings: dict[str, Any]) -> None:
    console.print(
        f"Media: [cyan]{'on' if settings['enabled'] else 'off'}[/cyan]\n"
        f"Provider: [cyan]{settings['provider']}[/cyan]\n"
        f"Model: [cyan]{settings['model']}[/cyan]\n"
        f"Size: [cyan]{settings['size']}[/cyan]\n"
        f"Output dir: [cyan]{settings['output_dir']}[/cyan]"
    )


@media_app.command("show")
def media_show() -> None:
    """Show image-generation settings."""
    _print_media(get_media())


@media_app.command("enable")
def media_enable(
    mode: Annotated[str, typer.Argument(help="on or off")] = "on",
) -> None:
    """Enable or disable image generation (on|off)."""
    token = mode.strip().lower()
    if token in {"on", "true", "yes", "enable", "enabled"}:
        enabled = True
    elif token in {"off", "false", "no", "disable", "disabled"}:
        enabled = False
    else:
        console.print("[red]Usage: config media enable <on|off>[/red]")
        raise typer.Exit(1)
    try:
        settings = set_media(enabled=enabled)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Image generation "
        f"{'enabled' if settings['enabled'] else 'disabled'}.[/green]"
    )


@media_app.command("model")
def media_model(model: Annotated[str, typer.Argument(help="Image model id")]) -> None:
    """Set the model used for image generation."""
    try:
        set_media(model=model)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Media model set to[/green] [cyan]{model}[/cyan]."
    )


@media_app.command("provider")
def media_provider(
    provider: Annotated[str, typer.Argument(help="Provider id")],
) -> None:
    """Set the provider used for image generation."""
    try:
        set_media(provider=provider)
    except (ValueError, KeyError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Media provider set to[/green] [cyan]{provider}[/cyan]."
    )


@media_app.command("size")
def media_size(size: Annotated[str, typer.Argument(help="Image size as WxH")]) -> None:
    """Set the requested image size (WxH)."""
    try:
        set_media(size=size)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Media size set to[/green] [cyan]{size}[/cyan]."
    )


# --- `config telemetry ...` -------------------------------------------------


def _print_telemetry(settings: dict[str, Any]) -> None:
    from kiwimatecoder import telemetry

    console.print(
        f"Telemetry: [cyan]{'on' if settings['enabled'] else 'off'}[/cyan]\n"
        f"Level: [cyan]{settings['level']}[/cyan]\n"
        f"Log file: [cyan]{telemetry.current_log_path()}[/cyan]\n"
        f"Max log bytes: [cyan]{settings['max_log_bytes']:,}[/cyan]\n"
        f"Crash reports: [cyan]{len(telemetry.crash_report_paths())}[/cyan]"
    )


@telemetry_app.command("show")
def telemetry_show() -> None:
    """Show telemetry settings, log path, and crash-report count."""
    _print_telemetry(get_telemetry())


@telemetry_app.command("enable")
def telemetry_enable(
    mode: Annotated[str, typer.Argument(help="on or off")] = "on",
) -> None:
    """Enable or disable telemetry (on|off)."""
    token = mode.strip().lower()
    if token in {"on", "true", "yes", "enable", "enabled"}:
        enabled = True
    elif token in {"off", "false", "no", "disable", "disabled"}:
        enabled = False
    else:
        console.print("[red]Usage: config telemetry enable <on|off>[/red]")
        raise typer.Exit(1)
    try:
        settings = set_telemetry(enabled=enabled)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Telemetry "
        f"{'enabled' if settings['enabled'] else 'disabled'}.[/green]"
    )
    if settings["enabled"] and settings["level"] == "off":
        console.print(
            "[dim]Set a level with `config telemetry level error|info|debug` "
            "to start recording.[/dim]"
        )


@telemetry_app.command("level")
def telemetry_level(
    level: Annotated[str, typer.Argument(help="off, error, info, or debug")],
) -> None:
    """Set the telemetry level (off|error|info|debug)."""
    try:
        settings = set_telemetry(level=level)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Telemetry level set to[/green] "
        f"[cyan]{settings['level']}[/cyan]."
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


HEADLESS_OUTPUT_FORMATS = ("text", "json", "stream-json")


def _headless_error(message: str) -> None:
    sys.stderr.write(f"error: {message}\n")


def _run_headless(
    prompt: str,
    *,
    output_format: str,
    mode: str | None,
    yes: bool,
    max_turns: int,
    provider: str | None,
    model: str | None,
    workspace: Path | None,
    quiet: bool,
) -> int:
    """Run one headless agent turn; returns the process exit code (0/1/2)."""
    if output_format not in HEADLESS_OUTPUT_FORMATS:
        _headless_error(
            f"Invalid --output-format '{output_format}'. "
            "Choose: text, json, stream-json."
        )
        return 2
    if max_turns < 1:
        _headless_error("--max-turns must be at least 1.")
        return 2
    if not prompt.strip():
        _headless_error("Prompt is empty.")
        return 2
    try:
        resolved_mode: PermissionMode | None = (
            PermissionMode.from_str(mode) if mode is not None else None
        )
    except ValueError as exc:
        _headless_error(str(exc))
        return 2
    if resolved_mode is None and yes:
        # --yes is shorthand for auto-accept when --mode is not explicit.
        resolved_mode = PermissionMode.AUTO

    if provider is not None:
        try:
            get_provider_config(provider)
        except KeyError:
            _headless_error(f"Unknown provider '{provider}'.")
            return 2

    try:
        workspace_root = (workspace or Path.cwd()).expanduser()
    except (OSError, RuntimeError) as exc:
        _headless_error(str(exc))
        return 2
    if not workspace_root.is_dir():
        _headless_error(f"Workspace is not a directory: {workspace_root}")
        return 2

    from kiwimatecoder import headless

    stderr_console = Console(file=sys.stderr, no_color=True, highlight=False)
    on_event: Callable[[str, dict[str, Any]], None] | None = None
    if output_format == "text":
        console = stderr_console

        def render_text(name: str, payload: dict[str, Any]) -> None:
            if name == "text_delta":
                sys.stdout.write(str(payload.get("text") or ""))
                sys.stdout.flush()

        on_event = render_text

    elif output_format == "stream-json":
        console = None

        def render_stream_json(name: str, payload: dict[str, Any]) -> None:
            record: dict[str, Any] = {"type": name}
            record.update(payload)
            sys.stdout.write(
                json.dumps(record, separators=(",", ":"), default=str) + "\n"
            )
            sys.stdout.flush()

        on_event = render_stream_json

    else:
        console = None

    def deny(summary: str, preview: str | None) -> bool:
        """Deny approvals that cannot be prompted for, with a visible note."""
        sys.stderr.write(
            f"denied (headless mode has no approval prompt; pass --yes to "
            f"allow): {summary}\n"
        )
        return False

    outcome = asyncio.run(
        headless.run_agent_once(
            prompt,
            workspace=workspace_root,
            provider=provider,
            model=model,
            mode=resolved_mode,
            confirm=deny,
            on_event=on_event,
            max_turns=max_turns,
            console=console,
            render_text=False,
            output_mode="compact" if quiet and output_format == "text" else None,
        )
    )

    payload = {
        "result": outcome.text,
        "usage": outcome.usage,
        "cost_usd": outcome.cost_usd,
        "provider": outcome.provider,
        "model": outcome.model,
        "mode": outcome.mode,
        "tools_used": outcome.tools_used,
        "messages": outcome.messages,
        "success": outcome.success,
    }
    if output_format == "json":
        sys.stdout.write(
            json.dumps(payload, separators=(",", ":"), default=str) + "\n"
        )
    elif output_format == "stream-json":
        record: dict[str, Any] = {"type": "result"}
        record.update(payload)
        sys.stdout.write(
            json.dumps(record, separators=(",", ":"), default=str) + "\n"
        )
    else:
        sys.stdout.write("\n")
    if outcome.error and output_format != "text":
        sys.stderr.write(f"error: {outcome.error}\n")
    return 0 if outcome.success else 1


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
    profile: Annotated[
        str | None,
        typer.Option(
            "--profile",
            help="Apply a saved config profile for this session (not persisted).",
        ),
    ] = None,
    print_prompt: Annotated[
        str | None,
        typer.Option(
            "--print",
            "-p",
            help="Run one headless agent turn with this prompt ('-' reads stdin).",
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--output-format",
            help="Headless output: text, json, or stream-json.",
        ),
    ] = "text",
    mode_override: Annotated[
        str | None,
        typer.Option(
            "--mode",
            help="Override the permission mode: ask, auto-accept, or plan.",
        ),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Treat every approval as granted (auto-accept) without prompting.",
        ),
    ] = False,
    max_turns: Annotated[
        int,
        typer.Option(
            "--max-turns",
            help="Maximum tool-loop iterations before stopping (default 30).",
        ),
    ] = 30,
    provider_override: Annotated[
        str | None,
        typer.Option("--provider", help="Provider id override for this run."),
    ] = None,
    model_override: Annotated[
        str | None,
        typer.Option("--model", help="Model id override for this run."),
    ] = None,
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Workspace root for this run (default: current directory).",
        ),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Suppress tool/progress lines on stderr in headless text mode.",
        ),
    ] = False,
) -> None:
    """Launch the interactive session when run with no subcommand."""
    if version:
        console.print(f"kiwimatecoder {__version__}")
        raise typer.Exit(0)

    if update:
        raise typer.Exit(run_update(console))

    from kiwimatecoder import telemetry

    # Opt-in telemetry configures logging and crash capture for every run path
    # (interactive, headless, and management commands); disabled is a no-op.
    telemetry.configure()

    if ctx.invoked_subcommand is not None:
        return

    if print_prompt is not None:
        prompt = print_prompt
        if prompt == "-":
            prompt = sys.stdin.read()
        raise typer.Exit(
            _run_headless(
                prompt,
                output_format=output_format,
                mode=mode_override,
                yes=yes,
                max_turns=max_turns,
                provider=provider_override,
                model=model_override,
                workspace=workspace,
                quiet=quiet,
            )
        )

    from kiwimatecoder import repl

    session: Session | None = None
    resumed = False
    if resume or continue_:
        from kiwimatecoder.session import load_session

        target = resume or "last"
        try:
            session = load_session(target, workspace_root=Path.cwd())
            resumed = True
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

    # A --profile overlay is session-local: it never writes config. On resume
    # only the mode and model are overlaid so the saved conversation keeps its
    # original provider.
    if profile is not None:
        profile_values = get_profile(profile)
        if profile_values is None:
            console.print(f"[red]Unknown profile '{profile}'.[/red]")
            raise typer.Exit(1)
        if resumed:
            profile_values = {
                key: value
                for key, value in profile_values.items()
                if key in {"mode", "model"}
            }
        apply_session_profile(session, profile_values)

    # The REPL needs a real terminal: prompt_toolkit cannot read piped stdin.
    # Point scripts and CI at the headless path instead of crashing or hanging.
    if not _stdin_is_tty():
        sys.stderr.write(
            "Interactive session needs a TTY. Use: "
            'kiwimatecoder -p "..." or echo ... | kiwimatecoder -p -\n'
        )
        raise typer.Exit(2)

    # repl.run loads user (and opted-in project) plugins before the agent is
    # constructed, turning any failure into a dim warning rather than a crash.
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
            f"[green]{_check()} {provider.name} needs no API key[/green] — "
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
        f"[green]{_check()} API key saved for {provider_id}[/green] "
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


# --- background / scheduled agent jobs --------------------------------------


def _short_prompt(prompt: str, limit: int = 60) -> str:
    text = prompt.replace("\n", " ").strip()
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _print_jobs(records: Sequence[Any]) -> None:
    table = Table(title="Background jobs", show_header=True)
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("status")
    table.add_column("created", no_wrap=True)
    table.add_column("pid", no_wrap=True)
    table.add_column("prompt", overflow="fold")
    for record in records:
        style = {
            "running": "yellow",
            "succeeded": "green",
            "failed": "red",
            "cancelled": "dim",
        }.get(record.status, "white")
        table.add_row(
            record.id,
            f"[{style}]{record.status}[/{style}]",
            record.created_at,
            str(record.pid or ""),
            escape(_short_prompt(record.prompt)),
        )
    console.print(table)


def _print_job(record: Any) -> None:
    console.print(f"ID: [cyan]{record.id}[/cyan]")
    console.print(f"Status: [bold]{record.status}[/bold]")
    console.print(f"Prompt: {escape(record.prompt)}")
    console.print(f"Workspace: {escape(record.workspace)}")
    if record.provider:
        console.print(f"Provider: {record.provider}")
    if record.model:
        console.print(f"Model: {record.model}")
    console.print(f"Mode: {record.mode}")
    console.print(f"PID: {record.pid or '-'}")
    console.print(f"Created: {record.created_at}")
    if record.finished_at:
        console.print(f"Finished: {record.finished_at}")
    if record.exit_code is not None:
        console.print(f"Exit code: {record.exit_code}")
    if record.interval is not None:
        console.print(f"Every: {record.interval:g}s (next: {record.next_run_at})")
    if record.error:
        console.print(f"[red]Error: {escape(record.error)}[/red]")
    if record.result is not None:
        console.print(Panel(escape(record.result or "(empty result)"), title="Result"))
    if record.output_path:
        console.print(f"[dim]Output: {record.output_path}[/dim]")


def _job_start_error(exc: Exception) -> None:
    console.print(f"[red]{exc}[/red]")
    raise typer.Exit(1)


@jobs_app.command("run")
def jobs_run(
    prompt: Annotated[str, typer.Argument(help="Prompt for the detached agent run")],
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", help="Workspace root (default: current directory)"),
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", help="Provider id override")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Model id override")
    ] = None,
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="Permission mode (default: auto-accept, i.e. unattended).",
        ),
    ] = "auto-accept",
) -> None:
    """Start a detached agent job and return immediately."""
    from kiwimatecoder import jobs as jobs_module

    try:
        record = jobs_module.start_job(
            prompt,
            workspace=workspace or Path.cwd(),
            provider=provider,
            model=model,
            mode=mode,
        )
    except (ValueError, KeyError, jobs_module.JobError) as exc:
        _job_start_error(exc)
        return
    console.print(
        f"[green]{_check()} Started job[/green] [cyan]{record.id}[/cyan] "
        f"([dim]{_short_prompt(record.prompt)}[/dim])"
    )
    console.print(
        f"[dim]Watch with `kiwimatecoder jobs show {record.id}` or "
        f"`kiwimatecoder jobs output {record.id}`.[/dim]"
    )


@jobs_app.command("schedule")
def jobs_schedule(
    prompt: Annotated[str, typer.Argument(help="Prompt for the scheduled run")],
    every: Annotated[
        float, typer.Option("--every", help="Seconds between runs")
    ],
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", help="Workspace root (default: current directory)"),
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", help="Provider id override")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Model id override")
    ] = None,
    mode: Annotated[
        str,
        typer.Option("--mode", help="Permission mode (default: auto-accept)."),
    ] = "auto-accept",
) -> None:
    """Start a job now and re-run it whenever `jobs tick` finds it due."""
    from kiwimatecoder import jobs as jobs_module

    try:
        record = jobs_module.schedule_job(
            prompt,
            every,
            workspace=workspace or Path.cwd(),
            provider=provider,
            model=model,
            mode=mode,
        )
    except (ValueError, KeyError, jobs_module.JobError) as exc:
        _job_start_error(exc)
        return
    console.print(
        f"[green]{_check()} Scheduled job[/green] [cyan]{record.id}[/cyan] "
        f"every {record.interval:g}s (next: {record.next_run_at})."
    )
    console.print(
        "[dim]Run `kiwimatecoder jobs tick` (e.g. from cron) to start due runs.[/dim]"
    )


@jobs_app.command("list")
def jobs_list() -> None:
    """List tracked jobs, refreshing running ones first."""
    from kiwimatecoder import jobs as jobs_module

    records = jobs_module.list_jobs(refresh=True)
    if not records:
        console.print("[dim]No background jobs.[/dim]")
        return
    _print_jobs(records)


@jobs_app.command("show")
def jobs_show(job_id: Annotated[str, typer.Argument(help="Job id")]) -> None:
    """Show one job's record and result."""
    from kiwimatecoder import jobs as jobs_module

    record = jobs_module.refresh_job(job_id) or jobs_module.get_job(job_id)
    if record is None:
        console.print(f"[red]Unknown job '{job_id}'.[/red]")
        raise typer.Exit(1)
    _print_job(record)


@jobs_app.command("output")
def jobs_output(job_id: Annotated[str, typer.Argument(help="Job id")]) -> None:
    """Print a job's combined output (bounded)."""
    from kiwimatecoder import jobs as jobs_module

    text = jobs_module.read_job_output(job_id)
    if text is None:
        console.print(f"[red]Unknown job '{job_id}'.[/red]")
        raise typer.Exit(1)
    console.print(text or "(no output yet)", markup=False, highlight=False)


@jobs_app.command("cancel")
def jobs_cancel(job_id: Annotated[str, typer.Argument(help="Job id")]) -> None:
    """Terminate a running job (SIGTERM, then SIGKILL after a grace period)."""
    from kiwimatecoder import jobs as jobs_module

    record = jobs_module.cancel_job(job_id)
    if record is None:
        console.print(f"[red]Unknown job '{job_id}'.[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Job[/green] [cyan]{record.id}[/cyan] is now "
        f"[bold]{record.status}[/bold]."
    )


@jobs_app.command("clear")
def jobs_clear(
    all_: Annotated[
        bool,
        typer.Option("--all", help="Also clear running jobs (terminating them)."),
    ] = False,
) -> None:
    """Delete finished job records and their output logs."""
    from kiwimatecoder import jobs as jobs_module

    removed = jobs_module.clear_jobs(finished_only=not all_)
    console.print(f"[green]{_check()} Cleared {removed} job(s).[/green]")


@jobs_app.command("tick")
def jobs_tick() -> None:
    """Start every scheduled job whose interval has elapsed."""
    from kiwimatecoder import jobs as jobs_module

    started = jobs_module.run_due_jobs()
    if not started:
        console.print("[dim]No jobs are due.[/dim]")
        return
    for record in started:
        console.print(
            f"[green]{_check()} Started[/green] [cyan]{record.id}[/cyan] "
            f"([dim]{_short_prompt(record.prompt)}[/dim])"
        )


@app.command("acp")
def acp_serve(
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Default workspace root for new sessions (default: cwd).",
        ),
    ] = None,
) -> None:
    """Run the Agent Client Protocol server over stdio for editor integration."""
    from kiwimatecoder.acp.stdio import serve_stdio

    raise typer.Exit(serve_stdio(workspace=workspace))


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


# --- cross-machine session sync ---------------------------------------------


def _print_sync_report(report: Any) -> None:
    for line in report.lines:
        console.print(line, markup=False, highlight=False)
    for error in report.errors:
        console.print(f"[yellow]{error}[/yellow]")
    console.print(report.summary())


@sync_app.command("status")
def sync_status_cmd() -> None:
    """Show local/remote session counts and pending changes."""
    from kiwimatecoder import sync as sync_module

    console.print(sync_module.status().summary(), markup=False, highlight=False)


@sync_app.command("push")
def sync_push_cmd(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite shared sessions with local copies, even on conflict.",
        ),
    ] = False,
) -> None:
    """Copy local sessions into the shared folder."""
    from kiwimatecoder import sync as sync_module

    try:
        report = sync_module.push(force=force)
    except sync_module.SyncError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(1)
    _print_sync_report(report)


@sync_app.command("pull")
def sync_pull_cmd(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite local sessions with shared copies, even on conflict.",
        ),
    ] = False,
) -> None:
    """Copy newer shared sessions into the local sessions directory."""
    from kiwimatecoder import sync as sync_module

    try:
        report = sync_module.pull(force=force)
    except sync_module.SyncError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(1)
    _print_sync_report(report)


@sync_app.command("enable")
def sync_enable_cmd(
    path: Annotated[
        Path,
        typer.Argument(help="Existing folder to share (Dropbox/iCloud/git/...)."),
    ],
) -> None:
    """Enable session sync through an existing folder."""
    try:
        settings = set_sync(enabled=True, path=str(path))
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Session sync enabled[/green] for machine "
        f"[cyan]{escape(settings['machine'])}[/cyan] in "
        f"[cyan]{escape(settings['path'])}[/cyan]."
    )


@sync_app.command("disable")
def sync_disable_cmd() -> None:
    """Disable session sync (the shared folder is left untouched)."""
    set_sync(enabled=False)
    console.print(
        f"[green]{_check()} Session sync disabled.[/green] "
        "The shared folder was not modified."
    )


@eval_app.command("list")
def eval_list_cmd(
    cases_dir: Annotated[
        Path | None,
        typer.Option(
            "--dir",
            help="Directory of *.json eval cases (default: evals/cases).",
        ),
    ] = None,
) -> None:
    """List the discovered eval cases without running them."""
    from kiwimatecoder.evals import cases as eval_cases

    try:
        discovered = eval_cases.discover_cases(cases_dir)
    except eval_cases.CaseError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    if not discovered:
        console.print("[yellow]No eval cases found.[/yellow]")
        return
    table = Table(title="Eval cases")
    table.add_column("Case", style="cyan", no_wrap=True)
    table.add_column("Description")
    for case in discovered:
        table.add_row(escape(case.name), escape(case.description))
    console.print(table)


def _eval_failures_mention_credentials(report: Any) -> bool:
    """Best-effort detection of provider/key failures for extra CLI guidance."""
    hints = ("key", "unauthorized", "authentication", "credential")
    for result in report.results:
        for reason in result.reasons:
            lowered = reason.lower()
            if any(hint in lowered for hint in hints):
                return True
    return False


@eval_app.command("run")
def eval_run_cmd(
    cases_dir: Annotated[
        Path | None,
        typer.Option(
            "--dir",
            help="Directory of *.json eval cases (default: evals/cases).",
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Provider id override for every case."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model id override for every case."),
    ] = None,
    filter: Annotated[
        str | None,
        typer.Option("--filter", help="Only run cases whose name contains this text."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the report as JSON."),
    ] = False,
    report_path: Annotated[
        Path | None,
        typer.Option("--report", help="Write the JSON report to this path."),
    ] = None,
) -> None:
    """Run eval cases: exit 0 when all pass, 1 on failure, 2 on invalid input."""
    from kiwimatecoder.evals import render, runner

    if provider is not None:
        try:
            get_provider_config(provider)
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2)
    try:
        report = runner.run_suite(
            cases_dir,
            provider=provider,
            model=model,
            filter=filter,
            report_path=report_path,
        )
    except (runner.CaseError, OSError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    if json_output:
        console.print_json(render.report_to_json(report))
    else:
        render.print_report(report, console)
    if not report.success and _eval_failures_mention_credentials(report):
        console.print(
            "[yellow]Some failures look provider-related. Configure a key with "
            "[cyan]kiwimatecoder setup[/cyan] or the provider's environment "
            "variable, then rerun.[/yellow]"
        )
    raise typer.Exit(0 if report.success else 1)


@app.command("update")
def update_cmd(
    ref: Annotated[
        str | None,
        typer.Option(
            "--ref",
            help="Git ref to install: branch, tag, or commit SHA (default: current branch).",
        ),
    ] = None,
) -> None:
    """Update KiwiMateCoder in the current Python environment."""
    raise typer.Exit(run_update(console, ref=ref))


@app.command("version")
def version_cmd() -> None:
    """Show the installed KiwiMateCoder version."""
    console.print(f"kiwimatecoder {__version__}")