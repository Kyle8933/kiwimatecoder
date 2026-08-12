import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kiwimatecoder import __version__
from kiwimatecoder.ai import stream_response
from kiwimatecoder.catalog import summarize_ids
from kiwimatecoder.config import (
    add_provider,
    apply_model_filter,
    describe_key,
    get_default_mode,
    get_key,
    get_model_catalog,
    get_model_filter,
    get_provider_config,
    get_selected_provider_id,
    list_provider_configs,
    load_config,
    remove_key,
    remove_provider,
    reset_default_mode,
    set_default_mode,
    set_key,
    set_model_filter,
    set_selected_model,
    set_selected_provider,
    update_provider,
)
from kiwimatecoder.permissions import PermissionMode
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
def config_main(ctx: typer.Context):
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
    provider: str = typer.Argument(..., help="Provider id"),
    key: str = typer.Argument(..., help="Your API key"),
):
    """Save an API key for a provider."""
    try:
        warning = set_key(provider, key)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]✓ API key saved for[/green] [cyan]{provider}[/cyan] "
        f"— {describe_key(provider)}."
    )
    if warning:
        console.print(f"[yellow]{warning}[/yellow]")


@key_app.command("remove")
def key_remove(provider: str = typer.Argument(..., help="Provider id")):
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
def key_list():
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
def provider_use(provider: str = typer.Argument(..., help="Provider id")):
    """Persist and switch to a provider as the default."""
    try:
        set_selected_provider(provider)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓ Default provider set to [cyan]{provider}[/cyan].[/green]")


@provider_app.command("list")
def provider_list():
    """List all built-in and custom providers."""
    table = Table(title="Providers", show_header=True)
    table.add_column("id", style="cyan")
    table.add_column("name")
    table.add_column("default model")
    for provider in list_provider_configs():
        table.add_row(provider.id, provider.name, provider.default_model)
    console.print(table)


@provider_app.command("add")
def provider_add(
    provider_id: str = typer.Argument(..., help="Unique id (no spaces)"),
    name: str = typer.Argument(..., help="Display name"),
    base_url: str = typer.Argument(..., help="Base URL including /v1"),
    default_model: str = typer.Argument(..., help="Default model id"),
    key_env: str = typer.Option(
        None, "--key-env", help="API key environment variable (default: <ID>_API_KEY)"
    ),
):
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
def provider_remove(provider: str = typer.Argument(..., help="Provider id")):
    """Remove a custom provider."""
    try:
        remove_provider(provider)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓ Removed provider {provider}.[/green]")


@provider_app.command("edit")
def provider_edit(
    provider: str = typer.Argument(..., help="Provider id"),
    name: str = typer.Option(None, "--name", help="New display name"),
    base_url: str = typer.Option(None, "--base-url", help="New base URL"),
    default_model: str = typer.Option(None, "--default-model", help="New default model"),
    key_env: str = typer.Option(None, "--key-env", help="New key environment variable"),
    compat: str = typer.Option(None, "--compat", help="'openai' or 'anthropic'"),
):
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
def model_set(model: str = typer.Argument(..., help="Model id")):
    """Set the default model (overrides the provider default)."""
    set_selected_model(model)
    console.print(f"[green]✓ Default model set to [cyan]{model}[/cyan].[/green]")


@model_app.command("reset")
def model_reset():
    """Use the provider's default model again."""
    set_selected_model(None)
    console.print("[green]✓ Default model reset (using provider default).[/green]")


# --- canonical `config mode ...` --------------------------------------------


@mode_app.command("set")
def mode_set(
    mode: str = typer.Argument(
        ..., help="Permission mode: ask, auto-accept, or plan"
    )
):
    """Set the default permission mode for new sessions."""
    try:
        effective = set_default_mode(mode)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓ Default mode set to [cyan]{effective}[/cyan].[/green]")


@mode_app.command("reset")
def mode_reset():
    """Reset the default permission mode to 'ask'."""
    effective = reset_default_mode()
    console.print(f"[green]✓ Default mode reset to [cyan]{effective}[/cyan].[/green]")


# --- canonical `config models ...` ------------------------------------------


@models_app.command("show")
def models_show(
    provider: str = typer.Option(
        None, "--provider", "-p", help="Provider id (default: configured provider)"
    ),
):
    """List the models offered for a provider, newest first."""
    _print_models(provider, refresh=False)


@models_app.command("refresh")
def models_refresh(
    provider: str = typer.Option(
        None, "--provider", "-p", help="Provider id (default: configured provider)"
    ),
):
    """Fetch the provider's live model list, dropping deprecated ids."""
    _print_models(provider, refresh=True)


@models_app.command("allow")
def models_allow(
    models: list[str] = typer.Argument(..., help="Model ids to show"),
    provider: str = typer.Option(
        None, "--provider", "-p", help="Provider id (default: configured provider)"
    ),
):
    """Only show these models for a provider."""
    _set_filter(provider, "allow", models)


@models_app.command("deny")
def models_deny(
    models: list[str] = typer.Argument(..., help="Model ids to hide"),
    provider: str = typer.Option(
        None, "--provider", "-p", help="Provider id (default: configured provider)"
    ),
):
    """Hide these models for a provider."""
    _set_filter(provider, "deny", models)


@models_app.command("clear")
def models_clear(
    provider: str = typer.Option(
        None, "--provider", "-p", help="Provider id (default: configured provider)"
    ),
):
    """Clear model visibility for a provider."""
    pid = provider or get_selected_provider_id()
    _resolve_provider(pid)
    set_model_filter(pid, "all", [])
    console.print(f"[green]✓ Cleared model visibility for {pid}.[/green]")


# --- `config show` ----------------------------------------------------------


@config_app.command("show")
def config_show():
    """Show the active provider, model, key, and model filter."""
    cfg = load_config()
    provider_id = get_selected_provider_id(cfg)
    provider = get_provider_config(provider_id, cfg)
    console.print(
        f"Provider: [cyan]{provider.id}[/cyan] ({provider.name})\n"
        f"Model: [cyan]{cfg.get('selected_model') or provider.default_model}[/cyan]\n"
        f"Mode: [cyan]{get_default_mode(cfg)}[/cyan]\n"
        f"Key: [cyan]{describe_key(provider_id)}[/cyan] ({provider.key_env})\n"
        f"Model visibility: [cyan]{get_model_filter(provider_id)['mode']}[/cyan]"
    )


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
    key: str = typer.Argument(..., help="Your API key"),
    provider: str = typer.Option(
        "openrouter", "--provider", "-p", help="Provider id this key belongs to"
    ),
):
    """Deprecated: use `config key set <provider> <key>`."""
    key_set(provider, key)


@config_app.command("set-provider", hidden=True, deprecated=True)
def set_provider_cmd(provider: str = typer.Argument(..., help="Provider id")):
    """Deprecated: use `config provider use <id>`."""
    provider_use(provider)


@config_app.command("set-model", hidden=True, deprecated=True)
def set_model_cmd(model: str = typer.Argument(..., help="Model id")):
    """Deprecated: use `config model set <id>`."""
    model_set(model)


@config_app.command("check", hidden=True, deprecated=True)
def check():
    """Deprecated: use `config show`."""
    config_show()


@config_app.command("list", hidden=True, deprecated=True)
def list_cmd():
    """Deprecated: use `config provider list`."""
    provider_list()


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the installed KiwiMateCoder version and exit.",
        is_eager=True,
    ),
    update: bool = typer.Option(
        False,
        "-update",
        "--update",
        help="Update KiwiMateCoder in the current Python environment.",
    ),
):
    """Launch the interactive session when run with no subcommand."""
    if version:
        console.print(f"kiwimatecoder {__version__}")
        raise typer.Exit(0)

    if update:
        raise typer.Exit(run_update(console))

    if ctx.invoked_subcommand is not None:
        return

    from kiwimatecoder import repl

    cfg = load_config()
    provider_id = get_selected_provider_id(cfg)
    provider = get_provider_config(provider_id, cfg)
    model = cfg.get("selected_model") or provider.default_model

    try:
        mode = PermissionMode.from_str(cfg.get("default_mode", "ask"))
    except ValueError:
        mode = PermissionMode.ASK

    if not get_key(provider_id):
        console.print(
            Panel(
                f"[yellow]No API key set for {provider.name}.[/yellow]\n"
                f"Run [cyan]kiwimatecoder setup[/cyan] to choose a provider and "
                f"enter a key, or export [cyan]{provider.key_env}[/cyan]."
            ),
            title="Quick start",
        )
        if _stdin_is_tty() and _prompt_yes_no("Run setup now?"):
            _run_setup(provider_id, key=None)

    session = Session(
        provider_id=provider_id,
        model=model,
        mode=mode,
        workspace_root=Path.cwd(),
    )
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


def _interactive_select_provider(providers, selected: str) -> str | None:
    """Keyboard-driven provider picker, matching the REPL's selector style."""
    from prompt_toolkit.shortcuts import choice

    try:
        return choice(
            message="Choose a provider to configure",
            options=[
                (p.id, f"{p.name} — {p.default_model}")
                for p in providers
            ],
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
        get_provider_config(provider_id)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return False
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
        f"— {describe_key(provider_id)}."
    )
    if warning:
        console.print(f"[yellow]{warning}[/yellow]")
    console.print("Ready to go. Run [cyan]kiwimatecoder[/cyan] to start a session.")
    return True


@app.command("setup")
def setup(
    provider: str = typer.Option(
        None, "--provider", "-p", help="Provider id (default: configured provider)"
    ),
    key: str = typer.Option(
        None, "--key", "-k", help="API key (skips the interactive prompt)"
    ),
):
    """Choose a provider and save its API key — the quick-start guide."""
    cfg = load_config()
    provider_id = provider or get_selected_provider_id(cfg)
    if provider is None and key is None and _stdin_is_tty():
        chosen = _interactive_select_provider(
            list_provider_configs(cfg), provider_id
        )
        if chosen is None:
            console.print("[yellow]Setup cancelled.[/yellow]")
            raise typer.Exit(1)
        provider_id = chosen
    if not _run_setup(provider_id, key):
        raise typer.Exit(1)


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="Your coding question"),
    file: Path = typer.Option(
        None, "--file", "-f", help="Path to a code file to include"
    ),
    model: str = typer.Option(None, "--model", "-m", help="Override the default model"),
    provider: str = typer.Option(
        None, "--provider", "-p", help="Provider id (default: configured provider)"
    ),
):
    """Ask KiwiMateCoder a one-shot coding question."""
    cfg = load_config()
    provider_id = provider or get_selected_provider_id(cfg)
    try:
        provider_cfg = get_provider_config(provider_id, cfg)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    api_key = get_key(provider_id)
    if not api_key:
        console.print(
            f"[red]No API key for {provider_cfg.name}. "
            f"Run: kiwimatecoder setup --provider {provider_id}[/red]"
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
            api_key,
            model=model or cfg.get("selected_model"),
            provider=provider_cfg,
        )
    )
    console.print()


@app.command("update")
def update_cmd():
    """Update KiwiMateCoder in the current Python environment."""
    raise typer.Exit(run_update(console))


@app.command("version")
def version_cmd():
    """Show the installed KiwiMateCoder version."""
    console.print(f"kiwimatecoder {__version__}")
