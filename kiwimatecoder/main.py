import asyncio
from pathlib import Path

import typer
from rich.console import Console

from kiwimatecoder import __version__
from kiwimatecoder.ai import stream_response
from kiwimatecoder.catalog import summarize_ids
from kiwimatecoder.config import (
    apply_model_filter,
    get_key,
    get_key_env_override,
    get_model_catalog,
    get_provider_config,
    get_selected_provider_id,
    list_provider_configs,
    load_config,
    set_key,
    set_selected_model,
    set_selected_provider,
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
config_app = typer.Typer(help="Manage KiwiMateCoder configuration")
app.add_typer(config_app, name="config")


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
            f"[yellow]No API key for {provider.name}.[/yellow] "
            f"Set one with: [cyan]kiwimatecoder config set-key --provider "
            f"{provider_id} <KEY>[/cyan]"
        )

    session = Session(
        provider_id=provider_id,
        model=model,
        mode=mode,
        workspace_root=Path.cwd(),
    )
    repl.run(session)


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
            f"Run: kiwimatecoder config set-key --provider {provider_id} <KEY>[/red]"
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


@config_app.command("set-key")
def set_key_cmd(
    key: str = typer.Argument(..., help="Your API key"),
    provider: str = typer.Option(
        "openrouter", "--provider", "-p", help="Provider id this key belongs to"
    ),
):
    """Save an API key for a provider."""
    try:
        warning = set_key(provider, key)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓ API key saved for {provider}![/green]")
    if warning:
        console.print(f"[yellow]{warning}[/yellow]")


@config_app.command("set-provider")
def set_provider_cmd(provider: str = typer.Argument(..., help="Provider id")):
    """Set the default provider."""
    try:
        set_selected_provider(provider)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓ Default provider set to {provider}.[/green]")


@config_app.command("set-model")
def set_model_cmd(model: str = typer.Argument(..., help="Model id")):
    """Set the default model (overrides the provider default)."""
    set_selected_model(model)
    console.print(f"[green]✓ Default model set to {model}.[/green]")


@config_app.command("check")
def check():
    """Check which providers have a configured API key."""
    cfg = load_config()
    provider_id = get_selected_provider_id(cfg)
    mode = cfg.get("default_mode") or "ask"
    try:
        PermissionMode.from_str(mode)
    except ValueError:
        mode = "ask"
    console.print(
        f"Default provider: [cyan]{provider_id}[/cyan], "
        f"model: [cyan]{cfg.get('selected_model') or '(provider default)'}[/cyan], "
        f"mode: [cyan]{mode}[/cyan]"
    )
    for provider in list_provider_configs(cfg):
        key = get_key(provider.id)
        if not key:
            console.print(f"[dim]✗ {provider.id} — no key[/dim]")
            continue
        env_override = get_key_env_override(provider.id)
        source = (
            f", from env {env_override}" if env_override else ", stored in config"
        )
        console.print(f"[green]✓ {provider.id}[/green] (ending in ...{key[-4:]}{source})")


@config_app.command("models")
def models_cmd(
    provider: str = typer.Option(
        None, "--provider", "-p", help="Provider id (default: configured provider)"
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        "-r",
        help="Fetch the provider's live model list before showing it.",
    ),
):
    """List the models offered for a provider, newest first.

    With ``--refresh`` the provider is asked what it serves right now: newly
    released models are added and ones it no longer lists are dropped.
    """
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


@config_app.command("list")
def list_cmd():
    """List all built-in providers and their default models."""
    for provider in list_provider_configs():
        console.print(
            f"[cyan]{provider.id}[/cyan]: {provider.name} "
            f"(default: {provider.default_model}, key env: {provider.key_env})"
        )
