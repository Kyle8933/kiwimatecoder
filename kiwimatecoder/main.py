import asyncio
from pathlib import Path

import typer
from rich.console import Console

from kiwimatecoder.ai import stream_response
from kiwimatecoder.config import (
    get_key,
    get_selected_provider_id,
    load_config,
    set_key,
    set_selected_model,
    set_selected_provider,
)
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.providers import default_provider, get_provider, list_providers
from kiwimatecoder.session import Session

app = typer.Typer(
    help="KiwiMateCoder - agentic AI coding assistant CLI",
    invoke_without_command=True,
    no_args_is_help=False,
)
console = Console()
config_app = typer.Typer(help="Manage KiwiMateCoder configuration")
app.add_typer(config_app, name="config")


@app.callback()
def main(ctx: typer.Context):
    """Launch the interactive session when run with no subcommand."""
    if ctx.invoked_subcommand is not None:
        return

    from kiwimatecoder import repl

    cfg = load_config()
    provider_id = get_selected_provider_id(cfg)
    provider = get_provider(provider_id)
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
    provider_id = provider or cfg.get("selected_provider") or default_provider().id
    try:
        provider_cfg = get_provider(provider_id)
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


@config_app.command("set-key")
def set_key_cmd(
    key: str = typer.Argument(..., help="Your API key"),
    provider: str = typer.Option(
        "openrouter", "--provider", "-p", help="Provider id this key belongs to"
    ),
):
    """Save an API key for a provider."""
    try:
        set_key(provider, key)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓ API key saved for {provider}![/green]")


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
    for provider in list_providers():
        key = get_key(provider.id)
        if key:
            console.print(
                f"[green]✓ {provider.id}[/green] (ending in ...{key[-4:]})"
            )
        else:
            console.print(f"[dim]✗ {provider.id} — no key[/dim]")


@config_app.command("list")
def list_cmd():
    """List all built-in providers and their default models."""
    for provider in list_providers():
        console.print(
            f"[cyan]{provider.id}[/cyan]: {provider.name} "
            f"(default: {provider.default_model}, key env: {provider.key_env})"
        )
