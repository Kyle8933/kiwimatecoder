import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from kiwimatecoder.ai import stream_response
from kiwimatecoder.config import load_api_key, save_api_key

app = typer.Typer(help="KiwiMateCoder - AI coding assistant CLI")
console = Console()
config_app = typer.Typer(help="Manage KiwiMateCoder configuration")
app.add_typer(config_app, name="config")


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="Your coding question"),
    file: Path = typer.Option(
        None, "--file", "-f", help="Path to a code file to include"
    ),
    model: str = typer.Option(None, "--model", "-m", help="Override the default model"),
):
    """Ask KiwiMateCoder a coding question."""
    api_key = load_api_key()
    if not api_key:
        console.print("[red]No API key found. Run: kiwimatecoder config set-key[/red]")
        raise typer.Exit(1)

    full_prompt = prompt
    if file:
        if not file.exists():
            console.print(f"[red]File not found: {file}[/red]")
            raise typer.Exit(1)
        code = file.read_text()
        full_prompt = f"{prompt}\n\n```\n{code}\n```"

    console.print(
        Panel(
            "[bold green]KiwiMateCoder[/bold green]",
            subtitle=f"model: {model or 'default'}",
        )
    )
    asyncio.run(
        stream_response(full_prompt, api_key, model=model)
        if model
        else stream_response(full_prompt, api_key)
    )
    console.print()


@config_app.command("set-key")
def set_key(key: str = typer.Argument(..., help="Your OpenRouter API key")):
    """Save your OpenRouter API key."""
    save_api_key(key)
    console.print("[green]✓ API key saved successfully![/green]")


@config_app.command("check")
def check():
    """Check if an API key is configured."""
    key = load_api_key()
    if key:
        console.print(f"[green]✓ API key configured[/green] (ending in ...{key[-4:]})")
    else:
        console.print("[red]✗ No API key found[/red]")
