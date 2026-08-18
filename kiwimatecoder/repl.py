"""Interactive REPL: the main loop launched by the bare ``kiwimatecoder`` command."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.shortcuts import CompleteStyle, checkboxlist_dialog, choice
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from kiwimatecoder import __version__
from kiwimatecoder.agent import Agent
from kiwimatecoder.commands import (
    CommandResult,
    MultiSelectionPrompt,
    SelectionPrompt,
    dispatch,
    slash_argument_completions,
    slash_command_completions,
)
from kiwimatecoder.session import Session

console = Console()


class SlashCommandCompleter(Completer):
    """Prompt-toolkit completer for KiwiMate slash commands."""

    session: Session | None

    def __init__(self, session: Session | None = None) -> None:
        self.session = session

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if "\n" in text or not text.startswith("/"):
            return

        body = text[1:]
        if " " not in body:
            for command, description in slash_command_completions(body):
                yield Completion(
                    command,
                    start_position=-len(text),
                    display=command,
                    display_meta=description,
                )
            return

        command, arg_text = body.split(" ", 1)
        if " " in arg_text.strip():
            return
        for value, description in slash_argument_completions(
            command, arg_text, self.session
        ):
            yield Completion(
                value,
                start_position=-len(arg_text),
                display=value,
                display_meta=description,
            )


def _git_info(root: Path) -> str | None:
    """Read active git branch if inside a git repository."""
    head = root / ".git" / "HEAD"
    if head.is_file():
        try:
            ref = head.read_text(encoding="utf-8").strip()
            if ref.startswith("ref: refs/heads/"):
                return ref.split("/")[-1]
            return ref[:7]
        except OSError:
            pass
    return None


def _banner(session: Session) -> Panel:
    git_branch = _git_info(session.workspace_root)
    git_badge = f" · [magenta]git:{git_branch}[/magenta]" if git_branch else ""
    ctx_badge = (
        f" · [cyan]{len(session.context_files)} pinned[/cyan]"
        if session.context_files
        else ""
    )

    active = session.active_providers
    provider_summary = (
        f"[bold cyan]{session.provider.name}[/bold cyan] ([dim]{session.model}[/dim])"
    )
    if len(active) > 1:
        fallback_names = ", ".join(p.name for p in active[1:])
        provider_summary += f" [dim]+ {fallback_names}[/dim]"

    content = (
        f"[bold green]KiwiMateCoder[/bold green] [dim]v{__version__}[/dim] — "
        f"{provider_summary}\n"
        f"[dim]📁 {session.workspace_root.name}{git_badge} · mode:[bold]{session.mode.value}[/bold]{ctx_badge}\n"
        f"Type /help for commands · Alt+Enter for newline · Ctrl-C cancels · Ctrl-D exits[/dim]"
    )
    return Panel(
        content,
        border_style="green",
        padding=(0, 1),
    )


def _prompt_text(session: Session) -> HTML:
    mode_color = (
        "ansimagenta"
        if session.mode.value == "plan"
        else "ansiyellow"
        if session.mode.value == "ask"
        else "ansigreen"
    )
    provider_display = f"{session.provider_id}:{session.model}"
    if len(session.active_provider_ids) > 1:
        provider_display += f" +{len(session.active_provider_ids) - 1}"
    return HTML(
        f"<ansigreen><b>kiwi</b></ansigreen> "
        f"<{mode_color}>({provider_display} · {session.mode.value})</{mode_color}> "
        f"<ansicyan>›</ansicyan> "
    )


def _select_command_option(prompt: SelectionPrompt) -> str | None:
    """Render a keyboard-driven selector for a choice-based slash command."""
    try:
        return choice(
            message=f"{prompt.title}\n{prompt.text}",
            options=[(option.value, option.label) for option in prompt.options],
            default=prompt.selected,
            show_frame=True,
            bottom_toolbar="↑/↓ move • Enter select • Ctrl-C cancel",
        )
    except (EOFError, KeyboardInterrupt):
        return None


def _select_command_options(prompt: MultiSelectionPrompt) -> list[str] | None:
    """Render a keyboard-driven checklist for multi-select slash commands."""
    try:
        selected = checkboxlist_dialog(
            title=prompt.title,
            text=prompt.text,
            values=[(option.value, option.label) for option in prompt.options],
            default_values=list(prompt.selected),
        ).run()
    except (EOFError, KeyboardInterrupt):
        return None
    if selected is None:
        return None
    return list(selected)


def _make_confirm(session: Session):
    """Build the approval callback used by the permission gate."""

    def confirm(summary: str, preview_text: str | None) -> bool:
        console.print()
        if preview_text:
            is_diff = preview_text.lstrip().startswith(
                ("---", "+++", "@@", "+", "-")
            )
            lexer = "diff" if is_diff else "bash"

            if is_diff:
                lines = preview_text.splitlines()
                added = sum(
                    1
                    for line in lines
                    if line.startswith("+") and not line.startswith("+++")
                )
                removed = sum(
                    1
                    for line in lines
                    if line.startswith("-") and not line.startswith("---")
                )
                stats = (
                    f" ([bold green]+{added}[/bold green] [bold red]-{removed}[/bold red])"
                    if (added or removed)
                    else ""
                )
                title = f"[bold yellow]Approve Change: {summary}[/bold yellow]{stats}"
                border = "yellow"
            else:
                title = f"[bold magenta]Approve Shell: {summary}[/bold magenta]"
                border = "magenta"

            console.print(
                Panel(
                    Syntax(
                        preview_text,
                        lexer,
                        theme="ansi_dark",
                        word_wrap=True,
                        line_numbers=is_diff,
                    ),
                    title=title,
                    border_style=border,
                )
            )
        else:
            console.print(f"[yellow]Approve: {summary}[/yellow]")

        try:
            answer = console.input(
                "[bold]Allow?[/bold] ([green]y[/green]es / [red]n[/red]o / [cyan]a[/cyan]lways this tool): "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("[yellow]Denied.[/yellow]")
            return False

        if answer in ("a", "always"):
            tool_name = summary.split("(", 1)[0].strip()
            session.allow_always(tool_name)
            return True
        return answer in ("y", "yes")

    return confirm


def run(session: Session) -> None:
    """Run the interactive loop until the user exits."""
    console.print(_banner(session))
    confirm = _make_confirm(session)
    agent = Agent(session, console, confirm)

    kb = KeyBindings()

    @kb.add("escape", "enter")
    def _(event: KeyPressEvent) -> None:
        """Alt+Enter / Escape+Enter inserts newline for multi-line prompts."""
        event.current_buffer.insert_text("\n")

    pt_session: PromptSession[str] = PromptSession(
        history=InMemoryHistory(),
        completer=SlashCommandCompleter(session),
        complete_while_typing=True,
        complete_style=CompleteStyle.MULTI_COLUMN,
        key_bindings=kb,
    )

    multiline_buffer: list[str] = []
    in_multiline_block = False

    while True:
        try:
            prompt_str = (
                HTML("<ansicyan>... </ansicyan>")
                if in_multiline_block
                else _prompt_text(session)
            )
            line = pt_session.prompt(prompt_str)
        except KeyboardInterrupt:
            # Ctrl-C at the prompt: clear the line / buffer, keep going.
            multiline_buffer.clear()
            in_multiline_block = False
            continue
        except EOFError:
            # Ctrl-D: exit.
            console.print("[dim]Goodbye![/dim]")
            break

        # Check for triple-quote multiline block mode
        stripped = line.strip()
        if not in_multiline_block and stripped.startswith('"""') and not (
            len(stripped) > 3 and stripped.endswith('"""')
        ):
            in_multiline_block = True
            multiline_buffer.append(stripped[3:])
            continue

        if in_multiline_block:
            if stripped.endswith('"""'):
                in_multiline_block = False
                multiline_buffer.append(stripped[:-3])
                line = "\n".join(multiline_buffer).strip()
                multiline_buffer.clear()
            else:
                multiline_buffer.append(line)
                continue

        line = line.strip()
        if not line:
            continue

        if line.startswith("/"):
            if (
                dispatch(
                    line,
                    session,
                    console,
                    selector=_select_command_option,
                    multi_selector=_select_command_options,
                )
                == CommandResult.EXIT
            ):
                break
            continue

        try:
            asyncio.run(agent.run_turn(line))
        except KeyboardInterrupt:
            # Ctrl-C during a turn: cancel and return to the prompt.
            console.print("\n[yellow]Interrupted.[/yellow]")
