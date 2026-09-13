"""Interactive REPL: the main loop launched by the bare ``kiwimatecoder`` command."""

from __future__ import annotations

import asyncio
import os
import time

try:  # pragma: no cover - platform dependent
    # Importing readline makes blocking ``input()`` (approvals, ask-user)
    # coexist with the active steering prompt, which holds the terminal in
    # raw mode. Without it, ``input()`` never sees a newline while a turn runs.
    import readline  # noqa: F401
except ImportError:  # pragma: no cover - Windows
    pass

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import (
    Condition,
    is_done,
    renderer_height_is_known,
    to_filter,
)
from prompt_toolkit.formatted_text import (
    AnyFormattedText,
    HTML,
    StyleAndTextTuples,
    to_formatted_text,
)
from prompt_toolkit.history import FileHistory, History, InMemoryHistory
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout.containers import (
    AnyContainer,
    ConditionalContainer,
    HSplit,
    Window,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.shortcuts import CompleteStyle, choice
from prompt_toolkit.shortcuts.choice_input import create_default_choice_input_style
from prompt_toolkit.styles import BaseStyle, Style, merge_styles
from prompt_toolkit.widgets import Box, Frame, Label
from prompt_toolkit.widgets.base import _DialogList
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from kiwimatecoder import __version__, browser, events, hooks, images, lsp, mcp, notify, plugins, ui
from kiwimatecoder.agent import Agent
from kiwimatecoder.commands import (
    CommandResult,
    MultiSelectionPrompt,
    SelectionPrompt,
    dispatch,
    has_command,
    slash_argument_completions,
    slash_command_completions,
)
from kiwimatecoder.config import get_ui, get_vision
from kiwimatecoder.hunks import Hunk, parse_hunk_selection, split_hunks
from kiwimatecoder.i18n import apply_config_locale, t
from kiwimatecoder.permissions import ApprovalResult, ConfirmFn
from kiwimatecoder.redaction import redact
from kiwimatecoder.session import Session
from kiwimatecoder.shell import close_shell
from kiwimatecoder.templates import find_template, render_template
from kiwimatecoder.tools.paths import (
    PathError,
    display_path,
    get_workspace_ignore,
    resolve_in_workspace,
)

console = Console()

_AT_MENTION_LIMIT = 25
_AT_MENTION_SCAN_LIMIT = 5000


def _workspace_file_candidates(
    workspace_root: Path, prefix: str = "", limit: int = _AT_MENTION_LIMIT
) -> list[str]:
    """Return ``@``-prefixed workspace file paths starting with ``prefix``.

    The walk skips gitignored files and the usual build/cache directories
    (via :func:`get_workspace_ignore`) and stops once ``limit`` matches are
    found or ``_AT_MENTION_SCAN_LIMIT`` entries have been examined.
    """
    root = Path(workspace_root).resolve()
    ignore = get_workspace_ignore(root)
    wanted = prefix.replace("\\", "/").lstrip("./")
    matches: list[str] = []
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not ignore.is_ignored(Path(dirpath) / name, is_dir=True)
        )
        for name in sorted(filenames):
            full = Path(dirpath) / name
            scanned += 1
            if ignore.is_ignored(full):
                continue
            try:
                rel = full.relative_to(root).as_posix()
            except ValueError:
                continue
            if wanted and not rel.startswith(wanted):
                if scanned >= _AT_MENTION_SCAN_LIMIT:
                    return matches
                continue
            matches.append("@" + rel)
            if len(matches) >= limit:
                return matches
        if scanned >= _AT_MENTION_SCAN_LIMIT:
            break
    return matches


class SlashCommandCompleter(Completer):
    """Prompt-toolkit completer for KiwiMate slash commands and @file mentions."""

    session: Session | None

    def __init__(self, session: Session | None = None) -> None:
        self.session = session

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if "\n" in text:
            return
        if text.startswith("/"):
            yield from self._slash_completions(text)
            return
        yield from self._file_completions(text)

    def _slash_completions(self, text: str) -> Iterable[Completion]:
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

    def _file_completions(self, text: str) -> Iterable[Completion]:
        if self.session is None or not text or text[-1].isspace():
            return
        token = text.split()[-1]
        if not token.startswith("@"):
            return
        prefix = token[1:]
        for candidate in _workspace_file_candidates(
            self.session.workspace_root, prefix
        ):
            yield Completion(
                candidate,
                start_position=-len(token),
                display=candidate,
                display_meta="file",
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
    accent = ui.theme_accent()
    folder = ui.glyph("folder")
    folder_prefix = f"{folder} " if folder else ""
    git_branch = _git_info(session.workspace_root)
    git_badge = f" · [magenta]git:{git_branch}[/magenta]" if git_branch else ""
    ctx_badge = (
        f" · [cyan]{t('banner.pinned', count=len(session.context_files))}[/cyan]"
        if session.context_files
        else ""
    )
    dry_badge = f" · [yellow]{t('banner.dry_run')}[/yellow]" if session.dry_run else ""

    active = session.active_providers
    provider_summary = (
        f"[bold cyan]{session.provider.name}[/bold cyan] ([dim]{session.model}[/dim])"
    )
    if len(active) > 1:
        fallback_names = ", ".join(p.name for p in active[1:])
        provider_summary += f" [dim]+ {fallback_names}[/dim]"

    content = (
        f"[bold {accent}]KiwiMateCoder[/bold {accent}] [dim]v{__version__}[/dim] — "
        f"{provider_summary}\n"
        f"[dim]{folder_prefix}{session.workspace_root.name}{git_badge} · mode:[bold]{session.mode.value}[/bold]{ctx_badge}{dry_badge}\n"
        f"{t('banner.help_hint')}[/dim]"
    )
    return Panel(
        content,
        border_style=accent,
        padding=(0, 1),
    )


def _prompt_text(session: Session) -> HTML:
    accent = f"ansi{ui.theme_accent()}"
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
        f"<{accent}><b>kiwi</b></{accent}> "
        f"<{mode_color}>({provider_display} · {session.mode.value})</{mode_color}> "
        f"<{accent}>›</{accent}> "
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


_T = TypeVar("_T")


class _CheckboxChoiceList(_DialogList[_T]):
    """Inline checklist widget with keyboard navigation and checkbox toggles."""

    def __init__(
        self,
        values: Sequence[tuple[_T, AnyFormattedText]],
        default_values: Sequence[_T] | None = None,
    ) -> None:
        super().__init__(
            values=values,
            default_values=default_values,
            multiple_selection=True,
            show_scrollbar=False,
            show_cursor=False,
            show_numbers=False,
        )

    def _get_text_fragments(self) -> StyleAndTextTuples:
        def mouse_handler(mouse_event: MouseEvent) -> None:
            if mouse_event.event_type == MouseEventType.MOUSE_UP:
                self._selected_index = mouse_event.position.y
                self._handle_enter()

        result: StyleAndTextTuples = []
        for i, value in enumerate(self.values):
            checked = value[0] in self.current_values
            selected = i == self._selected_index

            cursor = "> " if selected else "  "
            box = "[*] " if checked else "[ ] "

            cursor_style = "class:cursor" if selected else ""
            box_style = (
                "class:checkbox-checked" if checked else "class:checkbox-unchecked"
            )
            text_style = "class:option"
            if checked:
                text_style += " class:selected-option"
            if selected:
                text_style += " class:focused-option"

            result.append((cursor_style, cursor))
            if selected:
                result.append(("[SetCursorPosition]", ""))
            result.append((box_style, box))
            result.extend(to_formatted_text(value[1], style=text_style))
            result.append(("", "\n"))

        for i in range(len(result)):
            result[i] = (result[i][0], result[i][1], mouse_handler)

        if result:
            result.pop()
        return result


def checkbox_choice(
    message: AnyFormattedText,
    *,
    options: Sequence[tuple[_T, AnyFormattedText]],
    default_values: Sequence[_T] | None = None,
    bottom_toolbar: AnyFormattedText = None,
    show_frame: bool = True,
    mouse_support: bool = False,
    style: BaseStyle | None = None,
    input: Input | None = None,
) -> list[_T]:
    """Inline multiple-choice checkbox prompt matching the style of ``choice``."""
    if not options:
        return []

    cb_list = _CheckboxChoiceList(options, default_values=default_values)
    container: AnyContainer = HSplit(
        [
            Box(
                Label(text=message, dont_extend_height=True),
                padding_top=0,
                padding_left=1,
                padding_right=1,
                padding_bottom=0,
            ),
            Box(
                cb_list,
                padding_top=0,
                padding_left=1,
                padding_right=1,
                padding_bottom=0,
            ),
        ]
    )

    @Condition
    def show_frame_filter() -> bool:
        return to_filter(show_frame)()

    show_bottom_toolbar = (
        Condition(lambda: bottom_toolbar is not None)
        & ~is_done
        & renderer_height_is_known
    )

    framed_container = ConditionalContainer(
        Frame(container),
        alternative_content=container,
        filter=show_frame_filter,
    )

    toolbar_container = ConditionalContainer(
        Window(
            FormattedTextControl(
                lambda: bottom_toolbar, style="class:bottom-toolbar.text"
            ),
            style="class:bottom-toolbar",
            dont_extend_height=True,
            height=Dimension(min=1),
        ),
        filter=show_bottom_toolbar,
    )

    layout = Layout(
        HSplit(
            [
                framed_container,
                ConditionalContainer(Window(), filter=show_bottom_toolbar),
                toolbar_container,
            ]
        ),
        focused_element=cb_list,
    )

    kb = KeyBindings()

    @kb.add("enter", eager=True)
    def _accept_input(event: KeyPressEvent) -> None:
        event.app.exit(result=list(cb_list.current_values), style="class:accepted")

    @kb.add("c-c", eager=True)
    @kb.add("<sigint>", eager=True)
    def _keyboard_interrupt(event: KeyPressEvent) -> None:
        event.app.exit(exception=KeyboardInterrupt(), style="class:aborting")

    effective_style = merge_styles(
        [
            create_default_choice_input_style(),
            Style.from_dict(
                {
                    "cursor": "bold cyan",
                    "checkbox-checked": "bold green",
                    "checkbox-unchecked": "dim",
                    "focused-option": "bold",
                }
            ),
            style if style is not None else Style([]),
        ]
    )

    app: Application[list[_T]] = Application(
        layout=layout,
        full_screen=False,
        mouse_support=mouse_support,
        key_bindings=kb,
        style=effective_style,
        input=input,
    )
    return app.run()


def _select_command_options(prompt: MultiSelectionPrompt) -> list[str] | None:
    """Render a keyboard-driven checklist for multi-select slash commands."""
    try:
        return checkbox_choice(
            message=f"{prompt.title}\n{prompt.text}",
            options=[(option.value, option.label) for option in prompt.options],
            default_values=prompt.selected,
            show_frame=True,
            bottom_toolbar="↑/↓ move • Space toggle • Enter confirm • Ctrl-C cancel",
        )
    except (EOFError, KeyboardInterrupt):
        return None


def _hunk_overview(hunk: Hunk, limit: int = 2) -> list[str]:
    """Return up to ``limit`` changed lines for display in the hunk list."""
    changed = [line.rstrip("\n") for line in hunk.lines if line.startswith(("+", "-"))]
    return changed[:limit]


def _review_hunks(hunk_list: Sequence[Hunk]) -> ApprovalResult:
    """Prompt for a 1-based hunk selection; deny after repeated bad input."""
    console.print(f"[bold]{t('approval.review_hunks')}[/bold]")
    for hunk in hunk_list:
        console.print(f"  [cyan]{hunk.index}[/cyan]. {hunk.header}")
        for line in _hunk_overview(hunk):
            console.print(f"     {line}", markup=False, highlight=False)

    for _ in range(3):
        try:
            answer = console.input(
                f"[bold]{t('approval.apply_hunks')}[/bold] "
            )
        except (EOFError, KeyboardInterrupt):
            console.print(f"[yellow]{t('approval.denied')}[/yellow]")
            return ApprovalResult(allowed=False)

        selection = parse_hunk_selection(answer, len(hunk_list))
        if selection is None:
            console.print(f"[yellow]{t('approval.retry_selection')}[/yellow]")
            continue
        if selection == "all":
            return ApprovalResult(allowed=True)
        if selection == "none":
            return ApprovalResult(allowed=False)
        return ApprovalResult(allowed=True, selected_hunks=selection)

    console.print(f"[yellow]{t('approval.too_many_attempts')}[/yellow]")
    return ApprovalResult(allowed=False)


def _make_confirm(session: Session) -> ConfirmFn:
    """Build the approval callback used by the permission gate."""

    def confirm(summary: str, preview_text: str | None) -> bool | ApprovalResult:
        console.print()
        hunk_list: list[Hunk] = []
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
                title = (
                    f"[bold yellow]{t('approval.approve_change', summary=summary)}"
                    f"[/bold yellow]{stats}"
                )
                border = "yellow"
                hunk_list = split_hunks(preview_text)
            else:
                title = (
                    f"[bold magenta]{t('approval.approve_shell', summary=summary)}"
                    f"[/bold magenta]"
                )
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
            console.print(
                f"[yellow]{t('approval.approve', summary=summary)}[/yellow]"
            )

        multi_hunk = len(hunk_list) >= 2
        choices = (
            "([green]y[/green]es / [red]n[/red]o / [cyan]a[/cyan]lways this tool"
        )
        if multi_hunk:
            choices = (
                "([green]y[/green])es / ([red]n[/red])o / "
                "([cyan]a[/cyan])lways / ([magenta]h[/magenta])unks"
            )
        try:
            answer = console.input(
                f"[bold]{t('approval.allow')}[/bold] {choices}): "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print(f"[yellow]{t('approval.denied')}[/yellow]")
            return False

        if answer in ("a", "always"):
            tool_name = summary.split("(", 1)[0].strip()
            session.allow_always(tool_name)
            try:
                from kiwimatecoder.config import persist_always_allowed_tool

                persist_always_allowed_tool(tool_name)
                console.print(
                    "[dim]"
                    + t("approval.always_saved", tool=tool_name)
                    + "[/dim]"
                )
            except OSError:
                pass
            return True
        if multi_hunk and answer in ("h", "hunks"):
            return _review_hunks(hunk_list)
        return answer in ("y", "yes")

    return confirm


def _build_history() -> History:
    """Return the prompt history, persisted between sessions when possible."""
    try:
        from kiwimatecoder.config import ensure_config_dir

        return FileHistory(str(ensure_config_dir() / "history"))
    except OSError:
        # Unwritable home directory: keep history for this session only.
        return InMemoryHistory()


def _autosave(session: Session) -> None:
    """Silently persist the session so ``--continue`` can pick it up."""
    from kiwimatecoder.session import save_autosave

    try:
        path = save_autosave(session)
    except OSError:
        return
    if path is not None:
        console.print(
            "[dim]Session saved — resume with "
            "[bold]kiwimatecoder --continue[/bold].[/dim]"
        )


def _make_ask_user(console: Console):
    """Build the interactive callback used by the ask_user tool."""

    def ask(question: str, options: list[str]) -> str:
        console.print()
        console.print(f"[bold yellow]{question}[/bold yellow]")
        for index, option in enumerate(options, 1):
            console.print(f"  [cyan]{index}[/cyan]. {option}")
        answer = console.input("answer> ").strip()
        if options and answer.isdigit():
            position = int(answer)
            if 1 <= position <= len(options):
                return options[position - 1]
        return answer

    return ask


_STEERING_STYLE = "ansibrightblack"


def _steering_prompt() -> HTML:
    from html import escape as html_escape

    return HTML(
        f'<style fg="{_STEERING_STYLE}">{html_escape(t("prompt.steering"))}</style> '
    )


def _resolve_slash_line(line: str, session: Session) -> tuple[str, str] | None:
    """Resolve a line to a custom-template prompt, if one matches.

    Registered slash commands always win. Returns ``("template", prompt)`` for
    a discovered template, or None when the line is a command or unknown.
    """
    text = line.strip()
    if not text.startswith("/"):
        return None
    parts = text[1:].strip().split(maxsplit=1)
    name = parts[0].lower() if parts else ""
    if not name or has_command(name):
        return None
    template = find_template(name, session.workspace_root)
    if template is None:
        return None
    arguments = parts[1] if len(parts) > 1 else ""
    return ("template", render_template(template.body, arguments))


def _extract_image_mentions(text: str, workspace_root: Path) -> tuple[str, list[str]]:
    """Pull ``@path`` image mentions out of a line of user input.

    A token is an image mention when it starts with ``@``, resolves to an
    existing file inside the workspace, and has a supported image suffix.
    Mentions are removed from the returned text; everything else, including
    paths that are missing or outside the workspace, is left untouched.
    """
    kept: list[str] = []
    found: list[str] = []
    for token in text.split():
        if not token.startswith("@") or len(token) < 2:
            kept.append(token)
            continue
        try:
            resolved = resolve_in_workspace(token[1:], workspace_root)
        except PathError:
            kept.append(token)
            continue
        if resolved.is_file() and images.detect_media_type(resolved):
            found.append(str(resolved))
        else:
            kept.append(token)
    return " ".join(kept), found


MAX_MENTION_FILES = 5
MAX_MENTION_FILE_BYTES = 64 * 1024


def _is_text_file(path: Path) -> bool:
    """Whether ``path`` looks like readable text (no NUL byte in the first KB)."""
    try:
        with path.open("rb") as handle:
            sample = handle.read(1024)
    except OSError:
        return False
    return b"\x00" not in sample


def _extract_file_mentions(
    text: str, workspace_root: Path
) -> tuple[str, list[str]]:
    """Pull ``@path`` text-file mentions out of a line of user input.

    A token is a file mention when it starts with ``@``, resolves to an
    existing non-image, non-binary file inside the workspace. Mentions are
    removed from the returned text; everything else, including paths that are
    missing, outside the workspace, images, or binary files, is left untouched.
    """
    kept: list[str] = []
    found: list[str] = []
    for token in text.split():
        if not token.startswith("@") or len(token) < 2:
            kept.append(token)
            continue
        try:
            resolved = resolve_in_workspace(token[1:], workspace_root)
        except PathError:
            kept.append(token)
            continue
        if (
            resolved.is_file()
            and images.detect_media_type(resolved) is None
            and _is_text_file(resolved)
        ):
            found.append(str(resolved))
        else:
            kept.append(token)
    return " ".join(kept), found


def _render_mentioned_file(path: Path, workspace_root: Path) -> str | None:
    """Return a numbered, byte-bounded block for one mentioned file."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            clipped = handle.read(MAX_MENTION_FILE_BYTES)
    except OSError:
        return None
    text = clipped.decode("utf-8", "replace")
    numbered = "\n".join(
        f"{index}\t{line}" for index, line in enumerate(text.splitlines(), 1)
    )
    header = display_path(path, workspace_root)
    if size > len(clipped):
        numbered += (
            f'\n<truncated original_bytes="{size}" shown_bytes="{len(clipped)}" />'
        )
    return f"--- {header} ---\n{numbered or '[empty file]'}"


def _attach_file_mentions(line: str, session: Session) -> str:
    """Prepend a ``[mentioned files]`` block for ``@path`` text mentions.

    Files are read as UTF-8 (bounded to :data:`MAX_MENTION_FILE_BYTES` each)
    and at most :data:`MAX_MENTION_FILES` are included per turn. This is user
    context, so it is deliberately not redacted.
    """
    cleaned, paths = _extract_file_mentions(line, session.workspace_root)
    if not paths:
        return line
    if len(paths) > MAX_MENTION_FILES:
        console.print(
            f"[yellow]File mention limit ({MAX_MENTION_FILES} per turn) reached; "
            f"{len(paths) - MAX_MENTION_FILES} file(s) were not included.[/yellow]"
        )
    blocks: list[str] = []
    for path in paths[:MAX_MENTION_FILES]:
        block = _render_mentioned_file(Path(path), session.workspace_root)
        if block is not None:
            blocks.append(block)
    if not blocks:
        return cleaned
    context = "[mentioned files]\n" + "\n\n".join(blocks)
    if cleaned.strip():
        return context + "\n\n" + cleaned
    return context


def _attach_images(line: str, session: Session) -> str:
    """Stash ``@path`` images from ``line`` for the next turn.

    Respects the configured per-turn image limit and byte cap, printing a dim
    note for each batch. Returns the line with image mentions removed, or a
    default prompt when the line was only image mentions.
    """
    cleaned, paths = _extract_image_mentions(line, session.workspace_root)
    if not paths:
        return line

    vision = get_vision()
    max_images = int(vision["max_images_per_turn"])
    max_bytes = int(vision["max_image_bytes"])
    attached: list[str] = []
    for path in paths:
        if len(session.pending_images) >= max_images:
            console.print(
                f"[yellow]Image limit ({max_images} per turn) reached; "
                f"{Path(path).name} was not attached.[/yellow]"
            )
            break
        try:
            entry = images.encode_image(path, max_bytes)
        except ValueError as exc:
            console.print(f"[yellow]{exc}[/yellow]")
            continue
        session.pending_images.append(entry)
        attached.append(entry["name"])

    if attached:
        label = "image" if len(attached) == 1 else "images"
        console.print(f"[dim]Attached {label}: {', '.join(attached)}[/dim]")
    if not cleaned.strip() and attached:
        return "Please analyze the attached image(s)."
    return cleaned


def _route_steering_line(session: Session, line: str) -> str:
    """Route a line typed while a turn is running.

    Blank input is ignored, slash commands are queued to run after the turn,
    custom template commands are steered like normal text, and anything else is
    queued as steering for the agent. Returns one of ``"steered"``,
    ``"deferred"``, or ``"ignored"``.
    """
    text = line.strip()
    if not text:
        return "ignored"
    if text.startswith("/"):
        resolved = _resolve_slash_line(text, session)
        if resolved is not None:
            session.steering.append(resolved[1])
            return "steered"
        session.deferred_commands.append(text)
        console.print(f"[dim]{t('prompt.command_queued')}[/dim]")
        return "deferred"
    session.steering.append(text)
    return "steered"


async def _dispatch_command(line: str, session: Session) -> str:
    """Run a slash command off the event loop.

    Selector commands call ``Application.run()`` internally, which cannot run
    inside the live asyncio loop, so the whole dispatch happens in a worker.
    """
    return await asyncio.to_thread(
        dispatch,
        line,
        session,
        console,
        _select_command_option,
        None,
        _select_command_options,
    )


async def _process_deferred_commands(session: Session) -> bool:
    """Run queued slash commands FIFO; returns True when one requests exit."""
    while session.deferred_commands:
        command = session.deferred_commands.popleft()
        console.print(f"[dim]→ {command}[/dim]")
        if await _dispatch_command(command, session) == CommandResult.EXIT:
            session.deferred_commands.clear()
            return True
    return False


async def _cancel_task(task: asyncio.Task[Any]) -> None:
    """Cancel ``task`` and wait for it, swallowing prompt interruptions."""
    if not task.done():
        task.cancel()
    try:
        await task
    except (asyncio.CancelledError, KeyboardInterrupt, EOFError):
        pass


async def _steering_input(pt_session: PromptSession[str]) -> tuple[str, str]:
    """Read one steering line, turning interrupts into sentinel outcomes.

    A ``KeyboardInterrupt`` raised inside a separate asyncio task is re-raised
    by ``Task.__step`` and escapes the event loop instead of being stored on the
    task, so it must not cross the task boundary.
    """
    try:
        return await pt_session.prompt_async(_steering_prompt()), "input"
    except KeyboardInterrupt:
        return "", "interrupt"
    except EOFError:
        return "", "eof"


def _notify_turn_finished(elapsed: float) -> None:
    """Send a turn-finished notification when configured and slow enough."""
    settings = get_ui()
    threshold = float(settings["notify_after_seconds"])
    if not notify.should_notify(elapsed, threshold):
        return
    notify.notify(
        "KiwiMateCoder",
        t("prompt.turn_finished", seconds=int(elapsed)),
        mode=settings["notify"],
    )


async def _run_turn_with_steering(
    agent: Agent,
    pt_session: PromptSession[str],
    session: Session,
    line: str,
) -> bool:
    """Run one agent turn while steering input is accepted concurrently.

    Returns True when the REPL should exit (Ctrl-D during the turn).
    """
    started_at = time.monotonic()
    turn_task: asyncio.Task[None] = asyncio.create_task(agent.run_turn(line))
    interrupted = False
    exit_requested = False

    try:
        while not turn_task.done():
            prompt_task: asyncio.Task[tuple[str, str]] = asyncio.create_task(
                _steering_input(pt_session)
            )
            pending: set[asyncio.Task[Any]] = {turn_task, prompt_task}
            try:
                done, _ = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
            except (KeyboardInterrupt, asyncio.CancelledError):
                # Ctrl-C surfaced from the event loop (rather than the prompt)
                # cancels the whole REPL task; treat it like a prompt interrupt.
                await _cancel_task(prompt_task)
                await _cancel_task(turn_task)
                interrupted = True
                break

            if turn_task in done:
                await _cancel_task(prompt_task)
                break

            typed, outcome = prompt_task.result()
            if outcome == "interrupt":
                await _cancel_task(turn_task)
                interrupted = True
                break
            if outcome == "eof":
                await _cancel_task(turn_task)
                console.print(f"[dim]{t('prompt.goodbye')}[/dim]")
                exit_requested = True
                break

            _route_steering_line(session, typed)

        if not interrupted and not exit_requested:
            # Surface unexpected agent errors from the completed turn.
            await turn_task
            _notify_turn_finished(time.monotonic() - started_at)
    except (KeyboardInterrupt, asyncio.CancelledError):
        await _cancel_task(turn_task)
        interrupted = True
    except BaseException:
        await _cancel_task(turn_task)
        raise

    if interrupted:
        console.print(f"\n[yellow]{t('prompt.interrupted')}[/yellow]")

    if await _process_deferred_commands(session):
        exit_requested = True
    return exit_requested


def _run_lifecycle_hooks(
    bus: events.EventBus, event: str, session: Session
) -> None:
    """Emit a session lifecycle event and run its hooks; never raises."""
    try:
        bus.emit(event, workspace=str(session.workspace_root))
        results = hooks.run_hooks(event, session=session, console=console)
    except Exception as exc:  # hooks must never prevent startup or shutdown
        console.print(f"[dim]Hook error during {event}: {exc}[/dim]")
        return
    for result in results:
        if not result.ok:
            status = "timed out" if result.timed_out else f"exit {result.exit_code}"
            console.print(
                f"[dim]Hook {event} failed ({status}): "
                f"{redact(result.command)}[/dim]"
            )


def _load_session_plugins(
    session: Session, bus: events.EventBus
) -> plugins.PluginLoadResult:
    """Load plugins before the agent exists; failures are dim, never fatal."""
    result = plugins.load_plugins(session.workspace_root, console=console, bus=bus)
    for name, reason in result.failed:
        console.print(f"[dim]Plugin '{name}' failed to load: {reason}[/dim]")
    return result


def _load_session_mcp(bus: events.EventBus) -> mcp.McpManager:
    """Connect configured MCP servers; failures are dim warnings, never fatal."""
    manager = mcp.McpManager(console=console, bus=bus)
    mcp.set_manager(manager)
    manager.load()
    return manager


async def _run_interactive(
    session: Session, bus: events.EventBus | None = None
) -> None:
    """Run the async interactive loop until the user exits."""
    bus = bus if bus is not None else events.BUS
    _load_session_plugins(session, bus)
    _load_session_mcp(bus)
    console.print(_banner(session))
    _run_lifecycle_hooks(bus, events.SESSION_START, session)
    confirm = _make_confirm(session)
    session.ask_user = _make_ask_user(console)
    ui_config = ui.ui_config()
    agent = Agent(
        session,
        console,
        confirm,
        bus=bus,
        ascii_mode=ui_config["ascii"],
        output_mode=ui_config["output_mode"],
        spinner=ui_config["spinner"],
    )

    kb = KeyBindings()

    @kb.add("escape", "enter")
    def _(event: KeyPressEvent) -> None:
        """Alt+Enter / Escape+Enter inserts newline for multi-line prompts."""
        event.current_buffer.insert_text("\n")

    pt_session: PromptSession[str] = PromptSession(
        history=_build_history(),
        completer=SlashCommandCompleter(session),
        complete_while_typing=True,
        complete_style=CompleteStyle.MULTI_COLUMN,
        key_bindings=kb,
        vi_mode=ui.vi_mode_enabled(),
    )

    multiline_buffer: list[str] = []
    in_multiline_block = False

    try:
        while True:
            try:
                prompt_str = (
                    HTML("<ansicyan>... </ansicyan>")
                    if in_multiline_block
                    else _prompt_text(session)
                )
                line = await pt_session.prompt_async(prompt_str)
            except (KeyboardInterrupt, asyncio.CancelledError):
                # Ctrl-C at the prompt: clear the line / buffer, keep going.
                multiline_buffer.clear()
                in_multiline_block = False
                continue
            except EOFError:
                # Ctrl-D: exit.
                console.print(f"[dim]{t('prompt.goodbye')}[/dim]")
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
                resolved = _resolve_slash_line(line, session)
                if resolved is not None:
                    # A custom template runs as a normal agent turn.
                    turn_line = _attach_file_mentions(
                        _attach_images(resolved[1], session), session
                    )
                    if await _run_turn_with_steering(
                        agent, pt_session, session, turn_line
                    ):
                        break
                    continue
                if await _dispatch_command(line, session) == CommandResult.EXIT:
                    break
                continue

            line = _attach_file_mentions(_attach_images(line, session), session)
            if not line.strip():
                continue
            if await _run_turn_with_steering(agent, pt_session, session, line):
                break
    finally:
        _run_lifecycle_hooks(bus, events.SESSION_END, session)
        current_mcp = mcp.get_manager()
        if current_mcp is not None:
            current_mcp.shutdown()
        mcp.set_manager(None)
        current_lsp = lsp.get_manager()
        if current_lsp is not None:
            current_lsp.shutdown()
        lsp.set_manager(None)
        browser.reset_driver()
        close_shell(session)
        _autosave(session)


def run(session: Session, bus: events.EventBus | None = None) -> None:
    """Run the interactive loop until the user exits."""
    global console
    # Apply the persisted locale before anything is rendered.
    apply_config_locale()
    # Rebuild the console for this session so the persisted color setting and
    # NO_COLOR/FORCE_COLOR take effect at launch (Rich auto-detects the TTY).
    console = ui.make_console()
    asyncio.run(_run_interactive(session, bus=bus))
