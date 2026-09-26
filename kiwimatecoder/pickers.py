"""Grouped option pickers: ``choice``-style prompts split into framed boxes.

Provider pickers show the experimental providers (currently KiwiMate) at the
top in their own box titled "Experimental", above the normal list in a second
box. prompt_toolkit's ``choice`` and ``repl.checkbox_choice`` render a single
flat list, so this module draws each group of options in its own
:class:`~prompt_toolkit.widgets.Frame` while one flat cursor moves across all
of them as if they were a single list.

:func:`grouped_choice` picks one value (rows numbered continuously across
groups, digits 1-9 jump); :func:`grouped_checkbox_choice` picks several and
returns them in the order they were checked, so the first checked value can be
treated as the primary one.

The module only depends on prompt_toolkit so ``main`` can import it without
pulling in the REPL.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any, Generic, TypeAlias, TypeVar

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.filters import Condition, is_done, renderer_height_is_known
from prompt_toolkit.formatted_text import (
    AnyFormattedText,
    StyleAndTextTuples,
    fragment_list_to_text,
    to_formatted_text,
)
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
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
from prompt_toolkit.output import Output
from prompt_toolkit.shortcuts.choice_input import create_default_choice_input_style
from prompt_toolkit.styles import BaseStyle, Style, merge_styles
from prompt_toolkit.widgets import Box, Frame, Label

__all__ = ["OptionGroup", "grouped_checkbox_choice", "grouped_choice"]

_T = TypeVar("_T")

# Rows moved by PageUp/PageDown.
_PAGE_ROWS = 10

# ``(title, options)``: a titled group is drawn in its own titled box; an
# untitled one (``None``) in a plain box, or unframed when ``show_frame`` is off.
OptionGroup: TypeAlias = tuple[str | None, Sequence[tuple[_T, AnyFormattedText]]]


class _GroupedPicker(Generic[_T]):
    """Flat cursor and check state spanning every group, plus one window per group."""

    def __init__(
        self,
        groups: Sequence[OptionGroup[_T]],
        *,
        multiple: bool,
        default_values: Sequence[_T] = (),
    ) -> None:
        self.multiple = multiple
        self.values: list[_T] = []
        self.labels: list[AnyFormattedText] = []
        # ``(title, start, end)`` per non-empty group; its rows are values[start:end].
        self.groups: list[tuple[str | None, int, int]] = []
        for title, options in groups:
            start = len(self.values)
            for value, label in options:
                self.values.append(value)
                self.labels.append(label)
            # Tested after iterating: an empty generator is truthy but adds no rows.
            if len(self.values) > start:
                self.groups.append((title, start, len(self.values)))

        present = [row for row in map(self._row_of, default_values) if row is not None]
        self.cursor = present[0] if present else 0
        # Checked rows in the order they were checked (multi-choice only).
        self.checked: list[int] = list(dict.fromkeys(present)) if multiple else []

        self.windows = [
            Window(
                FormattedTextControl(
                    partial(self._fragments, index), focusable=True, show_cursor=False
                ),
                style="class:input-selection",
                dont_extend_height=True,
            )
            for index in range(len(self.groups))
        ]

    def _row_of(self, value: _T) -> int | None:
        try:
            return self.values.index(value)
        except ValueError:
            return None

    @property
    def current_window(self) -> Window:
        """The window of the group holding the cursor."""
        for window, (_, start, end) in zip(self.windows, self.groups):
            if start <= self.cursor < end:
                return window
        return self.windows[0]

    @property
    def checked_values(self) -> list[_T]:
        return [self.values[row] for row in self.checked]

    def move_to(self, row: int) -> None:
        """Move the cursor (clamped, no wrap) and focus the window it lands in."""
        self.cursor = max(0, min(len(self.values) - 1, row))
        # Focusing the cursor's window keeps the terminal cursor on that row.
        get_app().layout.focus(self.current_window)

    def find(self, text: str) -> None:
        """Move to the next row whose label starts with ``text`` (type-to-find)."""
        needle = text.lower()
        if not needle.strip():
            return
        count = len(self.values)
        for step in range(1, count + 1):
            row = (self.cursor + step) % count
            label = fragment_list_to_text(to_formatted_text(self.labels[row]))
            if label.lower().startswith(needle):
                self.move_to(row)
                return

    def toggle(self) -> None:
        if self.cursor in self.checked:
            self.checked.remove(self.cursor)
        else:
            self.checked.append(self.cursor)

    def _click(self, row: int, mouse_event: MouseEvent) -> None:
        if mouse_event.event_type == MouseEventType.MOUSE_UP:
            self.move_to(row)
            if self.multiple:
                self.toggle()

    def _fragments(self, index: int) -> StyleAndTextTuples:
        _, start, end = self.groups[index]
        result: StyleAndTextTuples = []
        for row in range(start, end):
            fragments: StyleAndTextTuples = []
            if self.multiple:
                self._checkbox_row(row, fragments)
            else:
                self._choice_row(row, fragments)
            fragments.append(("", "\n"))
            # A handler bound to the row itself (not the clicked line number)
            # keeps clicks right when a label spans several lines.
            handler = partial(self._click, row)
            result.extend((fragment[0], fragment[1], handler) for fragment in fragments)
        result.pop()
        return result

    def _choice_row(self, row: int, result: StyleAndTextTuples) -> None:
        # Mirrors the RadioList rows drawn by prompt_toolkit's ``choice``.
        selected = row == self.cursor
        style = " class:selected-option" if selected else ""
        if selected:
            result.append(("[SetCursorPosition]", ""))
        result.append((style, ">" if selected else " "))
        result.append((f"{style} class:option", " "))
        result.append((f"{style} class:number", f"{row + 1:2d}. "))
        result.extend(to_formatted_text(self.labels[row], style=f"{style} class:option"))

    def _checkbox_row(self, row: int, result: StyleAndTextTuples) -> None:
        # Mirrors ``repl._CheckboxChoiceList`` rows.
        checked = row in self.checked
        selected = row == self.cursor
        text_style = "class:option"
        if checked:
            text_style += " class:selected-option"
        if selected:
            text_style += " class:focused-option"

        result.append(("class:cursor" if selected else "", "> " if selected else "  "))
        if selected:
            result.append(("[SetCursorPosition]", ""))
        result.append(
            ("class:checkbox-checked", "[*] ")
            if checked
            else ("class:checkbox-unchecked", "[ ] ")
        )
        result.extend(to_formatted_text(self.labels[row], style=text_style))


def _picker_style(style: BaseStyle | None) -> BaseStyle:
    """``choice`` defaults, the checkbox colours and the titled-group colours."""
    return merge_styles(
        [
            create_default_choice_input_style(),
            Style.from_dict(
                {
                    "cursor": "bold cyan",
                    "checkbox-checked": "bold green",
                    "checkbox-unchecked": "dim",
                    "focused-option": "bold",
                    # Frame styles its container "class:frame class:option-group"
                    # and its border/title windows "class:frame.border"/".label".
                    "option-group frame.border": "ansiyellow",
                    "option-group frame.label": "bold ansiyellow",
                }
            ),
            style if style is not None else Style([]),
        ]
    )


def _build_application(
    message: AnyFormattedText,
    picker: _GroupedPicker[_T],
    *,
    bottom_toolbar: AnyFormattedText = None,
    show_frame: bool = True,
    mouse_support: bool = False,
    style: BaseStyle | None = None,
    input: Input | None = None,
    output: Output | None = None,
) -> Application[Any]:
    """Lay out the message, one box per group and the toolbar; bind the keys."""
    boxes: list[AnyContainer] = [
        Box(
            Label(text=message, dont_extend_height=True),
            padding_top=0,
            padding_left=1,
            padding_right=1,
            padding_bottom=0,
        )
    ]
    for window, (title, _, _) in zip(picker.windows, picker.groups):
        if title or show_frame:
            body = Box(window, padding_top=0, padding_left=1, padding_right=1, padding_bottom=0)
            if title:
                boxes.append(Frame(body, title=title, style="class:option-group"))
            else:
                boxes.append(Frame(body))
        else:
            # One extra column keeps the rows aligned with framed groups.
            boxes.append(
                Box(window, padding_top=0, padding_left=2, padding_right=1, padding_bottom=0)
            )

    show_bottom_toolbar = (
        Condition(lambda: bottom_toolbar is not None) & ~is_done & renderer_height_is_known
    )
    toolbar_container = ConditionalContainer(
        Window(
            FormattedTextControl(lambda: bottom_toolbar, style="class:bottom-toolbar.text"),
            style="class:bottom-toolbar",
            dont_extend_height=True,
            height=Dimension(min=1),
        ),
        filter=show_bottom_toolbar,
    )
    layout = Layout(
        HSplit(
            [
                HSplit(boxes),
                ConditionalContainer(Window(), filter=show_bottom_toolbar),
                toolbar_container,
            ]
        ),
        focused_element=picker.current_window,
    )

    # Bound at the application level so they work whichever group has focus.
    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")  # Vi-like.
    def _up(event: KeyPressEvent) -> None:
        picker.move_to(picker.cursor - 1)

    @kb.add("down")
    @kb.add("j")  # Vi-like.
    def _down(event: KeyPressEvent) -> None:
        picker.move_to(picker.cursor + 1)

    @kb.add("pageup")
    def _page_up(event: KeyPressEvent) -> None:
        picker.move_to(picker.cursor - _PAGE_ROWS)

    @kb.add("pagedown")
    def _page_down(event: KeyPressEvent) -> None:
        picker.move_to(picker.cursor + _PAGE_ROWS)

    @kb.add("home")
    def _first(event: KeyPressEvent) -> None:
        picker.move_to(0)

    @kb.add("end")
    def _last(event: KeyPressEvent) -> None:
        picker.move_to(len(picker.values) - 1)

    @kb.add(Keys.Any)
    def _type_to_find(event: KeyPressEvent) -> None:
        # Like prompt_toolkit's lists: a letter jumps to the next matching row.
        picker.find(event.data)

    if picker.multiple:

        @kb.add(" ")
        def _toggle(event: KeyPressEvent) -> None:
            picker.toggle()

    else:
        for digit in range(1, 10):

            @kb.add(str(digit))
            def _jump(event: KeyPressEvent, row: int = digit - 1) -> None:
                picker.move_to(row)

    @kb.add("enter", eager=True)
    def _accept_input(event: KeyPressEvent) -> None:
        if picker.multiple:
            event.app.exit(result=picker.checked_values, style="class:accepted")
        else:
            event.app.exit(result=picker.values[picker.cursor], style="class:accepted")

    @kb.add("c-c", eager=True)
    @kb.add("<sigint>", eager=True)
    def _keyboard_interrupt(event: KeyPressEvent) -> None:
        event.app.exit(exception=KeyboardInterrupt(), style="class:aborting")

    return Application(
        layout=layout,
        full_screen=False,
        mouse_support=mouse_support,
        key_bindings=kb,
        style=_picker_style(style),
        input=input,
        output=output,
    )


def grouped_choice(
    message: AnyFormattedText,
    *,
    groups: Sequence[OptionGroup[_T]],
    default: _T | None = None,
    bottom_toolbar: AnyFormattedText = None,
    show_frame: bool = True,
    mouse_support: bool = False,
    style: BaseStyle | None = None,
    input: Input | None = None,
    output: Output | None = None,
) -> _T:
    """Single-choice prompt like ``choice`` whose options are split into boxes.

    Empty groups are skipped. Raises ``ValueError`` when no group has options
    and ``KeyboardInterrupt`` on Ctrl-C.
    """
    picker = _GroupedPicker(
        groups, multiple=False, default_values=[default] if default is not None else []
    )
    if not picker.values:
        raise ValueError("grouped_choice() needs at least one option")
    app: Application[_T] = _build_application(
        message,
        picker,
        bottom_toolbar=bottom_toolbar,
        show_frame=show_frame,
        mouse_support=mouse_support,
        style=style,
        input=input,
        output=output,
    )
    return app.run()


def grouped_checkbox_choice(
    message: AnyFormattedText,
    *,
    groups: Sequence[OptionGroup[_T]],
    default_values: Sequence[_T] | None = None,
    bottom_toolbar: AnyFormattedText = None,
    show_frame: bool = True,
    mouse_support: bool = False,
    style: BaseStyle | None = None,
    input: Input | None = None,
    output: Output | None = None,
) -> list[_T]:
    """Checklist like ``repl.checkbox_choice`` whose options are split into boxes.

    Returns the checked values in the order they were checked, starting with
    ``default_values`` in their given order. Empty groups are skipped; with no
    options at all it returns ``[]`` without prompting. Raises
    ``KeyboardInterrupt`` on Ctrl-C.
    """
    picker = _GroupedPicker(groups, multiple=True, default_values=default_values or ())
    if not picker.values:
        return []
    app: Application[list[_T]] = _build_application(
        message,
        picker,
        bottom_toolbar=bottom_toolbar,
        show_frame=show_frame,
        mouse_support=mouse_support,
        style=style,
        input=input,
        output=output,
    )
    return app.run()
