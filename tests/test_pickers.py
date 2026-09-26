"""Grouped pickers: one cursor across framed option groups, keys, mouse and rendering."""

from __future__ import annotations

import io
import re

import pytest
from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Point, Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import ColorDepth, DummyOutput
from prompt_toolkit.output.vt100 import Vt100_Output

from kiwimatecoder.pickers import (
    _build_application,
    _GroupedPicker,
    grouped_checkbox_choice,
    grouped_choice,
)

DOWN = "\x1b[B"
UP = "\x1b[A"
ENTER = "\r"

GROUPS = [
    ("Experimental", [("kiwimate", "KiwiMate")]),
    (
        None,
        [
            ("openrouter", "OpenRouter"),
            ("openai", "OpenAI"),
            ("anthropic", "Anthropic"),
        ],
    ),
]


def _choose(keys, **kwargs):
    with create_pipe_input() as pipe_input:
        pipe_input.send_text(keys)
        return grouped_choice(
            "Select a provider", input=pipe_input, output=DummyOutput(), **kwargs
        )


def _check(keys, **kwargs):
    with create_pipe_input() as pipe_input:
        pipe_input.send_text(keys)
        return grouped_checkbox_choice(
            "Select providers", input=pipe_input, output=DummyOutput(), **kwargs
        )


# ---------------------------------------------------------------------------
# Single choice
# ---------------------------------------------------------------------------


def test_down_from_experimental_row_lands_in_main_box():
    assert _choose(ENTER, groups=GROUPS) == "kiwimate"
    assert _choose(DOWN + ENTER, groups=GROUPS) == "openrouter"


def test_default_preselects_row_in_second_box_and_up_crosses_back():
    assert _choose(ENTER, groups=GROUPS, default="openai") == "openai"
    assert _choose(UP + ENTER, groups=GROUPS, default="openai") == "openrouter"
    assert _choose(UP + UP + ENTER, groups=GROUPS, default="openai") == "kiwimate"


def test_cursor_clamps_at_both_ends_and_supports_vi_keys():
    assert _choose(UP + UP + ENTER, groups=GROUPS) == "kiwimate"
    assert _choose("jjjjjj" + ENTER, groups=GROUPS) == "anthropic"
    assert _choose("jjk" + ENTER, groups=GROUPS) == "openrouter"


PAGE_UP = "\x1b[5~"
PAGE_DOWN = "\x1b[6~"
HOME = "\x1b[H"
END = "\x1b[F"

LONG_GROUPS = [
    ("Experimental", [("kiwimate", "KiwiMate")]),
    (None, [(f"p{index:02d}", f"Provider {index:02d}") for index in range(1, 25)]),
]


def test_page_keys_move_ten_rows_across_groups_and_clamp():
    assert _choose(PAGE_DOWN + ENTER, groups=LONG_GROUPS) == "p10"
    assert _choose(PAGE_DOWN * 5 + ENTER, groups=LONG_GROUPS) == "p24"
    assert _choose(PAGE_UP + ENTER, groups=LONG_GROUPS, default="p12") == "p02"
    assert _choose(PAGE_UP * 3 + ENTER, groups=LONG_GROUPS, default="p12") == "kiwimate"


def test_home_and_end_jump_to_the_first_and_last_rows():
    assert _choose(END + ENTER, groups=LONG_GROUPS) == "p24"
    assert _choose(HOME + ENTER, groups=LONG_GROUPS, default="p12") == "kiwimate"


def test_typing_a_letter_finds_the_next_matching_label_across_groups():
    groups = [
        ("Experimental", [("kiwimate", "KiwiMate — chat only")]),
        (None, [("openai", "OpenAI"), ("openrouter", "OpenRouter"), ("groq", "Groq")]),
    ]

    assert _choose("o" + ENTER, groups=groups) == "openai"
    # Repeating the letter cycles through the matches, then wraps around.
    assert _choose("oo" + ENTER, groups=groups) == "openrouter"
    assert _choose("ooo" + ENTER, groups=groups) == "openai"
    assert _choose("K" + ENTER, groups=groups, default="groq") == "kiwimate"
    # No match leaves the cursor where it was.
    assert _choose("z" + ENTER, groups=groups, default="groq") == "groq"


def test_checkbox_supports_page_and_find_keys():
    assert _check(END + " " + ENTER, groups=LONG_GROUPS) == ["p24"]
    assert _check("P" + " " + ENTER, groups=LONG_GROUPS) == ["p01"]


def test_unknown_default_starts_on_first_row():
    assert _choose(ENTER, groups=GROUPS, default="missing") == "kiwimate"


def test_digit_jumps_across_groups():
    assert _choose("3" + ENTER, groups=GROUPS, default="kiwimate") == "openai"
    assert _choose("1" + ENTER, groups=GROUPS, default="anthropic") == "kiwimate"
    # Past the last row clamps, like prompt_toolkit's choice().
    assert _choose("9" + ENTER, groups=GROUPS) == "anthropic"


def test_empty_group_is_skipped_and_does_not_affect_numbering():
    groups = [("Experimental", []), (None, [("a", "A"), ("b", "B")])]

    assert _choose(ENTER, groups=groups) == "a"
    assert _choose("2" + ENTER, groups=groups) == "b"


def test_empty_lazily_built_group_is_skipped():
    # Callers may pass generator expressions: an empty one is truthy but has no rows.
    def groups():
        experimental: list[tuple[str, str]] = []
        return [
            ("Experimental", (option for option in experimental)),
            (None, [("a", "A"), ("b", "B")]),
        ]

    assert _choose("2" + ENTER, groups=groups()) == "b"
    assert _check(DOWN + " " + ENTER, groups=groups()) == ["b"]


def test_grouped_choice_without_options_raises():
    with pytest.raises(ValueError):
        grouped_choice("Pick", groups=[("Experimental", []), (None, [])])


# ---------------------------------------------------------------------------
# Multiple choice
# ---------------------------------------------------------------------------


def test_checkbox_returns_values_in_check_order_across_boxes():
    # Check OpenAI (second box) first, then KiwiMate (first box).
    keys = DOWN + DOWN + " " + UP + UP + " " + ENTER

    assert _check(keys, groups=GROUPS) == ["openai", "kiwimate"]


def test_checkbox_defaults_come_first_in_given_order():
    # The cursor starts on the first default (Anthropic, row 4).
    keys = UP + UP + UP + " " + ENTER

    defaults = ["anthropic", "openrouter"]

    result = _check(keys, groups=GROUPS, default_values=defaults)

    assert result == ["anthropic", "openrouter", "kiwimate"]
    assert defaults == ["anthropic", "openrouter"]  # The caller's list is not mutated.


def test_checkbox_unchecking_removes_and_rechecking_appends():
    assert _check(" " + UP + " " + ENTER, groups=GROUPS, default_values=["openrouter"]) == [
        "kiwimate"
    ]
    assert _check(
        "  " + ENTER, groups=GROUPS, default_values=["openrouter", "openai"]
    ) == ["openai", "openrouter"]


def test_checkbox_ignores_unknown_and_duplicate_defaults():
    result = _check(ENTER, groups=GROUPS, default_values=["missing", "openai", "openai"])

    assert result == ["openai"]


def test_checkbox_without_options_returns_empty_list():
    assert grouped_checkbox_choice("Pick", groups=[("Experimental", []), (None, [])]) == []
    assert grouped_checkbox_choice("Pick", groups=[]) == []


def test_checkbox_empty_group_is_skipped():
    groups = [("Experimental", []), (None, [("a", "A"), ("b", "B")])]

    assert _check(DOWN + " " + ENTER, groups=groups) == ["b"]


@pytest.mark.parametrize("picker", [grouped_choice, grouped_checkbox_choice])
def test_ctrl_c_raises_keyboard_interrupt(picker):
    with create_pipe_input() as pipe_input:
        pipe_input.send_text("\x03")
        with pytest.raises(KeyboardInterrupt):
            picker("Pick", groups=GROUPS, input=pipe_input, output=DummyOutput())


# ---------------------------------------------------------------------------
# Focus and mouse
# ---------------------------------------------------------------------------


def _app(groups, *, multiple=False, **kwargs):
    picker = _GroupedPicker(groups, multiple=multiple, **kwargs)
    with create_pipe_input() as pipe_input:
        app = _build_application("Pick", picker, input=pipe_input, output=DummyOutput())
    return picker, app


def _paint(app, width=60, height=16):
    """Draw one frame of ``app`` into a screen; return its lines and mouse handlers."""
    screen = Screen()
    handlers = MouseHandlers()
    with set_app(app):
        # What ``Application._redraw`` does first: controls cache fragments per
        # render counter, and mouse handlers only reach windows it has linked.
        app.render_counter += 1
        app.layout.update_parents_relations()
        app.layout.container.write_to_screen(
            screen,
            handlers,
            WritePosition(xpos=0, ypos=0, width=width, height=height),
            parent_style="",
            erase_bg=False,
            z_index=None,
        )
    lines = [
        "".join(screen.data_buffer[y][x].char for x in range(width)).rstrip()
        for y in range(height)
    ]
    return lines, handlers


def _click(app, text):
    """Click the first screen line containing ``text`` through the real handlers."""
    lines, handlers = _paint(app)
    y = next(y for y, line in enumerate(lines) if text in line)
    x = lines[y].index(text)
    with set_app(app):
        handlers.mouse_handlers[y][x](
            MouseEvent(
                position=Point(x=x, y=y),
                event_type=MouseEventType.MOUSE_UP,
                button=MouseButton.LEFT,
                modifiers=frozenset(),
            )
        )


@pytest.mark.parametrize(
    ("keys", "default", "focused"),
    [(DOWN, None, 1), (UP, "openrouter", 0), ("4", None, 1)],
)
def test_keyboard_moves_focus_to_the_box_holding_the_cursor(keys, default, focused):
    picker = _GroupedPicker(GROUPS, multiple=False, default_values=[default] if default else [])
    with create_pipe_input() as pipe_input:
        app = _build_application("Pick", picker, input=pipe_input, output=DummyOutput())
        assert app.layout.current_window is picker.current_window
        pipe_input.send_text(keys + ENTER)
        app.run()

    assert app.layout.current_window is picker.windows[focused]
    # The cursor row (and only it) carries the terminal cursor position.
    marks = [
        index
        for index in range(len(picker.windows))
        if ("[SetCursorPosition]", "") in [f[:2] for f in picker._fragments(index)]
    ]
    assert marks == [focused]


def test_mouse_click_moves_cursor_toggles_and_focuses_the_clicked_box():
    picker, app = _app(GROUPS, multiple=True)
    assert app.layout.current_window is picker.windows[0]

    _click(app, "OpenAI")

    assert picker.cursor == 2
    assert picker.checked_values == ["openai"]
    assert app.layout.current_window is picker.windows[1]


@pytest.mark.parametrize("multiple", [False, True])
def test_mouse_click_on_second_line_of_label_selects_that_row(multiple):
    groups = [
        ("Experimental", [("kiwimate", "KiwiMate\n      chat only, no tools")]),
        (None, [("openrouter", "OpenRouter"), ("openai", "OpenAI")]),
    ]
    picker, app = _app(groups, multiple=multiple, default_values=["openai"])

    _click(app, "chat only")

    assert picker.cursor == 0
    assert app.layout.current_window is picker.windows[0]
    if multiple:
        assert picker.checked_values == ["openai", "kiwimate"]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _render(picker, **kwargs):
    """Run ``picker`` against a vt100 output captured in a string; accept at once."""
    stdout = io.StringIO()
    output = Vt100_Output(
        stdout,
        get_size=lambda: Size(rows=40, columns=60),
        default_color_depth=ColorDepth.DEPTH_8_BIT,
    )
    with create_pipe_input() as pipe_input:
        pipe_input.send_text(ENTER)
        picker("Select a provider", input=pipe_input, output=output, **kwargs)
    return stdout.getvalue()


@pytest.mark.parametrize("picker", [grouped_choice, grouped_checkbox_choice])
def test_render_draws_experimental_box_above_main_box(picker):
    raw = _render(picker, groups=GROUPS)
    text = _ESCAPE.sub("", raw)

    assert "Select a provider" in text
    title = text.index("Experimental")
    kiwimate = text.index("KiwiMate")
    openrouter = text.index("OpenRouter")
    assert title < kiwimate < openrouter
    # KiwiMate's box closes before the main box opens.
    between = text[kiwimate:openrouter]
    assert "┘" in between and "┌" in between

    # The titled box is coloured through the "option-group" style rules.
    match = re.search(r"\x1b\[([0-9;]*)m┌─*\|\x1b\[([0-9;]*)m Experimental ", raw)
    assert match is not None
    assert "33" in match.group(1).split(";")
    assert {"1", "33"} <= set(match.group(2).split(";"))


def test_render_numbers_rows_continuously():
    text = _ESCAPE.sub("", _render(grouped_choice, groups=GROUPS, default="openai"))

    assert "1. KiwiMate" in text
    assert "2. OpenRouter" in text
    assert "3. OpenAI" in text


@pytest.mark.parametrize(
    ("multiple", "rows"),
    [
        (False, ["│    1. KiwiMate", "│    2. OpenRouter", "│ >  3. OpenAI", "│    4. Anthropic"]),
        (True, ["│   [ ] KiwiMate", "│   [ ] OpenRouter", "│ > [*] OpenAI", "│   [ ] Anthropic"]),
    ],
)
def test_screen_shows_cursor_on_default_row_in_second_box(multiple, rows):
    _, app = _app(GROUPS, multiple=multiple, default_values=["openai"])
    lines, _ = _paint(app)

    assert "| Experimental |" in lines[1]
    assert lines[2].startswith(rows[0])
    assert lines[3].startswith("└") and lines[4].startswith("┌")
    for line, row in zip(lines[5:8], rows[1:]):
        assert line.startswith(row + " ")


def test_render_checkbox_rows():
    text = _ESCAPE.sub(
        "", _render(grouped_checkbox_choice, groups=GROUPS, default_values=["openai"])
    )

    assert "[ ] KiwiMate" in text
    assert "[*] OpenAI" in text


def test_render_skips_empty_titled_group():
    groups = [("Experimental", []), (None, [("openrouter", "OpenRouter")])]

    text = _ESCAPE.sub("", _render(grouped_choice, groups=groups))

    assert "Experimental" not in text
    assert "1. OpenRouter" in text


def test_render_without_frame_keeps_titled_box_only():
    text = _ESCAPE.sub("", _render(grouped_choice, groups=GROUPS, show_frame=False))
    lines = text.split("\n")

    assert "Experimental" in text
    assert any("│" in line for line in lines if "KiwiMate" in line)
    assert all("│" not in line for line in lines if "OpenRouter" in line)
