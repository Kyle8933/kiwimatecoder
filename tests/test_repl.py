import base64
import difflib
import io
import os
import re
import stat
import sys
from pathlib import Path

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.history import FileHistory
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.data_structures import Size
from prompt_toolkit.output import ColorDepth, DummyOutput
from prompt_toolkit.output.vt100 import Vt100_Output
from rich.console import Console

from kiwimatecoder import config
from kiwimatecoder.commands import (
    CONFIG_KEY_SECTIONS,
    CONFIG_KEY_SET_ACTIONS,
    CommandOption,
    CommandResult,
    MultiSelectionPrompt,
    SelectionPrompt,
    dispatch,
    line_carries_secret,
)
from kiwimatecoder.hunks import parse_hunk_selection
from kiwimatecoder.permissions import ApprovalResult
from kiwimatecoder.providers import REGISTRY
from kiwimatecoder.repl import (
    MAX_MENTION_FILE_BYTES,
    MAX_MENTION_FILES,
    SlashCommandCompleter,
    _attach_file_mentions,
    _attach_images,
    _banner,
    _build_history,
    _PrivateHistory,
    _dispatch_command,
    _extract_file_mentions,
    _extract_image_mentions,
    _make_confirm,
    _notify_turn_finished,
    _option_groups,
    _process_deferred_commands,
    _prompt_text,
    _read_command_input,
    _read_secret_input,
    _resolve_slash_line,
    _route_steering_line,
    _select_command_option,
    _select_command_options,
    _workspace_file_candidates,
    checkbox_choice,
)


def _completion_texts(text: str) -> list[str]:
    completer = SlashCommandCompleter()
    return [
        completion.text
        for completion in completer.get_completions(Document(text), CompleteEvent())
    ]


def test_slash_completer_lists_commands_at_slash():
    completions = _completion_texts("/")

    assert "/help" in completions
    assert "/context" in completions
    assert "/config" in completions
    assert "/cost" in completions


def test_slash_completer_filters_commands_as_user_types():
    completions = _completion_texts("/co")

    assert "/context" in completions
    assert "/cost" in completions
    assert "/model" not in completions


def test_slash_completer_completes_mode_values():
    completions = _completion_texts("/mode p")

    assert completions == ["plan"]


def test_slash_completer_completes_config_actions():
    completions = _completion_texts("/config m")

    assert "model" in completions
    assert "models" in completions


def _fail_grouped(**kwargs):
    raise AssertionError("ungrouped options must use the plain picker")


def test_command_selector_renders_prompt_options(monkeypatch):
    captured = {}

    def fake_choice(**kwargs):
        captured.update(kwargs)
        return "model-b"

    monkeypatch.setattr("kiwimatecoder.repl.choice", fake_choice)
    monkeypatch.setattr("kiwimatecoder.repl.grouped_choice", _fail_grouped)
    prompt = SelectionPrompt(
        title="Select model",
        text="Choose one",
        options=(
            CommandOption("model-a", "Model A"),
            CommandOption("model-b", "Model B"),
        ),
        selected="model-a",
    )

    assert _select_command_option(prompt) == "model-b"
    assert captured["options"] == [("model-a", "Model A"), ("model-b", "Model B")]
    assert captured["default"] == "model-a"
    assert captured["show_frame"] is True


def test_command_multi_selector_renders_prompt_options(monkeypatch):
    captured = {}

    def fake_checkbox_choice(**kwargs):
        captured.update(kwargs)
        return ["openrouter", "openai"]

    monkeypatch.setattr("kiwimatecoder.repl.checkbox_choice", fake_checkbox_choice)
    monkeypatch.setattr("kiwimatecoder.repl.grouped_checkbox_choice", _fail_grouped)
    prompt = MultiSelectionPrompt(
        title="Select active providers",
        text="Check every provider you want",
        options=(
            CommandOption("openrouter", "OpenRouter"),
            CommandOption("openai", "OpenAI"),
            CommandOption("anthropic", "Anthropic"),
        ),
        selected=("openrouter",),
    )

    assert _select_command_options(prompt) == ["openrouter", "openai"]
    assert captured["message"] == "Select active providers\nCheck every provider you want"
    assert captured["options"] == [
        ("openrouter", "OpenRouter"),
        ("openai", "OpenAI"),
        ("anthropic", "Anthropic"),
    ]
    assert captured["default_values"] == ("openrouter",)
    assert captured["show_frame"] is True
    assert "Space toggle" in captured["bottom_toolbar"]


def test_command_multi_selector_returns_none_on_interrupt(monkeypatch):
    def fake_checkbox_choice(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("kiwimatecoder.repl.checkbox_choice", fake_checkbox_choice)
    prompt = MultiSelectionPrompt(
        title="Select active providers",
        text="Check providers",
        options=(CommandOption("openrouter", "OpenRouter"),),
    )

    assert _select_command_options(prompt) is None


def _fail_plain(**kwargs):
    raise AssertionError("grouped options must use the grouped picker")


_GROUPED_PROVIDER_OPTIONS = (
    CommandOption("openrouter", "OpenRouter — no model chosen (primary)"),
    CommandOption(
        "kiwimate",
        "KiwiMate — kiwimate.net · chat only, no tool use yet · no model chosen",
        group="Experimental",
    ),
    CommandOption("openai", "OpenAI — gpt-5.5"),
)


def test_command_selector_routes_grouped_options_to_grouped_choice(monkeypatch):
    captured = {}

    def fake_grouped_choice(message, **kwargs):
        captured.update(kwargs, message=message)
        return "kiwimate"

    monkeypatch.setattr("kiwimatecoder.repl.grouped_choice", fake_grouped_choice)
    monkeypatch.setattr("kiwimatecoder.repl.choice", _fail_plain)
    prompt = SelectionPrompt(
        title="Select provider",
        text="Choose the provider to use for this session.",
        options=_GROUPED_PROVIDER_OPTIONS,
        selected="openrouter",
    )

    assert _select_command_option(prompt) == "kiwimate"
    assert captured["message"] == (
        "Select provider\nChoose the provider to use for this session."
    )
    assert captured["groups"] == [
        (
            "Experimental",
            [
                (
                    "kiwimate",
                    "KiwiMate — kiwimate.net · chat only, no tool use yet · "
                    "no model chosen",
                )
            ],
        ),
        (
            None,
            [
                ("openrouter", "OpenRouter — no model chosen (primary)"),
                ("openai", "OpenAI — gpt-5.5"),
            ],
        ),
    ]
    assert captured["default"] == "openrouter"
    assert captured["show_frame"] is True
    assert "Enter select" in captured["bottom_toolbar"]


def test_command_selector_grouped_choice_interrupt_returns_none(monkeypatch):
    def fake_grouped_choice(message, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("kiwimatecoder.repl.grouped_choice", fake_grouped_choice)
    prompt = SelectionPrompt(
        title="Select provider",
        text="Choose",
        options=_GROUPED_PROVIDER_OPTIONS,
    )

    assert _select_command_option(prompt) is None


def test_command_multi_selector_routes_grouped_options_to_grouped_checklist(
    monkeypatch,
):
    captured = {}

    def fake_grouped_checkbox_choice(message, **kwargs):
        captured.update(kwargs, message=message)
        return ["openrouter", "kiwimate"]

    monkeypatch.setattr(
        "kiwimatecoder.repl.grouped_checkbox_choice", fake_grouped_checkbox_choice
    )
    monkeypatch.setattr("kiwimatecoder.repl.checkbox_choice", _fail_plain)
    prompt = MultiSelectionPrompt(
        title="Select active providers",
        text="Check every provider you want",
        options=_GROUPED_PROVIDER_OPTIONS,
        selected=("openrouter",),
    )

    assert _select_command_options(prompt) == ["openrouter", "kiwimate"]
    assert captured["message"] == "Select active providers\nCheck every provider you want"
    groups = captured["groups"]
    assert [title for title, _ in groups] == ["Experimental", None]
    assert [value for value, _ in groups[0][1]] == ["kiwimate"]
    # Ungrouped options keep their (roster-first) order in the plain box.
    assert [value for value, _ in groups[1][1]] == ["openrouter", "openai"]
    assert captured["default_values"] == ("openrouter",)
    assert captured["show_frame"] is True
    assert "Space toggle" in captured["bottom_toolbar"]


def test_command_multi_selector_grouped_interrupt_returns_none(monkeypatch):
    def fake_grouped_checkbox_choice(message, **kwargs):
        raise EOFError

    monkeypatch.setattr(
        "kiwimatecoder.repl.grouped_checkbox_choice", fake_grouped_checkbox_choice
    )
    prompt = MultiSelectionPrompt(
        title="Select active providers",
        text="Check providers",
        options=_GROUPED_PROVIDER_OPTIONS,
    )

    assert _select_command_options(prompt) is None


def _isolate_provider_config(session, monkeypatch):
    """Point config at the temp workspace and hide real provider keys."""
    monkeypatch.setattr(config, "CONFIG_DIR", session.workspace_root / "cfg")
    monkeypatch.setattr(config, "CONFIG_FILE", session.workspace_root / "cfg.json")
    monkeypatch.setattr(
        config, "LEGACY_CONFIG_FILE", session.workspace_root / "legacy-config"
    )
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for provider in REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)


def test_provider_checklist_draws_kiwimate_in_the_experimental_box(session, monkeypatch):
    _isolate_provider_config(session, monkeypatch)
    captured = {}

    def fake_grouped_checkbox_choice(message, **kwargs):
        captured.update(kwargs)
        return None  # cancelled

    monkeypatch.setattr(
        "kiwimatecoder.repl.grouped_checkbox_choice", fake_grouped_checkbox_choice
    )
    monkeypatch.setattr("kiwimatecoder.repl.checkbox_choice", _fail_plain)

    dispatch(
        "/provider",
        session,
        Console(file=io.StringIO(), force_terminal=False, width=120),
        multi_selector=_select_command_options,
    )

    experimental, plain = captured["groups"]
    assert experimental[0] == "Experimental"
    assert [value for value, _ in experimental[1]] == ["kiwimate"]
    assert "kiwimate.net" in experimental[1][0][1]
    assert plain[0] is None
    assert [value for value, _ in plain[1]][0] == "openrouter"  # roster first
    assert "kiwimate" not in [value for value, _ in plain[1]]
    assert captured["default_values"] == ("openrouter",)
    assert session.provider_id == "openrouter"


def test_provider_picker_draws_kiwimate_in_the_experimental_box(session, monkeypatch):
    _isolate_provider_config(session, monkeypatch)
    captured = {}

    def fake_grouped_choice(message, **kwargs):
        captured.update(kwargs, message=message)
        return None  # cancelled

    monkeypatch.setattr("kiwimatecoder.repl.grouped_choice", fake_grouped_choice)
    monkeypatch.setattr("kiwimatecoder.repl.choice", _fail_plain)

    dispatch(
        "/provider",
        session,
        Console(file=io.StringIO(), force_terminal=False, width=120),
        selector=_select_command_option,
    )

    assert captured["message"].startswith("Select provider")
    experimental, plain = captured["groups"]
    assert experimental[0] == "Experimental"
    assert [value for value, _ in experimental[1]] == ["kiwimate"]
    assert experimental[1][0][1] == (
        "KiwiMate — no model chosen · kiwimate.net · chat only, no tool use yet"
    )
    assert plain[0] is None
    plain_ids = [value for value, _ in plain[1]]
    assert "kiwimate" not in plain_ids
    assert {"openrouter", "openai", "ollama"} <= set(plain_ids)
    assert captured["default"] == "openrouter"
    assert session.provider_id == "openrouter"


def test_option_groups_puts_titled_groups_first_in_first_seen_order():
    options = (
        CommandOption("a", "A"),
        CommandOption("x1", "X1", group="Experimental"),
        CommandOption("b", "B"),
        CommandOption("y1", "Y1", group="Beta"),
        CommandOption("x2", "X2", group="Experimental"),
        CommandOption("c", "C"),
    )

    assert _option_groups(options) == [
        ("Experimental", [("x1", "X1"), ("x2", "X2")]),
        ("Beta", [("y1", "Y1")]),
        (None, [("a", "A"), ("b", "B"), ("c", "C")]),
    ]


def test_option_groups_without_titles_is_one_plain_group():
    options = (CommandOption("a", "A"), CommandOption("b", "B"))

    assert _option_groups(options) == [(None, [("a", "A"), ("b", "B")])]


def test_option_groups_all_grouped_leaves_plain_group_empty():
    options = (CommandOption("x", "X", group="Experimental"),)

    assert _option_groups(options) == [("Experimental", [("x", "X")]), (None, [])]


def test_checkbox_choice_keyboard_interaction():
    with create_pipe_input() as pipe_input:
        # Initial focus is on 'openrouter' (default checked)
        # Send: down to openai, space (toggle openai), enter (confirm)
        pipe_input.send_text("\x1b[B \r")
        result = checkbox_choice(
            "Select active providers",
            options=[
                ("openrouter", "OpenRouter"),
                ("openai", "OpenAI"),
                ("anthropic", "Anthropic"),
            ],
            default_values=["openrouter"],
            input=pipe_input,
        )
        assert result == ["openrouter", "openai"]


def test_build_history_uses_config_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)

    history = _build_history()

    assert isinstance(history, FileHistory)
    assert Path(history.filename) == tmp_path / "history"


# ---------------------------------------------------------------------------
# Hunk-level approvals
# ---------------------------------------------------------------------------


def _multi_hunk_preview() -> str:
    old = "".join(f"line{i}\n" for i in range(1, 21))
    new = old.replace("line1\n", "LINE1\n", 1).replace("line20\n", "LINE20\n", 1)
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile="f.txt",
            tofile="f.txt",
            n=3,
        )
    )


def _single_hunk_preview() -> str:
    return "".join(
        difflib.unified_diff(
            ["alpha\n", "beta\n"],
            ["ALPHA\n", "beta\n"],
            fromfile="f.txt",
            tofile="f.txt",
            n=3,
        )
    )


def test_parse_hunk_selection_forms_and_errors():
    assert parse_hunk_selection("1,2", 3) == (1, 2)
    assert parse_hunk_selection("1-2", 3) == (1, 2)
    assert parse_hunk_selection("all", 3) == "all"
    assert parse_hunk_selection("none", 3) == "none"
    assert parse_hunk_selection("8", 3) is None


def test_confirm_returns_partial_hunk_selection(session, monkeypatch):
    answers = iter(["h", "1,2"])
    monkeypatch.setattr(
        "kiwimatecoder.repl.console.input", lambda *args, **kwargs: next(answers)
    )

    confirm = _make_confirm(session)
    result = confirm("write_file(path='f.txt')", _multi_hunk_preview())

    assert result == ApprovalResult(allowed=True, selected_hunks=(1, 2))


def test_confirm_hunk_mode_none_denies(session, monkeypatch):
    answers = iter(["h", "none"])
    monkeypatch.setattr(
        "kiwimatecoder.repl.console.input", lambda *args, **kwargs: next(answers)
    )

    result = _make_confirm(session)("write_file(path='f.txt')", _multi_hunk_preview())

    assert result == ApprovalResult(allowed=False)


def test_confirm_hunk_mode_reprompts_then_accepts(session, monkeypatch):
    answers = iter(["h", "nope", "2"])
    monkeypatch.setattr(
        "kiwimatecoder.repl.console.input", lambda *args, **kwargs: next(answers)
    )

    result = _make_confirm(session)("write_file(path='f.txt')", _multi_hunk_preview())

    assert result == ApprovalResult(allowed=True, selected_hunks=(2,))


def test_confirm_hunk_mode_all_allows_whole_action(session, monkeypatch):
    answers = iter(["h", "all"])
    monkeypatch.setattr(
        "kiwimatecoder.repl.console.input", lambda *args, **kwargs: next(answers)
    )

    result = _make_confirm(session)("write_file(path='f.txt')", _multi_hunk_preview())

    assert result == ApprovalResult(allowed=True)


def test_confirm_single_hunk_keeps_plain_prompt(session, monkeypatch):
    prompts: list[str] = []

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        return "y"

    monkeypatch.setattr("kiwimatecoder.repl.console.input", fake_input)

    result = _make_confirm(session)("write_file(path='f.txt')", _single_hunk_preview())

    assert result is True
    assert "hunks" not in prompts[0]


def test_confirm_plain_answers_still_return_bool(session, monkeypatch):
    monkeypatch.setattr("kiwimatecoder.repl.console.input", lambda *args, **kwargs: "n")

    assert _make_confirm(session)("write_file(path='f.txt')", None) is False


# ---------------------------------------------------------------------------
# Steering and deferred commands
# ---------------------------------------------------------------------------


def test_route_steering_line_ignores_blank_input(session):
    assert _route_steering_line(session, "   ") == "ignored"
    assert not session.steering
    assert not session.deferred_commands


def test_route_steering_line_queues_normal_text(session):
    assert _route_steering_line(session, "  focus on the tests  ") == "steered"
    assert list(session.steering) == ["focus on the tests"]
    assert not session.deferred_commands


def test_route_steering_line_defers_slash_commands(session):
    assert _route_steering_line(session, "/undo") == "deferred"
    assert list(session.deferred_commands) == ["/undo"]
    assert not session.steering


def test_route_steering_line_steers_template_commands(session, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", session.workspace_root / "cfg")
    commands_dir = session.workspace_root / ".kiwimatecoder" / "commands"
    commands_dir.mkdir(parents=True)
    (commands_dir / "review.md").write_text("Review $ARGUMENTS")

    assert _route_steering_line(session, "/review now") == "steered"
    assert list(session.steering) == ["Review now"]
    assert not session.deferred_commands


def test_resolve_slash_line_returns_rendered_template(session, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", session.workspace_root / "cfg")
    commands_dir = session.workspace_root / ".kiwimatecoder" / "commands"
    commands_dir.mkdir(parents=True)
    (commands_dir / "review.md").write_text("Review $ARGUMENTS carefully.")

    assert _resolve_slash_line("/review src/main.py", session) == (
        "template",
        "Review src/main.py carefully.",
    )


def test_resolve_slash_line_leaves_builtins_and_unknown_alone(session, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", session.workspace_root / "cfg")
    commands_dir = session.workspace_root / ".kiwimatecoder" / "commands"
    commands_dir.mkdir(parents=True)
    (commands_dir / "undo.md").write_text("shadow the builtin")

    assert _resolve_slash_line("/undo", session) is None
    assert _resolve_slash_line("/no-such-command", session) is None
    assert _resolve_slash_line("plain text", session) is None


async def test_deferred_commands_run_fifo(session, monkeypatch):
    calls: list[str] = []

    async def fake_dispatch(line, session):
        calls.append(line)
        return CommandResult.CONTINUE

    monkeypatch.setattr("kiwimatecoder.repl._dispatch_command", fake_dispatch)
    session.deferred_commands.extend(["/todos", "/cost"])

    assert await _process_deferred_commands(session) is False
    assert calls == ["/todos", "/cost"]
    assert list(session.deferred_commands) == []


async def test_deferred_command_exit_stops_processing(session, monkeypatch):
    calls: list[str] = []

    async def fake_dispatch(line, session):
        calls.append(line)
        return CommandResult.EXIT

    monkeypatch.setattr("kiwimatecoder.repl._dispatch_command", fake_dispatch)
    session.deferred_commands.extend(["/exit", "/cost"])

    assert await _process_deferred_commands(session) is True
    assert calls == ["/exit"]
    assert list(session.deferred_commands) == []


# ---------------------------------------------------------------------------
# Themes, color, and ASCII mode
# ---------------------------------------------------------------------------


def test_banner_uses_folder_glyph_and_ascii_mode(session, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", session.workspace_root / "cfg")
    monkeypatch.setattr(config, "CONFIG_FILE", session.workspace_root / "cfg.json")
    monkeypatch.setattr(
        config, "LEGACY_CONFIG_FILE", session.workspace_root / "legacy-config"
    )

    assert "📁" in str(_banner(session).renderable)

    config.set_ui(ascii=True)

    rendered = str(_banner(session).renderable)
    assert "📁" not in rendered
    assert session.workspace_root.name in rendered


def test_banner_flags_experimental_providers_and_a_missing_model(session, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", session.workspace_root / "cfg")
    monkeypatch.setattr(config, "CONFIG_FILE", session.workspace_root / "cfg.json")
    monkeypatch.setattr(
        config, "LEGACY_CONFIG_FILE", session.workspace_root / "legacy-config"
    )

    rendered = str(_banner(session).renderable)
    assert "(experimental)" not in rendered
    assert "test-model" in rendered

    session.provider_id = "kiwimate"
    session.model = ""
    rendered = str(_banner(session).renderable)
    assert "KiwiMate[/bold cyan] [yellow](experimental)[/yellow]" in rendered
    assert "no model chosen — use /model" in rendered
    prompt = "".join(fragment[1] for fragment in to_formatted_text(_prompt_text(session)))
    assert "kiwimate:no model" in prompt


def test_banner_and_prompt_use_theme_accent(session, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", session.workspace_root / "cfg")
    monkeypatch.setattr(config, "CONFIG_FILE", session.workspace_root / "cfg.json")
    monkeypatch.setattr(
        config, "LEGACY_CONFIG_FILE", session.workspace_root / "legacy-config"
    )
    config.set_ui(theme="magenta")

    assert "magenta" in str(_banner(session).renderable)

    fragments = to_formatted_text(_prompt_text(session))
    styles = " ".join(fragment[0] for fragment in fragments)
    assert "ansimagenta" in styles


def test_prompt_keeps_mode_colors(session, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", session.workspace_root / "cfg")
    monkeypatch.setattr(config, "CONFIG_FILE", session.workspace_root / "cfg.json")
    monkeypatch.setattr(
        config, "LEGACY_CONFIG_FILE", session.workspace_root / "legacy-config"
    )

    fragments = to_formatted_text(_prompt_text(session))
    styles = " ".join(fragment[0] for fragment in fragments)
    assert "ansiyellow" in styles


def test_run_rebuilds_console_for_color_config(session, monkeypatch):
    from kiwimatecoder import repl

    monkeypatch.setattr(config, "CONFIG_DIR", session.workspace_root / "cfg")
    monkeypatch.setattr(config, "CONFIG_FILE", session.workspace_root / "cfg.json")
    monkeypatch.setattr(
        config, "LEGACY_CONFIG_FILE", session.workspace_root / "legacy-config"
    )
    config.set_ui(color="never")

    captured: dict[str, object] = {}

    def fake_asyncio_run(coro):
        coro.close()
        captured["console"] = repl.console

    monkeypatch.setattr(repl.asyncio, "run", fake_asyncio_run)
    original = repl.console
    try:
        repl.run(session)
    finally:
        repl.console = original

    assert captured["console"].no_color is True  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# @path image mentions
# ---------------------------------------------------------------------------

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_extract_image_mentions_removes_workspace_image(session):
    image_path = session.workspace_root / "shot.png"
    image_path.write_bytes(PNG_1PX)

    cleaned, found = _extract_image_mentions(
        "look at @shot.png please", session.workspace_root
    )

    assert cleaned == "look at please"
    assert found == [str(image_path.resolve())]


def test_extract_image_mentions_keeps_missing_file(session):
    cleaned, found = _extract_image_mentions(
        "@ghost.png hello", session.workspace_root
    )

    assert cleaned == "@ghost.png hello"
    assert found == []


def test_extract_image_mentions_keeps_outside_workspace(session, tmp_path):
    outside = tmp_path.parent / "outside-shot.png"
    outside.write_bytes(PNG_1PX)

    cleaned, found = _extract_image_mentions(
        f"@{outside} ok", session.workspace_root
    )

    assert cleaned == f"@{outside} ok"
    assert found == []


def test_extract_image_mentions_multiple_images(session):
    (session.workspace_root / "a.png").write_bytes(PNG_1PX)
    (session.workspace_root / "b.gif").write_bytes(
        base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")
    )

    cleaned, found = _extract_image_mentions(
        "@a.png and @b.gif", session.workspace_root
    )

    assert cleaned == "and"
    assert len(found) == 2


def test_extract_image_mentions_ignores_non_images(session):
    (session.workspace_root / "notes.txt").write_text("hi")

    cleaned, found = _extract_image_mentions(
        "@notes.txt hello", session.workspace_root
    )

    assert cleaned == "@notes.txt hello"
    assert found == []


def test_attach_images_stashes_and_cleans_line(session, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", session.workspace_root / "cfg")
    monkeypatch.setattr(config, "CONFIG_FILE", session.workspace_root / "cfg.json")
    monkeypatch.setattr(
        config, "LEGACY_CONFIG_FILE", session.workspace_root / "legacy-config"
    )
    (session.workspace_root / "shot.png").write_bytes(PNG_1PX)

    line = _attach_images("describe @shot.png", session)

    assert line == "describe"
    assert [entry["name"] for entry in session.pending_images] == ["shot.png"]


def test_attach_images_uses_default_prompt_when_only_image(session, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", session.workspace_root / "cfg")
    monkeypatch.setattr(config, "CONFIG_FILE", session.workspace_root / "cfg.json")
    monkeypatch.setattr(
        config, "LEGACY_CONFIG_FILE", session.workspace_root / "legacy-config"
    )
    (session.workspace_root / "shot.png").write_bytes(PNG_1PX)

    line = _attach_images("@shot.png", session)

    assert line == "Please analyze the attached image(s)."
    assert len(session.pending_images) == 1


def test_attach_images_respects_per_turn_limit(session, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", session.workspace_root / "cfg")
    monkeypatch.setattr(config, "CONFIG_FILE", session.workspace_root / "cfg.json")
    monkeypatch.setattr(
        config, "LEGACY_CONFIG_FILE", session.workspace_root / "legacy-config"
    )
    config.set_vision(max_images_per_turn=1)
    (session.workspace_root / "a.png").write_bytes(PNG_1PX)
    (session.workspace_root / "b.png").write_bytes(PNG_1PX)

    _attach_images("@a.png @b.png", session)

    assert len(session.pending_images) == 1


# ---------------------------------------------------------------------------
# @path text-file mentions
# ---------------------------------------------------------------------------


def test_extract_file_mentions_removes_workspace_text_file(session):
    notes = session.workspace_root / "notes.txt"
    notes.write_text("hello\n")

    cleaned, found = _extract_file_mentions(
        "read @notes.txt please", session.workspace_root
    )

    assert cleaned == "read please"
    assert found == [str(notes.resolve())]


def test_extract_file_mentions_keeps_missing_file(session):
    cleaned, found = _extract_file_mentions(
        "@ghost.txt hello", session.workspace_root
    )

    assert cleaned == "@ghost.txt hello"
    assert found == []


def test_extract_file_mentions_keeps_outside_workspace(session, tmp_path):
    outside = tmp_path.parent / "outside-notes.txt"
    outside.write_text("secret")

    cleaned, found = _extract_file_mentions(
        f"@{outside} ok", session.workspace_root
    )

    assert cleaned == f"@{outside} ok"
    assert found == []


def test_extract_file_mentions_keeps_binary_file(session):
    (session.workspace_root / "blob.bin").write_bytes(b"\x00\x01\x02binary")

    cleaned, found = _extract_file_mentions("@blob.bin hi", session.workspace_root)

    assert cleaned == "@blob.bin hi"
    assert found == []


def test_extract_file_mentions_keeps_images_for_the_image_path(session):
    (session.workspace_root / "shot.png").write_bytes(PNG_1PX)

    cleaned, found = _extract_file_mentions("@shot.png hi", session.workspace_root)

    assert cleaned == "@shot.png hi"
    assert found == []


def test_extract_file_mentions_multiple_files(session):
    (session.workspace_root / "a.txt").write_text("a")
    (session.workspace_root / "b.md").write_text("b")

    cleaned, found = _extract_file_mentions("@a.txt and @b.md", session.workspace_root)

    assert cleaned == "and"
    assert len(found) == 2


def test_attach_file_mentions_prepends_numbered_context(session):
    (session.workspace_root / "notes.txt").write_text("alpha\nbeta\n")

    line = _attach_file_mentions("summarize @notes.txt", session)

    assert line.startswith("[mentioned files]\n")
    assert "--- notes.txt ---" in line
    assert "1\talpha" in line
    assert "2\tbeta" in line
    assert line.endswith("summarize")


def test_attach_file_mentions_caps_the_number_of_files(session):
    mentions = " ".join(
        f"@f{index}.txt" for index in range(MAX_MENTION_FILES + 2)
    )
    for index in range(MAX_MENTION_FILES + 2):
        (session.workspace_root / f"f{index}.txt").write_text("x")

    line = _attach_file_mentions(f"check {mentions}", session)

    assert line.count("--- ") == MAX_MENTION_FILES


def test_attach_file_mentions_bounds_file_bytes(session):
    (session.workspace_root / "big.txt").write_text("x" * (MAX_MENTION_FILE_BYTES + 50))

    line = _attach_file_mentions("read @big.txt", session)

    assert "<truncated" in line
    assert f'shown_bytes="{MAX_MENTION_FILE_BYTES}"' in line


def test_attach_file_mentions_leaves_plain_lines_alone(session):
    assert _attach_file_mentions("just text", session) == "just text"


# ---------------------------------------------------------------------------
# @ completion
# ---------------------------------------------------------------------------


def _at_completions(session, text: str) -> list[str]:
    completer = SlashCommandCompleter(session)
    return [
        completion.text
        for completion in completer.get_completions(Document(text), CompleteEvent())
    ]


def test_at_completer_lists_workspace_files(session):
    (session.workspace_root / "readme.md").write_text("x")
    (session.workspace_root / "notes.txt").write_text("x")
    src = session.workspace_root / "src"
    src.mkdir()
    (src / "main.py").write_text("x")

    assert _at_completions(session, "look at @re") == ["@readme.md"]


def test_at_completer_lists_nested_paths(session):
    src = session.workspace_root / "src"
    src.mkdir()
    (src / "main.py").write_text("x")

    assert _at_completions(session, "@src/ma") == ["@src/main.py"]


def test_at_completer_respects_gitignore(session):
    (session.workspace_root / ".gitignore").write_text("secret.txt\n")
    (session.workspace_root / "secret.txt").write_text("x")

    assert _at_completions(session, "@sec") == []


def test_at_completer_without_session_returns_nothing():
    completer = SlashCommandCompleter(None)

    assert (
        list(completer.get_completions(Document("@re"), CompleteEvent())) == []
    )


def test_workspace_file_candidates_cap(session):
    for index in range(10):
        (session.workspace_root / f"f{index}.txt").write_text("x")

    assert len(_workspace_file_candidates(session.workspace_root, "", limit=3)) == 3


# ---------------------------------------------------------------------------
# Turn-finished notifications
# ---------------------------------------------------------------------------


def test_notify_turn_finished_respects_threshold_and_mode(monkeypatch):
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        "kiwimatecoder.repl.get_ui",
        lambda: {"notify": "bell", "notify_after_seconds": 5},
    )
    monkeypatch.setattr(
        "kiwimatecoder.repl.notify.notify",
        lambda title, body, mode: calls.append((title, body, mode)),
    )

    _notify_turn_finished(10)
    _notify_turn_finished(1)

    assert calls == [("KiwiMateCoder", "Turn finished in 10s", "bell")]


def test_notify_turn_finished_off_mode_writes_nothing(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr("kiwimatecoder.notify._stderr", lambda: stream)
    monkeypatch.setattr(
        "kiwimatecoder.repl.get_ui",
        lambda: {"notify": "off", "notify_after_seconds": 0},
    )

    _notify_turn_finished(99)

    assert stream.getvalue() == ""


def test_notify_turn_finished_bell_mode_rings(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr("kiwimatecoder.notify._stderr", lambda: stream)
    monkeypatch.setattr(
        "kiwimatecoder.repl.get_ui",
        lambda: {"notify": "bell", "notify_after_seconds": 0},
    )

    _notify_turn_finished(3)

    assert stream.getvalue() == "\a"


def test_read_command_input_returns_the_typed_line():
    with create_pipe_input() as pipe, create_app_session(
        input=pipe, output=DummyOutput()
    ):
        pipe.send_text("pytest -q\r")

        assert _read_command_input("value> ") == "pytest -q"


@pytest.mark.parametrize(
    ("key", "error"), [("\x03", KeyboardInterrupt), ("\x04", EOFError)]
)
def test_read_command_input_ctrl_c_and_ctrl_d_cancel(key, error):
    # Commands run in a worker thread, where a plain input() never sees Ctrl-C
    # (it goes to the event loop) and the REPL exited on the next Enter. The
    # reader must raise instead so the command can say "Cancelled."
    with create_pipe_input() as pipe, create_app_session(
        input=pipe, output=DummyOutput()
    ):
        pipe.send_text(key)

        with pytest.raises(error):
            _read_command_input("value> ")


async def test_dispatch_command_reads_typed_answers_through_prompt_toolkit(
    session, monkeypatch
):
    seen = {}

    def fake_dispatch(line, _session, _console, selector, reader, multi, secret):
        seen.update(selector=selector, reader=reader, multi=multi, secret=secret)
        return CommandResult.CONTINUE

    monkeypatch.setattr("kiwimatecoder.repl.dispatch", fake_dispatch)

    assert await _dispatch_command("/config", session) == CommandResult.CONTINUE

    assert seen["selector"] is _select_command_option
    assert seen["reader"] is _read_command_input
    assert seen["multi"] is _select_command_options
    # An API key is read by a different reader, one that does not echo.
    assert seen["secret"] is _read_secret_input
    assert seen["secret"] is not seen["reader"]


_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _typed_on_screen(reader, keys: str) -> tuple[str, str]:
    """Run ``reader`` on a captured vt100 screen; return (value, text it drew)."""
    stdout = io.StringIO()
    output = Vt100_Output(
        stdout,
        get_size=lambda: Size(rows=24, columns=80),
        default_color_depth=ColorDepth.DEPTH_8_BIT,
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        pipe.send_text(keys)
        value = reader("key> ")
    return value, _ANSI.sub("", stdout.getvalue())


def test_the_capture_sees_ordinary_typed_text():
    # Control for the test below: the visible reader does draw what is typed,
    # so "the key is absent from the screen" cannot pass for lack of a screen.
    value, drawn = _typed_on_screen(_read_command_input, "pytest -q\r")

    assert value == "pytest -q"
    assert "pytest -q" in drawn


def test_read_secret_input_returns_the_key_but_never_draws_it():
    key = "sk-live-Zx9f2Qk7"

    value, drawn = _typed_on_screen(_read_secret_input, key + "\r")

    assert value == key
    assert "key> " in drawn  # the prompt is shown
    assert key not in drawn
    assert "Zx9f2Qk7" not in drawn  # nor any recognisable part of it


@pytest.mark.parametrize(
    ("key", "error"), [("\x03", KeyboardInterrupt), ("\x04", EOFError)]
)
def test_read_secret_input_ctrl_c_and_ctrl_d_cancel(key, error):
    with create_pipe_input() as pipe, create_app_session(
        input=pipe, output=DummyOutput()
    ):
        pipe.send_text(key)

        with pytest.raises(error):
            _read_secret_input("key> ")


# ---------------------------------------------------------------------------
# The history file never holds an API key
# ---------------------------------------------------------------------------

SECRET_LINES = [
    "/config key set openrouter sk-live-123",
    "/config keys save openai sk-x",
    "/config api-key add anthropic sk-ant",
    "/config api-keys set a b",
    "/CONFIG KEY SET openrouter sk-x",
    "/ config key set openrouter sk-x",
    '/config key set openrouter "sk-with space"',
    "  /config key set openrouter sk-x  ",
    "/config\tkey\tset\topenrouter\tsk-x",
    "/config key set openrouter 'sk-unterminated",  # unparsable: err on the safe side
    '/config "key" "set" openrouter sk-x',  # the real parser unquotes these
    "/config 'keys' save openai sk-x",
]
ORDINARY_LINES = [
    "/config key list",
    "/config key remove openrouter",
    "/config key edit openrouter",
    "/config model set sk-looks-like-a-key",
    "/config",
    "/help",
    "fix the bug in /config key set",
    "how do I set a key?",
    "/model key set",
    "config key set openrouter sk-x",  # no slash: a message to the model, not a command
]


@pytest.mark.parametrize("line", SECRET_LINES)
def test_a_key_setting_line_is_recognised(line):
    assert line_carries_secret(line)


@pytest.mark.parametrize("line", ORDINARY_LINES)
def test_ordinary_lines_are_not_mistaken_for_a_key(line):
    assert not line_carries_secret(line)


@pytest.mark.parametrize("section", sorted(CONFIG_KEY_SECTIONS))
@pytest.mark.parametrize("action", sorted(CONFIG_KEY_SET_ACTIONS))
def test_every_spelling_the_parser_accepts_is_recognised_as_carrying_a_key(
    section, action, session, tmp_path, monkeypatch
):
    # The history filter and /config share their alias lists; this fails if the
    # parser ever accepts a spelling the filter would let through to disk.
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    key = f"sk-{section}-{action}"
    line = f"/config {section} {action} openrouter {key}"

    dispatch(line, session, Console(file=io.StringIO()))

    assert config.get_key("openrouter") == key  # the parser really took it
    assert line_carries_secret(line)


@pytest.fixture
def history_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    return tmp_path / "history"


def test_history_keeps_a_key_out_of_the_file_but_not_out_of_the_session(history_file):
    history = _build_history()

    history.append_string("/config key set openrouter sk-live-Zx9f2Qk7")
    history.append_string("/config model set vendor/x")

    on_disk = history_file.read_text()
    assert "sk-live-Zx9f2Qk7" not in on_disk
    assert "/config model set vendor/x" in on_disk
    # Still reachable with the up-arrow for the rest of this session.
    assert "/config key set openrouter sk-live-Zx9f2Qk7" in history.get_strings()


def test_history_survives_a_restart_without_the_key(history_file):
    first = _build_history()
    first.append_string("/help")
    first.append_string("/config key set openrouter sk-live-Zx9f2Qk7")
    first.append_string("/model x")

    second = _build_history()

    assert list(second.load_history_strings()) == ["/model x", "/help"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_history_file_is_private(history_file):
    history = _build_history()
    history.append_string("/help")

    assert stat.S_IMODE(os.stat(history_file).st_mode) == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_an_existing_world_readable_history_is_made_private(history_file):
    history_file.write_text("\n# 2024-01-01 00:00:00\n+/help\n")
    os.chmod(history_file, 0o644)

    _build_history()

    assert stat.S_IMODE(os.stat(history_file).st_mode) == 0o600
    assert "/help" in history_file.read_text()  # content untouched


def test_keys_saved_by_an_earlier_version_are_scrubbed_on_open(history_file):
    from prompt_toolkit.history import FileHistory

    old = FileHistory(str(history_file))  # what older versions used
    for line in (
        "/help",
        "/config key set openrouter sk-OLD-SECRET",
        "first line\nsecond line",  # multi-line entries must survive intact
        "/config keys add openai sk-OTHER-SECRET",
        "/model x",
    ):
        old.store_string(line)
    os.chmod(history_file, 0o644)

    history = _build_history()

    assert list(history.load_history_strings()) == [
        "/model x",
        "first line\nsecond line",
        "/help",
    ]
    text = history_file.read_text()
    assert "SECRET" not in text
    assert not history_file.with_name("history.tmp").exists()
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(history_file).st_mode) == 0o600


def test_a_clean_history_is_not_rewritten(history_file):
    from prompt_toolkit.history import FileHistory

    FileHistory(str(history_file)).store_string("/help")
    before = (history_file.read_bytes(), os.stat(history_file).st_ino)

    _build_history()

    assert (history_file.read_bytes(), os.stat(history_file).st_ino) == before


def test_history_falls_back_to_memory_when_the_directory_is_unwritable(monkeypatch):
    from prompt_toolkit.history import InMemoryHistory

    def broken():
        raise OSError("read-only home")

    monkeypatch.setattr(config, "ensure_config_dir", broken)

    assert isinstance(_build_history(), InMemoryHistory)


def test_the_private_history_is_what_the_repl_uses(history_file):
    assert isinstance(_build_history(), _PrivateHistory)
