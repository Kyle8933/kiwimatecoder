import base64
import difflib
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.history import FileHistory
from prompt_toolkit.input import create_pipe_input

from kiwimatecoder import config
from kiwimatecoder.commands import (
    CommandOption,
    CommandResult,
    MultiSelectionPrompt,
    SelectionPrompt,
)
from kiwimatecoder.hunks import parse_hunk_selection
from kiwimatecoder.permissions import ApprovalResult
from kiwimatecoder.repl import (
    SlashCommandCompleter,
    _attach_images,
    _banner,
    _build_history,
    _extract_image_mentions,
    _make_confirm,
    _process_deferred_commands,
    _prompt_text,
    _resolve_slash_line,
    _route_steering_line,
    _select_command_option,
    _select_command_options,
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


def test_command_selector_renders_prompt_options(monkeypatch):
    captured = {}

    def fake_choice(**kwargs):
        captured.update(kwargs)
        return "model-b"

    monkeypatch.setattr("kiwimatecoder.repl.choice", fake_choice)
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
