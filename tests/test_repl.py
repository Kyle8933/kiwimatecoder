import difflib
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.input import create_pipe_input

from kiwimatecoder import config
from kiwimatecoder.commands import CommandOption, MultiSelectionPrompt, SelectionPrompt
from kiwimatecoder.hunks import parse_hunk_selection
from kiwimatecoder.permissions import ApprovalResult
from kiwimatecoder.repl import (
    SlashCommandCompleter,
    _build_history,
    _make_confirm,
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
