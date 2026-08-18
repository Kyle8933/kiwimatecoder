from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input

from kiwimatecoder.commands import CommandOption, MultiSelectionPrompt, SelectionPrompt
from kiwimatecoder.repl import (
    SlashCommandCompleter,
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
