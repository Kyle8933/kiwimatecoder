from prompt_toolkit.document import Document

from kiwimatecoder.repl import SlashCommandCompleter


def _completion_texts(text: str) -> list[str]:
    completer = SlashCommandCompleter()
    return [
        completion.text
        for completion in completer.get_completions(Document(text), None)
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
