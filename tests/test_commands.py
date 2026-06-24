import io

from rich.console import Console

from kiwimatecoder.commands import CommandResult, dispatch, slash_command_completions


def _console():
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def test_context_add_lists_and_deduplicates(session):
    (session.workspace_root / "README.md").write_text("hello\n")
    console = _console()

    assert (
        dispatch("/context add README.md README.md", session, console)
        == CommandResult.CONTINUE
    )

    assert session.context_files == ["README.md"]


def test_context_add_glob(session):
    (session.workspace_root / "a.py").write_text("print('a')\n")
    (session.workspace_root / "b.txt").write_text("b\n")
    console = _console()

    dispatch("/context add *.py", session, console)

    assert session.context_files == ["a.py"]


def test_context_rejects_binary_files(session):
    (session.workspace_root / "image.bin").write_bytes(b"\x00\x01")
    console = _console()

    dispatch("/context add image.bin", session, console)

    assert session.context_files == []


def test_context_remove_and_clear(session):
    (session.workspace_root / "a.py").write_text("print('a')\n")
    (session.workspace_root / "b.py").write_text("print('b')\n")
    console = _console()
    dispatch("/context add *.py", session, console)

    dispatch("/context remove a.py", session, console)
    assert session.context_files == ["b.py"]

    dispatch("/context clear", session, console)
    assert session.context_files == []


def test_slash_command_completions_include_core_commands():
    completions = {command for command, _ in slash_command_completions("")}

    assert {"/help", "/model", "/provider", "/mode", "/context", "/cost"} <= completions
