import pytest

from kiwimatecoder import tools
from kiwimatecoder.tools.ask import _ask_user
from kiwimatecoder.tools.list_dir import _list_dir
from kiwimatecoder.tools.paths import PathError, resolve_in_workspace
from kiwimatecoder.tools.read_file import _read_file
from kiwimatecoder.tools.run_bash import _run_bash
from kiwimatecoder.tools.search import _search
from kiwimatecoder.tools.todo import _update_todos
from kiwimatecoder.tools.write_file import _write_file


def test_path_sandbox_rejects_escape(tmp_path):
    with pytest.raises(PathError):
        resolve_in_workspace("../outside.txt", tmp_path)


def test_path_sandbox_allows_inside(tmp_path):
    resolved = resolve_in_workspace("sub/file.txt", tmp_path)
    assert str(resolved).startswith(str(tmp_path.resolve()))


def test_read_file(session):
    (session.workspace_root / "a.txt").write_text("hello\nworld\n")
    result = _read_file({"path": "a.txt"}, session)
    assert result.ok
    assert "1\thello" in result.content
    assert "2\tworld" in result.content


def test_read_file_missing(session):
    result = _read_file({"path": "nope.txt"}, session)
    assert not result.ok


def test_read_file_rejects_binary(session):
    (session.workspace_root / "b.bin").write_bytes(b"\x00\x01\x02")
    result = _read_file({"path": "b.bin"}, session)
    assert not result.ok


def test_read_file_outside_workspace_blocked(session, tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")

    result = _read_file({"path": str(outside)}, session)

    assert not result.ok
    assert "outside the workspace" in result.content


def test_read_file_outside_workspace_allowed_when_trusted(session, tmp_path):
    outside = tmp_path.parent / "outside-read.txt"
    outside.write_text("shared")
    session.trusted_workspace = True

    result = _read_file({"path": str(outside)}, session)

    assert result.ok
    assert "shared" in result.content


def test_write_file_stays_sandboxed_when_trusted(session, tmp_path):
    outside = tmp_path.parent / "outside-write.txt"
    session.trusted_workspace = True

    result = _write_file({"path": str(outside), "content": "nope"}, session)

    assert not result.ok
    assert not outside.exists()


def test_write_file_creates_and_records(session):
    result = _write_file({"path": "out/new.txt", "content": "data"}, session)
    assert result.ok
    assert (session.workspace_root / "out/new.txt").read_text() == "data"
    assert "out/new.txt" in session.touched_files


def test_list_dir(session):
    (session.workspace_root / "x.py").write_text("")
    (session.workspace_root / "sub").mkdir()
    (session.workspace_root / ".git").mkdir()
    result = _list_dir({"path": "."}, session)
    assert "x.py" in result.content
    assert "sub/" in result.content
    assert ".git" not in result.content


def test_search_grep(session):
    (session.workspace_root / "f.py").write_text("def foo():\n    return 1\n")
    result = _search({"pattern": "def foo", "mode": "grep"}, session)
    assert "f.py:1" in result.content


def test_search_glob(session):
    (session.workspace_root / "a.py").write_text("")
    (session.workspace_root / "b.txt").write_text("")
    result = _search({"pattern": "*.py", "mode": "glob"}, session)
    assert "a.py" in result.content
    assert "b.txt" not in result.content


def test_run_bash(session):
    result = _run_bash({"command": "echo hello"}, session)
    assert result.ok
    assert "hello" in result.content
    assert "[exit code: 0]" in result.content


def test_run_bash_nonzero_exit(session):
    result = _run_bash({"command": "exit 3"}, session)
    assert not result.ok
    assert "[exit code: 3]" in result.content


def test_tool_schemas_read_only_excludes_writers():
    names = {s["function"]["name"] for s in tools.tool_schemas(read_only=True)}
    assert "read_file" in names
    assert "write_file" not in names
    assert "run_bash" not in names


def test_dispatch_unknown_tool(session):
    result = tools.dispatch("ghost", {}, session)
    assert not result.ok


def test_update_todos_records_and_renders(session):
    result = _update_todos(
        {
            "todos": [
                {"content": "step 1", "status": "in_progress"},
                {"content": "step 2", "status": "pending"},
            ]
        },
        session,
    )

    assert result.ok
    assert session.todos == [
        {"content": "step 1", "status": "in_progress"},
        {"content": "step 2", "status": "pending"},
    ]
    assert "step 1" in result.content
    assert "0/2 complete" in result.content


def test_update_todos_rejects_bad_status(session):
    result = _update_todos(
        {"todos": [{"content": "x", "status": "nope"}]}, session
    )

    assert not result.ok
    assert session.todos == []


def test_update_todos_clears_list(session):
    session.todos = [{"content": "x", "status": "pending"}]

    result = _update_todos({"todos": []}, session)

    assert result.ok
    assert session.todos == []


def test_ask_user_requires_callback(session):
    result = _ask_user({"question": "Which?"}, session)

    assert not result.ok
    assert "No interactive user" in result.content


def test_ask_user_returns_answer(session):
    session.ask_user = lambda question, options: "option-b"

    result = _ask_user({"question": "Pick", "options": ["a", "option-b"]}, session)

    assert result.ok
    assert "option-b" in result.content


def test_ask_user_rejects_empty_answer(session):
    session.ask_user = lambda question, options: ""

    result = _ask_user({"question": "Pick"}, session)

    assert not result.ok
