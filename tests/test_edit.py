import pytest

from kiwimatecoder.tools.edit_file import EditError, _edit_file, compute_edit, preview


def test_compute_edit_simple():
    assert compute_edit("a b c", "b", "X", False) == "a X c"


def test_compute_edit_not_found():
    with pytest.raises(EditError, match="not found"):
        compute_edit("abc", "z", "y", False)


def test_compute_edit_ambiguous():
    with pytest.raises(EditError, match="not unique"):
        compute_edit("x x x", "x", "y", False)


def test_compute_edit_replace_all():
    assert compute_edit("x x x", "x", "y", True) == "y y y"


def test_compute_edit_identical():
    with pytest.raises(EditError, match="identical"):
        compute_edit("abc", "a", "a", False)


def test_edit_file_applies(session):
    (session.workspace_root / "f.txt").write_text("hello world")
    result = _edit_file(
        {"path": "f.txt", "old_string": "world", "new_string": "there"}, session
    )
    assert result.ok
    assert (session.workspace_root / "f.txt").read_text() == "hello there"
    assert "f.txt" in session.touched_files


def test_edit_file_missing(session):
    result = _edit_file(
        {"path": "nope.txt", "old_string": "a", "new_string": "b"}, session
    )
    assert not result.ok


def test_preview_produces_diff(session):
    (session.workspace_root / "f.txt").write_text("line one\nline two\n")
    diff = preview(
        {"path": "f.txt", "old_string": "line two", "new_string": "line 2"}, session
    )
    assert "-line two" in diff
    assert "+line 2" in diff
