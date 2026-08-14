from __future__ import annotations

from kiwimatecoder.tools.list_dir import _list_dir
from kiwimatecoder.tools.paths import WorkspaceIgnore, get_workspace_ignore
from kiwimatecoder.tools.search import _glob_search, _grep_search


def test_workspace_ignore_skip_dirs(tmp_path):
    ignore = WorkspaceIgnore(tmp_path)
    assert ignore.is_ignored(tmp_path / ".git", is_dir=True)
    assert ignore.is_ignored(tmp_path / "node_modules" / "foo.js", is_dir=False)
    assert ignore.is_ignored(tmp_path / ".venv", is_dir=True)
    assert not ignore.is_ignored(tmp_path / "src" / "index.py", is_dir=False)


def test_workspace_ignore_gitignore_rules(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log\ndist/\n!important.log\n")
    ignore = get_workspace_ignore(tmp_path)

    assert ignore.is_ignored(tmp_path / "app.log", is_dir=False)
    assert not ignore.is_ignored(tmp_path / "important.log", is_dir=False)
    assert ignore.is_ignored(tmp_path / "dist", is_dir=True)
    assert ignore.is_ignored(tmp_path / "dist" / "bundle.js", is_dir=False)
    assert not ignore.is_ignored(tmp_path / "src" / "app.py", is_dir=False)


def test_search_and_list_dir_respects_gitignore(session):
    root = session.workspace_root
    (root / ".gitignore").write_text("ignored_folder/\n*.secret\n")
    (root / "ignored_folder").mkdir()
    (root / "ignored_folder" / "bad.py").write_text("SECRET_KEY = 123")
    (root / "secret.secret").write_text("SECRET_KEY = 456")
    (root / "good.py").write_text("SECRET_KEY = 789")

    # Grep search
    grep_res = _grep_search(root, "SECRET_KEY", None, session)
    assert len(grep_res) == 1
    assert "good.py:1" in grep_res[0]

    # Glob search
    glob_res = _glob_search(root, "*.secret", session)
    assert glob_res == []

    # List dir
    list_res = _list_dir({"path": "."}, session)
    assert "good.py" in list_res.content
    assert "ignored_folder" not in list_res.content
    assert "secret.secret" not in list_res.content
