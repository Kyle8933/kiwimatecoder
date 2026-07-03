from pathlib import Path

from rich.console import Console

from kiwimatecoder import updater


def test_find_source_root_detects_project_root(tmp_path):
    package_dir = tmp_path / "kiwimatecoder"
    package_dir.mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='kiwimatecoder'\n")
    package_file = package_dir / "updater.py"
    package_file.write_text("")

    assert updater.find_source_root(package_file) == tmp_path


def test_build_source_install_command_uses_editable_source_root():
    root = Path("/tmp/kiwimatecoder")

    command = updater.build_source_install_command(root)

    assert command[1:5] == ["-m", "pip", "install", "--upgrade"]
    assert command[-2:] == ["-e", str(root)]


def test_build_update_command_uses_github_repo_url():
    command = updater.build_update_command()

    assert command[1:4] == ["-m", "pip", "install"]
    assert "--upgrade" in command
    assert command[-1] == f"git+{updater.GITHUB_REPO_URL}"


def _patch_git_helpers(monkeypatch, *, root, behind, shas, branch="main"):
    """Patch the Git helper functions used by run_update's Git branch."""
    monkeypatch.setattr(updater, "find_source_root", lambda: root)
    monkeypatch.setattr(updater, "_has_git_remote", lambda source_root: True)
    monkeypatch.setattr(updater, "_fetch", lambda source_root, console: 0)
    monkeypatch.setattr(updater, "_get_branch", lambda source_root: branch)
    monkeypatch.setattr(updater, "_commits_behind", lambda source_root, b: behind)
    sha_iter = iter(shas)
    monkeypatch.setattr(updater, "_get_short_sha", lambda source_root: next(sha_iter))


def test_run_update_uses_git_source_checkout(monkeypatch):
    calls = []
    root = Path("/tmp/kiwimatecoder")

    _patch_git_helpers(monkeypatch, root=root, behind=2, shas=["abc1234", "def5678"])
    monkeypatch.setattr(
        updater,
        "_run",
        lambda command, console: calls.append(command) or 0,
    )

    code = updater.run_update(Console(record=True))

    assert code == 0
    assert calls == [
        updater.build_git_pull_command(root),
        updater.build_source_install_command(root),
    ]


def test_run_update_falls_back_to_git_url(monkeypatch):
    calls = []

    monkeypatch.setattr(updater, "find_source_root", lambda: None)
    monkeypatch.setattr(
        updater,
        "_run",
        lambda command, console: calls.append(command) or 0,
    )

    code = updater.run_update(Console(record=True))

    assert code == 0
    assert calls == [updater.build_update_command()]


def test_run_update_stops_when_git_pull_fails(monkeypatch):
    calls = []
    root = Path("/tmp/kiwimatecoder")

    _patch_git_helpers(monkeypatch, root=root, behind=2, shas=["abc1234", "abc1234"])

    def fake_run(command, console):
        calls.append(command)
        return 7

    monkeypatch.setattr(updater, "_run", fake_run)

    code = updater.run_update(Console(record=True))

    assert code == 7
    assert calls == [updater.build_git_pull_command(root)]


def test_run_update_already_up_to_date_skips_reinstall(monkeypatch):
    calls = []
    root = Path("/tmp/kiwimatecoder")

    _patch_git_helpers(monkeypatch, root=root, behind=0, shas=["abc1234"])
    monkeypatch.setattr(
        updater,
        "_run",
        lambda command, console: calls.append(command) or 0,
    )

    console = Console(record=True)
    code = updater.run_update(console)

    assert code == 0
    assert calls == []
    output = console.export_text()
    assert "Already on the latest version" in output
    assert "abc1234" in output


def test_run_update_reports_old_to_new_sha(monkeypatch):
    calls = []
    root = Path("/tmp/kiwimatecoder")

    _patch_git_helpers(monkeypatch, root=root, behind=3, shas=["abc1234", "def5678"])
    monkeypatch.setattr(
        updater,
        "_run",
        lambda command, console: calls.append(command) or 0,
    )

    console = Console(record=True)
    code = updater.run_update(console)

    assert code == 0
    output = console.export_text()
    assert "abc1234" in output
    assert "def5678" in output
    assert "Updated abc1234" in output
    assert "3 commit(s) behind origin/main" in output


def test_run_update_skips_reinstall_when_pull_changes_nothing(monkeypatch):
    calls = []
    root = Path("/tmp/kiwimatecoder")

    # behind unknown (fetch failed) -> proceed to pull, but sha unchanged afterwards.
    _patch_git_helpers(monkeypatch, root=root, behind=None, shas=["abc1234", "abc1234"])
    monkeypatch.setattr(
        updater,
        "_run",
        lambda command, console: calls.append(command) or 0,
    )

    console = Console(record=True)
    code = updater.run_update(console)

    assert code == 0
    # pull ran, but install was skipped because the SHA did not change.
    assert calls == [updater.build_git_pull_command(root)]
    assert "Already on the latest version" in console.export_text()


def test_run_update_git_fallback_failure_prints_guidance(monkeypatch):
    monkeypatch.setattr(updater, "find_source_root", lambda: None)
    monkeypatch.setattr(updater, "_run", lambda command, console: 1)

    console = Console(record=True)
    code = updater.run_update(console)

    assert code == 1
    output = console.export_text()
    assert "not on PyPI" in output
    assert updater.GITHUB_REPO_URL in output
