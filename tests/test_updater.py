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


def test_run_update_uses_git_source_checkout(monkeypatch):
    calls = []
    root = Path("/tmp/kiwimatecoder")

    monkeypatch.setattr(updater, "find_source_root", lambda: root)
    monkeypatch.setattr(updater, "_has_git_remote", lambda source_root: True)
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


def test_run_update_falls_back_to_pip(monkeypatch):
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

    monkeypatch.setattr(updater, "find_source_root", lambda: root)
    monkeypatch.setattr(updater, "_has_git_remote", lambda source_root: True)

    def fake_run(command, console):
        calls.append(command)
        return 7

    monkeypatch.setattr(updater, "_run", fake_run)

    code = updater.run_update(Console(record=True))

    assert code == 7
    assert calls == [updater.build_git_pull_command(root)]
