"""Cross-machine session sync tests.

Everything runs in temp directories: ``config.CONFIG_DIR`` is isolated and
``kiwimatecoder.session._sessions_dir`` is swapped per fake machine, so the
real home directory is never read or written.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from kiwimatecoder import config, sync
from kiwimatecoder import session as session_store
from kiwimatecoder.commands import CommandResult, dispatch
from kiwimatecoder.main import app
from kiwimatecoder.session import load_session


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config" / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config" / "legacy")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    return tmp_path


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def _output(console: Console) -> str:
    return console.file.getvalue()


def _write_session(path: Path, text: str, saved_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "format_version": 2,
                "provider_id": "openai",
                "model": "test-model",
                "mode": "ask",
                "workspace_root": ".",
                "messages": [{"role": "user", "content": text}],
                "saved_at": saved_at,
            }
        ),
        encoding="utf-8",
    )


def _content(path: Path) -> str:
    return json.loads(path.read_text(encoding="utf-8"))["messages"][0]["content"]


def _make_machine(root: Path, name: str, shared: Path, monkeypatch) -> Path:
    """Point config and the sessions store at one fake machine."""
    sessions = root / name / "sessions"
    sessions.mkdir(parents=True)
    cfg_dir = root / name / "config"
    cfg_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(config, "CONFIG_FILE", cfg_dir / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", cfg_dir / "legacy")
    monkeypatch.setattr(session_store, "_sessions_dir", lambda: sessions)
    config.set_sync(enabled=True, path=str(shared), machine=name)
    return sessions


def _remote_root(shared: Path) -> Path:
    return shared / sync.SYNC_DIR_NAME


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_get_sync_defaults_and_machine_slug():
    settings = config.get_sync()

    assert settings["enabled"] is False
    assert settings["path"] == ""
    assert settings["include_autosave"] is False
    assert re.fullmatch(r"[a-z0-9-]+", settings["machine"])


def test_set_sync_validates_and_roundtrips(tmp_path):
    with pytest.raises(ValueError):
        config.set_sync(enabled=True)
    with pytest.raises(ValueError):
        config.set_sync(enabled=True, path=str(tmp_path / "missing"))
    with pytest.raises(ValueError):
        config.set_sync(enabled="yes")
    with pytest.raises(ValueError):
        config.set_sync(machine="   ")
    with pytest.raises(ValueError):
        config.set_sync(include_autosave="yes")
    with pytest.raises(ValueError):
        config.set_sync(path=5)

    shared = tmp_path / "shared"
    shared.mkdir()
    settings = config.set_sync(
        enabled=True,
        path=str(shared),
        machine="laptop",
        include_autosave=True,
    )

    assert settings["enabled"] is True
    assert settings["path"] == str(shared)
    assert settings["machine"] == "laptop"
    assert settings["include_autosave"] is True
    assert config.get_sync() == settings

    config.set_sync(enabled=False)
    assert config.get_sync()["enabled"] is False
    assert config.get_sync()["path"] == str(shared)


def test_get_sync_tolerates_garbage_section(tmp_path):
    cfg = config.load_config()
    cfg["sync"] = ["not", "an", "object"]
    config.save_config(cfg)

    settings = config.get_sync()

    assert settings["enabled"] is False
    assert settings["path"] == ""


def test_validate_config_flags_sync_errors(tmp_path):
    cfg = config.load_config()
    cfg["sync"] = {
        "enabled": "yes",
        "path": 5,
        "machine": 7,
        "include_autosave": "no",
    }
    issues = config.validate_config(cfg)
    keys = {issue["key"] for issue in issues if issue["level"] == "error"}

    assert {
        "sync.enabled",
        "sync.path",
        "sync.machine",
        "sync.include_autosave",
    } <= keys

    cfg["sync"] = {"enabled": True, "path": str(tmp_path / "missing")}
    issues = config.validate_config(cfg)
    assert any(
        issue["key"] == "sync.path" for issue in issues if issue["level"] == "error"
    )

    cfg["sync"] = {"enabled": True}
    issues = config.validate_config(cfg)
    assert any(
        issue["key"] == "sync.path" for issue in issues if issue["level"] == "error"
    )


def test_validate_config_accepts_a_valid_sync_section(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    cfg = config.load_config()
    cfg["sync"] = {
        "enabled": True,
        "path": str(shared),
        "machine": "laptop",
        "include_autosave": False,
    }

    errors = [
        issue
        for issue in config.validate_config(cfg)
        if issue["level"] == "error"
    ]

    assert errors == []


# ---------------------------------------------------------------------------
# Disabled sync
# ---------------------------------------------------------------------------


def test_status_disabled_is_guidance_and_touches_nothing(tmp_path):
    status = sync.status()

    assert status.enabled is False
    assert "disabled" in status.summary().lower()
    assert "sync enable" in status.summary()
    assert not (tmp_path / "config").exists()


def test_push_and_pull_disabled_raise_without_touching_files(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    config.set_sync(path=str(shared))

    with pytest.raises(sync.SyncError):
        sync.push()
    with pytest.raises(sync.SyncError):
        sync.pull()

    assert not _remote_root(shared).exists()


# ---------------------------------------------------------------------------
# Round trips and status
# ---------------------------------------------------------------------------


def test_status_reports_counts_and_pending_changes(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    laptop = _make_machine(tmp_path, "laptop", shared, monkeypatch)
    _write_session(laptop / "alpha.json", "v1", "2026-01-01T00:00:00")

    status = sync.status()

    assert status.enabled is True
    assert status.machine == "laptop"
    assert status.local == 1
    assert status.remote == 0
    assert status.pending_pushes == ["alpha.json"]
    assert status.pending_pulls == []
    assert status.conflicts == []
    assert status.last_sync is None
    assert status.root == _remote_root(shared)

    sync.push()
    status = sync.status()
    assert status.pending_pushes == []
    assert status.pending_pulls == []
    assert status.last_sync


def test_push_pull_roundtrip_between_machines(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    laptop = _make_machine(tmp_path, "laptop", shared, monkeypatch)
    _write_session(laptop / "alpha.json", "hello from laptop", "2026-01-01T00:00:00")

    report = sync.push()

    assert report.pushed == 1
    assert report.errors == []
    root = _remote_root(shared)
    assert (root / "alpha.json").is_file()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == sync.MANIFEST_VERSION
    assert manifest["machine"] == "laptop"
    assert manifest["last_sync"]
    assert manifest["files"]["alpha.json"]["sha1"]

    desktop = _make_machine(tmp_path, "desktop", shared, monkeypatch)
    report = sync.pull()

    assert report.pulled == 1
    assert report.errors == []
    assert (desktop / "alpha.json").is_file()
    loaded = load_session("alpha", workspace_root=tmp_path)
    assert loaded.messages == [{"role": "user", "content": "hello from laptop"}]


def test_push_and_pull_are_idempotent(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    laptop = _make_machine(tmp_path, "laptop", shared, monkeypatch)
    _write_session(laptop / "alpha.json", "v1", "2026-01-01T00:00:00")

    assert sync.push().pushed == 1
    second = sync.push()
    assert second.pushed == 0
    assert second.conflicts == 0
    assert second.errors == []

    _make_machine(tmp_path, "desktop", shared, monkeypatch)
    assert sync.pull().pulled == 1
    assert sync.pull().pulled == 0


# ---------------------------------------------------------------------------
# Last-write-wins
# ---------------------------------------------------------------------------


def test_push_lets_the_newer_local_copy_win(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    laptop = _make_machine(tmp_path, "laptop", shared, monkeypatch)
    root = _remote_root(shared)
    _write_session(root / "alpha.json", "remote older", "2026-01-01T00:00:00")
    _write_session(laptop / "alpha.json", "local newer", "2026-01-02T00:00:00")

    report = sync.push()

    assert report.pushed == 1
    assert _content(root / "alpha.json") == "local newer"


def test_push_keeps_the_newer_remote_copy(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    laptop = _make_machine(tmp_path, "laptop", shared, monkeypatch)
    root = _remote_root(shared)
    _write_session(root / "alpha.json", "remote newer", "2026-01-02T00:00:00")
    _write_session(laptop / "alpha.json", "local older", "2026-01-01T00:00:00")

    report = sync.push()

    assert report.pushed == 0
    assert report.conflicts == 0
    assert _content(root / "alpha.json") == "remote newer"


def test_pull_replaces_the_local_copy_with_a_newer_remote(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    laptop = _make_machine(tmp_path, "laptop", shared, monkeypatch)
    _write_session(laptop / "alpha.json", "v1", "2026-01-01T00:00:00")
    sync.push()

    desktop = _make_machine(tmp_path, "desktop", shared, monkeypatch)
    sync.pull()
    assert _content(desktop / "alpha.json") == "v1"

    _write_session(
        _remote_root(shared) / "alpha.json", "v2 from laptop", "2026-01-03T00:00:00"
    )
    report = sync.pull()

    assert report.pulled == 1
    assert _content(desktop / "alpha.json") == "v2 from laptop"


# ---------------------------------------------------------------------------
# Conflicts
# ---------------------------------------------------------------------------


def test_push_keeps_both_when_both_changed(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    laptop = _make_machine(tmp_path, "laptop", shared, monkeypatch)
    _write_session(laptop / "alpha.json", "v1", "2026-01-01T00:00:00")
    sync.push()

    root = _remote_root(shared)
    _write_session(laptop / "alpha.json", "v2 local", "2026-01-02T00:00:00")
    _write_session(root / "alpha.json", "v3 elsewhere", "2026-01-03T00:00:00")

    report = sync.push()

    assert report.conflicts == 1
    assert report.pushed == 0
    assert _content(root / "alpha.json") == "v3 elsewhere"
    conflict = root / "alpha__laptop.json"
    assert conflict.is_file()
    assert _content(conflict) == "v2 local"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) >= {"alpha.json", "alpha__laptop.json"}


def test_push_force_overrides_the_conflict_rule(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    laptop = _make_machine(tmp_path, "laptop", shared, monkeypatch)
    _write_session(laptop / "alpha.json", "v1", "2026-01-01T00:00:00")
    sync.push()

    root = _remote_root(shared)
    _write_session(laptop / "alpha.json", "v2 local", "2026-01-02T00:00:00")
    _write_session(root / "alpha.json", "v3 elsewhere", "2026-01-03T00:00:00")

    report = sync.push(force=True)

    assert report.conflicts == 0
    assert report.pushed == 1
    assert _content(root / "alpha.json") == "v2 local"
    assert not (root / "alpha__laptop.json").exists()


def test_pull_keeps_both_when_both_changed(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    laptop = _make_machine(tmp_path, "laptop", shared, monkeypatch)
    _write_session(laptop / "alpha.json", "v1", "2026-01-01T00:00:00")
    sync.push()

    desktop = _make_machine(tmp_path, "desktop", shared, monkeypatch)
    sync.pull()
    _write_session(desktop / "alpha.json", "v2 desktop", "2026-01-02T00:00:00")
    _write_session(
        _remote_root(shared) / "alpha.json", "v3 laptop", "2026-01-03T00:00:00"
    )

    report = sync.pull()

    assert report.conflicts == 1
    assert report.pulled == 0
    assert _content(desktop / "alpha.json") == "v2 desktop"
    conflict = desktop / "alpha__laptop.json"
    assert conflict.is_file()
    assert _content(conflict) == "v3 laptop"


def test_pull_force_overrides_the_conflict_rule(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    laptop = _make_machine(tmp_path, "laptop", shared, monkeypatch)
    _write_session(laptop / "alpha.json", "v1", "2026-01-01T00:00:00")
    sync.push()

    desktop = _make_machine(tmp_path, "desktop", shared, monkeypatch)
    sync.pull()
    _write_session(desktop / "alpha.json", "v2 desktop", "2026-01-02T00:00:00")
    _write_session(
        _remote_root(shared) / "alpha.json", "v3 laptop", "2026-01-03T00:00:00"
    )

    report = sync.pull(force=True)

    assert report.conflicts == 0
    assert report.pulled == 1
    assert _content(desktop / "alpha.json") == "v3 laptop"


# ---------------------------------------------------------------------------
# Autosave, corruption, and robustness
# ---------------------------------------------------------------------------


def test_autosave_excluded_by_default_and_included_when_configured(
    tmp_path, monkeypatch
):
    shared = tmp_path / "shared"
    shared.mkdir()
    laptop = _make_machine(tmp_path, "laptop", shared, monkeypatch)
    _write_session(laptop / "alpha.json", "v1", "2026-01-01T00:00:00")
    _write_session(laptop / "last.json", "autosave", "2026-01-02T00:00:00")

    sync.push()
    root = _remote_root(shared)
    assert (root / "alpha.json").is_file()
    assert not (root / "last.json").exists()

    config.set_sync(include_autosave=True)
    report = sync.push()

    assert report.pushed == 1
    assert (root / "last.json").is_file()


def test_corrupt_manifest_is_tolerated_and_rebuilt(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    laptop = _make_machine(tmp_path, "laptop", shared, monkeypatch)
    _write_session(laptop / "alpha.json", "v1", "2026-01-01T00:00:00")
    root = _remote_root(shared)
    root.mkdir(parents=True)
    (root / "manifest.json").write_text("{not json", encoding="utf-8")

    status = sync.status()
    assert status.errors == []
    assert status.last_sync is None
    assert status.pending_pushes == ["alpha.json"]

    report = sync.push()

    assert report.errors == []
    assert report.pushed == 1
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == sync.MANIFEST_VERSION
    assert "alpha.json" in manifest["files"]


def test_local_session_named_manifest_is_never_synced(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    laptop = _make_machine(tmp_path, "laptop", shared, monkeypatch)
    _write_session(laptop / "manifest.json", "not the index", "2026-01-01T00:00:00")

    report = sync.push()

    assert report.pushed == 0
    assert report.errors == []
    root = _remote_root(shared)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == sync.MANIFEST_VERSION
    assert "manifest.json" not in manifest["files"]


def test_corrupt_session_is_skipped_with_an_error_entry(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    laptop = _make_machine(tmp_path, "laptop", shared, monkeypatch)
    _write_session(laptop / "good.json", "v1", "2026-01-01T00:00:00")
    (laptop / "bad.json").write_text("{oops", encoding="utf-8")

    report = sync.push()

    assert report.pushed == 1
    assert any("bad.json" in error for error in report.errors)
    root = _remote_root(shared)
    assert (root / "good.json").is_file()
    assert not (root / "bad.json").exists()

    (root / "bogus.json").write_text("{nope", encoding="utf-8")
    report = sync.pull()

    assert any("bogus.json" in error for error in report.errors)
    assert not (laptop / "bogus.json").exists()
    assert sync.status().errors


# ---------------------------------------------------------------------------
# Slash command and CLI
# ---------------------------------------------------------------------------


def test_slash_sync_status_shows_guidance_when_disabled(session):
    console = _console()

    assert dispatch("/sync status", session, console) == CommandResult.CONTINUE

    assert "disabled" in _output(console).lower()


def test_slash_sync_status_and_push(tmp_path, monkeypatch, session):
    shared = tmp_path / "shared"
    shared.mkdir()
    sessions = _make_machine(tmp_path, "laptop", shared, monkeypatch)
    _write_session(sessions / "alpha.json", "v1", "2026-01-01T00:00:00")

    status_console = _console()
    assert dispatch("/sync", session, status_console) == CommandResult.CONTINUE
    assert "Local sessions: 1" in _output(status_console)

    push_console = _console()
    assert dispatch("/sync push", session, push_console) == CommandResult.CONTINUE
    assert "Pushed alpha.json" in _output(push_console)
    assert (_remote_root(shared) / "alpha.json").is_file()


def test_slash_sync_unknown_action_prints_usage(session):
    console = _console()

    dispatch("/sync dance", session, console)

    assert "Usage: /sync" in _output(console)


def test_sync_slash_completions():
    from kiwimatecoder.commands import (
        slash_argument_completions,
        slash_command_completions,
    )

    assert "/sync" in dict(slash_command_completions("sy"))
    assert "push" in dict(slash_argument_completions("sync", "pus"))


def test_cli_sync_roundtrip(tmp_path, monkeypatch):
    runner = CliRunner()
    shared = tmp_path / "shared"
    shared.mkdir()
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(session_store, "_sessions_dir", lambda: sessions)

    result = runner.invoke(app, ["sync", "status"])
    assert result.exit_code == 0
    assert "disabled" in result.output.lower()

    result = runner.invoke(app, ["sync", "push"])
    assert result.exit_code == 1
    assert "disabled" in result.output.lower()

    result = runner.invoke(app, ["sync", "enable", str(shared)])
    assert result.exit_code == 0
    settings = config.get_sync()
    assert settings["enabled"] is True
    assert settings["path"] == str(shared)
    assert settings["machine"]

    _write_session(sessions / "alpha.json", "cli session", "2026-01-01T00:00:00")

    result = runner.invoke(app, ["sync", "status"])
    assert result.exit_code == 0
    assert "Local sessions: 1" in result.output

    result = runner.invoke(app, ["sync", "push"])
    assert result.exit_code == 0
    assert "pushed 1" in result.output
    assert (_remote_root(shared) / "alpha.json").is_file()

    result = runner.invoke(app, ["sync", "pull"])
    assert result.exit_code == 0
    assert "pulled 0" in result.output

    result = runner.invoke(app, ["sync", "disable"])
    assert result.exit_code == 0
    assert config.get_sync()["enabled"] is False

    result = runner.invoke(app, ["sync", "enable", str(tmp_path / "missing")])
    assert result.exit_code == 1
    assert "not an existing directory" in result.output.lower()
