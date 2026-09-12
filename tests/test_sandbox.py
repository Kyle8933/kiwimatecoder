"""Sandbox command construction, config, and tool wiring.

Nothing here ever executes ``sandbox-exec`` or ``bwrap``: builder tests assert
argv shape only, and execution tests replace ``subprocess.Popen`` or use the
unsandboxed fallback with a harmless command.
"""

from __future__ import annotations

import io
import subprocess

import pytest
from rich.console import Console
from typer.testing import CliRunner

from kiwimatecoder import config, sandbox
from kiwimatecoder.sandbox import (
    FALLBACK_WARNING,
    bwrap_command,
    sandbox_available,
    seatbelt_command,
    seatbelt_profile,
    wrap_command,
)
from kiwimatecoder.tools import run_bash as run_bash_module
from kiwimatecoder.tools.run_bash import _run_bash, preview


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def test_sandbox_available_darwin_uses_sandbox_exec(monkeypatch):
    monkeypatch.setattr(
        sandbox.shutil, "which", lambda name: f"/usr/bin/{name}"
    )

    assert sandbox_available("darwin") == ("seatbelt", "/usr/bin/sandbox-exec")


def test_sandbox_available_linux_uses_bwrap(monkeypatch):
    monkeypatch.setattr(
        sandbox.shutil, "which", lambda name: f"/usr/bin/{name}"
    )

    assert sandbox_available("linux") == ("bwrap", "/usr/bin/bwrap")


def test_sandbox_available_missing_or_unsupported(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)

    assert sandbox_available("darwin") is None
    assert sandbox_available("linux") is None
    assert sandbox_available("win32") is None


# ---------------------------------------------------------------------------
# Seatbelt builder
# ---------------------------------------------------------------------------


def test_seatbelt_profile_allows_network_and_broad_reads(tmp_path):
    profile = seatbelt_profile(workspace=tmp_path)

    assert "(version 1)" in profile
    assert "(deny default)" in profile
    assert "(allow process*)" in profile
    assert "(allow file-read*)" in profile
    assert "(allow network*)" in profile
    assert "(deny network*)" not in profile
    assert f'(subpath "{tmp_path}")' in profile
    assert '(subpath "/tmp")' in profile
    assert '(subpath "/private/tmp")' in profile


def test_seatbelt_profile_denies_network_and_adds_writable_paths(tmp_path):
    extra = tmp_path / "extra"
    profile = seatbelt_profile(
        workspace=tmp_path, extra_writable=[str(extra)], network=False
    )

    assert "(deny network*)" in profile
    assert "(allow network*)" not in profile
    assert f'(subpath "{extra}")' in profile
    write_rule = next(
        line for line in profile.splitlines() if "file-write" in line
    )
    assert f'(subpath "{tmp_path}")' in write_rule
    assert f'(subpath "{extra}")' in write_rule
    assert '(subpath "/srv/secret")' not in profile


def test_seatbelt_command_shape(tmp_path):
    argv = seatbelt_command(
        "pytest -q", workspace=tmp_path, extra_writable=(), network=True
    )

    assert argv[0] == "sandbox-exec"
    assert argv[1] == "-p"
    assert "(version 1)" in argv[2]
    assert argv[3:] == ["/bin/sh", "-c", "pytest -q"]


# ---------------------------------------------------------------------------
# Bubblewrap builder
# ---------------------------------------------------------------------------


def test_bwrap_command_binds_root_workspace_and_tmp(tmp_path):
    extra = tmp_path / "extra"
    argv = bwrap_command(
        "pytest -q",
        workspace=tmp_path,
        extra_writable=[str(extra)],
        network=True,
    )

    assert argv[0] == "bwrap"
    assert argv[1:4] == ["--ro-bind", "/", "/"]
    assert argv[4:7] == ["--bind", str(tmp_path), str(tmp_path)]
    assert "--tmpfs" in argv and "/tmp" in argv
    extra_index = argv.index(str(extra))
    assert argv[extra_index - 1] == "--bind"
    assert argv[extra_index + 1] == str(extra)
    assert "--unshare-net" not in argv
    assert argv[-3:] == ["/bin/sh", "-c", "pytest -q"]


def test_bwrap_command_unshares_network_when_disabled(tmp_path):
    argv = bwrap_command("pytest -q", workspace=tmp_path, network=False)

    assert "--unshare-net" in argv
    assert argv[-3:] == ["/bin/sh", "-c", "pytest -q"]


# ---------------------------------------------------------------------------
# wrap_command
# ---------------------------------------------------------------------------


def test_wrap_command_disabled_returns_plain_shell(tmp_path):
    argv, warning = wrap_command("echo hi", workspace=tmp_path)

    assert argv == ["/bin/sh", "-c", "echo hi"]
    assert warning is None


def test_wrap_command_enabled_without_backend_warns(tmp_path, monkeypatch):
    config.set_sandbox(enabled=True)
    monkeypatch.setattr(sandbox, "sandbox_available", lambda platform=None: None)

    argv, warning = wrap_command("echo hi", workspace=tmp_path)

    assert argv == ["/bin/sh", "-c", "echo hi"]
    assert warning == FALLBACK_WARNING


def test_wrap_command_seatbelt_uses_resolved_path(tmp_path, monkeypatch):
    config.set_sandbox(enabled=True, network=False)
    monkeypatch.setattr(
        sandbox,
        "sandbox_available",
        lambda platform=None: ("seatbelt", "/usr/bin/sandbox-exec"),
    )

    argv, warning = wrap_command("echo hi", workspace=tmp_path)

    assert warning is None
    assert argv[0] == "/usr/bin/sandbox-exec"
    assert "(deny network*)" in argv[2]
    assert argv[3:] == ["/bin/sh", "-c", "echo hi"]


def test_wrap_command_bwrap_uses_resolved_path(tmp_path, monkeypatch):
    config.set_sandbox(enabled=True)
    monkeypatch.setattr(
        sandbox,
        "sandbox_available",
        lambda platform=None: ("bwrap", "/usr/bin/bwrap"),
    )

    argv, warning = wrap_command("echo hi", workspace=tmp_path)

    assert warning is None
    assert argv[0] == "/usr/bin/bwrap"
    assert "--unshare-net" not in argv
    assert argv[-3:] == ["/bin/sh", "-c", "echo hi"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_sandbox_config_defaults_and_roundtrip():
    assert config.get_sandbox() == {
        "enabled": False,
        "network": True,
        "extra_writable": [],
    }

    updated = config.set_sandbox(
        enabled=True, network=False, extra_writable=["/srv/data"]
    )

    assert updated == {
        "enabled": True,
        "network": False,
        "extra_writable": ["/srv/data"],
    }
    assert config.get_sandbox() == updated


def test_sandbox_config_tolerates_junk_values():
    cfg = config.load_config()
    cfg["sandbox"] = {"enabled": "yes", "network": 0, "extra_writable": "nope"}

    assert config.get_sandbox(cfg) == {
        "enabled": False,
        "network": True,
        "extra_writable": [],
    }

    cfg["sandbox"] = ["nope"]
    assert config.get_sandbox(cfg) == {
        "enabled": False,
        "network": True,
        "extra_writable": [],
    }


def test_sandbox_config_rejects_invalid_updates():
    with pytest.raises(ValueError):
        config.set_sandbox(enabled="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        config.set_sandbox(network="maybe")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        config.set_sandbox(extra_writable=[""])
    with pytest.raises(ValueError):
        config.set_sandbox(extra_writable=[1])  # type: ignore[list-item]


def test_validate_config_flags_bad_sandbox_section():
    cfg = config.load_config()
    cfg["sandbox"] = {
        "enabled": "yes",
        "network": "no",
        "extra_writable": ["", 5],
    }

    errors = {
        issue["key"]
        for issue in config.validate_config(cfg)
        if issue["level"] == "error"
    }
    assert {
        "sandbox.enabled",
        "sandbox.network",
        "sandbox.extra_writable[0]",
        "sandbox.extra_writable[1]",
    } <= errors

    cfg["sandbox"] = ["nope"]
    assert any(
        issue["key"] == "sandbox" and issue["level"] == "error"
        for issue in config.validate_config(cfg)
    )

    assert config.validate_config(config.load_config()) == []


# ---------------------------------------------------------------------------
# run_bash preview and execution
# ---------------------------------------------------------------------------


def test_preview_is_plain_when_sandbox_disabled(session):
    assert preview({"command": "pytest -q"}, session) == "pytest -q"


def test_preview_shows_sandbox_note_when_enabled(session, monkeypatch):
    config.set_sandbox(enabled=True, network=False)
    monkeypatch.setattr(
        sandbox,
        "sandbox_available",
        lambda platform=None: ("seatbelt", "sandbox-exec"),
    )

    text = preview({"command": "pytest -q"}, session)

    assert "pytest -q" in text
    assert "[sandbox]" in text
    assert "network off" in text
    assert "sandbox-exec" in text


def test_preview_warns_when_backend_unavailable(session, monkeypatch):
    config.set_sandbox(enabled=True)
    monkeypatch.setattr(sandbox, "sandbox_available", lambda platform=None: None)

    text = preview({"command": "pytest -q"}, session)

    assert "[sandbox]" in text
    assert FALLBACK_WARNING in text


class _FakeProcess:
    pid = 4321
    returncode = 0

    def __init__(self, args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        return "", ""


def test_run_bash_executes_wrapped_argv_when_enabled(session, monkeypatch):
    config.set_sandbox(enabled=True)
    monkeypatch.setattr(
        sandbox,
        "sandbox_available",
        lambda platform=None: ("seatbelt", "/usr/bin/sandbox-exec"),
    )
    created: list[tuple[object, dict[str, object]]] = []

    def fake_popen(*args: object, **kwargs: object) -> _FakeProcess:
        created.append((args, kwargs))
        return _FakeProcess(args, **kwargs)

    monkeypatch.setattr(run_bash_module.subprocess, "Popen", fake_popen)

    result = _run_bash({"command": "echo hi"}, session)

    assert result.ok
    popen_args, popen_kwargs = created[0]
    argv = popen_args[0]
    assert isinstance(argv, list)
    assert argv[0] == "/usr/bin/sandbox-exec"
    assert argv[-3:] == ["/bin/sh", "-c", "echo hi"]
    assert popen_kwargs.get("shell") is False


def test_run_bash_falls_back_and_warns_when_backend_missing(session, monkeypatch):
    config.set_sandbox(enabled=True)
    monkeypatch.setattr(sandbox, "sandbox_available", lambda platform=None: None)

    result = _run_bash({"command": "echo fallback-ok"}, session)

    assert result.ok
    assert "[sandbox]" in result.content
    assert "without a sandbox" in result.content
    assert "fallback-ok" in result.content


def test_run_bash_falls_back_when_sandbox_start_fails(session, monkeypatch):
    config.set_sandbox(enabled=True)
    monkeypatch.setattr(
        sandbox,
        "sandbox_available",
        lambda platform=None: ("seatbelt", "/nope/sandbox-exec"),
    )
    real_popen = subprocess.Popen

    def fake_popen(args: object, **kwargs: object) -> object:
        if isinstance(args, list):
            raise FileNotFoundError(2, "No such file or directory", args[0])
        return real_popen(args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(run_bash_module.subprocess, "Popen", fake_popen)

    result = _run_bash({"command": "echo still-ok"}, session)

    assert result.ok
    assert "running unsandboxed" in result.content
    assert "still-ok" in result.content


# ---------------------------------------------------------------------------
# CLI and slash command
# ---------------------------------------------------------------------------


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def test_config_sandbox_slash_roundtrip(session):
    from kiwimatecoder.commands import dispatch

    console = _console()
    assert dispatch("/config sandbox show", session, console) == "continue"
    assert "Sandbox" in console.file.getvalue()

    dispatch("/config sandbox enable on", session, _console())
    assert config.get_sandbox()["enabled"] is True

    dispatch("/config sandbox network off", session, _console())
    assert config.get_sandbox()["network"] is False

    dispatch("/config sandbox add-path /srv/data", session, _console())
    assert config.get_sandbox()["extra_writable"] == ["/srv/data"]

    dispatch("/config sandbox remove-path /srv/data", session, _console())
    assert config.get_sandbox()["extra_writable"] == []

    bad = _console()
    dispatch("/config sandbox enable maybe", session, bad)
    assert "Usage" in bad.file.getvalue()


def test_config_sandbox_cli_roundtrip():
    from kiwimatecoder.main import app

    runner = CliRunner()

    result = runner.invoke(app, ["config", "sandbox", "show"])
    assert result.exit_code == 0
    assert "Sandbox" in result.output

    result = runner.invoke(app, ["config", "sandbox", "enable", "on"])
    assert result.exit_code == 0
    assert config.get_sandbox()["enabled"] is True

    runner.invoke(app, ["config", "sandbox", "network", "off"])
    assert config.get_sandbox()["network"] is False

    runner.invoke(app, ["config", "sandbox", "add-path", "/srv/data"])
    assert config.get_sandbox()["extra_writable"] == ["/srv/data"]

    runner.invoke(app, ["config", "sandbox", "remove-path", "/srv/data"])
    assert config.get_sandbox()["extra_writable"] == []

    runner.invoke(app, ["config", "sandbox", "clear-paths"])
    assert config.get_sandbox()["extra_writable"] == []

    result = runner.invoke(app, ["config", "sandbox", "enable", "maybe"])
    assert result.exit_code == 1
