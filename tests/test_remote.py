"""Remote/SSH and devcontainer command construction, config, and wiring.

Nothing here ever executes ``ssh``, ``docker``, or ``devcontainer``: builder
tests assert argv only, ``shutil.which`` is mocked for every test, and
execution tests replace ``subprocess.Popen`` with fakes.
"""

from __future__ import annotations

import io
import json
import shlex
import shutil

import pytest
from rich.console import Console
from typer.testing import CliRunner

from kiwimatecoder import config, remote
from kiwimatecoder.remote import (
    DEVCONTAINER_WARNING,
    DOCKER_WARNING,
    NO_TARGET_WARNING,
    SSH_WARNING,
    RemoteTarget,
    detect_devcontainer,
    devcontainer_argv,
    docker_argv,
    remote_enabled,
    remote_preview,
    ssh_argv,
    wrap_remote_command,
)
from kiwimatecoder.shell import PersistentShell, ShellManager, close_shell
from kiwimatecoder.tools import run_bash as run_bash_module
from kiwimatecoder.tools.run_bash import _run_bash, preview as bash_preview

DEFAULTS = {
    "enabled": False,
    "host": "",
    "user": "",
    "port": 22,
    "identity": "",
    "workspace": "",
    "devcontainer": "auto",
}


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    """Point config storage at a temp dir and never resolve remote CLIs."""
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config" / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config" / "legacy")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    real_which = shutil.which

    def guarded_which(name: str) -> str | None:
        if name in {"ssh", "docker", "devcontainer"}:
            return None
        return real_which(name)

    monkeypatch.setattr(remote.shutil, "which", guarded_which)


@pytest.fixture(autouse=True)
def close_session_shell(session):
    """Never leak a persistent shell or background job between tests."""
    yield
    close_shell(session)


def _which_only(*names: str):
    def which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in names else None

    return which


def _make_devcontainer(workspace, payload: dict[str, object] | str) -> None:
    directory = workspace / ".devcontainer"
    directory.mkdir(parents=True, exist_ok=True)
    directory.joinpath("devcontainer.json").write_text(
        payload if isinstance(payload, str) else json.dumps(payload)
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_remote_config_defaults_and_roundtrip(tmp_path):
    assert config.get_remote() == DEFAULTS
    assert remote_enabled() is False

    key = tmp_path / "id_ed25519"
    key.write_text("private")
    updated = config.set_remote(
        enabled=True,
        host="devbox",
        user="kiwi",
        port=2222,
        identity=str(key),
        workspace="/srv/app",
        devcontainer="app-container",
    )

    assert updated == {
        "enabled": True,
        "host": "devbox",
        "user": "kiwi",
        "port": 2222,
        "identity": str(key),
        "workspace": "/srv/app",
        "devcontainer": "app-container",
    }
    assert config.get_remote() == updated
    assert remote_enabled() is True

    cleared = config.set_remote(host="", identity="", workspace="")
    assert cleared["host"] == ""
    assert cleared["identity"] == ""
    assert cleared["workspace"] == ""


def test_remote_config_enable_requires_host_or_container():
    with pytest.raises(ValueError):
        config.set_remote(enabled=True)

    assert config.set_remote(enabled=True, host="devbox")["enabled"] is True
    config.set_remote(enabled=False)

    # An explicit container name is enough to reach a container without SSH.
    assert config.set_remote(enabled=True, devcontainer="app")["enabled"] is True


def test_remote_config_rejects_invalid_updates(tmp_path):
    with pytest.raises(ValueError):
        config.set_remote(enabled="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        config.set_remote(host=5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        config.set_remote(port=0)
    with pytest.raises(ValueError):
        config.set_remote(port=70000)
    with pytest.raises(ValueError):
        config.set_remote(port="soon")
    with pytest.raises(ValueError):
        config.set_remote(identity=str(tmp_path / "missing"))
    with pytest.raises(ValueError):
        config.set_remote(devcontainer="")
    with pytest.raises(ValueError):
        config.set_remote(devcontainer=5)  # type: ignore[arg-type]


def test_remote_config_tolerates_junk_values():
    cfg = config.load_config()
    cfg["remote"] = {
        "enabled": "yes",
        "host": 5,
        "user": [],
        "port": "soon",
        "identity": 7,
        "workspace": None,
        "devcontainer": [],
    }

    assert config.get_remote(cfg) == DEFAULTS

    cfg["remote"] = ["nope"]
    assert config.get_remote(cfg) == DEFAULTS


def test_validate_config_flags_bad_remote_section(tmp_path):
    cfg = config.load_config()
    cfg["remote"] = {
        "enabled": "yes",
        "host": 5,
        "user": 7,
        "port": 70000,
        "identity": str(tmp_path / "missing"),
        "workspace": 9,
        "devcontainer": "",
    }

    errors = {
        issue["key"]
        for issue in config.validate_config(cfg)
        if issue["level"] == "error"
    }
    assert {
        "remote.enabled",
        "remote.host",
        "remote.user",
        "remote.workspace",
        "remote.port",
        "remote.identity",
        "remote.devcontainer",
    } <= errors

    cfg["remote"] = {"enabled": True}
    assert any(
        issue["key"] == "remote.host" and issue["level"] == "error"
        for issue in config.validate_config(cfg)
    )

    cfg["remote"] = ["nope"]
    assert any(
        issue["key"] == "remote" and issue["level"] == "error"
        for issue in config.validate_config(cfg)
    )

    assert config.validate_config(config.load_config()) == []


# ---------------------------------------------------------------------------
# Devcontainer detection
# ---------------------------------------------------------------------------


def test_detect_devcontainer_reads_name_and_workspace_folder(tmp_path):
    _make_devcontainer(
        tmp_path,
        "{\n  // comment line\n  \"name\": \"My Dev Container\",\n"
        "  \"workspaceFolder\": \"/work/app\"\n}\n",
    )

    info = detect_devcontainer(tmp_path)

    assert info is not None
    assert info.name == "My Dev Container"
    assert info.workspace_folder == "/work/app"
    assert info.config_path.name == "devcontainer.json"


def test_detect_devcontainer_falls_back_to_service_then_basename(tmp_path):
    workspace = tmp_path / "myapp"
    workspace.mkdir()
    _make_devcontainer(workspace, {"service": "web"})
    info = detect_devcontainer(workspace)
    assert info is not None
    assert info.name == "web"
    assert info.workspace_folder == "/workspaces/myapp"

    _make_devcontainer(workspace, {"image": "python:3"})
    info = detect_devcontainer(workspace)
    assert info is not None
    assert info.name == "myapp"
    assert info.workspace_folder == "/workspaces/myapp"


def test_detect_devcontainer_returns_none_for_missing_or_bad_files(tmp_path):
    assert detect_devcontainer(tmp_path) is None

    _make_devcontainer(tmp_path, "{not json")
    assert detect_devcontainer(tmp_path) is None

    directory = tmp_path / ".devcontainer"
    directory.mkdir(parents=True, exist_ok=True)
    directory.joinpath("devcontainer.json").write_text("[1, 2, 3]")
    assert detect_devcontainer(tmp_path) is None


# ---------------------------------------------------------------------------
# argv builders
# ---------------------------------------------------------------------------


def test_ssh_argv_minimal():
    argv = ssh_argv("pytest -q", RemoteTarget(enabled=True, host="devbox"))

    assert argv == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "devbox",
        "--",
        "sh",
        "-lc",
        "'pytest -q'",
    ]


def test_ssh_argv_with_user_port_identity_and_workspace(tmp_path):
    key = tmp_path / "id_ed25519"
    key.write_text("private")
    target = RemoteTarget(
        enabled=True,
        host="devbox",
        user="kiwi",
        port=2222,
        identity=str(key),
        workspace="/srv/app",
    )

    argv = ssh_argv("echo hi", target)

    assert argv == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-p",
        "2222",
        "-i",
        str(key),
        "kiwi@devbox",
        "--",
        "sh",
        "-lc",
        "'cd /srv/app && echo hi'",
    ]


def test_ssh_argv_quotes_awkward_workspace_and_command():
    target = RemoteTarget(host="devbox", workspace="/srv/my apps")

    argv = ssh_argv("echo 'hi there'", target)

    assert argv[-1] == shlex.quote("cd '/srv/my apps' && echo 'hi there'")


def test_docker_argv_shapes():
    assert docker_argv("pytest -q", "app") == [
        "docker",
        "exec",
        "app",
        "sh",
        "-lc",
        "pytest -q",
    ]
    assert docker_argv("pytest -q", "app", "/workspaces/app") == [
        "docker",
        "exec",
        "-w",
        "/workspaces/app",
        "app",
        "sh",
        "-lc",
        "pytest -q",
    ]
    assert docker_argv("exec /bin/sh", "app", "/ws", interactive=True) == [
        "docker",
        "exec",
        "-i",
        "-w",
        "/ws",
        "app",
        "sh",
        "-lc",
        "exec /bin/sh",
    ]


def test_devcontainer_argv_shape(tmp_path):
    argv = devcontainer_argv("pytest -q", tmp_path)

    assert argv == [
        "devcontainer",
        "exec",
        "--workspace-folder",
        str(tmp_path),
        "sh",
        "-lc",
        "pytest -q",
    ]


# ---------------------------------------------------------------------------
# wrap_remote_command
# ---------------------------------------------------------------------------


def test_wrap_remote_command_disabled_returns_local(tmp_path):
    assert wrap_remote_command("echo hi", workspace=tmp_path) == (None, None)


def test_wrap_remote_command_uses_ssh_when_configured(tmp_path, monkeypatch):
    config.set_remote(host="devbox", user="kiwi", workspace="/srv/app")
    config.set_remote(enabled=True)
    monkeypatch.setattr(remote.shutil, "which", _which_only("ssh"))

    argv, warning = wrap_remote_command("pytest -q", workspace=tmp_path)

    assert warning is None
    assert argv is not None
    assert argv[:4] == ["ssh", "-o", "BatchMode=yes", "kiwi@devbox"]
    assert argv[-1] == "'cd /srv/app && pytest -q'"


def test_wrap_remote_command_warns_when_ssh_missing(tmp_path, monkeypatch):
    config.set_remote(host="devbox")
    config.set_remote(enabled=True)
    monkeypatch.setattr(remote.shutil, "which", _which_only())

    argv, warning = wrap_remote_command("pytest -q", workspace=tmp_path)

    assert argv is None
    assert warning == SSH_WARNING


def test_wrap_remote_command_warns_without_host_or_container(tmp_path, monkeypatch):
    raw = config.load_config()
    raw["remote"] = {"enabled": True, "devcontainer": "off", "host": ""}
    monkeypatch.setattr(remote.shutil, "which", _which_only())

    argv, warning = wrap_remote_command("pytest -q", workspace=tmp_path, cfg=raw)

    assert argv is None
    assert warning == NO_TARGET_WARNING


def test_wrap_remote_command_prefers_devcontainer_cli(tmp_path, monkeypatch):
    _make_devcontainer(tmp_path, {"name": "dev", "workspaceFolder": "/work/dev"})
    config.set_remote(host="devbox")
    config.set_remote(enabled=True)
    monkeypatch.setattr(remote.shutil, "which", _which_only("ssh", "devcontainer"))

    argv, warning = wrap_remote_command("pytest -q", workspace=tmp_path)

    assert warning is None
    assert argv is not None
    assert argv[0] == "devcontainer"
    assert "--workspace-folder" in argv
    assert str(tmp_path) in argv


def test_wrap_remote_command_falls_back_to_docker_exec(tmp_path, monkeypatch):
    _make_devcontainer(tmp_path, {"name": "dev", "workspaceFolder": "/work/dev"})
    config.set_remote(host="devbox")
    config.set_remote(enabled=True)
    monkeypatch.setattr(remote.shutil, "which", _which_only("ssh", "docker"))

    argv, warning = wrap_remote_command("pytest -q", workspace=tmp_path)

    assert warning is None
    assert argv == [
        "docker",
        "exec",
        "-w",
        "/work/dev",
        "dev",
        "sh",
        "-lc",
        "pytest -q",
    ]


def test_wrap_remote_command_falls_back_to_ssh_for_devcontainer(tmp_path, monkeypatch):
    _make_devcontainer(tmp_path, {"name": "dev"})
    config.set_remote(host="devbox", workspace="/srv/app")
    config.set_remote(enabled=True)
    monkeypatch.setattr(remote.shutil, "which", _which_only("ssh"))

    argv, warning = wrap_remote_command("pytest -q", workspace=tmp_path)

    assert warning is None
    assert argv is not None
    assert argv[0] == "ssh"


def test_wrap_remote_command_warns_when_no_container_cli(tmp_path, monkeypatch):
    _make_devcontainer(tmp_path, {"name": "dev"})
    raw = config.load_config()
    raw["remote"] = {"enabled": True, "devcontainer": "auto", "host": ""}
    monkeypatch.setattr(remote.shutil, "which", _which_only())

    argv, warning = wrap_remote_command("pytest -q", workspace=tmp_path, cfg=raw)

    assert argv is None
    assert warning == DEVCONTAINER_WARNING


def test_wrap_remote_command_explicit_container_name(tmp_path, monkeypatch):
    config.set_remote(
        enabled=True,
        host="devbox",
        workspace="/srv/app",
        devcontainer="app-container",
    )
    monkeypatch.setattr(remote.shutil, "which", _which_only("docker"))

    argv, warning = wrap_remote_command("pytest -q", workspace=tmp_path)

    assert warning is None
    assert argv == [
        "docker",
        "exec",
        "-w",
        "/srv/app",
        "app-container",
        "sh",
        "-lc",
        "pytest -q",
    ]


def test_wrap_remote_command_explicit_container_missing_docker(tmp_path, monkeypatch):
    config.set_remote(host="devbox", devcontainer="app-container")
    config.set_remote(enabled=True)
    monkeypatch.setattr(remote.shutil, "which", _which_only())

    argv, warning = wrap_remote_command("pytest -q", workspace=tmp_path)

    assert argv is None
    assert warning == DOCKER_WARNING


def test_wrap_remote_command_interactive_threads_to_docker(tmp_path, monkeypatch):
    config.set_remote(host="devbox", devcontainer="app")
    config.set_remote(enabled=True)
    monkeypatch.setattr(remote.shutil, "which", _which_only("docker"))

    argv, _warning = wrap_remote_command(
        "exec /bin/sh", workspace=tmp_path, interactive=True
    )

    assert argv is not None
    assert argv[:3] == ["docker", "exec", "-i"]


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------


def test_remote_preview_plain_when_disabled(tmp_path):
    assert remote_preview("pytest -q", workspace=tmp_path) == "pytest -q"


def test_remote_preview_shows_ssh_target(tmp_path, monkeypatch):
    config.set_remote(host="devbox", user="kiwi", workspace="/srv/app")
    config.set_remote(enabled=True)
    monkeypatch.setattr(remote.shutil, "which", _which_only("ssh"))

    text = remote_preview("pytest -q", workspace=tmp_path)

    assert "pytest -q" in text
    assert "[remote]" in text
    assert "kiwi@devbox" in text
    assert "/srv/app" in text


def test_remote_preview_notes_missing_cli(tmp_path, monkeypatch):
    config.set_remote(host="devbox")
    config.set_remote(enabled=True)

    text = remote_preview("pytest -q", workspace=tmp_path)

    assert SSH_WARNING in text
    assert text.startswith("pytest -q")


# ---------------------------------------------------------------------------
# run_bash wiring
# ---------------------------------------------------------------------------


class _FakeRunProcess:
    pid = 4321
    returncode = 0

    def __init__(self, args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        return "", ""


def test_run_bash_executes_remote_argv(session, monkeypatch):
    config.set_remote(host="devbox", workspace="/srv/app")
    config.set_remote(enabled=True)
    monkeypatch.setattr(remote.shutil, "which", _which_only("ssh"))
    created: list[tuple[object, dict[str, object]]] = []

    def fake_popen(*args: object, **kwargs: object) -> _FakeRunProcess:
        created.append((args, kwargs))
        return _FakeRunProcess(args, **kwargs)

    monkeypatch.setattr(run_bash_module.subprocess, "Popen", fake_popen)

    result = _run_bash({"command": "echo hi"}, session)

    assert result.ok
    popen_args, popen_kwargs = created[0]
    argv = popen_args[0]
    assert isinstance(argv, list)
    assert argv[:4] == ["ssh", "-o", "BatchMode=yes", "devbox"]
    assert argv[-1] == "'cd /srv/app && echo hi'"
    assert popen_kwargs.get("shell") is False


def test_run_bash_remote_preview_shows_wrapper(session, monkeypatch):
    config.set_remote(host="devbox", user="kiwi")
    config.set_remote(enabled=True)
    monkeypatch.setattr(remote.shutil, "which", _which_only("ssh"))

    text = bash_preview({"command": "pytest -q"}, session)

    assert "pytest -q" in text
    assert "[remote]" in text
    assert "kiwi@devbox" in text


def test_run_bash_warns_and_runs_locally_when_cli_missing(session, monkeypatch):
    config.set_remote(host="devbox")
    config.set_remote(enabled=True)
    created: list[tuple[object, dict[str, object]]] = []

    def fake_popen(*args: object, **kwargs: object) -> _FakeRunProcess:
        created.append((args, kwargs))
        return _FakeRunProcess(args, **kwargs)

    monkeypatch.setattr(run_bash_module.subprocess, "Popen", fake_popen)

    result = _run_bash({"command": "echo hi"}, session)

    assert result.ok
    assert f"[remote] {SSH_WARNING}" in result.content
    popen_args, popen_kwargs = created[0]
    assert popen_args[0] == "echo hi"
    assert popen_kwargs.get("shell") is True


def test_run_bash_falls_back_when_remote_start_fails(session, monkeypatch):
    config.set_remote(host="devbox")
    config.set_remote(enabled=True)
    monkeypatch.setattr(remote.shutil, "which", _which_only("ssh"))
    created: list[tuple[object, dict[str, object]]] = []

    def fake_popen(*args: object, **kwargs: object) -> _FakeRunProcess:
        created.append((args, kwargs))
        if isinstance(args[0], list):
            raise FileNotFoundError(2, "No such file or directory", args[0][0])
        return _FakeRunProcess(args, **kwargs)

    monkeypatch.setattr(run_bash_module.subprocess, "Popen", fake_popen)

    result = _run_bash({"command": "echo still-ok"}, session)

    assert result.ok
    assert "running locally" in result.content
    assert created[1][0][0] == "echo still-ok"
    assert created[1][1].get("shell") is True


def test_run_bash_remote_and_sandbox_do_not_double_wrap(session, monkeypatch):
    config.set_sandbox(enabled=True)
    config.set_remote(host="devbox")
    config.set_remote(enabled=True)
    monkeypatch.setattr(remote.shutil, "which", _which_only("ssh"))
    created: list[tuple[object, dict[str, object]]] = []

    def fake_popen(*args: object, **kwargs: object) -> _FakeRunProcess:
        created.append((args, kwargs))
        return _FakeRunProcess(args, **kwargs)

    monkeypatch.setattr(run_bash_module.subprocess, "Popen", fake_popen)

    result = _run_bash({"command": "echo hi"}, session)

    assert result.ok
    argv = created[0][0][0]
    assert isinstance(argv, list)
    assert argv[0] == "ssh"
    assert "sandbox-exec" not in argv
    text = bash_preview({"command": "echo hi"}, session)
    assert "[sandbox]" not in text


# ---------------------------------------------------------------------------
# Persistent shell and background jobs
# ---------------------------------------------------------------------------


class _FakeShellProcess:
    pid = 4242
    returncode = 0

    def __init__(self, args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self.stdin = io.StringIO()
        self.stdout = io.StringIO("")
        self.stderr = io.StringIO()

    def poll(self) -> int:
        return 0


def _remote_shell_config() -> None:
    config.set_remote(host="devbox", user="kiwi", workspace="/srv/app")
    config.set_remote(enabled=True)


def test_persistent_shell_starts_through_remote_wrapper(session, monkeypatch):
    _remote_shell_config()
    monkeypatch.setattr(remote.shutil, "which", _which_only("ssh"))
    created: list[tuple[object, dict[str, object]]] = []

    def fake_popen(*args: object, **kwargs: object) -> _FakeShellProcess:
        created.append((args, kwargs))
        return _FakeShellProcess(args, **kwargs)

    monkeypatch.setattr(
        "kiwimatecoder.shell.subprocess.Popen", fake_popen
    )

    shell = PersistentShell(session.workspace_root)
    try:
        assert created
        args, _kwargs = created[0]
        assert args[0] == [
            "ssh",
            "-o",
            "BatchMode=yes",
            "kiwi@devbox",
            "--",
            "sh",
            "-lc",
            "'cd /srv/app && exec /bin/sh'",
        ]
    finally:
        shell.close()


def test_background_job_starts_through_remote_wrapper(session, monkeypatch):
    _remote_shell_config()
    monkeypatch.setattr(remote.shutil, "which", _which_only("ssh"))
    created: list[tuple[object, dict[str, object]]] = []

    def fake_popen(*args: object, **kwargs: object) -> _FakeShellProcess:
        created.append((args, kwargs))
        return _FakeShellProcess(args, **kwargs)

    monkeypatch.setattr(
        "kiwimatecoder.shell.subprocess.Popen", fake_popen
    )

    manager = ShellManager(session.workspace_root)
    try:
        manager.start_background("pytest -q")
        assert created
        args, kwargs = created[0]
        assert args[0] == [
            "ssh",
            "-o",
            "BatchMode=yes",
            "kiwi@devbox",
            "--",
            "sh",
            "-lc",
            "'cd /srv/app && pytest -q'",
        ]
        assert kwargs.get("shell") is False
    finally:
        manager.close()


def test_persistent_shell_ignores_sandbox_when_remote(session, monkeypatch):
    config.set_sandbox(enabled=True)
    _remote_shell_config()
    monkeypatch.setattr(remote.shutil, "which", _which_only("ssh"))
    created: list[tuple[object, dict[str, object]]] = []

    def fake_popen(*args: object, **kwargs: object) -> _FakeShellProcess:
        created.append((args, kwargs))
        return _FakeShellProcess(args, **kwargs)

    monkeypatch.setattr(
        "kiwimatecoder.shell.subprocess.Popen", fake_popen
    )

    shell = PersistentShell(session.workspace_root)
    try:
        args, _kwargs = created[0]
        assert isinstance(args[0], list)
        assert args[0][0] == "ssh"
        assert "sandbox-exec" not in args[0]
    finally:
        shell.close()


def test_shell_jobs_preview_shows_remote_wrapper(session, monkeypatch):
    from kiwimatecoder import tools

    _remote_shell_config()
    monkeypatch.setattr(remote.shutil, "which", _which_only("ssh"))

    text = tools.preview(
        "shell_jobs", {"action": "start", "command": "pytest -q"}, session
    )

    assert "[remote]" in text
    assert "kiwi@devbox" in text


# ---------------------------------------------------------------------------
# Slash command and CLI
# ---------------------------------------------------------------------------


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def test_config_remote_slash_roundtrip(session):
    from kiwimatecoder.commands import dispatch, slash_argument_completions

    console = _console()
    assert dispatch("/config remote show", session, console) == "continue"
    assert "Remote" in console.file.getvalue()

    dispatch("/config remote host devbox", session, _console())
    assert config.get_remote()["host"] == "devbox"

    dispatch("/config remote user kiwi", session, _console())
    assert config.get_remote()["user"] == "kiwi"

    dispatch("/config remote port 2222", session, _console())
    assert config.get_remote()["port"] == 2222

    dispatch("/config remote workspace /srv/app", session, _console())
    assert config.get_remote()["workspace"] == "/srv/app"

    dispatch("/config remote devcontainer off", session, _console())
    assert config.get_remote()["devcontainer"] == "off"

    dispatch("/config remote enable on", session, _console())
    assert config.get_remote()["enabled"] is True

    dispatch("/config remote enable off", session, _console())
    assert config.get_remote()["enabled"] is False

    bad = _console()
    dispatch("/config remote enable maybe", session, bad)
    assert "Usage" in bad.file.getvalue()

    assert "remote" in dict(
        slash_argument_completions("config", "rem", session)
    )


def test_config_remote_slash_identity(session, tmp_path):
    from kiwimatecoder.commands import dispatch

    key = tmp_path / "id_ed25519"
    key.write_text("private")

    dispatch(f"/config remote identity {key}", session, _console())
    assert config.get_remote()["identity"] == str(key)

    bad = _console()
    dispatch("/config remote identity /nope/missing", session, bad)
    assert "does not exist" in bad.file.getvalue()


def test_config_remote_cli_roundtrip():
    from kiwimatecoder.main import app

    runner = CliRunner()

    result = runner.invoke(app, ["config", "remote", "show"])
    assert result.exit_code == 0
    assert "Remote" in result.output

    result = runner.invoke(app, ["config", "remote", "host", "devbox"])
    assert result.exit_code == 0
    assert config.get_remote()["host"] == "devbox"

    runner.invoke(app, ["config", "remote", "user", "kiwi"])
    assert config.get_remote()["user"] == "kiwi"

    runner.invoke(app, ["config", "remote", "port", "2222"])
    assert config.get_remote()["port"] == 2222

    runner.invoke(app, ["config", "remote", "workspace", "/srv/app"])
    assert config.get_remote()["workspace"] == "/srv/app"

    runner.invoke(app, ["config", "remote", "devcontainer", "off"])
    assert config.get_remote()["devcontainer"] == "off"

    result = runner.invoke(app, ["config", "remote", "enable", "on"])
    assert result.exit_code == 0
    assert config.get_remote()["enabled"] is True

    result = runner.invoke(app, ["config", "remote", "enable", "off"])
    assert result.exit_code == 0
    assert config.get_remote()["enabled"] is False

    result = runner.invoke(app, ["config", "remote", "port", "0"])
    assert result.exit_code == 1

    result = runner.invoke(app, ["config", "remote", "identity", "/nope/missing"])
    assert result.exit_code == 1

    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "Remote:" in result.output
