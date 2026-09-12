"""Remote (SSH) and devcontainer command execution.

This module is pure command construction: it decides *where* a shell command
should run (a remote host over SSH, inside a devcontainer, or in a Docker
container) but never runs anything itself. That keeps it trivially testable and
lets callers fall back to local execution when the required CLI is missing.

Order of preference when ``remote.devcontainer`` is ``"auto"``:

1. a detected ``.devcontainer/devcontainer.json`` plus the ``devcontainer`` CLI,
2. the detected container through ``docker exec``,
3. SSH when ``remote.host`` is configured,
4. local execution (the default) with a warning.

Only the *command* runs remotely. File tools still read and write the locally
accessible workspace, so point them at an sshfs/bind mount of the remote tree
(or accept that edits land locally). See the README section "Remote and
devcontainer execution" for the full limitations.
"""

from __future__ import annotations

import json
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kiwimatecoder.config import get_remote

SSH_WARNING = (
    "Remote execution is enabled but the 'ssh' CLI was not found; "
    "running the command locally."
)
DOCKER_WARNING = (
    "Remote execution is enabled but the 'docker' CLI was not found; "
    "running the command locally."
)
DEVCONTAINER_WARNING = (
    "Remote execution is enabled but neither the 'devcontainer' nor the "
    "'docker' CLI was found; running the command locally."
)
NO_TARGET_WARNING = (
    "Remote execution is enabled but no host or container is configured; "
    "running the command locally."
)

DEVCONTAINER_CONFIG = Path(".devcontainer") / "devcontainer.json"
DEFAULT_PORT = 22


@dataclass(frozen=True)
class RemoteTarget:
    """Where remote commands should run, built from the ``remote`` config."""

    enabled: bool = False
    host: str = ""
    user: str = ""
    port: int = DEFAULT_PORT
    identity: str = ""
    workspace: str = ""
    devcontainer: str = "auto"

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None) -> "RemoteTarget":
        """Build a target from the active (or given) configuration."""
        return cls(**get_remote(cfg))

    @property
    def address(self) -> str:
        """``user@host`` when a user is set, else just the host."""
        return f"{self.user}@{self.host}" if self.user else self.host


@dataclass(frozen=True)
class DevcontainerInfo:
    """The parts of ``.devcontainer/devcontainer.json`` this module uses."""

    name: str
    workspace_folder: str
    config_path: Path


def remote_enabled(cfg: dict[str, Any] | None = None) -> bool:
    """Whether remote/devcontainer execution is turned on."""
    return bool(get_remote(cfg)["enabled"])


def _strip_line_comments(text: str) -> str:
    """Drop full-line ``//`` comments so devcontainer JSON parses."""
    lines = [
        line for line in text.splitlines() if not line.lstrip().startswith("//")
    ]
    return "\n".join(lines)


def detect_devcontainer(workspace: str | Path) -> DevcontainerInfo | None:
    """Read ``.devcontainer/devcontainer.json`` under ``workspace``, if valid.

    Returns ``None`` when the file is missing, unreadable, or malformed. The
    container name is ``name`` then ``service`` from the file, falling back to
    the workspace directory's basename; the workspace folder is
    ``workspaceFolder`` when set, else ``/workspaces/<basename>``.
    """
    root = Path(workspace).expanduser()
    path = root / DEVCONTAINER_CONFIG
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(_strip_line_comments(text))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    basename = root.name or "workspace"
    container = ""
    for key in ("name", "service"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            container = value.strip()
            break
    if not container:
        container = basename
    folder = data.get("workspaceFolder")
    workspace_folder = (
        folder.strip()
        if isinstance(folder, str) and folder.strip()
        else f"/workspaces/{basename}"
    )
    return DevcontainerInfo(
        name=container, workspace_folder=workspace_folder, config_path=path
    )


def _remote_command(command: str, workspace: str = "") -> str:
    """Prefix ``command`` with a ``cd`` when a remote workspace is set."""
    if workspace:
        return f"cd {shlex.quote(workspace)} && {command}"
    return command


def ssh_argv(command: str, target: RemoteTarget) -> list[str]:
    """Build the SSH invocation for ``command``.

    The remote command is quoted for the SSH wire: the client joins the remote
    argv with spaces and hands the result to the remote login shell, so an
    unquoted ``sh -lc`` argument would be split apart.
    """
    argv = ["ssh", "-o", "BatchMode=yes"]
    if target.port and target.port != DEFAULT_PORT:
        argv += ["-p", str(target.port)]
    if target.identity:
        argv += ["-i", target.identity]
    remote = _remote_command(command, target.workspace)
    argv += [target.address, "--", "sh", "-lc", shlex.quote(remote)]
    return argv


def docker_argv(
    command: str,
    container: str,
    workspace: str = "",
    *,
    interactive: bool = False,
) -> list[str]:
    """Build ``docker exec`` for ``command`` in ``container``.

    ``interactive`` adds ``-i`` so a persistent shell can read commands from
    stdin; one-shot commands do not need it.
    """
    argv = ["docker", "exec"]
    if interactive:
        argv.append("-i")
    if workspace:
        argv += ["-w", workspace]
    argv += [container, "sh", "-lc", command]
    return argv


def devcontainer_argv(command: str, workspace: str | Path) -> list[str]:
    """Build ``devcontainer exec --workspace-folder <workspace> ...``."""
    return [
        "devcontainer",
        "exec",
        "--workspace-folder",
        str(workspace),
        "sh",
        "-lc",
        command,
    ]


def _resolve_remote(
    command: str,
    *,
    workspace: str | Path,
    cfg: dict[str, Any] | None = None,
    interactive: bool = False,
) -> tuple[list[str] | None, str | None, str | None]:
    """Return ``(argv, warning, label)`` for the best remote route.

    ``argv`` is ``None`` when the command should run locally; ``warning`` is
    then set when remote execution was requested but could not be arranged.
    """
    target = RemoteTarget.from_config(cfg)
    if not target.enabled:
        return None, None, None
    root = Path(workspace).expanduser()
    info = detect_devcontainer(root) if target.devcontainer == "auto" else None

    if target.devcontainer not in {"auto", "off"}:
        if shutil.which("docker"):
            label = f"docker exec {target.devcontainer}"
            if target.workspace:
                label += f" (workspace {target.workspace})"
            return (
                docker_argv(
                    command,
                    target.devcontainer,
                    target.workspace,
                    interactive=interactive,
                ),
                None,
                label,
            )
        return None, DOCKER_WARNING, None

    if info is not None:
        if shutil.which("devcontainer"):
            return (
                devcontainer_argv(command, root),
                None,
                f"devcontainer {info.name} (folder {root})",
            )
        if shutil.which("docker"):
            return (
                docker_argv(
                    command,
                    info.name,
                    info.workspace_folder,
                    interactive=interactive,
                ),
                None,
                f"docker exec {info.name} (workspace {info.workspace_folder})",
            )

    if target.host:
        if shutil.which("ssh"):
            label = f"ssh {target.address}"
            if target.workspace:
                label += f" (cd {target.workspace})"
            return ssh_argv(command, target), None, label
        return None, SSH_WARNING, None

    if info is not None:
        return None, DEVCONTAINER_WARNING, None
    return None, NO_TARGET_WARNING, None


def wrap_remote_command(
    command: str,
    *,
    workspace: str | Path,
    cfg: dict[str, Any] | None = None,
    interactive: bool = False,
) -> tuple[list[str] | None, str | None]:
    """Return the remote argv to execute plus an optional warning.

    ``(None, None)`` means the caller should run the command locally with its
    normal shell (remote execution is off). ``(None, warning)`` means remote
    execution was requested but could not be arranged and the caller should
    surface the warning before falling back to local execution.
    """
    argv, warning, _label = _resolve_remote(
        command, workspace=workspace, cfg=cfg, interactive=interactive
    )
    return argv, warning


def remote_preview(
    command: str, *, workspace: str | Path, cfg: dict[str, Any] | None = None
) -> str:
    """Return command text plus a readable remote note for approval prompts."""
    argv, warning, label = _resolve_remote(command, workspace=workspace, cfg=cfg)
    if argv is None:
        if warning:
            return f"{command}\n[remote] {warning}"
        return command
    return f"{command}\n[remote] via {label}"
