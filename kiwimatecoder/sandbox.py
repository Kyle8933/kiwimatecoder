"""Optional OS-level command sandboxing for ``run_bash`` and the shell tools.

This module is pure command construction: it decides *what* to execute (macOS
seatbelt via ``sandbox-exec`` or Linux bubblewrap via ``bwrap``) but never runs
anything itself. That keeps it trivially testable and lets callers fall back to
an unsandboxed execution when a backend is missing.

The sandbox is **off by default** and is a damage-limiter, not a security
boundary against a hostile model: the profile still allows broad reads and,
unless network is disabled, full network access. Only enable it in workspaces
you trust. Windows has no supported backend here.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kiwimatecoder.config import get_sandbox

FALLBACK_WARNING = (
    "Sandboxing is enabled but no supported backend is available on this "
    "platform; running the command without a sandbox."
)


def sandbox_available(platform: str | None = None) -> tuple[str, str] | None:
    """Detect the sandbox backend for ``platform`` (defaults to ``sys.platform``).

    Returns ``(backend, path)`` where ``backend`` is ``"seatbelt"`` on macOS or
    ``"bwrap"`` on Linux, or ``None`` when the platform is unsupported or the
    backend executable is not installed.
    """
    system = platform if platform is not None else sys.platform
    if system == "darwin":
        path = shutil.which("sandbox-exec")
        return ("seatbelt", path) if path else None
    if system.startswith("linux"):
        path = shutil.which("bwrap")
        return ("bwrap", path) if path else None
    return None


def _sbpl_string(value: object) -> str:
    """Quote a path for an SBPL profile string."""
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def seatbelt_profile(
    *,
    workspace: str | Path,
    extra_writable: Sequence[str] = (),
    network: bool = True,
) -> str:
    """Build the seatbelt (SBPL) profile used by :func:`seatbelt_command`.

    Reads are allowed broadly; writes are limited to the workspace, ``/tmp``
    (and its macOS alias ``/private/tmp``), and any ``extra_writable`` paths.
    Network access is allowed only when ``network`` is true.
    """
    writable = [str(Path(workspace).expanduser()), "/tmp", "/private/tmp"]
    writable.extend(str(Path(path).expanduser()) for path in extra_writable)
    write_rule = " ".join(f"(subpath {_sbpl_string(path)})" for path in writable)
    network_rule = "(allow network*)" if network else "(deny network*)"
    return "\n".join(
        [
            "(version 1)",
            "(deny default)",
            "(allow process*)",
            "(allow file-read*)",
            f"(allow file-write* {write_rule})",
            "(allow sysctl-read)",
            network_rule,
        ]
    )


def seatbelt_command(
    command: str,
    *,
    workspace: str | Path,
    extra_writable: Sequence[str] = (),
    network: bool = True,
) -> list[str]:
    """Build ``sandbox-exec -p <profile> /bin/sh -c <command>``."""
    profile = seatbelt_profile(
        workspace=workspace, extra_writable=extra_writable, network=network
    )
    return ["sandbox-exec", "-p", profile, "/bin/sh", "-c", command]


def bwrap_command(
    command: str,
    *,
    workspace: str | Path,
    extra_writable: Sequence[str] = (),
    network: bool = True,
) -> list[str]:
    """Build a bubblewrap invocation with a read-only root and writable workspace.

    ``/`` is bind-mounted read-only, the workspace read-write, ``/tmp`` is a
    private tmpfs, and ``--unshare-net`` is added when ``network`` is false.
    """
    ws = str(Path(workspace).expanduser())
    argv = [
        "bwrap",
        "--ro-bind",
        "/",
        "/",
        "--bind",
        ws,
        ws,
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
    ]
    for path in extra_writable:
        resolved = str(Path(path).expanduser())
        argv.extend(["--bind", resolved, resolved])
    if not network:
        argv.append("--unshare-net")
    argv.extend(["/bin/sh", "-c", command])
    return argv


def wrap_command(
    command: str,
    *,
    workspace: str | Path,
    cfg: dict[str, Any] | None = None,
) -> tuple[list[str], str | None]:
    """Return the argv to execute plus an optional warning for the caller.

    Disabled settings return the plain ``["/bin/sh", "-c", command]`` and no
    warning. Enabled settings with no usable backend also return the plain
    argv, with a warning the caller should surface before running.
    """
    settings = get_sandbox(cfg)
    fallback = ["/bin/sh", "-c", command]
    if not settings["enabled"]:
        return fallback, None
    available = sandbox_available()
    if available is None:
        return fallback, FALLBACK_WARNING
    backend, path = available
    if backend == "seatbelt":
        argv = seatbelt_command(
            command,
            workspace=workspace,
            extra_writable=settings["extra_writable"],
            network=settings["network"],
        )
    else:
        argv = bwrap_command(
            command,
            workspace=workspace,
            extra_writable=settings["extra_writable"],
            network=settings["network"],
        )
    if path:
        argv[0] = path
    return argv, None
