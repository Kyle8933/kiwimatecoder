"""Clipboard helpers: pick a platform tool and read/write text safely.

The helpers never raise from :func:`read_clipboard`/:func:`write_clipboard`;
they return ``(ok, text_or_error)`` so tools can surface a friendly message.
Commands run without a shell, with a short timeout, and are selected from the
platform's standard tools (``pbcopy``/``pbpaste`` on macOS, ``wl-clipboard`` or
``xclip`` on Linux, PowerShell on Windows).
"""

from __future__ import annotations

import shutil
import subprocess
import sys

TIMEOUT_SECONDS = 10.0

# (executable, argv) candidates per action, in preference order.
_LINUX_READ = (
    ("wl-paste", ["wl-paste"]),
    ("xclip", ["xclip", "-selection", "clipboard", "-o"]),
)
_LINUX_WRITE = (
    ("wl-copy", ["wl-copy"]),
    ("xclip", ["xclip", "-selection", "clipboard", "-i"]),
)


def clipboard_command(action: str, platform: str | None = None) -> list[str] | None:
    """Return the argv that reads or writes the clipboard, or None if absent.

    ``action`` is ``"read"`` or ``"write"``. ``platform`` overrides
    ``sys.platform`` for testing.
    """
    if action not in {"read", "write"}:
        raise ValueError("Clipboard action must be 'read' or 'write'.")
    system = (platform or sys.platform).lower()
    if system == "darwin":
        tool = "pbpaste" if action == "read" else "pbcopy"
        return [tool] if shutil.which(tool) else None
    if system.startswith("linux"):
        for candidate, argv in _LINUX_READ if action == "read" else _LINUX_WRITE:
            if shutil.which(candidate):
                return list(argv)
        return None
    if system.startswith("win"):
        command = "Get-Clipboard" if action == "read" else "Set-Clipboard"
        executable = shutil.which("powershell") or shutil.which("pwsh")
        return [executable, "-command", command] if executable else None
    return None


def _missing_message() -> str:
    return (
        "No clipboard tool found. Install one: pbpaste/pbcopy (macOS), "
        "wl-clipboard or xclip (Linux), or use PowerShell (Windows)."
    )


def _run(args: list[str], *, input_text: str | None = None) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            args,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        return False, f"Clipboard tool '{args[0]}' is not installed."
    except subprocess.TimeoutExpired:
        return False, f"Clipboard tool '{args[0]}' timed out after {TIMEOUT_SECONDS:g}s."
    except OSError as exc:
        return False, f"Clipboard tool '{args[0]}' failed: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip()
        message = f"Clipboard tool '{args[0]}' exited with code {proc.returncode}"
        return False, f"{message}: {detail}" if detail else f"{message}."
    return True, proc.stdout or ""


def read_clipboard() -> tuple[bool, str]:
    """Read the clipboard. Returns ``(ok, text_or_error_message)``."""
    command = clipboard_command("read")
    if command is None:
        return False, _missing_message()
    return _run(command)


def write_clipboard(text: str) -> tuple[bool, str]:
    """Write ``text`` to the clipboard. Never raises."""
    command = clipboard_command("write")
    if command is None:
        return False, _missing_message()
    ok, output = _run(command, input_text=str(text))
    if not ok:
        return False, output
    return True, f"Copied {len(str(text))} character(s) to the clipboard."
