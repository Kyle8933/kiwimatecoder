"""Turn-finished notifications: bell, OSC 9, and desktop fallbacks.

Every helper is best-effort by design: :func:`notify` never raises into the
caller and never blocks for more than :data:`DESKTOP_TIMEOUT` seconds. The
desktop mode first tries the OSC 9 terminal sequence when stderr is a TTY and
otherwise falls back to ``notify-send`` (Linux) or ``osascript`` (macOS) when
the tool is installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Any

NOTIFY_MODES: tuple[str, ...] = ("off", "bell", "desktop")

BELL = "\a"
OSC_PREFIX = "\x1b]9;"
OSC_SUFFIX = "\x07"
DESKTOP_TIMEOUT = 2.0


def should_notify(elapsed: float, threshold: float) -> bool:
    """Whether a turn that took ``elapsed`` seconds merits a notification.

    A threshold of zero notifies after every turn; a negative or malformed
    threshold disables notifications.
    """
    try:
        limit = float(threshold)
    except (TypeError, ValueError):
        return False
    return limit >= 0 and elapsed >= limit


def _stderr() -> Any:
    """Return the stream notifications write to (a test seam)."""
    return getattr(sys, "stderr", None)


def _write_stderr(text: str) -> bool:
    stream = _stderr()
    if stream is None:
        return False
    try:
        stream.write(text)
        stream.flush()
    except (OSError, ValueError):
        return False
    return True


def _is_tty(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def _bell() -> bool:
    return _write_stderr(BELL)


def _osc9(title: str, body: str) -> bool:
    if os.environ.get("TERM") == "dumb":
        return False
    stream = _stderr()
    if stream is None or not _is_tty(stream):
        return False
    return _write_stderr(f"{OSC_PREFIX}{title}: {body}{OSC_SUFFIX}")


def _applescript_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def desktop_command(
    title: str, body: str, platform: str | None = None
) -> list[str] | None:
    """Return the argv for a desktop notification, or None when unavailable.

    ``platform`` overrides ``sys.platform`` for testing.
    """
    system = (platform or sys.platform).lower()
    if system == "darwin":
        tool = shutil.which("osascript")
        if tool is None:
            return None
        script = (
            f'display notification "{_applescript_escape(body)}" '
            f'with title "{_applescript_escape(title)}"'
        )
        return [tool, "-e", script]
    tool = shutil.which("notify-send")
    if tool is None:
        return None
    return [tool, title, body]


def _desktop(title: str, body: str, platform: str | None = None) -> bool:
    if _osc9(title, body):
        return True
    command = desktop_command(title, body, platform)
    if command is None:
        return False
    try:
        subprocess.run(
            command,
            timeout=DESKTOP_TIMEOUT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def notify(
    title: str,
    body: str,
    *,
    mode: str = "off",
    platform: str | None = None,
) -> bool:
    """Deliver a notification according to ``mode``; never raises.

    Returns True when something was written or launched. ``platform`` is a
    test seam for the desktop fallback.
    """
    normalized = str(mode or "off").strip().lower()
    if normalized == "bell":
        return _bell()
    if normalized == "desktop":
        return _desktop(title, body, platform)
    return False
