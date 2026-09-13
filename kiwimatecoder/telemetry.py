"""Opt-in local telemetry, debug logging, and crash reports.

Nothing leaves the machine: events are appended as one compact JSON object per
line to a user-owned log (``~/.kiwimatecoder/logs/kiwimatecoder.log`` by
default) with secret-looking strings redacted. Telemetry is **off by default**
and does nothing until ``telemetry.enabled`` is true and a level other than
``off`` is configured. The log rotates to ``<log_file>.1`` once it would grow
past ``telemetry.max_log_bytes``.

``KIWIMATECODER_DEBUG=1`` (or ``true``/``yes``/``on``) forces the ``debug``
level for one run without changing config, which is handy when diagnosing a
stubborn provider or network issue.

Crash reports are opt-in too: :func:`install_crash_handler` (called by
:func:`configure` while telemetry is enabled) wraps ``sys.excepthook`` and
``threading.excepthook`` to write a redacted ``crash-<ts>.json`` bundle before
delegating to the previous hook.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

from kiwimatecoder import __version__, config, events
from kiwimatecoder.redaction import redact

DEBUG_ENV = "KIWIMATECODER_DEBUG"
LOG_DIR_NAME = "logs"
LOG_FILE_NAME = "kiwimatecoder.log"
CRASH_PREFIX = "crash-"
MAX_RECENT_LINES = 50
_RECENT_MAX_BYTES = 64 * 1024
_LEVEL_RANK = {"off": 0, "error": 1, "info": 2, "debug": 3}
_DEBUG_TRUTHY = {"1", "true", "yes", "on"}


def log_dir() -> Path:
    """Return the telemetry log directory (resolved lazily for tests)."""
    return config.CONFIG_DIR / LOG_DIR_NAME


def default_log_file() -> Path:
    """Return the default telemetry log path."""
    return log_dir() / LOG_FILE_NAME


def _debug_env_enabled() -> bool:
    return os.environ.get(DEBUG_ENV, "").strip().lower() in _DEBUG_TRUTHY


def _chmod(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _chmod(path, 0o700)


def _write_private_text(path: Path, text: str) -> None:
    _ensure_private_dir(path.parent)
    tmp = path.with_name(path.name + ".kiwi.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    _chmod(path, 0o600)


def _redact_value(value: Any) -> Any:
    """Recursively redact secret-looking strings in a JSON-friendly value."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {str(key): _redact_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    return value


class Telemetry:
    """Local JSON-line telemetry with rotation and lifecycle subscriptions."""

    def __init__(self) -> None:
        self._enabled = False
        self._level = "off"
        self._log_file: Path | None = None
        self._max_bytes = int(config.TELEMETRY_DEFAULTS["max_log_bytes"])
        self._unsubscribers: list[Callable[[], None]] = []

    @property
    def enabled(self) -> bool:
        """Whether telemetry is currently recording."""
        return self._enabled

    @property
    def level(self) -> str:
        """The effective level: off, error, info, or debug."""
        return self._level

    @property
    def log_file(self) -> Path:
        """The active log path (the default when unset)."""
        return self._log_file or default_log_file()

    def configure(
        self,
        cfg: dict[str, Any] | None = None,
        *,
        bus: events.EventBus | None = None,
    ) -> None:
        """Apply telemetry config, (re)subscribe, and arm crash reports.

        ``KIWIMATECODER_DEBUG`` forces ``enabled``/``debug`` for this run.
        Never raises: a broken config leaves telemetry off.
        """
        try:
            settings = config.get_telemetry(cfg)
        except Exception:
            settings = dict(config.TELEMETRY_DEFAULTS)
        enabled = bool(settings["enabled"])
        level = str(settings["level"])
        if _debug_env_enabled():
            enabled = True
            level = "debug"
        self._enabled = enabled
        self._level = level if enabled else "off"
        log_file = str(settings["log_file"] or "").strip()
        self._log_file = Path(log_file).expanduser() if log_file else default_log_file()
        self._max_bytes = int(settings["max_log_bytes"])
        self._detach()
        if self._enabled and self._level != "off":
            target_bus = bus if bus is not None else events.BUS
            for name in events.EVENT_NAMES:
                self._unsubscribers.append(
                    target_bus.subscribe(name, self._on_event)
                )
            install_crash_handler()
        else:
            uninstall_crash_handler()

    def _detach(self) -> None:
        for unsubscribe in self._unsubscribers:
            try:
                unsubscribe()
            except Exception:
                pass
        self._unsubscribers.clear()

    def _on_event(self, event: events.Event) -> None:
        try:
            self.log_event(event.name, **dict(event.payload))
        except Exception:
            pass

    def log_event(self, name: str, **fields: Any) -> None:
        """Append one JSON line when the current level admits the event.

        Pass ``level="error"``/``"debug"`` to filter by severity; the default
        is ``info``. Every string field is redacted. Never raises.
        """
        level = str(fields.pop("level", "info")).strip().lower() or "info"
        if not self._enabled or self._level == "off":
            return
        if _LEVEL_RANK.get(level, 2) > _LEVEL_RANK.get(self._level, 0):
            return
        try:
            entry: dict[str, Any] = {
                "timestamp": datetime.datetime.now().isoformat(),
                "level": level,
                "event": redact(str(name)),
            }
            for key, value in fields.items():
                entry[str(key)] = _redact_value(value)
            line = json.dumps(
                entry, ensure_ascii=False, separators=(",", ":"), default=str
            )
            self._append_line(line)
        except Exception:
            pass

    def log_exception(
        self, exc: BaseException, context: str | None = None
    ) -> None:
        """Log one exception with a redacted traceback at the error level."""
        try:
            tb = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            self.log_event(
                "exception",
                level="error",
                error=redact(f"{type(exc).__name__}: {exc}"),
                context=context or "",
                traceback=redact(tb),
            )
        except Exception:
            pass

    def _append_line(self, line: str) -> None:
        try:
            path = self.log_file
            _ensure_private_dir(path.parent)
            payload = (line + "\n").encode("utf-8", "replace")
            if path.exists() and path.stat().st_size + len(payload) > self._max_bytes:
                rotated = path.with_name(path.name + ".1")
                try:
                    os.replace(path, rotated)
                except OSError:
                    try:
                        path.unlink()
                    except OSError:
                        pass
                _chmod(rotated, 0o600)
            with path.open("ab") as handle:
                handle.write(payload)
            _chmod(path, 0o600)
        except OSError:
            pass


TELEMETRY = Telemetry()


def configure(
    cfg: dict[str, Any] | None = None, *, bus: events.EventBus | None = None
) -> None:
    """Configure the default telemetry manager."""
    TELEMETRY.configure(cfg, bus=bus)


def enabled() -> bool:
    """Whether the default manager is recording."""
    return TELEMETRY.enabled


def log_event(name: str, **fields: Any) -> None:
    """Log one event on the default manager."""
    TELEMETRY.log_event(name, **fields)


def log_exception(exc: BaseException, context: str | None = None) -> None:
    """Log one exception on the default manager."""
    TELEMETRY.log_exception(exc, context)


def current_log_path() -> Path:
    """Return the default manager's active log path."""
    return TELEMETRY.log_file


def _recent_lines(limit: int = MAX_RECENT_LINES) -> list[str]:
    """Return the tail of the log, bounded by bytes and lines."""
    try:
        path = TELEMETRY.log_file
        if not path.is_file():
            return []
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - _RECENT_MAX_BYTES))
            data = handle.read()
    except OSError:
        return []
    return data.decode("utf-8", "replace").splitlines()[-limit:]


def _unique_crash_path(now: datetime.datetime) -> Path:
    base = log_dir() / f"{CRASH_PREFIX}{now.strftime('%Y%m%d_%H%M%S')}.json"
    if not base.exists():
        return base
    counter = 1
    while True:
        candidate = base.with_name(f"{base.stem}-{counter}.json")
        if not candidate.exists():
            return candidate
        counter += 1


def _write_crash_report(exc: BaseException | None) -> Path | None:
    """Write a redacted crash bundle; returns its path (None when skipped)."""
    try:
        if not TELEMETRY.enabled or TELEMETRY.level == "off":
            return None
        now = datetime.datetime.now()
        if isinstance(exc, BaseException):
            trace = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            error = f"{type(exc).__name__}: {exc}"
        else:
            trace = ""
            error = type(exc).__name__ if exc is not None else "unknown"
        payload: dict[str, Any] = {
            "timestamp": now.isoformat(),
            "version": __version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "exception": redact(error),
            "traceback": redact(trace),
            "recent": [redact(line) for line in _recent_lines()],
        }
        data = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        path = _unique_crash_path(now)
        _write_private_text(path, data + "\n")
        return path
    except Exception:
        return None


def crash_report_paths() -> list[Path]:
    """Return the recorded crash bundles, oldest first."""
    try:
        return sorted(log_dir().glob(f"{CRASH_PREFIX}*.json"))
    except OSError:
        return []


_lock = threading.Lock()
_installed_sys_hook: Any = None
_installed_thread_hook: Any = None
_previous_sys_hook: Any = None
_previous_thread_hook: Any = None


def install_crash_handler() -> None:
    """Wrap the excepthooks to write a crash bundle, then call the originals.

    Idempotent; while telemetry is disabled the hooks are not installed (and
    :func:`uninstall_crash_handler` restores whatever was there before).
    """
    global _installed_sys_hook, _installed_thread_hook
    global _previous_sys_hook, _previous_thread_hook
    with _lock:
        if _installed_sys_hook is not None:
            return
        _previous_sys_hook = sys.excepthook
        _previous_thread_hook = threading.excepthook

        def _sys_hook(
            exc_type: type[BaseException],
            exc_value: BaseException | None,
            tb: Any,
        ) -> None:
            try:
                _write_crash_report(exc_value)
            except Exception:
                pass
            _previous_sys_hook(exc_type, exc_value, tb)

        def _thread_hook(args: threading.ExceptHookArgs) -> None:
            try:
                _write_crash_report(args.exc_value)
            except Exception:
                pass
            _previous_thread_hook(args)

        sys.excepthook = _sys_hook
        threading.excepthook = _thread_hook
        _installed_sys_hook = _sys_hook
        _installed_thread_hook = _thread_hook


def uninstall_crash_handler() -> None:
    """Restore the excepthooks captured by :func:`install_crash_handler`."""
    global _installed_sys_hook, _installed_thread_hook
    with _lock:
        if _installed_sys_hook is not None and sys.excepthook is _installed_sys_hook:
            sys.excepthook = _previous_sys_hook
        if (
            _installed_thread_hook is not None
            and threading.excepthook is _installed_thread_hook
        ):
            threading.excepthook = _previous_thread_hook
        _installed_sys_hook = None
        _installed_thread_hook = None
