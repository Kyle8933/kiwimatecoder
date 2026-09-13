"""Telemetry: disabled/enabled logging, rotation, crash reports, and config.

Every test uses a temp config dir and a mocked bus/hook so nothing touches the
real home directory and no events leak between tests.
"""

from __future__ import annotations

import io
import json
import sys

import pytest
from rich.console import Console

from kiwimatecoder import config, diagnostics, events, telemetry
from kiwimatecoder.commands import dispatch
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.session import Session


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    monkeypatch.delenv(telemetry.DEBUG_ENV, raising=False)
    telemetry.uninstall_crash_handler()
    yield
    telemetry.uninstall_crash_handler()
    telemetry.TELEMETRY.configure(
        {"telemetry": {"enabled": False, "level": "off"}}
    )


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def _session(tmp_path) -> Session:
    return Session(
        provider_id="openrouter",
        model="test-model",
        mode=PermissionMode.ASK,
        workspace_root=tmp_path,
    )


def _log_lines() -> list[dict[str, object]]:
    path = telemetry.current_log_path()
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_disabled_writes_nothing():
    telemetry.configure()
    telemetry.log_event("hello", value="x")
    telemetry.log_exception(ValueError("nope"))

    assert telemetry.enabled() is False
    assert not telemetry.current_log_path().exists()


def test_enabled_writes_redacted_json_lines():
    secret = "sk-" + "a" * 24
    telemetry.configure({"telemetry": {"enabled": True, "level": "info"}})

    telemetry.log_event("session", workspace="/tmp/ws", api_key=secret)

    lines = _log_lines()
    assert len(lines) == 1
    entry = lines[0]
    assert entry["event"] == "session"
    assert entry["level"] == "info"
    assert entry["workspace"] == "/tmp/ws"
    assert entry["api_key"] == "[REDACTED]"
    assert "timestamp" in entry
    assert secret not in telemetry.current_log_path().read_text(encoding="utf-8")


def test_level_filters_events():
    telemetry.configure({"telemetry": {"enabled": True, "level": "error"}})

    telemetry.log_event("info-event")
    telemetry.log_event("error-event", level="error")

    assert [line["event"] for line in _log_lines()] == ["error-event"]


def test_log_rotation_keeps_backup():
    telemetry.configure(
        {
            "telemetry": {
                "enabled": True,
                "level": "info",
                "max_log_bytes": 10_240,
            }
        }
    )

    telemetry.log_event("one", blob="a" * 6000)
    telemetry.log_event("two", blob="b" * 6000)

    path = telemetry.current_log_path()
    rotated = path.with_name(path.name + ".1")
    assert rotated.exists()
    assert "one" in rotated.read_text(encoding="utf-8")
    assert "two" in path.read_text(encoding="utf-8")


def test_custom_log_file_is_used(tmp_path):
    target = tmp_path / "logs" / "custom.log"
    telemetry.configure(
        {
            "telemetry": {
                "enabled": True,
                "level": "info",
                "log_file": str(target),
            }
        }
    )

    telemetry.log_event("hello")

    assert target.exists()
    assert "hello" in target.read_text(encoding="utf-8")


def test_log_exception_captures_traceback():
    telemetry.configure({"telemetry": {"enabled": True, "level": "error"}})

    try:
        raise ValueError("boom")
    except ValueError as exc:
        telemetry.log_exception(exc, context="unit-test")

    entry = _log_lines()[0]
    assert entry["event"] == "exception"
    assert entry["level"] == "error"
    assert entry["context"] == "unit-test"
    assert "ValueError: boom" in entry["error"]
    assert "Traceback" in entry["traceback"]


def test_event_bus_subscription_logs_lifecycle_events():
    bus = events.EventBus()
    telemetry.configure(
        {"telemetry": {"enabled": True, "level": "info"}}, bus=bus
    )

    bus.emit(events.PRE_TOOL, tool="read_file", args={"path": "a.py"})
    bus.emit(events.SESSION_START, workspace="/ws")
    bus.emit(events.POST_TOOL, tool="read_file", ok=True)

    lines = _log_lines()
    assert [line["event"] for line in lines] == [
        "pre_tool",
        "session_start",
        "post_tool",
    ]
    assert lines[0]["tool"] == "read_file"
    assert lines[2]["ok"] is True


def test_debug_env_forces_debug_level(monkeypatch):
    monkeypatch.setenv(telemetry.DEBUG_ENV, "1")

    telemetry.configure({"telemetry": {"enabled": False, "level": "off"}})

    assert telemetry.enabled() is True
    assert telemetry.TELEMETRY.level == "debug"
    telemetry.log_event("debug-event", level="debug")
    assert _log_lines()[0]["event"] == "debug-event"


# ---------------------------------------------------------------------------
# Crash reports
# ---------------------------------------------------------------------------


def test_crash_handler_writes_redacted_report_and_calls_previous(monkeypatch):
    telemetry.configure({"telemetry": {"enabled": True, "level": "error"}})
    telemetry.uninstall_crash_handler()

    calls: list[tuple[object, ...]] = []

    def previous(*args: object) -> None:
        calls.append(args)

    monkeypatch.setattr(sys, "excepthook", previous)
    telemetry.install_crash_handler()

    try:
        raise RuntimeError("leaked sk-" + "b" * 24)
    except RuntimeError as exc:
        sys.excepthook(type(exc), exc, exc.__traceback__)

    reports = telemetry.crash_report_paths()
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    payload = json.loads(text)
    assert "RuntimeError" in payload["exception"]
    assert "Traceback" in payload["traceback"]
    assert ("sk-" + "b" * 24) not in text
    assert "[REDACTED]" in text
    assert calls and calls[0][0] is RuntimeError


def test_crash_report_skipped_when_disabled(monkeypatch):
    telemetry.uninstall_crash_handler()
    telemetry.configure({"telemetry": {"enabled": False}})
    telemetry.install_crash_handler()

    try:
        raise RuntimeError("no report")
    except RuntimeError as exc:
        sys.excepthook(type(exc), exc, exc.__traceback__)

    assert telemetry.crash_report_paths() == []


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_telemetry_config_roundtrip_and_validation(tmp_path):
    log_path = tmp_path / "custom.log"
    settings = config.set_telemetry(
        enabled=True,
        level="debug",
        log_file=str(log_path),
        max_log_bytes=50_000,
    )

    assert settings["enabled"] is True
    assert settings["level"] == "debug"
    assert settings["log_file"] == str(log_path)
    assert settings["max_log_bytes"] == 50_000
    assert config.get_telemetry()["level"] == "debug"

    with pytest.raises(ValueError, match="level"):
        config.set_telemetry(level="verbose")
    with pytest.raises(ValueError, match="absolute"):
        config.set_telemetry(log_file="relative.log")
    with pytest.raises(ValueError, match="max_log_bytes"):
        config.set_telemetry(max_log_bytes=10)
    with pytest.raises(ValueError, match="true or false"):
        config.set_telemetry(enabled="yes")


def test_validate_config_flags_bad_telemetry():
    cfg = config._empty_config()
    cfg["telemetry"] = {
        "enabled": "yes",
        "level": "verbose",
        "log_file": "relative.log",
        "max_log_bytes": 1,
    }

    issues = config.validate_config(cfg)
    keys = {issue["key"] for issue in issues if issue["level"] == "error"}

    assert {
        "telemetry.enabled",
        "telemetry.level",
        "telemetry.log_file",
        "telemetry.max_log_bytes",
    } <= keys


def test_validate_config_accepts_telemetry_defaults():
    cfg = config._empty_config()
    assert config.validate_config(cfg) == []


# ---------------------------------------------------------------------------
# /config and doctor
# ---------------------------------------------------------------------------


def test_config_telemetry_slash_show_and_enable(tmp_path):
    console = _console()
    session = _session(tmp_path)

    dispatch("/config telemetry show", session, console)
    assert "Telemetry:" in console.file.getvalue()

    dispatch("/config telemetry enable on", session, console)
    assert config.get_telemetry()["enabled"] is True

    dispatch("/config telemetry level debug", session, console)
    assert config.get_telemetry()["level"] == "debug"

    output = console.file.getvalue()
    assert "Telemetry enabled" in output
    assert "debug" in output


def test_doctor_includes_telemetry_line(tmp_path):
    session = _session(tmp_path)

    checks = diagnostics.run_checks(session)

    names = [check.name for check in checks]
    assert "Telemetry" in names
    detail = next(check for check in checks if check.name == "Telemetry").detail
    assert "level" in detail
    assert str(telemetry.current_log_path()) in detail


# ---------------------------------------------------------------------------
# Client wiring
# ---------------------------------------------------------------------------


class _StreamResponse:
    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self._body = body

    async def __aenter__(self) -> "_StreamResponse":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def aread(self) -> bytes:
        return self._body

    async def aiter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
        yield "data: [DONE]"


class _StreamClient:
    responses: list[tuple[int, bytes]] = []
    calls: list[str] = []

    def __init__(self, **kwargs: object) -> None:
        pass

    async def __aenter__(self) -> "_StreamClient":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def stream(self, method: str, url: str, **kwargs: object) -> _StreamResponse:
        type(self).calls.append(f"{method} {url}")
        status, body = type(self).responses.pop(0)
        return _StreamResponse(status, body)


async def test_client_logs_retries_and_http_metadata(monkeypatch):
    from kiwimatecoder import client as client_module
    from kiwimatecoder.client import TextDelta, UnifiedClient
    from kiwimatecoder.providers import REGISTRY

    _StreamClient.responses = [(500, b"try later"), (200, b"ok")]
    _StreamClient.calls = []

    async def _noop_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(client_module.httpx, "AsyncClient", _StreamClient)
    monkeypatch.setattr(client_module.asyncio, "sleep", _noop_sleep)
    telemetry.configure({"telemetry": {"enabled": True, "level": "debug"}})

    events = [
        event
        async for event in UnifiedClient(
            REGISTRY["openai"], "sk-test"
        ).stream_chat([], None, "gpt-x")
    ]

    assert any(isinstance(event, TextDelta) for event in events)
    lines = _log_lines()
    retries = [line for line in lines if line["event"] == "provider_retry"]
    requests = [line for line in lines if line["event"] == "http_request"]
    assert retries and retries[0]["status"] == 500
    assert requests and requests[-1]["status"] == 200
    for entry in requests:
        assert entry["method"] == "POST"
        assert entry["host"] == "api.openai.com"
        assert "duration_ms" in entry
        # Metadata only: no headers or bodies are ever logged.
        assert "headers" not in entry
        assert "body" not in entry
