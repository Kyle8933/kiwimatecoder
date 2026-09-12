from __future__ import annotations

import subprocess

import pytest

from kiwimatecoder import clipboard, tools
from kiwimatecoder.tools.io import (
    _read_clipboard,
    _write_clipboard,
    write_clipboard_preview,
)


def _install_which(present: set[str], monkeypatch) -> None:
    monkeypatch.setattr(
        clipboard.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in present else None,
    )


def _completed(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["tool"], returncode=returncode, stdout=stdout, stderr=stderr
    )


# ---------------------------------------------------------------------------
# clipboard_command
# ---------------------------------------------------------------------------


def test_macos_commands(monkeypatch):
    _install_which({"pbpaste", "pbcopy"}, monkeypatch)

    assert clipboard.clipboard_command("read", platform="darwin") == ["pbpaste"]
    assert clipboard.clipboard_command("write", platform="darwin") == ["pbcopy"]


def test_macos_missing_tool_returns_none(monkeypatch):
    _install_which(set(), monkeypatch)

    assert clipboard.clipboard_command("read", platform="darwin") is None
    assert clipboard.clipboard_command("write", platform="darwin") is None


def test_linux_prefers_wayland(monkeypatch):
    _install_which({"wl-paste", "wl-copy", "xclip"}, monkeypatch)

    assert clipboard.clipboard_command("read", platform="linux") == ["wl-paste"]
    assert clipboard.clipboard_command("write", platform="linux") == ["wl-copy"]


def test_linux_falls_back_to_xclip(monkeypatch):
    _install_which({"xclip"}, monkeypatch)

    assert clipboard.clipboard_command("read", platform="linux") == [
        "xclip",
        "-selection",
        "clipboard",
        "-o",
    ]
    assert clipboard.clipboard_command("write", platform="linux") == [
        "xclip",
        "-selection",
        "clipboard",
        "-i",
    ]


def test_linux_without_tools_returns_none(monkeypatch):
    _install_which(set(), monkeypatch)

    assert clipboard.clipboard_command("read", platform="linux") is None
    assert clipboard.clipboard_command("write", platform="linux") is None


def test_windows_commands(monkeypatch):
    _install_which({"powershell"}, monkeypatch)

    assert clipboard.clipboard_command("read", platform="win32") == [
        "/usr/bin/powershell",
        "-command",
        "Get-Clipboard",
    ]
    assert clipboard.clipboard_command("write", platform="win32") == [
        "/usr/bin/powershell",
        "-command",
        "Set-Clipboard",
    ]


def test_unsupported_platform_returns_none(monkeypatch):
    _install_which({"pbpaste", "pbcopy"}, monkeypatch)

    assert clipboard.clipboard_command("read", platform="freebsd") is None


def test_invalid_action_raises():
    with pytest.raises(ValueError, match="read.*write"):
        clipboard.clipboard_command("copy")


# ---------------------------------------------------------------------------
# read_clipboard / write_clipboard
# ---------------------------------------------------------------------------


def test_read_clipboard_success(monkeypatch):
    monkeypatch.setattr(clipboard, "clipboard_command", lambda action: ["/bin/pbpaste"])
    calls: list[dict] = []

    def fake_run(args, **kwargs):
        calls.append(kwargs)
        return _completed(stdout="clip text")

    monkeypatch.setattr(clipboard.subprocess, "run", fake_run)

    assert clipboard.read_clipboard() == (True, "clip text")
    assert calls[0]["capture_output"] is True
    assert calls[0]["text"] is True
    assert calls[0].get("shell") is not True


def test_read_clipboard_missing_tool(monkeypatch):
    monkeypatch.setattr(clipboard, "clipboard_command", lambda action: None)

    ok, message = clipboard.read_clipboard()

    assert ok is False
    assert "No clipboard tool found" in message


def test_read_clipboard_nonzero_exit(monkeypatch):
    monkeypatch.setattr(clipboard, "clipboard_command", lambda action: ["/bin/xclip"])
    monkeypatch.setattr(
        clipboard.subprocess,
        "run",
        lambda args, **kwargs: _completed(returncode=1, stderr="no display"),
    )

    ok, message = clipboard.read_clipboard()

    assert ok is False
    assert "exited with code 1" in message
    assert "no display" in message


def test_read_clipboard_never_raises(monkeypatch):
    monkeypatch.setattr(clipboard, "clipboard_command", lambda action: ["/bin/xclip"])

    def boom(args, **kwargs):
        raise OSError("denied")

    monkeypatch.setattr(clipboard.subprocess, "run", boom)

    ok, message = clipboard.read_clipboard()

    assert ok is False
    assert "failed" in message


def test_read_clipboard_timeout_is_reported(monkeypatch):
    monkeypatch.setattr(clipboard, "clipboard_command", lambda action: ["/bin/xclip"])

    def timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(args, clipboard.TIMEOUT_SECONDS)

    monkeypatch.setattr(clipboard.subprocess, "run", timeout)

    ok, message = clipboard.read_clipboard()

    assert ok is False
    assert "timed out" in message


def test_write_clipboard_success_pipes_text(monkeypatch):
    monkeypatch.setattr(clipboard, "clipboard_command", lambda action: ["/bin/pbcopy"])
    captured: dict = {}

    def fake_run(args, **kwargs):
        captured.update(kwargs)
        return _completed()

    monkeypatch.setattr(clipboard.subprocess, "run", fake_run)

    ok, message = clipboard.write_clipboard("hello world")

    assert ok is True
    assert "11 character(s)" in message
    assert captured["input"] == "hello world"
    assert captured.get("shell") is not True


def test_write_clipboard_missing_tool(monkeypatch):
    monkeypatch.setattr(clipboard, "clipboard_command", lambda action: None)

    ok, message = clipboard.write_clipboard("hello")

    assert ok is False
    assert "No clipboard tool found" in message


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_clipboard_tools_registered_and_approval_flags():
    read_tool = tools.get_tool("read_clipboard")
    write_tool = tools.get_tool("write_clipboard")
    assert read_tool is not None and write_tool is not None
    assert read_tool.writes is False and read_tool.runs is False
    assert read_tool.needs_approval is False
    assert write_tool.runs is True
    assert write_tool.needs_approval is True

    read_only_names = {
        schema["function"]["name"] for schema in tools.tool_schemas(read_only=True)
    }
    assert "read_clipboard" in read_only_names
    assert "write_clipboard" not in read_only_names


def test_read_clipboard_tool_returns_text(session, monkeypatch):
    monkeypatch.setattr(
        clipboard, "read_clipboard", lambda: (True, "from clipboard")
    )

    result = _read_clipboard({}, session)

    assert result.ok
    assert result.content == "from clipboard"


def test_read_clipboard_tool_empty(session, monkeypatch):
    monkeypatch.setattr(clipboard, "read_clipboard", lambda: (True, ""))

    result = _read_clipboard({}, session)

    assert result.ok
    assert result.content == "(clipboard is empty)"


def test_read_clipboard_tool_error(session, monkeypatch):
    monkeypatch.setattr(clipboard, "read_clipboard", lambda: (False, "no tool"))

    result = _read_clipboard({}, session)

    assert not result.ok
    assert result.content == "Error: no tool"


def test_write_clipboard_tool_requires_text(session):
    result = _write_clipboard({}, session)

    assert not result.ok
    assert "'text' is required" in result.content


def test_write_clipboard_tool_success(session, monkeypatch):
    captured: list[str] = []

    def fake_write(text):
        captured.append(text)
        return True, f"Copied {len(text)} character(s) to the clipboard."

    monkeypatch.setattr(clipboard, "write_clipboard", fake_write)

    result = _write_clipboard({"text": "hello"}, session)

    assert result.ok
    assert captured == ["hello"]
    assert "Copied 5" in result.content


def test_write_clipboard_preview_redacts_and_clips(session):
    secret = "sk-or-abcdefghijklmnop"
    preview = write_clipboard_preview({"text": f"token {secret}"}, session)

    assert preview.startswith("Copy to clipboard:\n")
    assert secret not in preview
    assert "[REDACTED]" in preview

    long = write_clipboard_preview({"text": "x" * 500}, session)
    assert long.endswith("…")
    assert len(long) < 260

    assert "(empty)" in write_clipboard_preview({"text": ""}, session)
