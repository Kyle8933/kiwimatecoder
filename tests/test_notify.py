from __future__ import annotations

import io

import pytest

from kiwimatecoder import notify


class FakeStderr(io.StringIO):
    def __init__(self, tty: bool = False) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@pytest.fixture
def fake_stderr(monkeypatch):
    stream = FakeStderr()
    monkeypatch.setattr(notify, "_stderr", lambda: stream)
    return stream


def test_should_notify_threshold():
    assert notify.should_notify(21, 20) is True
    assert notify.should_notify(20, 20) is True
    assert notify.should_notify(19.9, 20) is False
    assert notify.should_notify(0, 0) is True
    assert notify.should_notify(30, -1) is False
    assert notify.should_notify(30, "junk") is False


def test_notify_off_is_a_noop(fake_stderr):
    assert notify.notify("Kiwi", "done", mode="off") is False
    assert fake_stderr.getvalue() == ""


def test_notify_bell_writes_bel_byte(fake_stderr):
    assert notify.notify("Kiwi", "done", mode="bell") is True
    assert fake_stderr.getvalue() == "\a"


def test_notify_desktop_writes_osc9_on_a_tty(monkeypatch):
    stream = FakeStderr(tty=True)
    monkeypatch.setattr(notify, "_stderr", lambda: stream)

    assert notify.notify("Kiwi", "done", mode="desktop") is True
    assert stream.getvalue() == "\x1b]9;Kiwi: done\x07"


def test_notify_desktop_skips_osc9_when_term_is_dumb(monkeypatch):
    stream = FakeStderr(tty=True)
    monkeypatch.setattr(notify, "_stderr", lambda: stream)
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setattr(notify.shutil, "which", lambda tool: None)

    assert notify.notify("Kiwi", "done", mode="desktop") is False
    assert stream.getvalue() == ""


def test_notify_desktop_falls_back_to_osascript(monkeypatch, fake_stderr):
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(
        notify.shutil, "which", lambda tool: f"/usr/bin/{tool}" if tool == "osascript" else None
    )
    monkeypatch.setattr(notify.subprocess, "run", fake_run)

    assert notify.notify("Kiwi", 'say "hi"', mode="desktop", platform="darwin") is True
    command, kwargs = calls[0]
    assert command[0] == "/usr/bin/osascript"
    assert command[1] == "-e"
    assert "display notification" in command[2]
    assert '\\"hi\\"' in command[2]
    assert kwargs["timeout"] == notify.DESKTOP_TIMEOUT
    assert fake_stderr.getvalue() == ""


def test_notify_desktop_falls_back_to_notify_send(monkeypatch, fake_stderr):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(
        notify.shutil,
        "which",
        lambda tool: "/usr/bin/notify-send" if tool == "notify-send" else None,
    )
    monkeypatch.setattr(notify.subprocess, "run", fake_run)

    assert notify.notify("Kiwi", "done", mode="desktop", platform="linux") is True
    assert calls == [["/usr/bin/notify-send", "Kiwi", "done"]]


def test_notify_desktop_returns_false_without_a_tool(monkeypatch, fake_stderr):
    monkeypatch.setattr(notify.shutil, "which", lambda tool: None)

    assert notify.notify("Kiwi", "done", mode="desktop", platform="linux") is False


def test_notify_desktop_survives_a_failing_subprocess(monkeypatch, fake_stderr):
    def fake_run(command, **kwargs):
        raise OSError("no display")

    monkeypatch.setattr(notify.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(notify.subprocess, "run", fake_run)

    assert notify.notify("Kiwi", "done", mode="desktop", platform="darwin") is False
