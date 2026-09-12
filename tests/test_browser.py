"""Optional browser automation: driver helpers, the browser tool, and config.

Every test is hermetic: a fake :class:`~kiwimatecoder.browser.Driver` is injected
by monkeypatching :func:`kiwimatecoder.browser.get_driver`, and no test imports
Playwright or opens a network connection.
"""

from __future__ import annotations

import io
import re
import struct
import sys
import zlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from kiwimatecoder import browser as browser_module
from kiwimatecoder import config, tools
from kiwimatecoder.agent import Agent
from kiwimatecoder.commands import dispatch, slash_argument_completions
from kiwimatecoder.main import app as cli_app
from kiwimatecoder.tools.browser import (
    DISABLED_MESSAGE,
    NO_PAGE_MESSAGE,
    _browser,
    preview,
)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


# A minimal, valid 1x1 RGBA PNG assembled with the standard library.
PNG_1PX = (
    b"\x89PNG\r\n\x1a\n"
    + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
    + _png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00\x00"))
    + _png_chunk(b"IEND", b"")
)


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    home = tmp_path / "config-home"
    monkeypatch.setattr(config, "CONFIG_DIR", home)
    monkeypatch.setattr(config, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", home / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    monkeypatch.setattr(browser_module, "_DRIVER", None)


@pytest.fixture
def enabled():
    config.set_browser(enabled=True)


class FakeDriver:
    """Records every call so tests can assert argument passing."""

    def __init__(self, text: str = "hello page") -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.text_value = text
        self.open_result = ("https://example.com/final", "Example Title")
        self.evaluate_result: Any = {"ok": True}
        self.timeouts: list[int] = []
        self.closed = False

    def open(self, url: str) -> tuple[str, str]:
        self.calls.append(("open", url))
        return self.open_result

    def text(self) -> str:
        self.calls.append(("text",))
        return self.text_value

    def screenshot(self, path) -> None:
        self.calls.append(("screenshot", str(path)))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PNG_1PX)

    def click(self, selector: str) -> None:
        self.calls.append(("click", selector))

    def type(self, selector: str, text: str) -> None:
        self.calls.append(("type", selector, text))

    def evaluate(self, script: str) -> Any:
        self.calls.append(("evaluate", script))
        return self.evaluate_result

    def set_timeout(self, timeout_ms: int) -> None:
        self.timeouts.append(timeout_ms)

    def close(self) -> None:
        self.calls.append(("close",))
        self.closed = True


@pytest.fixture
def fake_driver(monkeypatch):
    driver = FakeDriver()
    monkeypatch.setattr(browser_module, "get_driver", lambda *a, **k: driver)
    monkeypatch.setattr(browser_module, "current_driver", lambda: driver)
    return driver


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120)


# ---------------------------------------------------------------------------
# Driver module helpers
# ---------------------------------------------------------------------------


def test_screenshot_path_creates_workspace_dir(tmp_path):
    path = browser_module.screenshot_path(tmp_path)

    assert path.parent == tmp_path / ".kiwimatecoder" / "screenshots"
    assert path.parent.is_dir()
    assert re.fullmatch(r"\d{8}_\d{6}\.png", path.name)


def test_reset_driver_closes_and_clears_singleton(monkeypatch):
    driver = FakeDriver()
    monkeypatch.setattr(browser_module, "_DRIVER", driver)

    browser_module.reset_driver()

    assert driver.closed is True
    assert browser_module.current_driver() is None


def test_reset_driver_without_driver_is_noop():
    browser_module.reset_driver()
    assert browser_module.current_driver() is None


def test_get_driver_without_playwright_raises_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "playwright", None)

    with pytest.raises(browser_module.BrowserError) as excinfo:
        browser_module.get_driver(config.get_browser())

    message = str(excinfo.value)
    assert "Playwright is not installed" in message
    assert "pip install 'kiwimatecoder[browser]'" in message
    assert browser_module.current_driver() is None


# ---------------------------------------------------------------------------
# Tool registration and discovery
# ---------------------------------------------------------------------------


def test_browser_tool_registered_and_approval_gated():
    tool = tools.get_tool("browser")

    assert tool is not None
    assert tool.writes is False
    assert tool.runs is True
    assert tool.needs_approval is True
    schema = tool.schema()["function"]
    assert schema["parameters"]["required"] == ["action"]
    assert schema["parameters"]["properties"]["action"]["enum"] == [
        "open",
        "text",
        "screenshot",
        "click",
        "type",
        "evaluate",
        "close",
    ]


def test_browser_preview_registered():
    assert tools.preview("browser", {"action": "open"}, MagicMock()) is not None


def test_agent_hides_browser_schema_until_enabled(session):
    agent = Agent(session, Console(quiet=True), MagicMock())

    names = {schema["function"]["name"] for schema in agent._tool_schemas()}
    assert "browser" not in names

    config.set_browser(enabled=True)
    names = {schema["function"]["name"] for schema in agent._tool_schemas()}
    assert "browser" in names


def test_agent_summary_for_browser(session):
    agent = Agent(session, Console(quiet=True), MagicMock())

    assert (
        agent._format_call_summary(
            "browser", {"action": "open", "url": "https://example.com/docs"}
        )
        == "browser [dim]open https://example.com/docs[/dim]"
    )
    assert (
        agent._format_call_summary("browser", {"action": "close"})
        == "browser [dim]close[/dim]"
    )


# ---------------------------------------------------------------------------
# Tool behavior
# ---------------------------------------------------------------------------


def test_disabled_browser_returns_clear_error(session):
    result = _browser({"action": "open", "url": "https://example.com"}, session)

    assert not result.ok
    assert result.content == f"Error: {DISABLED_MESSAGE}"


def test_open_returns_final_url_and_title(session, enabled, fake_driver):
    result = _browser(
        {"action": "open", "url": "https://example.com/start"}, session
    )

    assert result.ok
    assert "https://example.com/final" in result.content
    assert "Example Title" in result.content
    assert fake_driver.calls == [("open", "https://example.com/start")]


def test_open_requires_url(session, enabled, fake_driver):
    result = _browser({"action": "open"}, session)

    assert not result.ok
    assert "'url' is required" in result.content
    assert fake_driver.calls == []


def test_open_rejects_non_http_scheme(session, enabled, fake_driver):
    result = _browser({"action": "open", "url": "ftp://example.com/file"}, session)

    assert not result.ok
    assert "http" in result.content
    assert fake_driver.calls == []


def test_open_blocks_local_hosts_by_default(session, enabled, fake_driver):
    result = _browser({"action": "open", "url": "http://localhost:8080/"}, session)

    assert not result.ok
    assert "local or private" in result.content
    assert fake_driver.calls == []


def test_open_allows_local_when_web_config_enables_it(session, enabled, fake_driver):
    config.set_web(allow_local=True)

    result = _browser({"action": "open", "url": "http://127.0.0.1:8080/"}, session)

    assert result.ok
    assert fake_driver.calls == [("open", "http://127.0.0.1:8080/")]


def test_actions_before_open_return_friendly_error(session, enabled):
    for action in ("text", "screenshot", "click", "type", "evaluate"):
        result = _browser({"action": action}, session)
        assert not result.ok
        assert result.content == f"Error: {NO_PAGE_MESSAGE}"


def test_text_is_returned_verbatim(session, enabled, fake_driver):
    fake_driver.text_value = "visible body"

    result = _browser({"action": "text"}, session)

    assert result.ok
    assert result.content == "visible body"


def test_text_is_truncated_with_note(session, enabled, fake_driver):
    fake_driver.text_value = "x" * 40_000

    result = _browser({"action": "text"}, session)

    assert result.ok
    assert result.content.count("x") == 30_000
    assert "[truncated at 30000 characters]" in result.content


def test_empty_page_text_has_placeholder(session, enabled, fake_driver):
    fake_driver.text_value = ""

    result = _browser({"action": "text"}, session)

    assert result.ok
    assert result.content == "(no visible text)"


def test_screenshot_saves_and_attaches_to_pending_images(
    session, enabled, fake_driver
):
    result = _browser({"action": "screenshot"}, session)

    assert result.ok
    assert "Saved screenshot to .kiwimatecoder/screenshots/" in result.content
    assert len(fake_driver.calls) == 1
    kind, saved = fake_driver.calls[0]
    assert kind == "screenshot"
    assert saved.endswith(".png")
    saved_path = Path(saved)
    assert saved_path.is_file()

    assert len(session.pending_images) == 1
    entry = session.pending_images[0]
    assert entry["media_type"] == "image/png"
    assert entry["name"] == saved_path.name
    assert entry["data"]


def test_screenshot_respects_per_turn_image_limit(session, enabled, fake_driver):
    session.pending_images = [{"media_type": "image/png", "data": "", "name": "x"}] * 4

    result = _browser({"action": "screenshot"}, session)

    assert result.ok
    assert "not attached" in result.content
    assert len(session.pending_images) == 4


def test_click_passes_selector(session, enabled, fake_driver):
    result = _browser({"action": "click", "selector": "#submit"}, session)

    assert result.ok
    assert fake_driver.calls == [("click", "#submit")]


def test_click_requires_selector(session, enabled, fake_driver):
    result = _browser({"action": "click"}, session)

    assert not result.ok
    assert "'selector' is required" in result.content
    assert fake_driver.calls == []


def test_type_passes_selector_and_text(session, enabled, fake_driver):
    result = _browser(
        {"action": "type", "selector": "input[name=q]", "text": "kiwi"}, session
    )

    assert result.ok
    assert fake_driver.calls == [("type", "input[name=q]", "kiwi")]


def test_type_requires_selector_and_text(session, enabled, fake_driver):
    missing_selector = _browser({"action": "type", "text": "kiwi"}, session)
    assert not missing_selector.ok
    assert "'selector' is required" in missing_selector.content

    missing_text = _browser({"action": "type", "selector": "#q"}, session)
    assert not missing_text.ok
    assert "'text' is required" in missing_text.content
    assert fake_driver.calls == []


def test_evaluate_passes_script_and_formats_result(session, enabled, fake_driver):
    result = _browser(
        {"action": "evaluate", "script": "document.title"}, session
    )

    assert result.ok
    assert fake_driver.calls == [("evaluate", "document.title")]
    assert '"ok": true' in result.content


def test_evaluate_requires_script(session, enabled, fake_driver):
    result = _browser({"action": "evaluate"}, session)

    assert not result.ok
    assert "'script' is required" in result.content
    assert fake_driver.calls == []


def test_evaluate_result_is_capped(session, enabled, fake_driver):
    fake_driver.evaluate_result = "y" * 20_000

    result = _browser({"action": "evaluate", "script": "1"}, session)

    assert result.ok
    assert result.content.count("y") == 10_000
    assert "[truncated at 10000 characters]" in result.content


def test_per_action_timeout_override(session, enabled, fake_driver):
    result = _browser(
        {"action": "click", "selector": "#go", "timeout_ms": 5_000}, session
    )

    assert result.ok
    assert fake_driver.timeouts == [5_000]


@pytest.mark.parametrize("value", [10, 200_000, "soon"])
def test_invalid_timeout_override(session, enabled, fake_driver, value):
    result = _browser({"action": "text", "timeout_ms": value}, session)

    assert not result.ok
    assert "timeout_ms" in result.content
    assert fake_driver.calls == []


def test_unknown_action_is_rejected(session, enabled, fake_driver):
    result = _browser({"action": "fly"}, session)

    assert not result.ok
    assert "Unknown browser action" in result.content
    assert fake_driver.calls == []


def test_close_resets_driver(session, enabled, monkeypatch):
    calls: list[bool] = []
    monkeypatch.setattr(browser_module, "reset_driver", lambda: calls.append(True))

    result = _browser({"action": "close"}, session)

    assert result.ok
    assert result.content == "Browser closed."
    assert calls == [True]


def test_driver_errors_are_wrapped_friendly(session, enabled, monkeypatch):
    def boom(*args, **kwargs):
        raise browser_module.BrowserError("Browser action failed: selector missing")

    driver = FakeDriver()
    monkeypatch.setattr(driver, "click", boom)
    monkeypatch.setattr(browser_module, "current_driver", lambda: driver)

    result = _browser({"action": "click", "selector": "#nope"}, session)

    assert not result.ok
    assert "Browser action failed" in result.content


# ---------------------------------------------------------------------------
# Approval preview
# ---------------------------------------------------------------------------


def test_preview_shows_action_and_target(session):
    text = preview({"action": "open", "url": "https://example.com/docs"}, session)

    assert text == "browser open https://example.com/docs"


def test_preview_redacts_and_caps_scripts(session):
    secret = "sk-" + "a" * 40
    script = f"fetch('https://api.example.com?key={secret}')"

    text = preview({"action": "evaluate", "script": script}, session)

    assert secret not in text
    assert "[REDACTED]" in text
    assert text.startswith("browser evaluate ")
    assert len(text) <= len("browser evaluate ") + 80


def test_preview_without_target(session):
    assert preview({"action": "close"}, session) == "browser close"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_get_browser_defaults():
    assert config.get_browser() == {
        "enabled": False,
        "headless": True,
        "timeout_ms": 15000,
    }


def test_set_browser_roundtrip():
    config.set_browser(enabled=True, headless=False, timeout_ms=30_000)

    stored = config.load_config()["browser"]
    assert stored == {"enabled": True, "headless": False, "timeout_ms": 30_000}
    assert config.get_browser()["timeout_ms"] == 30_000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": "yes"},
        {"headless": 1},
        {"timeout_ms": 10},
        {"timeout_ms": 999_999},
        {"timeout_ms": "soon"},
    ],
)
def test_set_browser_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        config.set_browser(**kwargs)


def test_get_browser_normalizes_bad_stored_values():
    config.save_config(
        {
            "browser": {
                "enabled": "yes",
                "headless": "no",
                "timeout_ms": 5,
            }
        }
    )

    assert config.get_browser() == {
        "enabled": False,
        "headless": True,
        "timeout_ms": 15000,
    }


def test_validate_config_accepts_clean_browser_section():
    config.set_browser(enabled=True, headless=False, timeout_ms=20_000)

    issues = [
        issue for issue in config.validate_config() if issue["key"].startswith("browser")
    ]
    assert issues == []


def test_validate_config_catches_bad_browser_section():
    config.save_config(
        {
            "browser": {
                "enabled": "yes",
                "headless": "no",
                "timeout_ms": -1,
            }
        }
    )

    keys = {issue["key"] for issue in config.validate_config()}
    assert {"browser.enabled", "browser.headless", "browser.timeout_ms"} <= keys


# ---------------------------------------------------------------------------
# Slash command and CLI
# ---------------------------------------------------------------------------


def test_config_browser_slash_roundtrip(session):
    console = _console()
    dispatch("/config browser show", session, console)
    assert "off" in console.file.getvalue()

    dispatch("/config browser enable on", session, _console())
    assert config.get_browser()["enabled"] is True

    dispatch("/config browser headless off", session, _console())
    assert config.get_browser()["headless"] is False

    show = _console()
    dispatch("/config browser show", session, show)
    output = show.file.getvalue()
    assert "on" in output

    dispatch("/config browser timeout 20000", session, _console())
    assert config.get_browser()["timeout_ms"] == 20_000


def test_config_browser_slash_rejects_bad_values(session):
    console = _console()
    dispatch("/config browser enable maybe", session, console)
    dispatch("/config browser headless maybe", session, console)
    dispatch("/config browser timeout 5", session, console)

    output = console.file.getvalue()
    assert "Usage: /config browser enable" in output
    assert "Usage: /config browser headless" in output
    assert "between 1000 and 120000" in output
    assert config.get_browser() == {
        "enabled": False,
        "headless": True,
        "timeout_ms": 15000,
    }


def test_config_browser_change_resets_live_driver(session, monkeypatch):
    resets: list[bool] = []
    monkeypatch.setattr(browser_module, "reset_driver", lambda: resets.append(True))

    dispatch("/config browser enable on", session, _console())
    dispatch("/config browser headless off", session, _console())
    dispatch("/config browser timeout 9000", session, _console())

    assert resets == [True, True, True]


def test_config_help_and_completions_include_browser(session):
    console = _console()
    dispatch("/config help", session, console)
    assert "/config browser" in console.file.getvalue()

    choices = dict(slash_argument_completions("config", "bro", None))
    assert "browser" in choices


def test_cli_config_browser_roundtrip():
    runner = CliRunner()

    result = runner.invoke(cli_app, ["config", "browser", "show"])
    assert result.exit_code == 0
    assert "off" in result.output

    result = runner.invoke(cli_app, ["config", "browser", "enable", "on"])
    assert result.exit_code == 0
    assert config.get_browser()["enabled"] is True

    result = runner.invoke(cli_app, ["config", "browser", "headless", "off"])
    assert result.exit_code == 0
    assert config.get_browser()["headless"] is False

    result = runner.invoke(cli_app, ["config", "browser", "timeout", "20000"])
    assert result.exit_code == 0
    assert config.get_browser()["timeout_ms"] == 20_000

    result = runner.invoke(cli_app, ["config", "browser", "show"])
    assert "20000ms" in result.output


def test_cli_config_browser_rejects_invalid_values():
    runner = CliRunner()

    result = runner.invoke(cli_app, ["config", "browser", "timeout", "5"])
    assert result.exit_code == 1
    assert config.get_browser()["timeout_ms"] == 15000

    result = runner.invoke(cli_app, ["config", "browser", "headless", "maybe"])
    assert result.exit_code == 1

    result = runner.invoke(cli_app, ["config", "browser", "enable", "maybe"])
    assert result.exit_code == 1
