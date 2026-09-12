"""Optional headless-browser automation backed by Playwright.

Playwright is an optional dependency::

    pip install 'kiwimatecoder[browser]' && playwright install chromium

Importing this module never imports ``playwright``; the import happens lazily
on the first :func:`get_driver` call, so the base install stays dependency-free.
The sync Playwright API refuses to run inside a running asyncio event loop
(which is where the agent invokes synchronous tools), so the real driver owns a
single worker thread and proxies every action onto it. Chromium therefore talks
to its own thread while the tool call blocks, and successive tool calls share
one browser page until :func:`reset_driver` closes it.
"""

from __future__ import annotations

import datetime
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

from kiwimatecoder import config

INSTALL_HINT = (
    "Playwright is not installed. Run: "
    "pip install 'kiwimatecoder[browser]' && playwright install chromium"
)

WORKSPACE_STATE_DIR = ".kiwimatecoder"
SCREENSHOT_DIR_NAME = "screenshots"


class BrowserError(Exception):
    """A user-facing browser failure with a ready-to-show message."""


class Driver(Protocol):
    """Minimal page interface implemented by the Playwright driver and fakes."""

    def open(self, url: str) -> tuple[str, str]:
        """Navigate to ``url`` and return ``(final_url, title)``."""
        ...

    def text(self) -> str:
        """Return the visible text of the current page."""
        ...

    def screenshot(self, path: Path) -> None:
        """Save a PNG screenshot of the current page to ``path``."""
        ...

    def click(self, selector: str) -> None:
        """Click the first element matching ``selector``."""
        ...

    def type(self, selector: str, text: str) -> None:
        """Replace the value of the element matching ``selector``."""
        ...

    def evaluate(self, script: str) -> Any:
        """Run ``script`` in the page and return its (JSON-ish) result."""
        ...

    def close(self) -> None:
        """Close the page, browser, and Playwright driver."""
        ...


def _friendly(action: str, exc: Exception) -> str:
    """Collapse an exception into one bounded, user-facing line."""
    detail = " ".join(str(exc).split())
    if len(detail) > 500:
        detail = detail[:497] + "..."
    if not detail:
        detail = exc.__class__.__name__
    return f"{action} failed: {detail}"


class _PlaywrightDriver:
    """A Chromium page hosted on a dedicated worker thread."""

    def __init__(self, settings: dict[str, Any], console: Any = None) -> None:
        self._settings = dict(settings)
        self._console = console
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="kiwi-browser"
        )
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None

    # -- thread plumbing ---------------------------------------------------

    def _run(self, action: str, fn: Callable[..., Any], *args: Any) -> Any:
        future = self._executor.submit(fn, *args)
        try:
            return future.result()
        except BrowserError:
            raise
        except Exception as exc:
            raise BrowserError(_friendly(action, exc)) from exc

    def start(self) -> None:
        """Import Playwright and launch Chromium, raising ``BrowserError``."""
        self._run("Could not start the browser", self._start)

    def _start(self) -> None:
        try:
            from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BrowserError(INSTALL_HINT) from exc
        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                headless=bool(self._settings.get("headless", True))
            )
            self._page = self._browser.new_page()
            timeout = int(self._settings.get("timeout_ms", 15000))
            self._page.set_default_timeout(timeout)
            self._page.set_default_navigation_timeout(timeout)
        except Exception as exc:
            self._close_impl()
            raise BrowserError(_friendly("Could not start Chromium", exc)) from exc

    def _ensure_page(self) -> None:
        if self._page is None:
            raise BrowserError("No page is open. Use action='open' with a URL first.")

    def _close_impl(self) -> None:
        page, self._page = self._page, None
        browser, self._browser = self._browser, None
        playwright, self._playwright = self._playwright, None
        for obj in (page, browser):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        if playwright is not None:
            try:
                playwright.stop()
            except Exception:
                pass

    def close(self) -> None:
        try:
            self._run("Could not close the browser", self._close_impl)
        finally:
            self._executor.shutdown(wait=False)

    # -- Driver actions ----------------------------------------------------

    def open(self, url: str) -> tuple[str, str]:
        def action() -> tuple[str, str]:
            self._ensure_page()
            response = self._page.goto(url, wait_until="domcontentloaded")
            final_url = str(
                getattr(response, "url", None) or getattr(self._page, "url", "") or url
            )
            try:
                title = str(self._page.title() or "")
            except Exception:
                title = ""
            return final_url, title

        return self._run("Opening the page", action)

    def text(self) -> str:
        def action() -> str:
            self._ensure_page()
            return str(self._page.inner_text("body") or "")

        return self._run("Reading the page", action)

    def screenshot(self, path: Path) -> None:
        def action() -> None:
            self._ensure_page()
            self._page.screenshot(path=str(path), full_page=True)

        self._run("Taking the screenshot", action)

    def click(self, selector: str) -> None:
        def action() -> None:
            self._ensure_page()
            self._page.click(selector)

        self._run("Clicking the element", action)

    def type(self, selector: str, text: str) -> None:
        def action() -> None:
            self._ensure_page()
            self._page.fill(selector, text)

        self._run("Typing into the element", action)

    def evaluate(self, script: str) -> Any:
        def action() -> Any:
            self._ensure_page()
            return self._page.evaluate(script)

        return self._run("Evaluating the script", action)

    def set_timeout(self, timeout_ms: int) -> None:
        def action() -> None:
            self._ensure_page()
            self._page.set_default_timeout(timeout_ms)

        self._run("Setting the timeout", action)


_DRIVER: Driver | None = None
_LOCK = threading.Lock()


def get_driver(
    settings: dict[str, Any] | None = None, console: Any = None
) -> Driver:
    """Return the singleton driver, launching Chromium on first use."""
    global _DRIVER
    with _LOCK:
        if _DRIVER is not None:
            return _DRIVER
        effective = settings if settings is not None else config.get_browser()
        driver = _PlaywrightDriver(effective, console)
        try:
            driver.start()
        except Exception:
            driver.close()
            raise
        _DRIVER = driver
        return _DRIVER


def current_driver() -> Driver | None:
    """Return the live driver without creating one (None before ``open``)."""
    return _DRIVER


def reset_driver() -> None:
    """Close the singleton driver, if any. Never raises."""
    global _DRIVER
    with _LOCK:
        driver, _DRIVER = _DRIVER, None
    if driver is not None:
        try:
            driver.close()
        except Exception:
            pass


def screenshot_path(workspace_root: str | Path) -> Path:
    """Return a fresh screenshot path under the workspace, creating dirs."""
    directory = (
        Path(workspace_root) / WORKSPACE_STATE_DIR / SCREENSHOT_DIR_NAME
    )
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return directory / f"{stamp}.png"
