"""browser tool: drive an optional headless Chromium page via Playwright.

The tool is approval-gated (``runs=True``) and only advertised while browser
automation is enabled in config. It never imports Playwright itself; the
optional driver lives in :mod:`kiwimatecoder.browser`.
"""

from __future__ import annotations

import json
from typing import Any

from kiwimatecoder import browser, config, images, redaction
from kiwimatecoder import web as web_module
from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult
from kiwimatecoder.tools.paths import display_path

ACTIONS = ("open", "text", "screenshot", "click", "type", "evaluate", "close")

TEXT_MAX_CHARS = 30_000
EVALUATE_MAX_CHARS = 10_000
MIN_TIMEOUT_MS = 1_000
MAX_TIMEOUT_MS = 120_000

DISABLED_MESSAGE = (
    "Browser automation is disabled. Enable it with `/config browser enable on` "
    "(and install the optional extra: pip install 'kiwimatecoder[browser]' && "
    "playwright install chromium)."
)
NO_PAGE_MESSAGE = "No page is open. Use action='open' with a URL first."


def _timeout_override(args: dict[str, Any]) -> tuple[int | None, str | None]:
    raw = args.get("timeout_ms")
    if raw is None:
        return None, None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None, "'timeout_ms' must be an integer number of milliseconds"
    if not MIN_TIMEOUT_MS <= value <= MAX_TIMEOUT_MS:
        return None, (
            f"'timeout_ms' must be between {MIN_TIMEOUT_MS} and {MAX_TIMEOUT_MS}"
        )
    return value, None


def _validate_url(url: str) -> str:
    try:
        cleaned = web_module.clean_url(url)
    except web_module.WebError as exc:
        raise browser.BrowserError(str(exc)) from exc
    if web_module.is_local_address(cleaned) and not config.get_web()["allow_local"]:
        raise browser.BrowserError(
            "Refusing to open a local or private address. Enable it with "
            "/config web allow-local on (or config web allow-local on)."
        )
    return cleaned


def _required(args: dict[str, Any], key: str) -> tuple[str, str | None]:
    value = str(args.get(key) or "").strip()
    if not value:
        return "", f"'{key}' is required"
    return value, None


def _apply_timeout(driver: browser.Driver, timeout_ms: int) -> None:
    setter = getattr(driver, "set_timeout", None)
    if callable(setter):
        setter(timeout_ms)


def _format_result(value: Any) -> str:
    if isinstance(value, str):
        rendered = value
    else:
        try:
            rendered = json.dumps(value, indent=2, default=str)
        except (TypeError, ValueError):
            rendered = repr(value)
    if len(rendered) > EVALUATE_MAX_CHARS:
        rendered = rendered[:EVALUATE_MAX_CHARS] + (
            f"\n\n[truncated at {EVALUATE_MAX_CHARS} characters]"
        )
    return rendered or "(no result)"


def _screenshot(driver: browser.Driver, session: Session) -> ToolResult:
    path = browser.screenshot_path(session.workspace_root)
    driver.screenshot(path)
    label = display_path(path, session.workspace_root)

    vision = config.get_vision()
    limit = int(vision["max_images_per_turn"])
    if len(session.pending_images) >= limit:
        return ToolResult(
            content=(
                f"Saved screenshot to {label}; the per-turn image limit "
                f"({limit}) is reached, so it was not attached."
            )
        )
    try:
        entry = images.encode_image(path, int(vision["max_image_bytes"]))
    except (ValueError, OSError) as exc:
        return ToolResult(
            content=f"Saved screenshot to {label} but could not attach it: {exc}"
        )
    session.pending_images.append(entry)
    return ToolResult(content=f"Saved screenshot to {label} and attached it.")


def _browser(args: dict[str, Any], session: Session) -> ToolResult:
    action = str(args.get("action") or "").strip().lower()
    if action not in ACTIONS:
        return ToolResult.error(
            f"Unknown browser action '{action or '?'}'. "
            f"Choose: {', '.join(ACTIONS)}."
        )
    settings = config.get_browser()
    if not settings["enabled"]:
        return ToolResult.error(DISABLED_MESSAGE)

    timeout_ms, error = _timeout_override(args)
    if error:
        return ToolResult.error(error)

    if action == "close":
        browser.reset_driver()
        return ToolResult(content="Browser closed.")

    if action == "open":
        url = str(args.get("url") or "").strip()
        if not url:
            return ToolResult.error("'url' is required for action 'open'")
        try:
            cleaned = _validate_url(url)
            driver = browser.get_driver(settings)
            if timeout_ms is not None:
                _apply_timeout(driver, timeout_ms)
            final_url, title = driver.open(cleaned)
        except browser.BrowserError as exc:
            return ToolResult.error(str(exc))
        return ToolResult(
            content=f"Opened {final_url}\nTitle: {title or '(no title)'}"
        )

    page = browser.current_driver()
    if page is None:
        return ToolResult.error(NO_PAGE_MESSAGE)

    try:
        if timeout_ms is not None:
            _apply_timeout(page, timeout_ms)

        if action == "text":
            page_text = page.text()
            if len(page_text) > TEXT_MAX_CHARS:
                page_text = page_text[:TEXT_MAX_CHARS] + (
                    f"\n\n[truncated at {TEXT_MAX_CHARS} characters]"
                )
            return ToolResult(content=page_text or "(no visible text)")

        if action == "screenshot":
            return _screenshot(page, session)

        if action == "click":
            selector, error = _required(args, "selector")
            if error:
                return ToolResult.error(error)
            page.click(selector)
            return ToolResult(content=f"Clicked {selector}.")

        if action == "type":
            selector, error = _required(args, "selector")
            if error:
                return ToolResult.error(error)
            text = str(args.get("text") or "")
            if not text:
                return ToolResult.error("'text' is required for action 'type'")
            page.type(selector, text)
            return ToolResult(content=f"Typed into {selector}.")

        if action == "evaluate":
            script, error = _required(args, "script")
            if error:
                return ToolResult.error(error)
            return ToolResult(content=_format_result(page.evaluate(script)))
    except browser.BrowserError as exc:
        return ToolResult.error(str(exc))

    return ToolResult.error(f"Unsupported browser action '{action}'.")


def preview(args: dict[str, Any], session: Session) -> str:
    """Render the approval preview: ``browser <action> <target>``."""
    del session
    action = str(args.get("action") or "?")
    target = (
        args.get("url")
        or args.get("selector")
        or args.get("script")
        or args.get("text")
        or ""
    )
    cleaned = redaction.redact(str(target))
    if len(cleaned) > 80:
        cleaned = cleaned[:77] + "..."
    return f"browser {action} {cleaned}".rstrip()


browser_tool = FunctionTool(
    name="browser",
    description=(
        "Drive a headless Chromium page when browser automation is enabled. "
        "Start with action='open' and a URL, then read the page with 'text', "
        "capture it with 'screenshot' (the image is sent to you), interact "
        "with 'click'/'type', run JavaScript with 'evaluate', or finish with "
        "'close'. Local and private addresses are blocked unless enabled in "
        "/config web. Requires the optional Playwright extra."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(ACTIONS),
                "description": "Browser action to perform.",
            },
            "url": {
                "type": "string",
                "description": "Absolute http(s) URL for action 'open'.",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector for 'click' or 'type'.",
            },
            "text": {
                "type": "string",
                "description": "Text to enter for action 'type'.",
            },
            "script": {
                "type": "string",
                "description": "JavaScript expression for action 'evaluate'.",
            },
            "timeout_ms": {
                "type": "integer",
                "description": (
                    "Optional per-action timeout in milliseconds "
                    f"({MIN_TIMEOUT_MS}-{MAX_TIMEOUT_MS}); defaults to the "
                    "configured browser timeout."
                ),
            },
        },
        "required": ["action"],
    },
    func=_browser,
    writes=False,
    runs=True,
)
