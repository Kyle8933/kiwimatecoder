"""Clipboard and HTTP/API testing tools.

``read_clipboard`` and ``http_get`` are read-only (safe in plan mode).
``write_clipboard`` and ``http_request`` execute a clipboard command or can
mutate remote state, so they set ``runs=True`` and go through the approval
gate. HTTP responses are rendered with a status line, selected headers, and a
truncated body; raw ``httpx`` errors are never leaked.
"""

from __future__ import annotations

import json as json_module
from typing import Any

import httpx

from kiwimatecoder import clipboard as clipboard_module
from kiwimatecoder import config
from kiwimatecoder import web as web_module
from kiwimatecoder.redaction import redact
from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult

HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")
_PREVIEW_CHARS = 200
_RESPONSE_HEADER_NAMES = (
    "content-type",
    "content-length",
    "location",
    "server",
    "date",
    "etag",
    "last-modified",
    "cache-control",
    "www-authenticate",
    "x-request-id",
)


def request_http(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    body: str | None = None,
    json_body: Any = None,
    timeout: float,
    max_chars: int,
    allow_local: bool,
    transport: httpx.BaseTransport | None = None,
) -> ToolResult:
    """Perform an HTTP request and render a bounded, model-friendly result.

    The response status is returned as-is (4xx/5xx are informative for API
    testing), the body is clipped to ``max_chars``, and JSON payloads are
    pretty-printed. ``transport`` exists for tests.
    """
    method = str(method).strip().upper()
    if method not in HTTP_METHODS:
        return ToolResult.error(
            f"Unknown HTTP method '{method}'. Choose: {', '.join(HTTP_METHODS)}."
        )
    if body is not None and json_body is not None:
        return ToolResult.error("'body' and 'json' are mutually exclusive")
    try:
        cleaned = web_module.clean_url(url)
    except web_module.WebError as exc:
        return ToolResult.error(str(exc))
    if web_module.is_local_address(cleaned) and not allow_local:
        return ToolResult.error(
            "Refusing to call a local or private address. Enable it with "
            "/config web allow-local on (or config web allow-local on)."
        )

    request_headers = {"User-Agent": web_module.USER_AGENT}
    request_headers.update(headers or {})

    byte_cap = max(1024, int(max_chars) * web_module.MAX_BYTES_PER_CHAR)
    status_line = "HTTP ?"
    final_url = cleaned
    response_headers: dict[str, str] = {}
    data = bytearray()
    charset = "utf-8"
    try:
        with httpx.Client(
            timeout=timeout, transport=transport, follow_redirects=True
        ) as client:
            with client.stream(
                method,
                cleaned,
                headers=request_headers,
                content=body,
                json=json_body,
            ) as response:
                status_line = (
                    f"HTTP {response.status_code} {response.reason_phrase}"
                ).strip()
                final_url = str(response.url)
                response_headers = {
                    key.lower(): value for key, value in response.headers.items()
                }
                charset = response.charset_encoding or "utf-8"
                for chunk in response.iter_bytes():
                    data.extend(chunk)
                    if len(data) >= byte_cap:
                        del data[byte_cap:]
                        break
    except httpx.HTTPError as exc:
        return ToolResult.error(f"Request failed: {exc.__class__.__name__}.")

    text = bytes(data).decode(charset, errors="replace")
    mime = (response_headers.get("content-type") or "").split(";", 1)[0].strip().lower()
    if mime == "application/json" or mime.endswith("+json"):
        try:
            text = json_module.dumps(
                json_module.loads(text), indent=2, ensure_ascii=False
            )
        except (json_module.JSONDecodeError, ValueError):
            pass
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[truncated at {max_chars} characters]"

    lines = [status_line]
    if final_url != cleaned:
        lines.append(f"URL: {final_url}")
    lines.extend(
        f"{name}: {response_headers[name]}"
        for name in _RESPONSE_HEADER_NAMES
        if name in response_headers
    )
    rendered = "\n".join(lines)
    if text:
        rendered += "\n\n" + text
    return ToolResult(content=rendered)


def _string_headers(raw: Any) -> tuple[dict[str, str] | None, str | None]:
    if raw is None:
        return {}, None
    if not isinstance(raw, dict):
        return None, "'headers' must be an object of string values"
    return {str(key): str(value) for key, value in raw.items()}, None


def _http_call(
    args: dict[str, Any], session: Session, *, method: str | None = None
) -> ToolResult:
    del session
    url = str(args.get("url") or "").strip()
    if not url:
        return ToolResult.error("'url' is required")
    headers, error = _string_headers(args.get("headers"))
    if error:
        return ToolResult.error(error)

    settings = config.get_web()
    timeout = float(settings["timeout"])
    raw_timeout = args.get("timeout")
    if raw_timeout is not None:
        try:
            timeout = float(raw_timeout)
        except (TypeError, ValueError):
            return ToolResult.error("'timeout' must be a number of seconds")
        if not config.WEB_TIMEOUT_MIN <= timeout <= config.WEB_TIMEOUT_MAX:
            return ToolResult.error(
                f"'timeout' must be between {config.WEB_TIMEOUT_MIN:g} and "
                f"{config.WEB_TIMEOUT_MAX:g} seconds"
            )

    body = args.get("body")
    if body is not None and not isinstance(body, str):
        return ToolResult.error("'body' must be a string")
    json_body = args.get("json")
    if body is not None and json_body is not None:
        return ToolResult.error("'body' and 'json' are mutually exclusive")

    chosen = method or str(args.get("method") or "").strip()
    return request_http(
        chosen,
        url,
        headers=headers,
        body=body,
        json_body=json_body,
        timeout=timeout,
        max_chars=int(settings["max_chars"]),
        allow_local=bool(settings["allow_local"]),
    )


def _read_clipboard(args: dict[str, Any], session: Session) -> ToolResult:
    del args, session
    ok, payload = clipboard_module.read_clipboard()
    if not ok:
        return ToolResult.error(payload)
    return ToolResult(content=payload if payload else "(clipboard is empty)")


def _write_clipboard(args: dict[str, Any], session: Session) -> ToolResult:
    del session
    text = args.get("text")
    if text is None or str(text) == "":
        return ToolResult.error("'text' is required")
    ok, message = clipboard_module.write_clipboard(str(text))
    if not ok:
        return ToolResult.error(message)
    return ToolResult(content=message)


def _clip(text: str) -> str:
    return text if len(text) <= _PREVIEW_CHARS else text[:_PREVIEW_CHARS] + "…"


def write_clipboard_preview(args: dict[str, Any], session: Session) -> str:
    """Show (redacted, clipped) text for the write_clipboard approval prompt."""
    del session
    return f"Copy to clipboard:\n{_clip(redact(str(args.get('text') or ''))) or '(empty)'}"


def http_request_preview(args: dict[str, Any], session: Session) -> str:
    """Show the redacted request for the http_request approval prompt."""
    del session
    method = str(args.get("method") or "GET").strip().upper()
    lines = [f"{method} {str(args.get('url') or '')}"]
    headers = args.get("headers")
    if isinstance(headers, dict):
        lines.extend(f"{key}: {redact(str(value))}" for key, value in headers.items())
    body = args.get("body")
    json_body = args.get("json")
    if body is not None:
        lines.append(_clip(redact(str(body))))
    elif json_body is not None:
        try:
            payload = json_module.dumps(json_body, ensure_ascii=False)
        except (TypeError, ValueError):
            payload = str(json_body)
        lines.append(_clip(redact(payload)))
    return "\n".join(lines)


def _http_get(args: dict[str, Any], session: Session) -> ToolResult:
    return _http_call(args, session, method="GET")


def _http_request(args: dict[str, Any], session: Session) -> ToolResult:
    return _http_call(args, session)


read_clipboard_tool = FunctionTool(
    name="read_clipboard",
    description=(
        "Read the system clipboard as text. Read-only, so it runs without "
        "approval and stays available in plan mode."
    ),
    parameters={"type": "object", "properties": {}},
    func=_read_clipboard,
)

write_clipboard_tool = FunctionTool(
    name="write_clipboard",
    description=(
        "Copy text to the system clipboard. Requires approval; the preview "
        "shows the exact text that will be copied."
    ),
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to place on the clipboard.",
            },
        },
        "required": ["text"],
    },
    func=_write_clipboard,
    runs=True,
)

http_get_tool = FunctionTool(
    name="http_get",
    description=(
        "Fetch an http(s) URL and return its status line, selected response "
        "headers, and body (JSON is pretty-printed). Read-only. Local and "
        "private addresses are blocked unless enabled in /config web; for "
        "other methods use http_request."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Absolute http(s) URL to request.",
            },
            "headers": {
                "type": "object",
                "description": "Optional request headers (name -> value).",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["url"],
    },
    func=_http_get,
)

http_request_tool = FunctionTool(
    name="http_request",
    description=(
        "Send an HTTP request with GET, POST, PUT, PATCH, or DELETE and return "
        "the status line, selected response headers, and body (JSON is "
        "pretty-printed). Requires approval because it can change remote state. "
        "Use 'body' for raw text or 'json' for a JSON payload, not both. Local "
        "and private addresses are blocked unless enabled in /config web."
    ),
    parameters={
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": list(HTTP_METHODS),
                "description": "HTTP method.",
            },
            "url": {
                "type": "string",
                "description": "Absolute http(s) URL to request.",
            },
            "headers": {
                "type": "object",
                "description": "Optional request headers (name -> value).",
                "additionalProperties": {"type": "string"},
            },
            "body": {
                "type": "string",
                "description": "Raw request body (mutually exclusive with json).",
            },
            "json": {
                "description": "JSON request body (mutually exclusive with body).",
            },
            "timeout": {
                "type": "number",
                "description": "Seconds before the request times out (default: configured).",
            },
        },
        "required": ["method", "url"],
    },
    func=_http_request,
    runs=True,
)
