from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
from rich.console import Console

from kiwimatecoder import config, tools
from kiwimatecoder.agent import Agent
from kiwimatecoder.tools import io as io_module
from kiwimatecoder.tools.io import _http_get, _http_request, request_http


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)


def _install_transport(monkeypatch, handler) -> httpx.MockTransport:
    """Route tool-level ``request_http`` calls through a MockTransport."""
    transport = httpx.MockTransport(handler)
    original = io_module.request_http

    def patched(method, url, **kwargs):
        kwargs["transport"] = transport
        return original(method, url, **kwargs)

    monkeypatch.setattr(io_module, "request_http", patched)
    return transport


def _forbid_transport(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request to {request.url}")

    _install_transport(monkeypatch, handler)


# ---------------------------------------------------------------------------
# request_http helper
# ---------------------------------------------------------------------------


def test_json_response_is_pretty_printed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            json={"ok": True, "items": [1, 2]},
            request=request,
        )

    result = request_http(
        "GET",
        "https://api.example/data",
        timeout=5.0,
        max_chars=10_000,
        allow_local=False,
        transport=httpx.MockTransport(handler),
    )

    assert result.ok
    assert "HTTP 200" in result.content
    assert '"ok": true' in result.content
    assert '  "items": [' in result.content


def test_json_suffix_content_type_is_pretty_printed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "application/problem+json"},
            text='{"title":"bad"}',
            request=request,
        )

    result = request_http(
        "GET",
        "https://api.example/problem",
        timeout=5.0,
        max_chars=10_000,
        allow_local=False,
        transport=httpx.MockTransport(handler),
    )

    assert '"title": "bad"' in result.content


def test_request_http_rejects_bad_method():
    result = request_http(
        "TRACE",
        "https://api.example/",
        timeout=5.0,
        max_chars=1_000,
        allow_local=False,
    )

    assert not result.ok
    assert "Unknown HTTP method" in result.content


def test_request_http_rejects_body_and_json_together():
    result = request_http(
        "POST",
        "https://api.example/",
        body="raw",
        json_body={"a": 1},
        timeout=5.0,
        max_chars=1_000,
        allow_local=False,
    )

    assert not result.ok
    assert "mutually exclusive" in result.content


# ---------------------------------------------------------------------------
# Tool wrappers
# ---------------------------------------------------------------------------


def test_http_get_renders_status_headers_and_body(session, monkeypatch):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            headers={
                "Content-Type": "application/json",
                "Server": "test-server",
                "X-Request-Id": "abc123",
                "X-Internal": "hidden",
            },
            json={"ok": True},
            request=request,
        )

    _install_transport(monkeypatch, handler)

    result = _http_get({"url": "https://api.example/data"}, session)

    assert result.ok
    assert "HTTP 200" in result.content
    assert "content-type: application/json" in result.content
    assert "server: test-server" in result.content
    assert "x-request-id: abc123" in result.content
    assert "x-internal" not in result.content
    assert '"ok": true' in result.content
    assert seen[0].method == "GET"


def test_http_request_sends_method_headers_and_raw_body(session, monkeypatch):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201, text="created", request=request)

    _install_transport(monkeypatch, handler)

    result = _http_request(
        {
            "method": "post",
            "url": "https://api.example/items",
            "headers": {"Content-Type": "text/plain", "X-Test": "1"},
            "body": "raw payload",
        },
        session,
    )

    assert result.ok
    assert seen[0].method == "POST"
    assert seen[0].content == b"raw payload"
    assert seen[0].headers["x-test"] == "1"
    assert "HTTP 201" in result.content


def test_http_request_sends_json_body(session, monkeypatch):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"received": True}, request=request)

    _install_transport(monkeypatch, handler)

    result = _http_request(
        {"method": "PUT", "url": "https://api.example/items/1", "json": {"a": 1}},
        session,
    )

    assert result.ok
    assert seen[0].method == "PUT"
    assert seen[0].content == b'{"a":1}'
    assert '"received": true' in result.content


def test_http_tools_reject_missing_url_and_bad_headers(session):
    assert not _http_get({}, session).ok
    assert "'url' is required" in _http_get({}, session).content

    bad_headers = _http_get(
        {"url": "https://api.example/", "headers": ["nope"]}, session
    )
    assert not bad_headers.ok
    assert "'headers' must be an object" in bad_headers.content


def test_http_request_body_json_conflict_tool_level(session, monkeypatch):
    _forbid_transport(monkeypatch)

    result = _http_request(
        {
            "method": "POST",
            "url": "https://api.example/",
            "body": "raw",
            "json": {"a": 1},
        },
        session,
    )

    assert not result.ok
    assert "mutually exclusive" in result.content


def test_http_request_requires_known_method(session):
    result = _http_request({"method": "TRACE", "url": "https://api.example/"}, session)

    assert not result.ok
    assert "Unknown HTTP method" in result.content


def test_http_response_is_truncated(session, monkeypatch):
    config.set_web(max_chars=1_000)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/plain"},
            text="a" * 5_000,
            request=request,
        )

    _install_transport(monkeypatch, handler)

    result = _http_get({"url": "https://api.example/big"}, session)

    assert "[truncated at 1000 characters]" in result.content
    assert "a" * 1_000 in result.content
    assert "a" * 1_001 not in result.content


def test_http_error_status_is_returned_not_raised(session, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            headers={"Content-Type": "text/plain"},
            text="missing",
            request=request,
        )

    _install_transport(monkeypatch, handler)

    result = _http_get({"url": "https://api.example/missing"}, session)

    assert result.ok
    assert "HTTP 404" in result.content
    assert "missing" in result.content


def test_http_network_error_is_friendly(session, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    _install_transport(monkeypatch, handler)

    result = _http_get({"url": "https://api.example/"}, session)

    assert not result.ok
    assert "ConnectError" in result.content


def test_http_rejects_non_http_scheme(session, monkeypatch):
    _forbid_transport(monkeypatch)

    result = _http_get({"url": "ftp://example.com/file"}, session)

    assert not result.ok
    assert "http://" in result.content


def test_http_reports_redirect_target(session, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(
                302,
                headers={"Location": "https://api.example/final"},
                request=request,
            )
        return httpx.Response(200, text="arrived", request=request)

    _install_transport(monkeypatch, handler)

    result = _http_get({"url": "https://api.example/start"}, session)

    assert "URL: https://api.example/final" in result.content
    assert "arrived" in result.content


@pytest.mark.parametrize("value", ["soon", 0.5, 999])
def test_http_bad_timeout_override(session, monkeypatch, value):
    _forbid_transport(monkeypatch)

    result = _http_request(
        {"method": "GET", "url": "https://api.example/", "timeout": value}, session
    )

    assert not result.ok
    assert "'timeout' must be" in result.content


# ---------------------------------------------------------------------------
# Local-address policy
# ---------------------------------------------------------------------------


def test_http_blocks_local_address_by_default(session, monkeypatch):
    _forbid_transport(monkeypatch)

    result = _http_get({"url": "http://127.0.0.1:8000/status"}, session)

    assert not result.ok
    assert "local or private" in result.content


def test_http_allows_local_address_when_enabled(session, monkeypatch):
    config.set_web(allow_local=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/plain"},
            text="local ok",
            request=request,
        )

    _install_transport(monkeypatch, handler)

    result = _http_get({"url": "http://127.0.0.1:8000/status"}, session)

    assert result.ok
    assert "local ok" in result.content


# ---------------------------------------------------------------------------
# Registration and summaries
# ---------------------------------------------------------------------------


def test_http_tools_registered_and_approval_flags():
    get_tool = tools.get_tool("http_get")
    request_tool = tools.get_tool("http_request")
    assert get_tool is not None and request_tool is not None
    assert get_tool.writes is False and get_tool.runs is False
    assert get_tool.needs_approval is False
    assert request_tool.runs is True
    assert request_tool.needs_approval is True

    read_only_names = {
        schema["function"]["name"] for schema in tools.tool_schemas(read_only=True)
    }
    assert "http_get" in read_only_names
    assert "http_request" not in read_only_names


def test_agent_http_call_summaries(session):
    agent = Agent(session, Console(quiet=True), MagicMock())

    assert agent._format_call_summary(
        "http_get", {"url": "https://api.example/data"}
    ) == "http [dim]GET https://api.example/data[/dim]"
    assert agent._format_call_summary(
        "http_request", {"method": "post", "url": "https://api.example/items"}
    ) == "http [dim]POST https://api.example/items[/dim]"


def test_agent_clipboard_call_summaries(session):
    agent = Agent(session, Console(quiet=True), MagicMock())

    assert agent._format_call_summary("read_clipboard", {}) == (
        "clipboard [dim]read[/dim]"
    )
    assert agent._format_call_summary("write_clipboard", {"text": "hi"}) == (
        "clipboard [dim]write hi[/dim]"
    )


def test_http_request_preview_redacts_and_clips(session):
    preview = io_module.http_request_preview(
        {
            "method": "post",
            "url": "https://api.example/items",
            "headers": {"Authorization": "Bearer sk-or-abcdefghijklmnop"},
            "json": {"name": "kiwi"},
        },
        session,
    )

    assert preview.splitlines()[0] == "POST https://api.example/items"
    assert "Authorization: Bearer [REDACTED]" in preview
    assert '"name": "kiwi"' in preview
    assert "sk-or-abcdefghijklmnop" not in preview

    long_preview = io_module.http_request_preview(
        {"method": "POST", "url": "https://api.example/", "body": "x" * 500}, session
    )
    assert long_preview.endswith("…")

    registered = tools.preview(
        "http_request",
        {"method": "DELETE", "url": "https://api.example/items/1"},
        session,
    )
    assert registered == "DELETE https://api.example/items/1"
    assert tools.preview("write_clipboard", {"text": "hi"}, session) is not None
