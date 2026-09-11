from __future__ import annotations

import io

import httpx
import pytest
from rich.console import Console

from kiwimatecoder import config, tools
from kiwimatecoder import web as web_module
from kiwimatecoder.commands import dispatch
from kiwimatecoder.tools.web import _web_fetch, _web_search

HTML_FIXTURE = """<!doctype html>
<html>
  <head>
    <title>  Kiwi &amp; Mate  </title>
    <style>body { color: red; }</style>
    <script>console.log("tracking")</script>
    <noscript>enable js</noscript>
    <svg><text>vector</text></svg>
  </head>
  <body>
    <h1>Hello &amp; welcome</h1>
    <p>First paragraph.</p>
    <div>Second <b>bold</b> block.</div>



    <p>Last paragraph.</p>
  </body>
</html>
"""

DDG_FIXTURE = """
<html><body>
<div class="result results_links results_links_deep web-result">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F&amp;rut=abc">
      Python 3 Documentation
    </a>
  </h2>
  <div class="result__extras">
    <a class="result__url"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F">docs.python.org/3/</a>
  </div>
  <a class="result__snippet" href="https://docs.python.org/3/">
    The official <b>Python</b> documentation &amp; tutorials.
  </a>
</div>
<div class="result results_links results_links_deep web-result">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a" href="https://example.com/guide?x=1&amp;y=2">
      Example Guide
    </a>
  </h2>
  <a class="result__snippet" href="https://example.com/guide">A direct link.</a>
</div>
</body></html>
"""


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# html_to_text
# ---------------------------------------------------------------------------


def test_html_to_text_extracts_title_and_drops_scripts():
    title, text = web_module.html_to_text(HTML_FIXTURE)

    assert title == "Kiwi & Mate"
    assert "Hello & welcome" in text
    assert "First paragraph." in text
    assert "Second bold block." in text
    assert "Last paragraph." in text
    assert "tracking" not in text
    assert "color: red" not in text
    assert "enable js" not in text
    assert "vector" not in text
    assert "&amp;" not in text


def test_html_to_text_collapses_blank_lines():
    _title, text = web_module.html_to_text(HTML_FIXTURE)

    assert "\n\n\n" not in text


def test_html_to_text_without_title():
    title, text = web_module.html_to_text("<p>just text</p>")

    assert title is None
    assert text == "just text"


def test_html_to_text_decodes_entities():
    _title, text = web_module.html_to_text("<p>a &lt; b &amp;&amp; c &gt; d</p>")

    assert text == "a < b && c > d"


# ---------------------------------------------------------------------------
# is_local_address
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://localhost:8080/path",
        "localhost:8080",
        "https://127.0.0.1/",
        "https://127.0.0.1:443/",
        "http://[::1]/",
        "http://10.0.0.5/",
        "http://172.16.4.2/",
        "http://192.168.1.10/",
        "http://169.254.10.10/",
        "http://0.0.0.0/",
        "https://box.local/",
        "https://api.local:9000/",
    ],
)
def test_is_local_address_true(url):
    assert web_module.is_local_address(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/",
        "http://example.com:8080/",
        "https://8.8.8.8/",
        "https://docs.python.org/3/",
        "",
        "not a url",
    ],
)
def test_is_local_address_false(url):
    assert web_module.is_local_address(url) is False


# ---------------------------------------------------------------------------
# fetch_url
# ---------------------------------------------------------------------------


def _html_response(request: httpx.Request, body: str = HTML_FIXTURE) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"Content-Type": "text/html; charset=utf-8"},
        text=body,
        request=request,
    )


def test_fetch_url_follows_redirect_and_converts_html():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path == "/start":
            return httpx.Response(
                302,
                headers={"Location": "https://example.com/final"},
                request=request,
            )
        return _html_response(request)

    final_url, text = web_module.fetch_url(
        "https://example.com/start",
        timeout=5.0,
        max_chars=10_000,
        allow_local=False,
        transport=_transport(handler),
    )

    assert seen == ["https://example.com/start", "https://example.com/final"]
    assert final_url == "https://example.com/final"
    assert "Hello & welcome" in text
    assert "tracking" not in text


def test_fetch_url_text_plain_is_decoded():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/plain; charset=utf-8"},
            content=b"plain body",
            request=request,
        )

    final_url, text = web_module.fetch_url(
        "https://example.com/file.txt",
        timeout=5.0,
        max_chars=1_000,
        allow_local=False,
        transport=_transport(handler),
    )

    assert final_url == "https://example.com/file.txt"
    assert text == "plain body"


def test_fetch_url_binary_summary():
    payload = b"%PDF-1.4 fake pdf"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "application/pdf"},
            content=payload,
            request=request,
        )

    _final, text = web_module.fetch_url(
        "https://example.com/doc.pdf",
        timeout=5.0,
        max_chars=1_000,
        allow_local=False,
        transport=_transport(handler),
    )

    assert text == f"[binary content: application/pdf, {len(payload)} bytes]"


def test_fetch_url_enforces_byte_cap():
    payload = "a" * 100_000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/plain"},
            content=payload.encode(),
            request=request,
        )

    _final, text = web_module.fetch_url(
        "https://example.com/big.txt",
        timeout=5.0,
        max_chars=1_000,
        allow_local=False,
        transport=_transport(handler),
    )

    assert len(text) == 1_000 * web_module.MAX_BYTES_PER_CHAR


def test_fetch_url_http_error_is_friendly():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, request=request)

    with pytest.raises(web_module.WebError, match="HTTP 404"):
        web_module.fetch_url(
            "https://example.com/missing",
            timeout=5.0,
            max_chars=1_000,
            allow_local=False,
            transport=_transport(handler),
        )


def test_fetch_url_network_error_is_friendly():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(web_module.WebError) as excinfo:
        web_module.fetch_url(
            "https://example.com/",
            timeout=5.0,
            max_chars=1_000,
            allow_local=False,
            transport=_transport(handler),
        )

    assert "ConnectError" in str(excinfo.value)


def test_fetch_url_rejects_local_address():
    with pytest.raises(web_module.WebError, match="local or private"):
        web_module.fetch_url(
            "http://127.0.0.1:8000/",
            timeout=5.0,
            max_chars=1_000,
            allow_local=False,
        )


def test_fetch_url_allows_local_address_when_enabled():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/plain"},
            content=b"local ok",
            request=request,
        )

    _final, text = web_module.fetch_url(
        "http://127.0.0.1:8000/",
        timeout=5.0,
        max_chars=1_000,
        allow_local=True,
        transport=_transport(handler),
    )

    assert text == "local ok"


def test_fetch_url_rejects_non_http_scheme():
    with pytest.raises(web_module.WebError, match="http"):
        web_module.fetch_url(
            "ftp://example.com/file",
            timeout=5.0,
            max_chars=1_000,
            allow_local=False,
        )


# ---------------------------------------------------------------------------
# DuckDuckGo parsing
# ---------------------------------------------------------------------------


def test_parse_ddg_results_unwraps_urls_and_keeps_snippets():
    results = web_module.parse_ddg_results(DDG_FIXTURE, limit=10)

    assert len(results) == 2
    first = results[0]
    assert first.title == "Python 3 Documentation"
    assert first.url == "https://docs.python.org/3/"
    assert first.snippet == "The official Python documentation & tutorials."
    second = results[1]
    assert second.title == "Example Guide"
    assert second.url == "https://example.com/guide?x=1&y=2"
    assert second.snippet == "A direct link."


def test_parse_ddg_results_respects_limit():
    results = web_module.parse_ddg_results(DDG_FIXTURE, limit=1)

    assert [result.title for result in results] == ["Python 3 Documentation"]


def test_search_web_queries_duckduckgo():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text=DDG_FIXTURE,
            request=request,
        )

    results = web_module.search_web(
        "python docs",
        count=3,
        timeout=5.0,
        provider="duckduckgo",
        transport=_transport(handler),
    )

    assert len(results) == 2
    assert captured[0].url.host == "html.duckduckgo.com"
    assert captured[0].url.params["q"] == "python docs"


def test_search_web_rejects_unknown_provider():
    with pytest.raises(web_module.WebError, match="Unknown search provider"):
        web_module.search_web("x", timeout=5.0, provider="google")


def test_search_web_rejects_empty_query():
    with pytest.raises(web_module.WebError, match="empty"):
        web_module.search_web("", timeout=5.0, provider="duckduckgo")


# ---------------------------------------------------------------------------
# Tool wrappers and registration
# ---------------------------------------------------------------------------


def test_web_tools_registered_and_read_only():
    for name in ("web_fetch", "web_search"):
        tool = tools.get_tool(name)
        assert tool is not None
        assert tool.writes is False
        assert tool.runs is False
        assert tool.needs_approval is False

    read_only_names = {
        schema["function"]["name"] for schema in tools.tool_schemas(read_only=True)
    }
    assert {"web_fetch", "web_search"} <= read_only_names


def test_web_fetch_tool_requires_url(session):
    result = _web_fetch({}, session)

    assert not result.ok
    assert "'url' is required" in result.content


def test_web_fetch_tool_rejects_local_url_by_default(session):
    result = _web_fetch({"url": "http://localhost:8080/status"}, session)

    assert not result.ok
    assert "local or private" in result.content


def test_web_fetch_tool_allows_local_when_configured(session, monkeypatch):
    config.set_web(allow_local=True)
    monkeypatch.setattr(
        web_module,
        "fetch_url",
        lambda url, **kwargs: (url, "local service ok"),
    )

    result = _web_fetch({"url": "http://localhost:8080/status"}, session)

    assert result.ok
    assert "local service ok" in result.content


def test_web_fetch_tool_truncates_with_note(session, monkeypatch):
    config.set_web(max_chars=1_000)
    monkeypatch.setattr(
        web_module,
        "fetch_url",
        lambda url, **kwargs: (url, "x" * 5_000),
    )

    result = _web_fetch({"url": "https://example.com/"}, session)

    assert result.ok
    assert "[truncated at 1000 characters]" in result.content
    assert result.content.count("x") == 1_000


def test_web_fetch_tool_reports_redirect_target(session, monkeypatch):
    monkeypatch.setattr(
        web_module,
        "fetch_url",
        lambda url, **kwargs: ("https://example.com/final", "body"),
    )

    result = _web_fetch({"url": "https://example.com/start"}, session)

    assert "URL: https://example.com/final" in result.content
    assert "body" in result.content


def test_web_search_tool_formats_results(session, monkeypatch):
    monkeypatch.setattr(
        web_module,
        "search_web",
        lambda query, **kwargs: [
            web_module.SearchResult("Title A", "https://a.example", "snippet a"),
            web_module.SearchResult("Title B", "https://b.example", ""),
        ],
    )

    result = _web_search({"query": "kiwi"}, session)

    assert result.ok
    assert "Title A" in result.content
    assert "https://a.example" in result.content
    assert "snippet a" in result.content
    assert "Cite the URLs" in result.content


def test_web_search_tool_empty_results(session, monkeypatch):
    monkeypatch.setattr(web_module, "search_web", lambda query, **kwargs: [])

    result = _web_search({"query": "nothing"}, session)

    assert result.ok
    assert "No results found" in result.content


def test_web_search_tool_requires_query(session):
    result = _web_search({}, session)

    assert not result.ok
    assert "'query' is required" in result.content


def test_web_search_tool_rejects_bad_count(session):
    result = _web_search({"query": "x", "count": "many"}, session)

    assert not result.ok
    assert "'count' must be an integer" in result.content


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_get_web_defaults():
    settings = config.get_web()

    assert settings == {
        "max_chars": 50000,
        "timeout": 20.0,
        "allow_local": False,
        "search_provider": "duckduckgo",
        "search_api_key": "",
    }


def test_set_web_roundtrip():
    config.set_web(max_chars=120_000, timeout=9, allow_local=True)

    stored = config.load_config()["web"]
    assert stored["max_chars"] == 120_000
    assert stored["timeout"] == 9.0
    assert stored["allow_local"] is True

    settings = config.get_web()
    assert settings["max_chars"] == 120_000
    assert settings["timeout"] == 9.0
    assert settings["allow_local"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_chars": 10},
        {"max_chars": 3_000_000},
        {"max_chars": "many"},
        {"timeout": 0.5},
        {"timeout": 500},
        {"timeout": "soon"},
        {"search_provider": "google"},
    ],
)
def test_set_web_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        config.set_web(**kwargs)


def test_get_web_normalizes_bad_stored_values():
    config.save_config(
        {
            "web": {
                "max_chars": "junk",
                "timeout": 999,
                "allow_local": "yes",
                "search_provider": "nope",
                "search_api_key": 123,
            }
        }
    )

    settings = config.get_web()

    assert settings["max_chars"] == 50000
    assert settings["timeout"] == 20.0
    assert settings["allow_local"] is False
    assert settings["search_provider"] == "duckduckgo"
    assert settings["search_api_key"] == "123"


def test_validate_config_accepts_clean_web_section():
    config.set_web(max_chars=60_000, timeout=15, allow_local=False)

    issues = [issue for issue in config.validate_config() if issue["key"].startswith("web")]

    assert issues == []


def test_validate_config_catches_bad_web_section():
    config.save_config(
        {
            "web": {
                "max_chars": -1,
                "timeout": "later",
                "allow_local": "yes",
                "search_provider": "bing",
                "search_api_key": 5,
            }
        }
    )

    keys = {issue["key"] for issue in config.validate_config()}

    assert {"web.max_chars", "web.timeout", "web.allow_local"} <= keys
    assert "web.search_provider" in keys
    assert "web.search_api_key" in keys


# ---------------------------------------------------------------------------
# Slash command
# ---------------------------------------------------------------------------


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def test_config_web_slash_roundtrip(session):
    console = _console()

    dispatch("/config web max-chars 60000", session, console)
    dispatch("/config web timeout 9", session, console)
    dispatch("/config web allow-local on", session, console)

    settings = config.get_web()
    assert settings["max_chars"] == 60000
    assert settings["timeout"] == 9.0
    assert settings["allow_local"] is True

    show = _console()
    dispatch("/config web show", session, show)
    output = show.file.getvalue()
    assert "60000" in output
    assert "on" in output


def test_config_web_slash_rejects_bad_values(session):
    console = _console()

    dispatch("/config web max-chars 5", session, console)
    dispatch("/config web timeout nope", session, console)
    dispatch("/config web allow-local maybe", session, console)

    settings = config.get_web()
    assert settings["max_chars"] == 50000
    assert settings["timeout"] == 20.0
    assert settings["allow_local"] is False
    assert "between 1000 and 2000000" in console.file.getvalue()
    assert "Expected 'on' or 'off'." in console.file.getvalue()
