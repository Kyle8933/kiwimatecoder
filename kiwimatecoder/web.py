"""Web helpers: HTML-to-text, local-address detection, fetching, and search.

The pure helpers (:func:`html_to_text`, :func:`is_local_address`,
:func:`parse_ddg_results`) are kept free of network access so they can be
unit-tested from fixture strings. Network failures surface as
:class:`WebError` with a message that is safe to show the model; raw
``httpx`` errors are never leaked.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlsplit

import httpx

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 KiwiMateCoder/0.1"
)

#: A UTF-8 character can take up to four bytes, so a byte cap of
#: ``max_chars * 4`` guarantees at least ``max_chars`` characters are readable.
MAX_BYTES_PER_CHAR = 4

DDG_HTML_URL = "https://html.duckduckgo.com/html/"

DEFAULT_SEARCH_COUNT = 5
MAX_SEARCH_COUNT = 10


class WebError(Exception):
    """A user-facing web failure with a ready-to-show message."""


# ---------------------------------------------------------------------------
# HTML to text
# ---------------------------------------------------------------------------

_SKIP_TAGS = frozenset({"script", "style", "noscript", "svg", "template"})
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "dd",
        "details",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "summary",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
)
_WHITESPACE = re.compile(r"[ \t\f\v\xa0]+")


class _TextExtractor(HTMLParser):
    """Collect visible text and the document title from HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._title_chunks: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
            return
        if tag == "br" or tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS or self._skip_depth:
            return
        if tag == "br" or tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title_chunks.append(data)
            return
        self._chunks.append(data)

    def title(self) -> str | None:
        title = _WHITESPACE.sub(" ", "".join(self._title_chunks)).strip()
        return title or None

    def text(self) -> str:
        lines = [
            _WHITESPACE.sub(" ", line).strip()
            for line in "".join(self._chunks).splitlines()
        ]
        text = "\n".join(lines)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


def html_to_text(html: str) -> tuple[str | None, str]:
    """Convert ``html`` to ``(title, text)``.

    Script/style/noscript/svg content is dropped, block tags become newlines,
    entities are decoded, and runs of more than two blank lines collapse to
    one blank line. The title is ``None`` when the document has none.
    """
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # HTMLParser is forgiving; never break a tool call
        pass
    return parser.title(), parser.text()


# ---------------------------------------------------------------------------
# Local / private address detection
# ---------------------------------------------------------------------------


def _host_of(url: str) -> str:
    raw = str(url).strip()
    if not raw:
        return ""
    # ``urlsplit`` reads "localhost:8080" as scheme="localhost"; retry as a
    # host-relative form when there is no explicit scheme separator.
    parsed = urlsplit(raw if "://" in raw else f"//{raw}")
    return (parsed.hostname or "").strip().lower().rstrip(".")


def is_local_address(url: str) -> bool:
    """Whether ``url`` points at localhost, a private IP, or a ``.local`` name."""
    host = _host_of(url)
    if not host:
        return False
    if host in {"localhost", "localhost.localdomain"} or host.endswith(
        (".local", ".localhost")
    ):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_reserved
        or address.is_unspecified
    )


def clean_url(url: str) -> str:
    """Validate an http(s) URL, raising :class:`WebError` for anything else."""
    cleaned = str(url).strip()
    if not cleaned:
        raise WebError("A non-empty URL is required.")
    parsed = urlsplit(cleaned)
    if parsed.scheme not in {"http", "https"}:
        raise WebError("Only http:// and https:// URLs are supported.")
    if not parsed.hostname:
        raise WebError(f"Could not parse a host from '{cleaned}'.")
    return cleaned


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _mime_type(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def _is_textual(mime: str) -> bool:
    if not mime:
        return True
    if mime.startswith("text/"):
        return True
    if mime in {
        "application/json",
        "application/ld+json",
        "application/xml",
        "application/xhtml+xml",
        "application/javascript",
        "application/x-javascript",
        "application/rss+xml",
        "application/atom+xml",
    }:
        return True
    return mime.endswith(("+json", "+xml"))


def _is_html(mime: str) -> bool:
    return mime in {"text/html", "application/xhtml+xml"} or mime.endswith("+html")


def _decode(body: bytes, charset: str) -> str:
    try:
        return body.decode(charset or "utf-8", errors="replace")
    except (LookupError, TypeError):
        return body.decode("utf-8", errors="replace")


def fetch_url(
    url: str,
    *,
    timeout: float,
    max_chars: int,
    allow_local: bool,
    transport: httpx.BaseTransport | None = None,
) -> tuple[str, str]:
    """Fetch ``url`` and return ``(final_url, text)``.

    Redirects are followed; the body is read up to a byte cap derived from
    ``max_chars``. HTML is converted to text, ``text/*`` is decoded, and
    anything else is summarized as ``[binary content: <type>, <n> bytes]``.
    Failures raise :class:`WebError` (never a raw ``httpx`` exception).
    """
    cleaned = clean_url(url)
    if is_local_address(cleaned) and not allow_local:
        raise WebError(
            "Refusing to fetch a local or private address. Enable it with "
            "/config web allow-local on (or config web allow-local on)."
        )

    byte_cap = max(1024, int(max_chars) * MAX_BYTES_PER_CHAR)
    final_url = cleaned
    mime = ""
    charset = "utf-8"
    body = b""
    try:
        with httpx.Client(
            timeout=timeout, transport=transport, follow_redirects=True
        ) as client:
            with client.stream(
                "GET",
                cleaned,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": (
                        "text/html,application/xhtml+xml,text/plain;q=0.9,"
                        "*/*;q=0.5"
                    ),
                },
            ) as response:
                final_url = str(response.url)
                if response.status_code >= 400:
                    raise WebError(
                        f"HTTP {response.status_code} when fetching {final_url}."
                    )
                mime = _mime_type(response.headers.get("content-type", ""))
                charset = response.charset_encoding or "utf-8"
                data = bytearray()
                for chunk in response.iter_bytes():
                    data.extend(chunk)
                    if len(data) >= byte_cap:
                        del data[byte_cap:]
                        break
                body = bytes(data)
    except WebError:
        raise
    except httpx.HTTPError as exc:
        raise WebError(
            f"Could not fetch {cleaned}: {exc.__class__.__name__}."
        ) from exc

    if not _is_textual(mime):
        return final_url, f"[binary content: {mime or 'unknown'}, {len(body)} bytes]"

    text = _decode(body, charset)
    if _is_html(mime):
        _title, text = html_to_text(text)
    return final_url, text


# ---------------------------------------------------------------------------
# Search (DuckDuckGo HTML endpoint)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """One search hit."""

    title: str
    url: str
    snippet: str = ""


class _DDGParser(HTMLParser):
    """Pull ``result__a`` / ``result__snippet`` / ``result__url`` blocks out of
    the DuckDuckGo HTML results page."""

    _FIELDS = {"title", "snippet", "display"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture: str | None = None
        self._depth = 0

    def _start_capture(self, field: str) -> None:
        self._capture = field
        self._depth = 1

    def _finish(self) -> None:
        if self._current is not None:
            self.results.append(self._current)
        self._current = None
        self._capture = None
        self._depth = 0

    def finish(self) -> None:
        """Flush the final in-progress result block."""
        self._finish()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._capture is not None:
            self._depth += 1
        if tag != "a":
            return
        attributes = {key.lower(): value or "" for key, value in attrs}
        classes = set(attributes.get("class", "").split())
        if "result__a" in classes:
            self._finish()
            self._current = {
                "title": "",
                "url": attributes.get("href", ""),
                "display": "",
                "snippet": "",
            }
            self._start_capture("title")
        elif self._current is not None and "result__snippet" in classes:
            self._start_capture("snippet")
        elif self._current is not None and "result__url" in classes:
            self._start_capture("display")

    def handle_endtag(self, tag: str) -> None:
        if self._capture is None:
            return
        if tag == "a" and self._depth <= 1:
            self._capture = None
            self._depth = 0
            return
        self._depth = max(0, self._depth - 1)
        if self._depth == 0:
            self._capture = None

    def handle_data(self, data: str) -> None:
        if self._capture is None or self._current is None:
            return
        if self._capture in self._FIELDS:
            self._current[self._capture] += data


def _collapse(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _unwrap_ddg_url(raw: str) -> str:
    """Undo DuckDuckGo's ``/l/?uddg=...`` click-tracking wrapper."""
    url = raw.strip()
    if not url:
        return ""
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlsplit(url)
    if parsed.path.startswith("/l/") or "uddg=" in parsed.query:
        values = parse_qs(parsed.query).get("uddg")
        if values and values[0]:
            value = values[0]
            return value if value.startswith(("http://", "https://")) else unquote(value)
    return url


def parse_ddg_results(html: str, limit: int = DEFAULT_SEARCH_COUNT) -> list[SearchResult]:
    """Parse a DuckDuckGo HTML results page into :class:`SearchResult`s."""
    parser = _DDGParser()
    parser.feed(html)
    parser.close()
    parser.finish()

    results: list[SearchResult] = []
    for raw in parser.results:
        title = _collapse(raw.get("title", ""))
        url = _unwrap_ddg_url(raw.get("url") or raw.get("display", ""))
        if not title or not url:
            continue
        results.append(
            SearchResult(title=title, url=url, snippet=_collapse(raw.get("snippet", "")))
        )
        if len(results) >= max(0, limit):
            break
    return results


def search_web(
    query: str,
    *,
    count: int = DEFAULT_SEARCH_COUNT,
    timeout: float,
    provider: str,
    api_key: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> list[SearchResult]:
    """Search the web and return up to ``count`` results.

    Only the DuckDuckGo HTML endpoint is supported for now; the provider and
    optional API key are parameters so alternate backends can slot in later.
    Failures raise :class:`WebError`.
    """
    del api_key  # reserved for providers that need a key
    cleaned = str(query).strip()
    if not cleaned:
        raise WebError("Search query must not be empty.")
    if provider != "duckduckgo":
        raise WebError(
            f"Unknown search provider '{provider}'. Only 'duckduckgo' is supported."
        )
    limit = max(1, min(int(count), MAX_SEARCH_COUNT))
    try:
        with httpx.Client(
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
        ) as client:
            response = client.get(DDG_HTML_URL, params={"q": cleaned})
    except httpx.HTTPError as exc:
        raise WebError(f"Web search failed: {exc.__class__.__name__}.") from exc
    if response.status_code != 200:
        raise WebError(f"Search provider returned HTTP {response.status_code}.")
    return parse_ddg_results(response.text, limit=limit)
