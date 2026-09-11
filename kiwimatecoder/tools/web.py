"""web_fetch and web_search tools: read the web with citation-friendly output."""

from __future__ import annotations

from typing import Any

from kiwimatecoder import config
from kiwimatecoder import web as web_module
from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult

DEFAULT_FETCH_MAX_CHARS = 50_000
DEFAULT_SEARCH_COUNT = 5
MAX_SEARCH_COUNT = 10


def _fetch_max_chars(args: dict[str, Any], default: int) -> tuple[int, str | None]:
    raw = args.get("max_chars")
    if raw is None:
        return default, None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default, "'max_chars' must be an integer"
    return max(config.WEB_MAX_CHARS_MIN, min(value, config.WEB_MAX_CHARS_MAX)), None


def _web_fetch(args: dict[str, Any], session: Session) -> ToolResult:
    del session  # fetching does not depend on the workspace
    url = str(args.get("url") or "").strip()
    if not url:
        return ToolResult.error("'url' is required")
    settings = config.get_web()
    max_chars, error = _fetch_max_chars(args, settings["max_chars"])
    if error:
        return ToolResult.error(error)
    try:
        final_url, text = web_module.fetch_url(
            url,
            timeout=settings["timeout"],
            max_chars=max_chars,
            allow_local=settings["allow_local"],
        )
    except web_module.WebError as exc:
        return ToolResult.error(str(exc))

    if not text.strip():
        return ToolResult(
            content=f"Fetched {final_url} but found no readable text content."
        )
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + f"\n\n[truncated at {max_chars} characters]"
    header = f"URL: {final_url}\n\n" if final_url != url else ""
    return ToolResult(content=header + text)


def _web_search(args: dict[str, Any], session: Session) -> ToolResult:
    del session
    query = str(args.get("query") or "").strip()
    if not query:
        return ToolResult.error("'query' is required")
    raw_count = args.get("count")
    try:
        count = int(raw_count) if raw_count is not None else DEFAULT_SEARCH_COUNT
    except (TypeError, ValueError):
        return ToolResult.error("'count' must be an integer")
    count = max(1, min(count, MAX_SEARCH_COUNT))

    settings = config.get_web()
    try:
        results = web_module.search_web(
            query,
            count=count,
            timeout=settings["timeout"],
            provider=settings["search_provider"],
            api_key=settings["search_api_key"] or None,
        )
    except web_module.WebError as exc:
        return ToolResult.error(str(exc))

    if not results:
        return ToolResult(content=f"No results found for {query!r}.")
    lines = [f"Results for {query!r} ({len(results)}):", ""]
    for index, result in enumerate(results, 1):
        lines.append(f"{index}. {result.title}")
        lines.append(f"   {result.url}")
        if result.snippet:
            lines.append(f"   {result.snippet}")
        lines.append("")
    lines.append("Cite the URLs you use; prefer official documentation over mirrors.")
    return ToolResult(content="\n".join(lines).rstrip())


web_fetch_tool = FunctionTool(
    name="web_fetch",
    description=(
        "Fetch a web page and return its readable text. Use it to consult "
        "official documentation and cite the URL in your answer. Prefer "
        "official docs and primary sources over aggregator pages. HTML is "
        "converted to text; binary files are summarized. Local and private "
        "addresses are blocked unless enabled in /config web."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Absolute http(s) URL to fetch.",
            },
            "max_chars": {
                "type": "integer",
                "description": (
                    "Maximum characters to return (defaults to the configured "
                    "limit)."
                ),
            },
        },
        "required": ["url"],
    },
    func=_web_fetch,
)

web_search_tool = FunctionTool(
    name="web_search",
    description=(
        "Search the web and return result titles, URLs, and snippets. Use it "
        "to find current information, then fetch the most promising URLs. "
        "Always cite the URLs you rely on and prefer official documentation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query.",
            },
            "count": {
                "type": "integer",
                "description": f"Number of results (1-{MAX_SEARCH_COUNT}, default 5).",
            },
        },
        "required": ["query"],
    },
    func=_web_search,
)
