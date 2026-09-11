"""Minimal JSON-RPC 2.0 and MCP content helpers.

Only the subset the client needs is implemented: request/notification
construction, response validation, SSE framing, and flattening MCP content
blocks into the plain text the agent consumes. The JSON-RPC method names and
result shapes follow the Model Context Protocol revision named in
:data:`MCP_PROTOCOL_VERSION`.
"""

from __future__ import annotations

import json
from typing import Any

from kiwimatecoder import __version__
from kiwimatecoder.mcp.errors import McpError

MCP_PROTOCOL_VERSION = "2025-06-18"

CLIENT_INFO = {"name": "kiwimatecoder", "version": __version__}


def request_message(
    request_id: int, method: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build a JSON-RPC request object."""
    message: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params if params is not None else {},
    }
    return message


def notification_message(
    method: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build a JSON-RPC notification (no ``id``, no response expected)."""
    message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        message["params"] = params
    return message


def parse_response(message: dict[str, Any], method: str) -> Any:
    """Return a JSON-RPC result or raise :class:`McpError` for its error."""
    error = message.get("error")
    if isinstance(error, dict):
        detail = str(error.get("message") or error)
        code = error.get("code")
        suffix = f" (code {code})" if code is not None else ""
        raise McpError(f"{method} failed: {detail}{suffix}")
    if "result" not in message:
        raise McpError(f"{method}: malformed JSON-RPC response")
    return message["result"]


def parse_sse_messages(text: str) -> list[dict[str, Any]]:
    """Extract JSON-RPC objects from an SSE body's ``data:`` lines."""
    messages: list[dict[str, Any]] = []
    data_lines: list[str] = []

    def flush() -> None:
        if not data_lines:
            return
        payload = "\n".join(data_lines)
        data_lines.clear()
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            return
        if isinstance(decoded, dict):
            messages.append(decoded)

    for line in text.splitlines():
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif not line.strip():
            flush()
    flush()
    return messages


def _flatten_block(block: Any) -> str:
    if not isinstance(block, dict):
        return str(block)
    kind = str(block.get("type") or "")
    if kind == "text":
        return str(block.get("text") or "")
    if kind == "image":
        return "[image content]"
    if kind == "audio":
        return "[audio content]"
    if kind == "resource":
        resource = block.get("resource")
        if isinstance(resource, dict) and resource.get("text") is not None:
            return str(resource["text"])
        return "[resource content]"
    if kind == "resource_link":
        return str(block.get("uri") or "[resource link]")
    return f"[{kind} content]" if kind else "[content]"


def flatten_content(content: Any) -> str:
    """Render an MCP ``content`` list as plain text for the agent."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    return "\n".join(part for part in (_flatten_block(item) for item in content) if part)


def flatten_resource_contents(contents: Any) -> str:
    """Render an MCP ``resources/read`` contents list as plain text."""
    if contents is None:
        return ""
    if not isinstance(contents, list):
        return flatten_content(contents)
    parts: list[str] = []
    for item in contents:
        if not isinstance(item, dict):
            parts.append(str(item))
        elif item.get("text") is not None:
            parts.append(str(item["text"]))
        elif item.get("blob") is not None:
            parts.append("[binary resource]")
        else:
            parts.append("[resource content]")
    return "\n".join(part for part in parts if part)
