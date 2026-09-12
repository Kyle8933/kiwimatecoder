"""Newline-delimited JSON-RPC 2.0 framing for the ACP server.

ACP messages are JSON-RPC 2.0 objects sent one per line over stdio, so this
module is a tiny, transport-agnostic codec: :func:`encode` renders a message
as a terminated line and :func:`decode_lines` incrementally consumes a buffer,
returning the messages found so far plus whatever partial line is left over.
Malformed lines are dropped rather than raising, so a broken client can never
take the server down; the server reports a parse error before falling back to
this codec (see :meth:`kiwimatecoder.acp.server.AcpServer._handle_line`).

Everything here is pure and synchronous; no I/O or asyncio.
"""

from __future__ import annotations

import json
from typing import Any

# Standard JSON-RPC 2.0 error codes plus the server-defined range.
ERROR_PARSE = -32700
ERROR_INVALID_REQUEST = -32600
ERROR_METHOD_NOT_FOUND = -32601
ERROR_INVALID_PARAMS = -32602
ERROR_INTERNAL = -32603
ERROR_SERVER = -32000


class AcpError(Exception):
    """A JSON-RPC error that should be returned to the caller."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class AcpTimeoutError(TimeoutError):
    """Raised when a request to the client is not answered in time."""


class IdAllocator:
    """Monotonic request-id source for agent -> client requests."""

    def __init__(self, start: int = 0) -> None:
        self._value = start

    def next(self) -> int:
        self._value += 1
        return self._value


def encode(message: Any) -> str:
    """Render ``message`` as one newline-terminated JSON line."""
    return json.dumps(message, separators=(",", ":"), ensure_ascii=False) + "\n"


def decode_lines(buffer: str) -> tuple[list[dict[str, Any]], str]:
    """Split complete JSON lines out of ``buffer``.

    Returns ``(messages, remainder)`` where ``remainder`` is the unterminated
    tail (empty when the buffer ended on a newline). Non-object and malformed
    lines are skipped so one bad message cannot poison the stream.
    """
    messages: list[dict[str, Any]] = []
    remainder = buffer
    while "\n" in remainder:
        line, remainder = remainder.split("\n", 1)
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            messages.append(payload)
    return messages, remainder


def request_message(
    request_id: int, method: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build a JSON-RPC request object."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params if params is not None else {},
    }


def notification_message(
    method: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build a JSON-RPC notification (no ``id``, no response expected)."""
    message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        message["params"] = params
    return message


def success_response(request_id: Any, result: Any) -> dict[str, Any]:
    """Build a JSON-RPC success response."""
    return {"jsonrpc": "2.0", "id": request_id, "result": result if result is not None else {}}


def error_response(
    request_id: Any, code: int, message: str, data: Any = None
) -> dict[str, Any]:
    """Build a JSON-RPC error response."""
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}
