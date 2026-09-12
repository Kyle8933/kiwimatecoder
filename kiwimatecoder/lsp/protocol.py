"""LSP base-protocol framing: ``Content-Length`` headers around JSON payloads.

The codec is pure (no I/O) so it can be unit-tested directly. ``encode_message``
produces one complete frame; ``decode_messages`` consumes a possibly-partial
byte buffer and returns every complete message plus the unparsed remainder, so
a reader can call it again as more bytes arrive.
"""

from __future__ import annotations

import json
from typing import Any

HEADER_SEPARATOR = b"\r\n\r\n"
ALT_HEADER_SEPARATOR = b"\n\n"


class LspProtocolError(ValueError):
    """Raised when a frame header or body cannot be decoded."""


def encode_message(message: dict[str, Any]) -> bytes:
    """Encode ``message`` as one LSP frame (header + UTF-8 JSON body)."""
    body = json.dumps(
        message, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def decode_messages(buffer: bytes) -> tuple[list[dict[str, Any]], bytes]:
    """Decode every complete frame in ``buffer``; return (messages, remainder).

    A partial header or body is left in the remainder for the next call. A
    malformed header or body raises :class:`LspProtocolError`; callers decide
    whether to discard the connection.
    """
    messages: list[dict[str, Any]] = []
    while buffer:
        split = buffer.find(HEADER_SEPARATOR)
        separator_length = len(HEADER_SEPARATOR)
        if split < 0:
            split = buffer.find(ALT_HEADER_SEPARATOR)
            separator_length = len(ALT_HEADER_SEPARATOR)
        if split < 0:
            return messages, buffer  # incomplete header
        header = buffer[:split].decode("ascii", "replace")
        length = _content_length(header)
        body_start = split + separator_length
        if len(buffer) - body_start < length:
            return messages, buffer  # incomplete body
        body = buffer[body_start : body_start + length]
        try:
            message = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LspProtocolError(f"invalid JSON body: {exc}") from exc
        if not isinstance(message, dict):
            raise LspProtocolError("message body must be a JSON object")
        messages.append(message)
        buffer = buffer[body_start + length :]
    return messages, buffer


def _content_length(header: str) -> int:
    """Return the ``Content-Length`` value from a frame header."""
    for line in header.replace("\r\n", "\n").split("\n"):
        name, _, value = line.partition(":")
        if name.strip().lower() != "content-length":
            continue
        try:
            length = int(value.strip())
        except ValueError as exc:
            raise LspProtocolError(
                f"invalid Content-Length: {value.strip()!r}"
            ) from exc
        if length < 0:
            raise LspProtocolError("negative Content-Length")
        return length
    raise LspProtocolError("missing Content-Length header")
