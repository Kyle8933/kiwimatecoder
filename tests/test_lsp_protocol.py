from __future__ import annotations

import pytest

from kiwimatecoder.lsp.protocol import (
    LspProtocolError,
    decode_messages,
    encode_message,
)


def test_encode_decode_roundtrip():
    message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"root": "prōject"},
    }

    encoded = encode_message(message)

    assert encoded.startswith(b"Content-Length: ")
    assert b"\r\n\r\n" in encoded
    messages, remainder = decode_messages(encoded)
    assert messages == [message]
    assert remainder == b""


def test_multiple_messages_in_one_buffer():
    first = {"jsonrpc": "2.0", "id": 1, "result": None}
    second = {
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {"uri": "file:///x.py"},
    }

    messages, remainder = decode_messages(
        encode_message(first) + encode_message(second)
    )

    assert messages == [first, second]
    assert remainder == b""


def test_partial_buffer_returns_remainder():
    payload = encode_message({"jsonrpc": "2.0", "id": 7, "result": {"ok": True}})

    messages, remainder = decode_messages(payload[:10])
    assert messages == []
    assert remainder == payload[:10]

    messages, remainder = decode_messages(remainder + payload[10:])
    assert messages == [{"jsonrpc": "2.0", "id": 7, "result": {"ok": True}}]
    assert remainder == b""


def test_partial_body_waits_for_more_bytes():
    payload = encode_message({"jsonrpc": "2.0", "id": 3, "result": "x"})
    body_start = payload.index(b"\r\n\r\n") + 4
    partial = payload[: body_start + 2]

    messages, remainder = decode_messages(partial)
    assert messages == []
    assert remainder == partial

    messages, remainder = decode_messages(remainder + payload[len(partial) :])
    assert [message["id"] for message in messages] == [3]
    assert remainder == b""


def test_accepts_lf_only_header_separator():
    body = b'{"jsonrpc":"2.0","id":9,"result":null}'
    frame = b"Content-Length: " + str(len(body)).encode("ascii") + b"\n\n" + body

    messages, remainder = decode_messages(frame)

    assert messages == [{"jsonrpc": "2.0", "id": 9, "result": None}]
    assert remainder == b""


def test_malformed_header_raises():
    with pytest.raises(LspProtocolError):
        decode_messages(b"Not-A-Length: 2\r\n\r\n{}")
    with pytest.raises(LspProtocolError):
        decode_messages(b"Content-Length: nope\r\n\r\n{}")
    with pytest.raises(LspProtocolError):
        decode_messages(b"Content-Length: -1\r\n\r\n")


def test_invalid_json_body_raises():
    with pytest.raises(LspProtocolError):
        decode_messages(b"Content-Length: 1\r\n\r\n{")
