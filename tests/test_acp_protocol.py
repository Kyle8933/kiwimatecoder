"""Unit tests for the ACP newline-delimited JSON-RPC codec."""

from __future__ import annotations

from kiwimatecoder.acp.protocol import (
    ERROR_INTERNAL,
    ERROR_METHOD_NOT_FOUND,
    ERROR_PARSE,
    AcpError,
    AcpTimeoutError,
    IdAllocator,
    decode_lines,
    encode,
    error_response,
    notification_message,
    request_message,
    success_response,
)


def test_encode_decode_roundtrip_is_one_line():
    message = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}

    encoded = encode(message)

    assert encoded.endswith("\n")
    assert encoded.count("\n") == 1
    messages, remainder = decode_lines(encoded)
    assert messages == [message]
    assert remainder == ""


def test_decode_lines_returns_partial_remainder():
    buffer = encode({"id": 1}) + '{"id": 2}'

    messages, remainder = decode_lines(buffer)

    assert messages == [{"id": 1}]
    assert remainder == '{"id": 2}'


def test_decode_lines_handles_several_messages_and_blank_lines():
    buffer = "\n" + encode({"id": 1}) + "\n" + encode({"id": 2})

    messages, remainder = decode_lines(buffer)

    assert messages == [{"id": 1}, {"id": 2}]
    assert remainder == ""


def test_decode_lines_skips_malformed_and_non_object_lines():
    buffer = "not json\n[1, 2]\n" + encode({"ok": True})

    messages, remainder = decode_lines(buffer)

    assert messages == [{"ok": True}]
    assert remainder == ""


def test_id_allocator_is_monotonic_and_resettable():
    allocator = IdAllocator()
    assert [allocator.next() for _ in range(3)] == [1, 2, 3]
    assert IdAllocator(start=10).next() == 11


def test_message_builders_have_jsonrpc_shapes():
    request = request_message(7, "session/prompt", {"a": 1})
    assert request == {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "session/prompt",
        "params": {"a": 1},
    }

    notification = notification_message("session/cancel", {"sessionId": "s"})
    assert notification == {
        "jsonrpc": "2.0",
        "method": "session/cancel",
        "params": {"sessionId": "s"},
    }
    assert "id" not in notification

    assert success_response(7, {"ok": True}) == {
        "jsonrpc": "2.0",
        "id": 7,
        "result": {"ok": True},
    }
    assert success_response(7, None)["result"] == {}

    assert error_response(7, ERROR_METHOD_NOT_FOUND, "nope") == {
        "jsonrpc": "2.0",
        "id": 7,
        "error": {"code": ERROR_METHOD_NOT_FOUND, "message": "nope"},
    }
    assert error_response(None, ERROR_PARSE, "bad")["id"] is None


def test_error_helpers_carry_code_message_and_data():
    error = AcpError(ERROR_INTERNAL, "boom", {"detail": 1})

    assert error.code == ERROR_INTERNAL
    assert error.message == "boom"
    assert error.data == {"detail": 1}
    assert str(error) == "boom"
    assert isinstance(AcpTimeoutError("late"), TimeoutError)
