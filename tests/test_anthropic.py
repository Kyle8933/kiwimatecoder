from __future__ import annotations

from typing import Any

import pytest

from kiwimatecoder.client import (
    Done,
    ProviderError,
    TextDelta,
    ToolCallDelta,
    Usage,
    format_anthropic_messages,
    format_anthropic_tools,
    parse_anthropic_sse_event,
)


def test_format_anthropic_messages():
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are Kiwi."},
        {"role": "user", "content": "Check file"},
        {
            "role": "assistant",
            "content": "Running read...",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "a.txt"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "file text"},
    ]

    system_prompt, converted = format_anthropic_messages(messages)
    assert system_prompt == "You are Kiwi."
    assert len(converted) == 3
    assert converted[0] == {"role": "user", "content": "Check file"}
    assert converted[1]["role"] == "assistant"
    assert converted[1]["content"][0] == {
        "type": "text",
        "text": "Running read...",
    }
    assert converted[1]["content"][1] == {
        "type": "tool_use",
        "id": "c1",
        "name": "read_file",
        "input": {"path": "a.txt"},
    }
    assert converted[2]["role"] == "user"
    assert converted[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "c1",
        "content": "file text",
    }


def test_format_anthropic_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
    ]
    anth_tools = format_anthropic_tools(tools)
    assert anth_tools == [
        {
            "name": "read_file",
            "description": "Read file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        }
    ]


def test_parse_anthropic_message_start():
    data = '{"type": "message_start", "message": {"usage": {"input_tokens": 15, "output_tokens": 2}}}'
    events = parse_anthropic_sse_event(None, data)
    assert events == [Usage(prompt_tokens=15, completion_tokens=2)]


def test_parse_anthropic_content_block_start_tool_use():
    data = '{"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "tu_1", "name": "search"}}'
    events = parse_anthropic_sse_event(None, data)
    assert events == [
        ToolCallDelta(index=0, id="tu_1", name="search", args_fragment="")
    ]


def test_parse_anthropic_content_block_delta_text_and_json():
    t_data = '{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}'
    assert parse_anthropic_sse_event(None, t_data) == [TextDelta(text="Hello")]

    j_data = '{"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\\"path\\":"}}'
    assert parse_anthropic_sse_event(None, j_data) == [
        ToolCallDelta(index=0, args_fragment='{"path":')
    ]


def test_parse_anthropic_message_delta():
    data = '{"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 25}}'
    events = parse_anthropic_sse_event(None, data)
    assert Usage(prompt_tokens=0, completion_tokens=25) in events
    assert Done(finish_reason="end_turn") in events


def test_parse_anthropic_error():
    data = '{"type": "error", "error": {"type": "invalid_request_error", "message": "Bad token"}}'
    with pytest.raises(ProviderError) as exc_info:
        parse_anthropic_sse_event(None, data)
    assert "Bad token" in str(exc_info.value)
