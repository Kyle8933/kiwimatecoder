from kiwimatecoder.client import (
    Done,
    TextDelta,
    ToolCallAssembler,
    ToolCallDelta,
    UnifiedClient,
    Usage,
    parse_sse_chunk,
)
from kiwimatecoder.providers import REGISTRY


def test_parse_text_delta():
    events = parse_sse_chunk('{"choices":[{"delta":{"content":"hi"}}]}')
    assert events == [TextDelta(text="hi")]


def test_parse_done_sentinel():
    events = parse_sse_chunk("[DONE]")
    assert events == [Done(finish_reason=None)]


def test_parse_finish_reason():
    events = parse_sse_chunk('{"choices":[{"delta":{},"finish_reason":"stop"}]}')
    assert Done(finish_reason="stop") in events


def test_parse_usage():
    events = parse_sse_chunk('{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":7}}')
    assert Usage(prompt_tokens=5, completion_tokens=7) in events


def test_parse_malformed_json_yields_nothing():
    assert parse_sse_chunk("{not json") == []


def test_tool_call_assembler_reassembles_fragments():
    asm = ToolCallAssembler()
    asm.add(ToolCallDelta(index=0, id="call_1", name="read_file", args_fragment='{"pa'))
    asm.add(ToolCallDelta(index=0, args_fragment='th":"a.txt"}'))
    calls = asm.finalize()
    assert len(calls) == 1
    assert calls[0].id == "call_1"
    assert calls[0].name == "read_file"
    assert calls[0].parse_arguments() == {"path": "a.txt"}


def test_tool_call_assembler_multiple_calls_ordered():
    asm = ToolCallAssembler()
    asm.add(ToolCallDelta(index=0, id="a", name="read_file", args_fragment="{}"))
    asm.add(ToolCallDelta(index=1, id="b", name="list_dir", args_fragment="{}"))
    calls = asm.finalize()
    assert [c.name for c in calls] == ["read_file", "list_dir"]


def test_assembler_synthesizes_missing_id():
    asm = ToolCallAssembler()
    asm.add(ToolCallDelta(index=2, name="search", args_fragment="{}"))
    calls = asm.finalize()
    assert calls[0].id == "call_2"


def test_parse_tool_call_delta():
    chunk = (
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1",'
        '"function":{"name":"read_file","arguments":"{}"}}]}}]}'
    )
    events = parse_sse_chunk(chunk)
    assert any(isinstance(e, ToolCallDelta) and e.name == "read_file" for e in events)


def test_headers_omit_authorization_when_keyless():
    keyless = UnifiedClient(REGISTRY["ollama"], "")
    assert "Authorization" not in keyless._headers()

    keyed = UnifiedClient(REGISTRY["openai"], "sk-test")
    assert keyed._headers()["Authorization"] == "Bearer sk-test"


def test_payload_omits_tool_choice_for_local_providers():
    tools = [{"type": "function", "function": {"name": "read_file"}}]

    local = UnifiedClient(REGISTRY["ollama"], "")
    local_payload = local._payload([], tools, "llama3.1:8b")
    assert local_payload["tools"] == tools
    assert "tool_choice" not in local_payload

    cloud = UnifiedClient(REGISTRY["openai"], "sk-test")
    assert cloud._payload([], tools, "gpt-5.6-sol")["tool_choice"] == "auto"


def test_payload_defaults_omit_sampling_params():
    client = UnifiedClient(REGISTRY["openai"], "sk-test")

    payload = client._payload([], None, "gpt-5.6-sol")

    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "max_tokens" not in payload
    assert "reasoning_effort" not in payload


def test_payload_applies_openai_sampling_params():
    client = UnifiedClient(
        REGISTRY["openai"],
        "sk-test",
        sampling={
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 4096,
            "reasoning_effort": "medium",
        },
    )

    payload = client._payload([], None, "gpt-5.6-sol")

    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.9
    assert payload["max_tokens"] == 4096
    assert payload["reasoning_effort"] == "medium"


def test_payload_anthropic_uses_sampling_and_max_tokens_default():
    default_client = UnifiedClient(REGISTRY["anthropic"], "sk-test")
    default_payload = default_client._payload(
        [{"role": "user", "content": "hi"}], None, "claude-sonnet-5"
    )
    assert default_payload["max_tokens"] == 8192

    client = UnifiedClient(
        REGISTRY["anthropic"],
        "sk-test",
        sampling={"max_tokens": 2048, "temperature": 0.1, "top_p": 0.8},
    )
    payload = client._payload([{"role": "user", "content": "hi"}], None, "claude-sonnet-5")

    assert payload["max_tokens"] == 2048
    assert payload["temperature"] == 0.1
    assert payload["top_p"] == 0.8


# ---------------------------------------------------------------------------
# Anthropic prompt caching
# ---------------------------------------------------------------------------


def _anthropic_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def test_anthropic_prompt_cache_disabled_keeps_plain_system_and_tools():
    client = UnifiedClient(REGISTRY["anthropic"], "sk-test", prompt_cache=False)

    payload = client._payload(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ],
        _anthropic_tools(),
        "claude-sonnet-5",
    )

    assert payload["system"] == "You are helpful."
    assert "cache_control" not in payload["tools"][-1]


def test_anthropic_prompt_cache_enabled_marks_system_and_last_tool():
    client = UnifiedClient(REGISTRY["anthropic"], "sk-test", prompt_cache=True)

    payload = client._payload(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ],
        _anthropic_tools(),
        "claude-sonnet-5",
    )

    assert payload["system"] == [
        {
            "type": "text",
            "text": "You are helpful.",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert "cache_control" not in payload["tools"][0]
    assert payload["tools"][-1]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_prompt_cache_without_system_omits_system_key():
    client = UnifiedClient(REGISTRY["anthropic"], "sk-test", prompt_cache=True)

    payload = client._payload(
        [{"role": "user", "content": "hi"}], None, "claude-sonnet-5"
    )

    assert "system" not in payload


def test_openai_payload_identical_with_prompt_cache():
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ]

    disabled = UnifiedClient(REGISTRY["openai"], "sk-test")
    enabled = UnifiedClient(REGISTRY["openai"], "sk-test", prompt_cache=True)

    assert disabled._payload(messages, tools, "gpt-5.6-sol") == enabled._payload(
        messages, tools, "gpt-5.6-sol"
    )
