import asyncio
import copy
import json
from datetime import date

import httpx
import pytest

from kiwimatecoder import network, telemetry
from kiwimatecoder.client import (
    Done,
    ProviderError,
    TextDelta,
    ToolCallAssembler,
    ToolCallDelta,
    UnifiedClient,
    Usage,
    flatten_tool_messages,
    format_anthropic_messages,
    limit_images,
    parse_sse_chunk,
)
from kiwimatecoder.providers import REGISTRY, ProviderConfig


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


# ---------------------------------------------------------------------------
# Custom auth headers and Azure-style api-version
# ---------------------------------------------------------------------------


def _azure_like(**overrides):
    fields = {
        "id": "my-azure",
        "name": "My Azure",
        "base_url": "https://my-resource.openai.azure.com/openai/v1",
        "key_env": "AZURE_OPENAI_API_KEY",
        "key_header": "api-key",
        "key_prefix": "",
        "api_version": "2024-10-21",
    }
    fields.update(overrides)
    return ProviderConfig(**fields)


def test_headers_emit_configured_api_key_header_without_bearer():
    client = UnifiedClient(_azure_like(), "secret-key")

    headers = client._headers()

    assert headers["api-key"] == "secret-key"
    assert "Authorization" not in headers


def test_headers_keep_bearer_for_default_providers():
    client = UnifiedClient(REGISTRY["deepseek"], "sk-test")

    headers = client._headers()

    assert headers["Authorization"] == "Bearer sk-test"
    assert "api-key" not in headers


def test_headers_azure_registry_entry_uses_api_key():
    client = UnifiedClient(REGISTRY["azure"], "az-key")

    assert client._headers()["api-key"] == "az-key"
    assert "Authorization" not in client._headers()


def test_headers_custom_prefix_is_used_verbatim():
    client = UnifiedClient(_azure_like(key_prefix="Token "), "abc")

    assert client._headers()["api-key"] == "Token abc"


def test_url_appends_api_version_query():
    client = UnifiedClient(_azure_like(), "secret-key")

    assert client._url == (
        "https://my-resource.openai.azure.com/openai/v1/chat/completions"
        "?api-version=2024-10-21"
    )


def test_url_unchanged_without_api_version():
    client = UnifiedClient(REGISTRY["openai"], "sk-test")

    assert client._url == "https://api.openai.com/v1/chat/completions"


def test_anthropic_headers_ignore_custom_auth_fields():
    custom = ProviderConfig(
        id="claude-proxy",
        name="Claude proxy",
        base_url="https://proxy.example.com/v1",
        key_env="CLAUDE_PROXY_KEY",
        compat="anthropic",
        key_header="api-key",
        key_prefix="",
    )
    client = UnifiedClient(custom, "sk-ant")

    headers = client._headers()

    assert headers["x-api-key"] == "sk-ant"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "api-key" not in headers
    assert "Authorization" not in headers


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


# ---------------------------------------------------------------------------
# Vision content parts
# ---------------------------------------------------------------------------


def test_anthropic_converts_user_content_parts_with_data_url_image():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,QUJD"},
                },
            ],
        }
    ]

    system_prompt, converted = format_anthropic_messages(messages)

    assert system_prompt == ""
    assert converted == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "QUJD",
                    },
                },
            ],
        }
    ]


def test_anthropic_converts_http_image_url_to_url_source():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/shot.png"},
                }
            ],
        }
    ]

    _system, converted = format_anthropic_messages(messages)

    assert converted[0]["content"] == [
        {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/shot.png"},
        }
    ]


def test_anthropic_content_parts_skip_unknown_parts():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "keep"},
                {"type": "input_audio", "data": "..."},
                "not-a-part",
            ],
        }
    ]

    _system, converted = format_anthropic_messages(messages)

    assert converted[0]["content"] == [{"type": "text", "text": "keep"}]


def test_anthropic_assistant_content_parts_keep_tool_use_order():
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "working"}],
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                }
            ],
        }
    ]

    _system, converted = format_anthropic_messages(messages)

    assert converted[0]["content"] == [
        {"type": "text", "text": "working"},
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "read_file",
            "input": {"path": "a.txt"},
        },
    ]


def test_anthropic_leaves_string_user_content_unchanged():
    _system, converted = format_anthropic_messages(
        [{"role": "user", "content": "plain text"}]
    )

    assert converted == [{"role": "user", "content": "plain text"}]


def test_openai_payload_passes_image_content_through_untouched():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,QUJD"},
                },
            ],
        }
    ]
    client = UnifiedClient(REGISTRY["openai"], "sk-test")

    payload = client._payload(messages, None, "gpt-5.6-sol")

    assert payload["messages"] == messages


# ---------------------------------------------------------------------------
# Chat-only providers (KiwiMate)
# ---------------------------------------------------------------------------


def _tool_round_trip() -> list[dict]:
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is in a.txt and b.txt?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "list_dir", "arguments": ""},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "alpha"},
        {"role": "tool", "tool_call_id": "call_2", "content": "b.txt"},
    ]


def test_flatten_turns_tool_round_trip_into_text():
    flattened = flatten_tool_messages(_tool_round_trip())

    assert flattened == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is in a.txt and b.txt?"},
        {
            "role": "assistant",
            "content": '[called tool read_file with {"path": "a.txt"}]\n\n'
            "[called tool list_dir with {}]",
        },
        {
            "role": "user",
            "content": "[result of read_file]\nalpha\n\n[result of list_dir]\nb.txt",
        },
    ]


def test_flatten_keeps_assistant_text_and_strips_tool_keys():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "Let me look.",
            "tool_calls": [
                {"id": "c1", "function": {"name": "grep", "arguments": '{"q": "x"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "match"},
        {"role": "tool", "tool_call_id": "unknown", "content": "orphan"},
    ]

    flattened = flatten_tool_messages(messages)

    assert flattened[1] == {
        "role": "assistant",
        "content": 'Let me look.\n\n[called tool grep with {"q": "x"}]',
    }
    # An unknown tool_call_id falls back to a generic name.
    assert flattened[2]["content"].endswith("[result of tool]\norphan")
    for message in flattened:
        assert set(message) == {"role", "content"}


def test_flatten_truncates_long_tool_results():
    messages = [
        {"role": "user", "content": "read it"},
        {"role": "tool", "tool_call_id": "x", "content": "a" * 20000},
    ]

    content = flatten_tool_messages(messages)[0]["content"]

    assert content.endswith("[result of tool]\n" + "a" * 8000 + "\n…[truncated]")


def test_flatten_truncates_long_arguments_and_encodes_non_string_ones():
    # A write_file call can carry a whole file; it must be cut like a result.
    big = json.dumps({"path": "x.py", "content": "y" * 20000})
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "w", "function": {"name": "write_file", "arguments": big}},
                # Non-string arguments (hand-edited sessions): readable, never a crash.
                {
                    "id": "d",
                    "function": {
                        "name": "note",
                        "arguments": {"q": "héllo", "on": date(2026, 1, 2)},
                    },
                },
            ],
        },
    ]

    content = flatten_tool_messages(messages)[1]["content"]

    assert len(content) < 8200
    assert "\n…[truncated]]" in content
    assert content.endswith('[called tool note with {"q": "héllo", "on": "2026-01-02"}]')


def test_flatten_merges_text_with_image_parts():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}
    messages = [
        {"role": "user", "content": "Look at this:"},
        {"role": "user", "content": [{"type": "text", "text": "the shot"}, image]},
    ]

    flattened = flatten_tool_messages(messages)

    assert flattened == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this:"},
                {"type": "text", "text": "the shot"},
                image,
            ],
        }
    ]


def test_flatten_drops_empty_assistant_and_appends_continue():
    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": None},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "partial repl"},
    ]

    flattened = flatten_tool_messages(messages)

    assert flattened == [
        {"role": "user", "content": "one\n\ntwo"},
        {"role": "assistant", "content": "partial repl"},
        {"role": "user", "content": "Continue."},
    ]


def test_flatten_maps_developer_to_system_and_never_merges_systems():
    messages = [
        {"role": "system", "content": "base"},
        {"role": "developer", "content": [{"type": "text", "text": "extra"}]},
        {"role": "user", "content": "hi"},
    ]

    flattened = flatten_tool_messages(messages)

    assert flattened == [
        {"role": "system", "content": "base"},
        {"role": "system", "content": "extra"},
        {"role": "user", "content": "hi"},
    ]
    assert flatten_tool_messages([{"role": "system", "content": "only"}]) == [
        {"role": "system", "content": "only"}
    ]


def test_flatten_does_not_mutate_input():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}
    messages = [
        *_tool_round_trip(),
        {"role": "user", "content": [{"type": "text", "text": "and this"}, image]},
        {"role": "user", "content": [image]},
        {"role": "assistant", "content": "ok"},
    ]
    before = copy.deepcopy(messages)

    flattened = flatten_tool_messages(messages)
    # Mutating the output must not reach back into the input either.
    for message in flattened:
        if isinstance(message["content"], list):
            message["content"].append({"type": "text", "text": "later"})

    assert messages == before


def test_payload_for_chat_only_provider_flattens_and_omits_tools():
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    client = UnifiedClient(REGISTRY["kiwimate"], "sk-km-test")

    payload = client._payload(_tool_round_trip(), tools, "kiwimate-small-1-0")

    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert payload["messages"] == flatten_tool_messages(_tool_round_trip())
    assert payload["messages"][-1]["role"] == "user"
    assert all(m["role"] in ("system", "user", "assistant") for m in payload["messages"])


def test_payload_for_tool_provider_keeps_tools_and_history():
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    client = UnifiedClient(REGISTRY["openai"], "sk-test")

    payload = client._payload(_tool_round_trip(), tools, "gpt-5.6-sol")

    assert payload["tools"] == tools
    assert payload["tool_choice"] == "auto"
    assert payload["messages"] == _tool_round_trip()


def _isolate_network(monkeypatch):
    """Keep stream_chat off the user's config and telemetry log."""
    monkeypatch.setattr(network, "offline_enabled", lambda: False)
    monkeypatch.setattr(network, "current_options", lambda: {})
    monkeypatch.setattr(telemetry, "log_event", lambda *args, **kwargs: None)


@pytest.mark.parametrize("model", ["", "   "])
def test_stream_chat_without_model_fails_before_any_request(monkeypatch, model):
    def no_http(*args, **kwargs):
        raise AssertionError("stream_chat must not open an HTTP client")

    _isolate_network(monkeypatch)
    monkeypatch.setattr(httpx, "AsyncClient", no_http)
    client = UnifiedClient(REGISTRY["kiwimate"], "sk-km-test")

    async def consume() -> None:
        async for _event in client.stream_chat([{"role": "user", "content": "hi"}], None, model):
            pass

    # Specific enough that the offline-mode refusal (which also names the
    # provider) cannot satisfy it.
    with pytest.raises(ProviderError, match=r"^No model chosen for KiwiMate; .*/model"):
        asyncio.run(consume())


def test_stream_chat_sends_kiwimate_a_request_it_accepts(monkeypatch):
    """End to end over a fake endpoint that enforces KiwiMate's rules."""
    bodies: list[dict] = []

    def kiwimate(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        roles = [m["role"] for m in body["messages"]]
        if {"tool", "developer"} & set(roles) or roles[-1] != "user":
            return httpx.Response(400, json={"error": {"message": f"bad roles {roles}"}})
        sse = (
            'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    _isolate_network(monkeypatch)
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(kiwimate)
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs)
    )
    client = UnifiedClient(REGISTRY["kiwimate"], "sk-km-test")
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    # A partial reply left by a failed stream after tools ran elsewhere.
    messages = [*_tool_round_trip(), {"role": "assistant", "content": "Partial"}]

    async def collect() -> list:
        return [event async for event in client.stream_chat(messages, tools, "kiwimate-small-1-0")]

    events = asyncio.run(collect())

    assert len(bodies) == 1
    assert "tools" not in bodies[0] and "tool_choice" not in bodies[0]
    assert bodies[0]["messages"][-1] == {"role": "user", "content": "Continue."}
    assert TextDelta(text="hi") in events
    assert Usage(prompt_tokens=3, completion_tokens=1) in events
    assert events[-1] == Done(finish_reason=None)


# ---------------------------------------------------------------------------
# Per-provider image limits (KiwiMate: <= 4 per message, data URLs < 3M chars)
# ---------------------------------------------------------------------------


def _image(size: int = 10) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * size}}


def test_limit_images_keeps_the_newest_images_within_the_per_message_cap():
    provider = REGISTRY["kiwimate"]
    images = [_image(index + 1) for index in range(6)]
    messages = [{"role": "user", "content": [{"type": "text", "text": "look"}, *images]}]
    original = copy.deepcopy(messages)

    limited = limit_images(messages, provider)

    parts = limited[0]["content"]
    assert parts[0] == {
        "type": "text",
        "text": "[omitted 2 earlier image(s): KiwiMate accepts at most 4 per message]",
    }
    assert parts[1] == {"type": "text", "text": "look"}
    assert parts[2:] == images[2:]
    assert messages == original


def test_limit_images_drops_oversized_data_urls():
    provider = REGISTRY["kiwimate"]
    small = _image()
    huge = _image(provider.max_image_url_chars)
    messages = [{"role": "user", "content": [huge, small]}]

    limited = limit_images(messages, provider)

    assert limited[0]["content"] == [
        {"type": "text", "text": "[omitted 1 image(s) too large for KiwiMate]"},
        small,
    ]


def test_limit_images_is_a_no_op_without_limits_or_parts():
    provider = REGISTRY["openai"]
    messages = [{"role": "user", "content": [_image() for _ in range(9)]}]
    assert limit_images(messages, provider) is messages

    text_only = [{"role": "user", "content": "hi"}]
    assert limit_images(text_only, REGISTRY["kiwimate"]) == text_only


def test_kiwimate_payload_caps_images_after_merging_unanswered_turns():
    # A failed turn leaves its images behind; flattening merges it with the
    # next turn, which would exceed KiwiMate's cap and fail every later turn.
    first = [_image(index + 1) for index in range(3)]
    second = [_image(index + 10) for index in range(3)]
    history = [
        {"role": "user", "content": [{"type": "text", "text": "a"}, *first]},
        {"role": "user", "content": [{"type": "text", "text": "b"}, *second]},
    ]
    client = UnifiedClient(REGISTRY["kiwimate"], "sk-km-test")

    payload = client._payload(history, None, "kiwimate-small-1-0")

    (message,) = payload["messages"]
    image_urls = [part for part in message["content"] if part.get("type") == "image_url"]
    assert image_urls == [first[2], *second]
    assert "2 earlier image(s)" in message["content"][0]["text"]


def test_openai_payload_keeps_every_image():
    history = [{"role": "user", "content": [_image(index + 1) for index in range(6)]}]
    client = UnifiedClient(REGISTRY["openai"], "sk-test")

    payload = client._payload(history, None, "gpt-5.5")

    assert payload["messages"] is history
