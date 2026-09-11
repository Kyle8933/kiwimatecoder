from __future__ import annotations

import pytest

from kiwimatecoder.pricing import (
    FREE,
    estimate_cost,
    estimate_messages_tokens,
    estimate_text_tokens,
    find_price,
    normalize_model,
)


def test_exact_model_pricing():
    price = find_price("gpt-5.6-sol")

    assert price is not None
    assert price.input_per_mtok > 0
    assert price.output_per_mtok > price.input_per_mtok


def test_openrouter_prefixed_model_resolves_to_bare_id():
    assert find_price("anthropic/claude-sonnet-5") == find_price("claude-sonnet-5")


def test_unknown_model_returns_none():
    assert find_price("totally-made-up-model") is None


def test_local_provider_is_free():
    assert find_price("some-local-gguf", is_local=True) is FREE


def test_free_suffix_is_free():
    assert find_price("meta-llama/llama-3.1-8b:free") is FREE


def test_normalize_model_strips_prefix_and_tag():
    assert normalize_model("openai/gpt-5.6-sol") == "gpt-5.6-sol"
    assert normalize_model("llama3.1:8b") == "llama3.1"
    assert normalize_model("  Grok-4.5 ") == "grok-4.5"


def test_estimate_cost_math():
    price = find_price("gpt-5.6-sol")
    assert price is not None

    cost = estimate_cost(1_000_000, 1_000_000, "gpt-5.6-sol")

    assert cost == pytest.approx(price.input_per_mtok + price.output_per_mtok)


def test_estimate_cost_unknown_is_none():
    assert estimate_cost(100, 100, "totally-made-up-model") is None


def test_estimate_cost_local_is_zero():
    assert estimate_cost(10_000, 10_000, "whatever", is_local=True) == 0.0


def test_estimate_text_tokens_empty_and_scaling():
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("hello world") >= 1
    assert estimate_text_tokens("word " * 100) > estimate_text_tokens("word " * 10)


def test_estimate_messages_tokens_counts_tool_calls():
    plain = estimate_messages_tokens([{"role": "user", "content": "hi"}])
    with_tools = estimate_messages_tokens(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "run_bash",
                            "arguments": '{"command": "pytest -q"}',
                        },
                    }
                ],
            },
        ]
    )

    assert with_tools > plain
