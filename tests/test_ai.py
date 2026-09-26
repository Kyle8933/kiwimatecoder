from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from rich.console import Console

from kiwimatecoder import config
from kiwimatecoder.ai import stream_response
from kiwimatecoder.client import Done, ProviderError, TextDelta
from kiwimatecoder.providers import REGISTRY, ProviderConfig
from tests.conftest import track_console


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    """Point config storage at a temp dir and clear provider env vars."""
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for provider in REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)


@pytest.mark.anyio
async def test_stream_response_success():
    events = [TextDelta(text="Code answer"), Done()]

    async def mock_stream(*args, **kwargs):
        for e in events:
            yield e

    provider = ProviderConfig(
        id="test",
        name="Test",
        base_url="https://api.test.com/v1",
        key_env="TEST_KEY",
    )

    with patch(
        "kiwimatecoder.client.UnifiedClient.stream_chat",
        side_effect=mock_stream,
    ):
        # Should complete without error
        await stream_response("How to print?", "key123", "m1", provider)


@pytest.mark.anyio
async def test_stream_response_error_handled():
    async def mock_stream(*args, **kwargs):
        raise ProviderError("Rate limit exceeded")
        yield Done()

    provider = ProviderConfig(
        id="test",
        name="Test",
        base_url="https://api.test.com/v1",
        key_env="TEST_KEY",
    )

    with patch(
        "kiwimatecoder.client.UnifiedClient.stream_chat",
        side_effect=mock_stream,
    ):
        await stream_response("Prompt", "key123", "m1", provider)


@pytest.mark.anyio
async def test_stream_response_shows_thinking_until_first_text():
    """The one-shot ask path must hide Thinking before the first token prints."""
    events = [TextDelta(text="Code answer"), Done()]

    async def mock_stream(*args, **kwargs):
        for event in events:
            yield event

    provider = ProviderConfig(
        id="test",
        name="Test",
        base_url="https://api.test.com/v1",
        key_env="TEST_KEY",
    )
    console = Console(file=io.StringIO(), force_terminal=False, width=120)
    log = track_console(console)

    with (
        patch("kiwimatecoder.ai.console", console),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await stream_response("How to print?", "key123", "m1", provider)

    start_idx = next(
        (
            i
            for i, event in enumerate(log)
            if event[0] == "start" and "Thinking" in event[1]
        ),
        None,
    )
    stop_idx = next(
        (
            i
            for i, event in enumerate(log)
            if event[0] == "stop" and "Thinking" in event[1]
        ),
        None,
    )
    text_idx = next(
        (
            i
            for i, event in enumerate(log)
            if event[0] == "print" and "Code answer" in event[1]
        ),
        None,
    )
    assert start_idx is not None, f"expected Thinking status, got {log}"
    assert stop_idx is not None, f"expected Thinking status to stop, got {log}"
    assert text_idx is not None, f"expected streamed text, got {log}"
    assert start_idx < stop_idx < text_idx


@pytest.mark.anyio
async def test_stream_response_stops_thinking_before_error():
    """A failed ask must stop Thinking before the error line prints."""

    async def mock_stream(*args, **kwargs):
        raise ProviderError("Rate limit exceeded")
        yield Done()

    provider = ProviderConfig(
        id="test",
        name="Test",
        base_url="https://api.test.com/v1",
        key_env="TEST_KEY",
    )
    console = Console(file=io.StringIO(), force_terminal=False, width=120)
    log = track_console(console)

    with (
        patch("kiwimatecoder.ai.console", console),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=mock_stream,
        ),
    ):
        await stream_response("Prompt", "key123", "m1", provider)

    stop_idx = next(
        (
            i
            for i, event in enumerate(log)
            if event[0] == "stop" and "Thinking" in event[1]
        ),
        None,
    )
    error_idx = next(
        (
            i
            for i, event in enumerate(log)
            if event[0] == "print" and "Rate limit exceeded" in event[1]
        ),
        None,
    )
    assert stop_idx is not None, f"expected Thinking status to stop, got {log}"
    assert error_idx is not None, f"expected error line, got {log}"
    assert stop_idx < error_idx


@pytest.mark.anyio
async def test_stream_response_passes_prompt_cache():
    config.set_prompt_cache(True)

    captured: dict = {}

    class FakeClient:
        def __init__(self, provider, api_key, **kwargs):
            captured.update(kwargs)

        async def stream_chat(self, messages, tools, model):
            yield Done()

    provider = ProviderConfig(
        id="test",
        name="Test",
        base_url="https://api.test.com/v1",
        key_env="TEST_KEY",
    )

    with patch("kiwimatecoder.ai.UnifiedClient", FakeClient):
        await stream_response("Prompt", "key123", "m1", provider)

    assert captured["prompt_cache"] is True


@pytest.mark.anyio
async def test_stream_response_without_a_model_prints_how_to_choose_one(monkeypatch):
    constructed: list[object] = []

    class ForbiddenClient:
        def __init__(self, *args, **kwargs):
            constructed.append(args)
            raise AssertionError("no client may be built without a model")

    def forbid_fetch(provider, api_key=None, **kwargs):
        raise AssertionError("the network must not be touched here")

    monkeypatch.setattr(config.catalog, "fetch_models", forbid_fetch)
    console = Console(file=io.StringIO(), force_terminal=False, width=400)

    with (
        patch("kiwimatecoder.ai.console", console),
        patch("kiwimatecoder.ai.UnifiedClient", ForbiddenClient),
    ):
        await stream_response("Prompt", "key123", None, REGISTRY["openai"])

    output = console.file.getvalue()
    assert config.no_model_message(REGISTRY["openai"]) in output
    assert "No model chosen for OpenAI" in output
    assert constructed == []


@pytest.mark.anyio
async def test_stream_response_uses_the_chosen_model_when_none_is_passed():
    config.set_provider_model("openai", "gpt-chosen")
    models: list[str] = []

    class FakeClient:
        def __init__(self, provider, api_key, **kwargs):
            pass

        async def stream_chat(self, messages, tools, model):
            models.append(model)
            yield Done()

    with patch("kiwimatecoder.ai.UnifiedClient", FakeClient):
        await stream_response("Prompt", "key123", None, REGISTRY["openai"])
        await stream_response("Prompt", "key123", "gpt-explicit", REGISTRY["openai"])

    assert models == ["gpt-chosen", "gpt-explicit"]
