from __future__ import annotations

from unittest.mock import patch

import pytest

from kiwimatecoder.ai import stream_response
from kiwimatecoder.client import Done, ProviderError, TextDelta
from kiwimatecoder.providers import ProviderConfig


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
        default_model="m1",
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
        default_model="m1",
        key_env="TEST_KEY",
    )

    with patch(
        "kiwimatecoder.client.UnifiedClient.stream_chat",
        side_effect=mock_stream,
    ):
        await stream_response("Prompt", "key123", "m1", provider)
