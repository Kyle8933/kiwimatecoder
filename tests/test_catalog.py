from typing import Any

import httpx
import pytest

from kiwimatecoder import catalog
from kiwimatecoder.providers import REGISTRY, ProviderConfig


def _transport(handler):
    return httpx.MockTransport(handler)


def _json_transport(payload, status_code=200, seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status_code, json=payload)

    return _transport(handler)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_openai_shape_keeps_chat_models_only():
    models = catalog.parse_models_response(
        {
            "data": [
                {"id": "gpt-9", "created": 200},
                {"id": "text-embedding-4", "created": 300},
                {"id": "whisper-2", "created": 400},
                {"id": "gpt-image-2", "created": 500},
                {"id": "omni-moderation-latest", "created": 600},
            ]
        }
    )

    assert [model.id for model in models] == ["gpt-9"]
    assert models[0].created == 200


def test_parse_handles_bare_lists_strings_and_duplicates():
    models = catalog.parse_models_response(
        ["alpha", {"id": "beta"}, "alpha", 17, {"name": "gamma"}]
    )

    assert [model.id for model in models] == ["alpha", "beta", "gamma"]


def test_parse_strips_google_models_prefix():
    models = catalog.parse_models_response(
        {"data": [{"id": "models/gemini-9-pro", "created": 1}]}
    )

    assert [model.id for model in models] == ["gemini-9-pro"]


def test_parse_reads_anthropic_iso_created_at():
    models = catalog.parse_models_response(
        {
            "data": [
                {"id": "claude-new", "created_at": "2026-05-01T00:00:00Z"},
                {"id": "claude-old", "created_at": "2024-01-01T00:00:00Z"},
                {"id": "claude-undated"},
            ]
        }
    )

    by_id = {model.id: model.created for model in models}
    assert by_id["claude-new"] > by_id["claude-old"] > 0
    assert by_id["claude-undated"] == 0.0


def test_parse_honors_provider_metadata():
    models = catalog.parse_models_response(
        {
            "data": [
                # OpenRouter-style: usable.
                {
                    "id": "vendor/good",
                    "architecture": {
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                    },
                    "supported_parameters": ["tools", "temperature"],
                },
                # No tool calling: the agent loop cannot drive it.
                {
                    "id": "vendor/no-tools",
                    "supported_parameters": ["temperature"],
                },
                # Text in, pictures out.
                {
                    "id": "vendor/painter",
                    "architecture": {"output_modalities": ["image"]},
                    "supported_parameters": ["tools"],
                },
                # Mistral-style capability flags.
                {"id": "vendor/not-chat", "capabilities": {"completion_chat": False}},
            ]
        }
    )

    assert [model.id for model in models] == ["vendor/good"]


def test_parse_ignores_unusable_payloads():
    assert catalog.parse_models_response({"error": "nope"}) == []
    assert catalog.parse_models_response("nonsense") == []
    assert catalog.parse_models_response(None) == []


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def test_fetch_models_uses_bearer_auth_and_models_endpoint():
    seen: list[httpx.Request] = []
    provider = REGISTRY["openrouter"]

    models = catalog.fetch_models(
        provider,
        "sk-test",
        transport=_json_transport({"data": [{"id": "vendor/one"}]}, seen=seen),
    )

    assert [model.id for model in models] == ["vendor/one"]
    request = seen[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/models"
    assert request.headers["Authorization"] == "Bearer sk-test"
    # Provider extra headers still ride along.
    assert request.headers["X-Title"] == "KiwiMateCoder"


def test_fetch_models_uses_anthropic_auth_scheme():
    seen: list[httpx.Request] = []

    catalog.fetch_models(
        REGISTRY["anthropic"],
        "sk-ant",
        transport=_json_transport({"data": [{"id": "claude-x"}]}, seen=seen),
    )

    request = seen[0]
    assert request.headers["x-api-key"] == "sk-ant"
    assert request.headers["anthropic-version"] == catalog.ANTHROPIC_VERSION
    assert "Authorization" not in request.headers
    assert request.url.params["limit"] == "1000"


def test_fetch_models_raises_on_http_error():
    provider = REGISTRY["openai"]
    transport = _json_transport({"error": "bad key"}, status_code=401)

    with pytest.raises(catalog.CatalogFetchError) as exc:
        catalog.fetch_models(provider, "sk-bad", transport=transport)

    assert "401" in str(exc.value)


def test_fetch_models_raises_on_transport_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    with pytest.raises(catalog.CatalogFetchError):
        catalog.fetch_models(REGISTRY["openai"], "sk", transport=_transport(handler))


def test_fetch_models_raises_on_non_json_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>nope</html>")

    with pytest.raises(catalog.CatalogFetchError):
        catalog.fetch_models(REGISTRY["openai"], "sk", transport=_transport(handler))


def test_fetch_models_raises_when_nothing_usable_is_listed():
    transport = _json_transport({"data": [{"id": "text-embedding-9"}]})

    with pytest.raises(catalog.CatalogFetchError):
        catalog.fetch_models(REGISTRY["openai"], "sk", transport=transport)


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def _provider(**overrides: Any) -> ProviderConfig:
    kwargs: dict[str, Any] = {
        "id": "demo",
        "name": "Demo",
        "base_url": "https://demo.test/v1",
        "default_model": "demo-default",
        "key_env": "DEMO_API_KEY",
        "models": ("demo-default", "demo-old"),
    }
    kwargs.update(overrides)
    return ProviderConfig(**kwargs)


def test_merge_orders_default_first_then_newest():
    provider = _provider()
    remote = [
        catalog.RemoteModel("demo-mid", 200),
        catalog.RemoteModel("demo-default", 100),
        catalog.RemoteModel("demo-new", 300),
    ]

    assert catalog.merge_catalog(provider, remote) == [
        "demo-default",
        "demo-new",
        "demo-mid",
    ]


def test_merge_drops_models_the_provider_no_longer_lists():
    provider = _provider()
    remote = [catalog.RemoteModel("demo-default", 1), catalog.RemoteModel("demo-new", 2)]

    merged = catalog.merge_catalog(provider, remote)

    assert "demo-old" not in merged  # curated but deprecated upstream
    assert merged == ["demo-default", "demo-new"]


def test_merge_pins_kept_models_but_cannot_resurrect_retired_ones():
    provider = _provider()
    remote = [
        catalog.RemoteModel("demo-default", 1),
        catalog.RemoteModel("demo-current", 2),
        catalog.RemoteModel("demo-new", 3),
    ]

    merged = catalog.merge_catalog(provider, remote, keep=("demo-current", "gone-model"))

    assert merged[:2] == ["demo-default", "demo-current"]
    assert "gone-model" not in merged


def test_merge_respects_the_catalog_limit():
    provider = _provider(default_model="m0", models=())
    remote = [catalog.RemoteModel(f"m{i}", i) for i in range(10)]

    merged = catalog.merge_catalog(provider, remote, limit=3)

    assert merged == ["m0", "m9", "m8"]


def test_merge_of_empty_listing_is_empty():
    assert catalog.merge_catalog(_provider(), []) == []


def test_search_models_matches_substrings_case_insensitively():
    models = ["anthropic/claude-sonnet-5", "anthropic/claude-opus-4-8", "openai/gpt-5.6-sol"]

    assert catalog.search_models(models, "sonnet") == ["anthropic/claude-sonnet-5"]
    assert catalog.search_models(models, "ANTHROPIC") == [
        "anthropic/claude-sonnet-5",
        "anthropic/claude-opus-4-8",
    ]


def test_search_models_requires_every_token():
    models = ["anthropic/claude-sonnet-5", "anthropic/claude-opus-4-8"]

    assert catalog.search_models(models, "claude opus") == ["anthropic/claude-opus-4-8"]
    assert catalog.search_models(models, "claude nope") == []


def test_search_models_with_blank_query_returns_everything():
    models = ["a-model", "b-model"]
    assert catalog.search_models(models, "   ") == models
    assert catalog.search_models(models, "") == models


# ---------------------------------------------------------------------------
# Server probes
# ---------------------------------------------------------------------------


def test_probe_true_on_200_even_with_no_models_loaded():
    transport = _json_transport({"data": []})
    assert catalog.probe(REGISTRY["ollama"], transport=transport)


def test_probe_false_on_http_error():
    transport = _json_transport({"error": "boom"}, status_code=500)
    assert not catalog.probe(REGISTRY["ollama"], transport=transport)


def test_probe_false_on_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    assert not catalog.probe(REGISTRY["ollama"], transport=_transport(handler))


def test_probe_sends_no_authorization_header_without_a_key():
    seen: list[httpx.Request] = []
    transport = _json_transport({"data": []}, seen=seen)
    catalog.probe(REGISTRY["lmstudio"], transport=transport)
    assert "authorization" not in seen[0].headers


def test_probe_forwards_the_api_key_when_given_one():
    seen: list[httpx.Request] = []
    transport = _json_transport({"data": []}, seen=seen)
    catalog.probe(REGISTRY["unsloth"], api_key="sk-unsloth-test", transport=transport)
    assert seen[0].headers["authorization"] == "Bearer sk-unsloth-test"


def test_probe_counts_401_as_running_for_key_requiring_local():
    """A 401 from Unsloth proves the server is up; only the key is missing."""
    transport = _json_transport({"error": "unauthorized"}, status_code=401)
    assert catalog.probe(REGISTRY["unsloth"], transport=transport)


def test_probe_does_not_count_401_as_running_for_keyless_local():
    transport = _json_transport({"error": "unauthorized"}, status_code=401)
    assert not catalog.probe(REGISTRY["ollama"], transport=transport)


def test_parse_ollama_style_listing_drops_embedding_models():
    models = catalog.parse_models_response(
        {
            "data": [
                {"id": "llama3.1:8b", "created": 200, "owned_by": "library"},
                {"id": "nomic-embed-text", "created": 300, "owned_by": "library"},
                {"id": "qwen3:8b", "created": 100, "owned_by": "library"},
            ]
        }
    )

    assert [model.id for model in models] == ["llama3.1:8b", "qwen3:8b"]
