import pytest

from kiwimatecoder import providers
from kiwimatecoder.session import Session


# Newest GA defaults re-verified July 2026 (see plan model research).
EXPECTED_DEFAULT_MODELS = {
    "openai": "gpt-5.6-sol",
    "anthropic": "claude-sonnet-5",
    "google": "gemini-3.5-flash",
    "xai": "grok-4.5",
    "mistral": "mistral-medium-3.5",
    "deepseek": "deepseek-v4-pro",
    "qwen": "qwen3.7-max",
    "moonshot": "kimi-k2.7-code",
    "openrouter": "anthropic/claude-sonnet-5",
}


def test_registry_integrity():
    for pid, p in providers.REGISTRY.items():
        assert p.id == pid
        assert p.base_url.startswith("http")
        assert not p.base_url.endswith("/chat/completions")
        assert p.default_model
        assert p.key_env.isupper()
        assert p.compat in {"openai", "anthropic"}


def test_default_provider_is_openrouter():
    assert providers.default_provider().id == "openrouter"


def test_get_provider_unknown_raises():
    with pytest.raises(KeyError):
        providers.get_provider("does-not-exist")


def test_list_providers_covers_registry():
    assert {p.id for p in providers.list_providers()} == set(providers.REGISTRY)


def test_openrouter_keeps_referer_headers():
    p = providers.get_provider("openrouter")
    assert p.extra_headers.get("X-Title") == "KiwiMateCoder"


def test_registry_default_models_match_newest_ga():
    """Every built-in provider ships the researched newest GA default_model."""
    assert set(providers.REGISTRY) == set(EXPECTED_DEFAULT_MODELS)
    for pid, expected in EXPECTED_DEFAULT_MODELS.items():
        assert providers.REGISTRY[pid].default_model == expected


def test_session_falls_through_to_provider_default_model():
    """No model override → Session uses the provider registry default_model."""
    for pid, expected in EXPECTED_DEFAULT_MODELS.items():
        provider = providers.get_provider(pid)
        session = Session(provider_id=pid, model=provider.default_model)
        assert session.model == expected
        # set_provider without model arg also falls through to default_model
        session.set_provider(pid)
        assert session.model == expected
