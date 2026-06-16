import pytest

from kiwimatecoder import providers


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
