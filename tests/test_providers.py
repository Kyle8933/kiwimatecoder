import dataclasses

import pytest

from kiwimatecoder import catalog, config, providers
from kiwimatecoder.session import Session


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    """Point config storage at a temp dir, clear provider env vars, no network.

    Resolving a session model reads the user's chosen models from config, so
    these tests must never see the real ~/.kiwimatecoder.
    """
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for provider in providers.REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)

    def fake_fetch(provider, api_key=None, **kwargs):
        raise AssertionError("the network must not be touched here")

    monkeypatch.setattr(config.catalog, "fetch_models", fake_fetch)
    return tmp_path


# Every built-in provider. There are no default models: the user chooses one
# per provider, and the ``models`` tuples are only offline suggestions.
EXPECTED_PROVIDER_IDS = {
    "kiwimate",
    "openai",
    "anthropic",
    "google",
    "xai",
    "mistral",
    "deepseek",
    "qwen",
    "moonshot",
    "openrouter",
    "azure",
    "bedrock",
    "groq",
    "together",
    "fireworks",
    "cerebras",
    "deepinfra",
    "ollama",
    "lmstudio",
    "unsloth",
}

KIWIMATE_BASE_URL = (
    "https://gznrhppouxwpfihlfgpb.supabase.co/functions/v1/api-chat-completions"
)

# Provider id -> key env var for the OpenAI-compatible expansion.
EXPECTED_KEY_ENVS = {
    "azure": "AZURE_OPENAI_API_KEY",
    "bedrock": "AWS_BEARER_TOKEN_BEDROCK",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "deepinfra": "DEEPINFRA_API_KEY",
}


def test_registry_integrity():
    assert set(providers.REGISTRY) == EXPECTED_PROVIDER_IDS
    for pid, p in providers.REGISTRY.items():
        assert p.id == pid
        assert p.base_url.startswith("http")
        assert not p.base_url.endswith("/chat/completions")
        assert not hasattr(p, "default_model")
        assert p.key_env.isupper()
        assert p.compat in {"openai", "anthropic"}


def test_default_provider_is_openrouter():
    # KiwiMate is listed first, but it is experimental: the default stays put.
    assert providers.DEFAULT_PROVIDER_ID == "openrouter"
    assert providers.default_provider().id == "openrouter"


def test_get_provider_unknown_raises():
    with pytest.raises(KeyError):
        providers.get_provider("does-not-exist")


def test_list_providers_covers_registry():
    assert {p.id for p in providers.list_providers()} == set(providers.REGISTRY)


def test_openrouter_keeps_referer_headers():
    p = providers.get_provider("openrouter")
    assert p.extra_headers.get("X-Title") == "KiwiMateCoder"


# ---------------------------------------------------------------------------
# No default models
# ---------------------------------------------------------------------------


def test_no_provider_exposes_a_default_model():
    field_names = {f.name for f in dataclasses.fields(providers.ProviderConfig)}
    assert "default_model" not in field_names
    for provider in providers.list_providers():
        assert not hasattr(provider, "default_model")
    with pytest.raises(TypeError):
        providers.ProviderConfig(  # type: ignore[call-arg]
            id="x", name="X", base_url="https://api.example.com/v1",
            default_model="m", key_env="X_KEY",
        )


def test_every_builtin_cloud_provider_still_ships_suggested_models():
    """The suggestions fill the offline picker; none of them is a default."""
    for provider in providers.list_providers():
        if provider.is_local:
            continue
        assert provider.models, f"{provider.id} should suggest at least one model"
        assert all(model.strip() for model in provider.models)
        assert len(provider.models) == len(set(provider.models))


# Newest GA ids researched July 2026. They used to be per-provider defaults;
# now they are only offered as suggestions, so the user can still pick them.
RESEARCHED_SUGGESTED_MODELS = {
    "openai": "gpt-5.6-sol",
    "anthropic": "claude-sonnet-5",
    "google": "gemini-3.5-flash",
    "xai": "grok-4.5",
    "mistral": "mistral-medium-3.5",
    "deepseek": "deepseek-v4-pro",
    "qwen": "qwen3.7-max",
    "moonshot": "kimi-k2.7-code",
    "openrouter": "anthropic/claude-sonnet-5",
    "azure": "gpt-5.6-sol",
    "bedrock": "openai.gpt-5.6-sol",
    "groq": "llama-4.1-70b-versatile",
    "together": "meta-llama/Llama-4.1-70B-Instruct-Turbo",
    "fireworks": "accounts/fireworks/models/llama-v4-70b-instruct",
    "cerebras": "llama-4.1-70b",
    "deepinfra": "meta-llama/Llama-4.1-70B-Instruct",
}


def test_researched_newest_ga_models_are_still_suggested():
    for pid, model in RESEARCHED_SUGGESTED_MODELS.items():
        assert model in providers.REGISTRY[pid].models, pid


def test_session_uses_chosen_model_and_has_no_default():
    """Without a chosen model a cloud provider resolves to "" (the user picks)."""
    for pid in sorted(EXPECTED_PROVIDER_IDS):
        provider = providers.get_provider(pid)
        if provider.is_local:
            continue  # local servers resolve live; covered in test_config
        session = Session(provider_id="openrouter", model="test-model")

        session.set_provider(pid)
        assert session.model == ""
        assert session.model_for(pid) == ""

        config.set_provider_model(pid, f"{pid}-chosen")
        session.set_provider(pid)
        assert session.model == f"{pid}-chosen"

        # An explicit model still wins over the stored choice.
        session.set_provider(pid, "explicit-model")
        assert session.model == "explicit-model"


def test_fallback_model_comes_from_the_chosen_model_or_is_empty():
    session = Session(provider_id="openrouter", model="test-model")

    assert session.model_for("openai") == ""

    config.set_provider_model("openai", "gpt-chosen")
    assert session.model_for("openai") == "gpt-chosen"

    session.models["openai"] = "session-override"
    assert session.model_for("openai") == "session-override"
    assert session.model_for("openrouter") == "test-model"


def test_session_switch_to_local_uses_the_running_servers_model(monkeypatch):
    """Local servers resolve live; their suggested tuple is never a default."""
    listed: list[str] = []

    def fake_fetch(provider, api_key=None, **kwargs):
        if not listed:
            raise catalog.CatalogFetchError("Ollama listed no usable chat models")
        return [catalog.RemoteModel(model_id, 0.0) for model_id in listed]

    monkeypatch.setattr(config.catalog, "fetch_models", fake_fetch)
    session = Session(provider_id="openrouter", model="test-model")

    # Nothing loaded on the server: no model, even though suggestions exist.
    session.set_provider("ollama")
    assert providers.get_provider("ollama").models
    assert session.model == ""

    config.clear_model_cache("ollama")
    listed.append("qwen3:8b")
    session.set_provider("ollama")
    assert session.model == "qwen3:8b"
    assert session.model_for("ollama") == "qwen3:8b"


# ---------------------------------------------------------------------------
# KiwiMate (experimental, chat only)
# ---------------------------------------------------------------------------


def test_kiwimate_is_listed_first():
    assert next(iter(providers.REGISTRY)) == "kiwimate"
    assert providers.list_providers()[0].id == "kiwimate"


def test_kiwimate_registry_entry():
    kiwimate = providers.get_provider("kiwimate")

    assert kiwimate.name == "KiwiMate"
    assert kiwimate.base_url == KIWIMATE_BASE_URL
    assert kiwimate.base_url.startswith("https://")
    assert kiwimate.key_env == "KIWIMATE_API_KEY"
    assert kiwimate.models == (
        "kiwimate-mini-1-0",
        "kiwimate-small-1-0",
        "kiwimate-medium-1-0",
        "kiwimate-large-1-0",
    )
    assert kiwimate.experimental is True
    assert kiwimate.supports_tools is False
    assert "kiwimate.net" in kiwimate.description
    assert kiwimate.compat == "openai"
    assert not kiwimate.is_local
    assert kiwimate.needs_key
    assert catalog.models_url(kiwimate) == f"{KIWIMATE_BASE_URL}/models"


def test_only_kiwimate_is_experimental_or_chat_only():
    for provider in providers.list_providers():
        if provider.id == "kiwimate":
            continue
        assert provider.experimental is False, provider.id
        assert provider.supports_tools is True, provider.id


def test_new_provider_fields_default_to_a_regular_tool_capable_provider():
    provider = providers.ProviderConfig(
        id="x", name="X", base_url="https://api.example.com/v1", key_env="X_KEY"
    )

    assert provider.experimental is False
    assert provider.supports_tools is True
    assert provider.description == ""
    assert provider.models == ()


# ---------------------------------------------------------------------------
# Local providers
# ---------------------------------------------------------------------------


def test_local_providers_are_marked_local():
    for pid in ("ollama", "lmstudio", "unsloth"):
        provider = providers.get_provider(pid)
        assert provider.is_local
        assert provider.compat == "openai"
        assert provider.models  # suggestions for the offline picker
    assert not providers.get_provider("openai").is_local


def test_unsloth_is_local_but_still_requires_a_key():
    """Unsloth enforces auth even on localhost; ollama/lmstudio stay keyless."""
    unsloth = providers.get_provider("unsloth")
    assert unsloth.key_env == "UNSLOTH_API_KEY"
    assert unsloth.is_local
    assert unsloth.requires_key
    assert unsloth.needs_key
    for pid in ("ollama", "lmstudio"):
        provider = providers.get_provider(pid)
        assert not provider.requires_key
        assert not provider.needs_key
    assert providers.get_provider("openai").needs_key  # cloud always needs one


def test_is_local_matches_host_heuristic():
    for base_url in (
        "http://127.0.0.1:8080/v1",
        "http://[::1]:8080/v1",
        "http://host.tailnet.local/v1",
    ):
        assert providers.ProviderConfig(
            id="x", name="X", base_url=base_url, key_env="X_KEY"
        ).is_local
    assert not providers.ProviderConfig(
        id="y", name="Y", base_url="https://api.example.com/v1", key_env="Y_KEY",
    ).is_local


# ---------------------------------------------------------------------------
# Auth header / api-version configuration
# ---------------------------------------------------------------------------


def test_provider_auth_fields_default_backwards_compatible():
    provider = providers.ProviderConfig(
        id="x", name="X", base_url="https://api.example.com/v1", key_env="X_KEY",
    )

    assert provider.key_header == "Authorization"
    assert provider.key_prefix == "Bearer "
    assert provider.api_version == ""


def test_azure_uses_api_key_header_without_bearer_prefix():
    azure = providers.get_provider("azure")

    assert azure.key_header == "api-key"
    assert azure.key_prefix == ""
    assert azure.api_version
    assert not azure.is_local
    assert azure.needs_key


def test_bedrock_uses_bearer_token_env():
    bedrock = providers.get_provider("bedrock")

    assert bedrock.key_env == "AWS_BEARER_TOKEN_BEDROCK"
    assert bedrock.key_header == "Authorization"
    assert bedrock.key_prefix == "Bearer "


def test_gateway_providers_registered_with_env_vars():
    for pid, env in EXPECTED_KEY_ENVS.items():
        provider = providers.get_provider(pid)
        assert provider.key_env == env
        assert provider.compat == "openai"
        assert not provider.is_local
        assert len(provider.models) >= 1


def test_versioned_url_appends_api_version_only_when_set():
    azure = providers.ProviderConfig(
        id="az", name="AZ", base_url="https://r.openai.azure.com/openai/v1",
        key_env="X_KEY", api_version="2024-10-21",
    )
    assert azure.versioned_url("https://r.openai.azure.com/openai/v1/models") == (
        "https://r.openai.azure.com/openai/v1/models?api-version=2024-10-21"
    )

    plain = providers.ProviderConfig(
        id="p", name="P", base_url="https://api.example.com/v1", key_env="X_KEY",
    )
    assert plain.versioned_url("https://api.example.com/v1/models") == (
        "https://api.example.com/v1/models"
    )


def test_versioned_url_uses_ampersand_when_query_exists():
    provider = providers.ProviderConfig(
        id="az", name="AZ", base_url="https://r.openai.azure.com/openai/v1",
        key_env="X_KEY", api_version="2024-10-21",
    )

    assert provider.versioned_url("https://x/v1/models?limit=10") == (
        "https://x/v1/models?limit=10&api-version=2024-10-21"
    )
