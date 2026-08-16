import json
import time

import pytest

from kiwimatecoder import catalog, config
from kiwimatecoder.providers import REGISTRY


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    """Point config storage at a temp dir and clear provider env vars."""
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    for env in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    return tmp_path


def test_empty_config_defaults():
    cfg = config.load_config()
    assert cfg["keys"] == {}
    assert cfg["selected_provider"] == "openrouter"
    assert cfg["default_mode"] == "ask"


def test_legacy_migration(isolate_config):
    (isolate_config / "config").write_text("OPENROUTER_API_KEY=legacy-key-123\n")
    cfg = config.load_config()
    assert cfg["keys"]["openrouter"] == "legacy-key-123"
    # Legacy file is not deleted.
    assert (isolate_config / "config").exists()


def test_set_and_get_key_roundtrip():
    config.set_key("openai", "sk-openai")
    assert config.get_key("openai") == "sk-openai"
    stored = json.loads((config.CONFIG_FILE).read_text())
    assert stored["keys"]["openai"] == "sk-openai"


def test_env_var_overrides_stored_key(monkeypatch):
    config.set_key("openrouter", "stored")
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    assert config.get_key("openrouter") == "from-env"


def test_set_key_persists_when_env_override_warns(monkeypatch):
    """Re-setting a key stores it on disk, but warns that the env var still wins."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    warning = config.set_key("openrouter", "sk-new")
    assert warning and "OPENROUTER_API_KEY" in warning
    assert config.load_config()["keys"]["openrouter"] == "sk-new"
    assert config.get_key("openrouter") == "from-env"


def test_set_key_returns_none_without_env_override():
    assert config.set_key("openai", "sk-openai") is None


def test_get_key_env_override(monkeypatch):
    config.set_key("openai", "sk-stored")
    assert config.get_key_env_override("openai") is None
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert config.get_key_env_override("openai") == "OPENAI_API_KEY"


def test_empty_env_var_takes_precedence_and_disables_stored_key(monkeypatch):
    """Exported empty env var must win over stored key (returns None -> friendly no-key path)."""
    config.set_key("openrouter", "stored-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    assert config.get_key("openrouter") is None


def test_absent_env_uses_stored_key(monkeypatch):
    """When env var is not present at all, stored key is used."""
    config.set_key("openai", "sk-stored")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert config.get_key("openai") == "sk-stored"


def test_key_source_stored(monkeypatch):
    config.set_key("openai", "sk-stored")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    source = config.key_source("openai")
    assert source["origin"] == "stored"
    assert source["value"] == "sk-stored"


def test_key_source_env_override(monkeypatch):
    config.set_key("openai", "sk-stored")
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    source = config.key_source("openai")
    assert source["origin"] == "env"
    assert source["env"] == "OPENAI_API_KEY"
    assert source["value"] == "from-env"


def test_key_source_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert config.key_source("openai")["origin"] == "missing"


def test_key_source_legacy(monkeypatch, tmp_path):
    """A key that only exists in the flat legacy file is reported as legacy."""
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / "config").write_text("OPENROUTER_API_KEY=sk-or-legacy\n")

    source = config.key_source("openrouter")
    assert source["origin"] == "legacy"
    assert source["value"] == "sk-or-legacy"
    desc = config.describe_key("openrouter")
    assert "legacy" in desc
    assert "sk-or-l" not in desc  # key is redacted


def test_describe_key_redacts_and_names_source(monkeypatch):
    config.set_key("openai", "sk-abcdefgh")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    desc = config.describe_key("openai")
    assert "config.json" in desc
    assert "…efgh" in desc
    assert "sk-abcdefgh" not in desc


def test_set_key_unknown_provider_raises():
    with pytest.raises(KeyError):
        config.set_key("nope", "x")


def test_legacy_shims():
    config.save_api_key("shim-key")
    assert config.load_api_key() == "shim-key"
    assert config.get_key("openrouter") == "shim-key"


def test_custom_provider_roundtrip():
    provider = config.add_provider(
        "local",
        "Local Models",
        "http://localhost:1234/v1/",
        "local-code",
        "LOCAL_API_KEY",
    )

    assert provider.id == "local"
    assert provider.base_url == "http://localhost:1234/v1"
    assert config.get_provider_config("local").default_model == "local-code"
    assert "local" in {p.id for p in config.list_provider_configs()}


def test_remove_custom_provider_removes_related_config():
    config.add_provider("local", "Local", "http://localhost:1234/v1", "local-code")
    config.set_key("local", "sk-local")
    config.set_selected_provider("local")
    config.set_model_filter("local", "allow", ["local-code"])

    config.remove_provider("local")
    cfg = config.load_config()

    assert "local" not in cfg["providers"]
    assert "local" not in cfg["keys"]
    assert "local" not in cfg["model_filters"]
    assert cfg["selected_provider"] == "openrouter"


def test_model_filters_control_visible_models():
    config.set_model_filter("openrouter", "allow", ["a", "b", "a"])
    assert config.get_model_filter("openrouter") == {
        "mode": "allow",
        "models": ["a", "b"],
    }
    assert config.list_visible_models("openrouter") == ["a", "b"]

    config.set_model_filter("openrouter", "deny", ["anthropic/claude-sonnet-5"])
    visible = config.list_visible_models("openrouter")
    assert "anthropic/claude-sonnet-5" not in visible
    assert visible  # the rest of the catalog is still offered

    config.set_model_filter("openrouter", "all", [])
    assert config.get_model_filter("openrouter") == {"mode": "all", "models": []}


def test_visible_models_default_to_full_provider_catalog():
    for provider_id, provider in REGISTRY.items():
        visible = config.list_visible_models(provider_id)
        # Local providers ship no static default; the curated tuple leads.
        expected_first = provider.default_model or provider.models[0]
        assert visible[0] == expected_first
        assert set(provider.models) <= set(visible)
        assert len(visible) == len(set(visible))
        assert len(visible) > 1, f"{provider_id} should offer more than one model"


def test_custom_provider_models_from_config():
    config.add_provider("local", "Local", "http://localhost:1234/v1", "local-code")
    assert config.list_visible_models("local") == ["local-code"]

    cfg = config.load_config()
    cfg["providers"]["local"]["models"] = ["local-fast", "local-code", " ", "local-fast"]
    config.save_config(cfg)

    assert config.list_visible_models("local") == ["local-code", "local-fast"]


def test_remove_key():
    config.set_key("openai", "sk-openai")
    assert config.remove_key("openai")
    assert config.get_key("openai") is None


# ---------------------------------------------------------------------------
# Live model catalogs
# ---------------------------------------------------------------------------


def _install_fetch(monkeypatch, model_ids=(), error=None):
    """Patch the network fetch and return the list of providers it was asked for.

    ``model_ids`` are dated newest-first so the catalog order matches the order
    they are written in each test.
    """
    calls: list[str] = []

    def fake_fetch(provider, api_key=None, **kwargs):
        calls.append(provider.id)
        if error is not None:
            raise catalog.CatalogFetchError(error)
        return [
            catalog.RemoteModel(model_id, float(len(model_ids) - index))
            for index, model_id in enumerate(model_ids)
        ]

    monkeypatch.setattr(config.catalog, "fetch_models", fake_fetch)
    return calls


def _forbid_fetch(monkeypatch):
    def fake_fetch(provider, api_key=None, **kwargs):
        raise AssertionError("the network must not be touched here")

    monkeypatch.setattr(config.catalog, "fetch_models", fake_fetch)


def test_refresh_adds_new_models_and_drops_deprecated_ones(monkeypatch):
    config.set_key("openrouter", "sk-test")
    default = REGISTRY["openrouter"].default_model
    calls = _install_fetch(monkeypatch, [default, "vendor/brand-new"])

    result = config.get_model_catalog("openrouter", refresh=True)

    assert calls == ["openrouter"]
    assert result.source == "live"
    assert result.models == [default, "vendor/brand-new"]
    assert result.added == ["vendor/brand-new"]
    # Curated ids the provider no longer serves are gone from the catalog.
    assert "openai/gpt-5.6-sol" in result.removed
    assert "openai/gpt-5.6-sol" not in config.list_visible_models("openrouter")


def test_live_catalog_is_cached_and_reused_without_network(monkeypatch):
    config.set_key("openrouter", "sk-test")
    _install_fetch(monkeypatch, ["vendor/one", "vendor/two"])
    config.get_model_catalog("openrouter", refresh=True)

    _forbid_fetch(monkeypatch)
    result = config.get_model_catalog("openrouter", refresh=True)

    assert result.source == "cache"
    assert result.models == ["vendor/one", "vendor/two"]
    assert config.list_visible_models("openrouter") == ["vendor/one", "vendor/two"]


def test_stale_cache_is_refetched(monkeypatch):
    config.set_key("openrouter", "sk-test")
    _install_fetch(monkeypatch, ["vendor/old"])
    config.get_model_catalog("openrouter", refresh=True)

    cache = config.load_model_cache()
    cache["providers"]["openrouter"]["fetched_at"] = (
        time.time() - catalog.CATALOG_TTL_SECONDS - 1
    )
    config.save_model_cache(cache)

    _install_fetch(monkeypatch, ["vendor/fresh"])
    result = config.get_model_catalog("openrouter", refresh=True)

    assert result.source == "live"
    assert result.models == ["vendor/fresh"]
    assert result.removed == ["vendor/old"]


def test_force_refetches_a_fresh_cache(monkeypatch):
    config.set_key("openrouter", "sk-test")
    _install_fetch(monkeypatch, ["vendor/one"])
    config.get_model_catalog("openrouter", refresh=True)

    calls = _install_fetch(monkeypatch, ["vendor/two"])
    result = config.get_model_catalog("openrouter", force=True)

    assert calls == ["openrouter"]
    assert result.models == ["vendor/two"]


def test_no_key_means_no_fetch(monkeypatch):
    _forbid_fetch(monkeypatch)

    result = config.get_model_catalog("openrouter", refresh=True)

    assert result.source == "curated"
    assert result.models[0] == REGISTRY["openrouter"].default_model
    # The automatic path stays quiet about providers that aren't set up.
    assert result.error is None


def test_forced_refresh_without_a_key_says_why(monkeypatch):
    _forbid_fetch(monkeypatch)

    result = config.get_model_catalog("openrouter", force=True)

    assert result.source == "curated"
    assert result.error is not None
    assert "no API key" in result.error
    assert "OPENROUTER_API_KEY" in result.error


def test_local_provider_is_fetched_without_a_key(monkeypatch):
    monkeypatch.delenv("LOCAL_API_KEY", raising=False)
    config.add_provider("local", "Local", "http://localhost:1234/v1", "local-code")
    calls = _install_fetch(monkeypatch, ["local-code", "local-new"])

    result = config.get_model_catalog("local", refresh=True)

    assert calls == ["local"]
    assert result.models == ["local-code", "local-new"]


def test_failed_fetch_falls_back_and_backs_off(monkeypatch):
    config.set_key("openrouter", "sk-test")
    calls = _install_fetch(monkeypatch, error="boom")

    result = config.get_model_catalog("openrouter", refresh=True)

    assert calls == ["openrouter"]
    assert result.source == "curated"
    assert result.error and "boom" in result.error
    assert result.models[0] == REGISTRY["openrouter"].default_model

    # A second automatic refresh is suppressed until the backoff expires.
    _forbid_fetch(monkeypatch)
    assert config.get_model_catalog("openrouter", refresh=True).source == "curated"


def test_failed_refresh_keeps_serving_the_cached_catalog(monkeypatch):
    config.set_key("openrouter", "sk-test")
    _install_fetch(monkeypatch, ["vendor/one"])
    config.get_model_catalog("openrouter", refresh=True)

    _install_fetch(monkeypatch, error="offline")
    result = config.get_model_catalog("openrouter", force=True)

    assert result.source == "cache"
    assert result.models == ["vendor/one"]
    assert result.error is not None
    assert "offline" in result.error


def test_visible_models_apply_filters_to_the_live_catalog(monkeypatch):
    config.set_key("openrouter", "sk-test")
    _install_fetch(monkeypatch, ["vendor/one", "vendor/two"])
    config.get_model_catalog("openrouter", refresh=True)
    config.set_model_filter("openrouter", "deny", ["vendor/two"])

    assert config.list_visible_models("openrouter") == ["vendor/one"]


def test_search_model_catalog_finds_models_below_the_selector_cap(monkeypatch):
    config.set_key("openrouter", "sk-test")
    ids = [f"model-{i:02d}" for i in range(70)]
    _install_fetch(monkeypatch, ids)

    # With 70 models the newest 60 are kept for the selector; the oldest ten
    # fall below the cap. Search still sees them in the full catalog.
    assert config.search_model_catalog("openrouter", "model-68", refresh=True) == [
        "model-68"
    ]
    assert "model-69" not in config.list_visible_models("openrouter")


def test_search_model_catalog_respects_allow_and_deny_filters(monkeypatch):
    config.set_key("openrouter", "sk-test")
    _install_fetch(monkeypatch, ["vendor/one", "vendor/two"])
    config.get_model_catalog("openrouter", refresh=True)

    config.set_model_filter("openrouter", "deny", ["vendor/two"])
    assert config.search_model_catalog("openrouter", "vendor") == ["vendor/one"]

    config.set_model_filter("openrouter", "allow", ["vendor/two"])
    assert config.search_model_catalog("openrouter", "vendor") == ["vendor/two"]


def test_list_visible_models_never_fetches_by_default(monkeypatch):
    config.set_key("openrouter", "sk-test")
    _forbid_fetch(monkeypatch)

    assert config.list_visible_models("openrouter")[0] == (
        REGISTRY["openrouter"].default_model
    )


def test_clear_model_cache(monkeypatch):
    config.set_key("openrouter", "sk-test")
    _install_fetch(monkeypatch, ["vendor/one"])
    config.get_model_catalog("openrouter", refresh=True)

    config.clear_model_cache("openrouter")

    assert config.load_model_cache()["providers"] == {}
    _forbid_fetch(monkeypatch)
    assert config.get_model_catalog("openrouter").source == "curated"


def test_removing_a_provider_clears_its_cached_catalog(monkeypatch):
    config.add_provider("local", "Local", "http://localhost:1234/v1", "local-code")
    _install_fetch(monkeypatch, ["local-code"])
    config.get_model_catalog("local", refresh=True)

    config.remove_provider("local")

    assert "local" not in config.load_model_cache()["providers"]


def test_corrupt_model_cache_is_ignored(monkeypatch):
    (config.CONFIG_DIR / config.MODEL_CACHE_NAME).write_text("{not json")
    _forbid_fetch(monkeypatch)

    assert config.load_model_cache() == {"version": 1, "providers": {}}
    assert config.get_model_catalog("openrouter").source == "curated"


# ---------------------------------------------------------------------------
# Local providers
# ---------------------------------------------------------------------------


def test_builtin_local_provider_fetches_without_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    calls = _install_fetch(monkeypatch, ["llama3.1:8b", "qwen3:8b"])

    result = config.get_model_catalog("ollama", refresh=True)

    assert calls == ["ollama"]
    assert result.source == "live"
    assert result.models == ["llama3.1:8b", "qwen3:8b"]


def test_offline_local_server_falls_back_to_curated(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    _install_fetch(monkeypatch, error="connection refused")

    result = config.get_model_catalog("ollama", refresh=True)

    assert result.source == "curated"
    assert result.error == "connection refused"
    assert result.models[0] == REGISTRY["ollama"].models[0]
    assert "" not in result.models  # the empty static default never leaks


def test_describe_key_for_local_provider_without_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    assert config.describe_key("ollama") == "not required (local server)"
    assert config.describe_key("openai") == "missing"


def test_resolve_default_model_returns_static_default_without_network(monkeypatch):
    _forbid_fetch(monkeypatch)
    provider = config.get_provider_config("openai")
    assert config.resolve_default_model(provider) == provider.default_model


def test_resolve_default_model_local_picks_first_live_model(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    _install_fetch(monkeypatch, ["qwen3:8b", "llama3.1:8b"])

    provider = config.get_provider_config("ollama")

    assert config.resolve_default_model(provider) == "qwen3:8b"


def test_resolve_default_model_local_falls_back_to_curated_when_offline(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    _install_fetch(monkeypatch, error="offline")

    provider = config.get_provider_config("ollama")

    assert config.resolve_default_model(provider) == REGISTRY["ollama"].models[0]
