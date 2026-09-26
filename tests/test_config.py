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
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for provider in REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)
    return tmp_path


def _pin_selected_model(model):
    """Store a ``selected_model`` pin as a profile or hand edit would."""
    cfg = config.load_config()
    cfg["selected_model"] = model
    config.save_config(cfg)


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
    assert not hasattr(config.get_provider_config("local"), "default_model")
    assert config.get_provider_model("local") == "local-code"
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
    assert "local" not in cfg["provider_models"]
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
        # Offline, the suggested models are offered as-is; none is pinned
        # first as a default.
        assert visible == list(provider.models)
        assert len(visible) == len(set(visible))
        assert len(visible) > 1, f"{provider_id} should offer more than one model"


def test_custom_provider_models_from_config():
    config.add_provider("local", "Local", "http://localhost:1234/v1", "local-code")
    # The chosen model is a choice, not a suggestion: nothing is offered
    # offline until the provider's config lists models.
    assert config.list_visible_models("local") == []

    cfg = config.load_config()
    cfg["providers"]["local"]["models"] = ["local-fast", "local-code", " ", "local-fast"]
    config.save_config(cfg)

    assert config.list_visible_models("local") == ["local-fast", "local-code"]
    assert config.get_provider_model("local") == "local-code"


def test_remove_key():
    config.set_key("openai", "sk-openai")
    assert config.remove_key("openai")
    assert config.get_key("openai") is None


# ---------------------------------------------------------------------------
# Active provider roster
# ---------------------------------------------------------------------------


def test_empty_config_defaults_to_single_active_provider():
    cfg = config.load_config()
    assert cfg["active_providers"] == ["openrouter"]


def test_get_active_provider_ids_validates_and_dedupes():
    config.set_active_providers(["openai", "openrouter", "openai"])
    assert config.get_active_provider_ids() == ["openai", "openrouter"]


def test_get_active_provider_ids_drops_unknown_ids():
    cfg = config.load_config()
    cfg["active_providers"] = ["nope", "openai", "openrouter"]
    config.save_config(cfg)
    assert config.get_active_provider_ids() == ["openai", "openrouter"]


def test_set_active_providers_requires_a_nonempty_list():
    with pytest.raises(ValueError):
        config.set_active_providers([])


def test_set_active_providers_keeps_selected_provider_in_sync():
    ids = config.set_active_providers(["openai", "deepseek"])
    assert ids == ["openai", "deepseek"]
    assert config.get_selected_provider_id() == "openai"


def test_set_active_providers_clears_selected_model_when_primary_changes():
    config.set_active_providers(["openai"])
    config.set_selected_model("gpt-chosen")
    _pin_selected_model("openai/gpt-test")
    config.set_active_providers(["deepseek", "openai"])
    cfg = config.load_config()
    assert cfg.get("selected_model") is None
    # The pin goes, but the model chosen for each provider stays.
    assert config.get_provider_model("openai", cfg) == "gpt-chosen"
    assert config.get_provider_model("deepseek", cfg) == ""


def test_set_active_providers_keeps_selected_model_when_primary_unchanged():
    config.set_active_providers(["openai", "deepseek"])
    _pin_selected_model("openai/gpt-test")
    config.set_active_providers(["openai", "openrouter"])
    assert config.load_config().get("selected_model") == "openai/gpt-test"
    assert config.get_provider_model("openai") == "openai/gpt-test"


def test_set_selected_provider_clears_selected_model_when_switching():
    config.set_selected_provider("openai")
    config.set_selected_model("gpt-chosen")
    _pin_selected_model("openai/gpt-test")
    config.set_selected_provider("deepseek")
    assert config.load_config().get("selected_model") is None
    assert config.get_provider_model("deepseek") == ""
    # Switching back restores the model chosen for that provider.
    config.set_selected_provider("openai")
    assert config.get_provider_model("openai") == "gpt-chosen"


def test_set_selected_provider_resets_roster_to_single():
    config.set_active_providers(["openai", "openrouter"])
    config.set_selected_provider("deepseek")
    assert config.get_active_provider_ids() == ["deepseek"]


def test_remove_provider_drops_it_from_active_roster():
    config.add_provider("local", "Local", "http://localhost:1234/v1", "local-code")
    config.set_active_providers(["openai", "local", "openrouter"])
    config.remove_provider("local")
    assert config.get_active_provider_ids() == ["openai", "openrouter"]


def test_remove_primary_provider_repoints_active_roster():
    config.add_provider("local", "Local", "http://localhost:1234/v1", "local-code")
    config.set_active_providers(["local", "openrouter"])
    config.remove_provider("local")
    assert config.get_active_provider_ids() == ["openrouter"]


def test_remove_primary_provider_syncs_selected_to_remaining():
    config.add_provider("local", "Local", "http://localhost:1234/v1", "local-code")
    config.set_active_providers(["local", "openai"])
    config.remove_provider("local")
    assert config.get_active_provider_ids() == ["openai"]
    assert config.get_selected_provider_id() == "openai"
    assert config.load_config()["selected_provider"] == "openai"


def test_get_active_provider_ids_does_not_append_selected():
    cfg = config.load_config()
    cfg["selected_provider"] = "openrouter"
    cfg["active_providers"] = ["openai"]
    config.save_config(cfg)
    assert config.get_active_provider_ids() == ["openai"]
    assert config.get_selected_provider_id() == "openai"


def test_load_config_repairs_malformed_active_providers():
    cfg = config._empty_config()
    cfg["selected_provider"] = "openai"
    cfg["active_providers"] = "openai"
    config.save_config(cfg)
    loaded = config.load_config()
    assert loaded["active_providers"] == ["openai"]
    assert config.get_active_provider_ids() == ["openai"]


def test_active_roster_migrates_from_legacy_selected_provider():
    # A config with only `selected_provider` (no `active_providers`) gets a
    # roster seeded from that single provider.
    cfg = config._empty_config()
    cfg["selected_provider"] = "openai"
    cfg.pop("active_providers", None)
    config.save_config(cfg)
    assert config.get_active_provider_ids() == ["openai"]


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
    suggested = REGISTRY["openrouter"].models[0]
    calls = _install_fetch(monkeypatch, ["vendor/brand-new", suggested])

    result = config.get_model_catalog("openrouter", refresh=True)

    assert calls == ["openrouter"]
    assert result.source == "live"
    # Newest first; the first suggested model is not pinned as a default.
    assert result.models == ["vendor/brand-new", suggested]
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
    assert result.models == list(REGISTRY["openrouter"].models)
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
    assert result.models == list(REGISTRY["openrouter"].models)

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

    assert config.list_visible_models("openrouter") == list(
        REGISTRY["openrouter"].models
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
    assert "" not in result.models  # no empty model id ever leaks


def test_describe_key_for_local_provider_without_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("UNSLOTH_API_KEY", raising=False)
    assert config.describe_key("ollama") == "not required (local server)"
    assert config.describe_key("unsloth") == "required by local server"
    assert config.describe_key("openai") == "missing"


def test_local_provider_requiring_key_is_not_fetched_without_one(monkeypatch):
    """Unsloth answers 401 without a key, so no doomed fetch is attempted."""
    monkeypatch.delenv("UNSLOTH_API_KEY", raising=False)
    calls = _install_fetch(monkeypatch, ["unsloth/Qwen3.6-27B-GGUF"])

    result = config.get_model_catalog("unsloth", refresh=True)

    assert calls == []
    assert result.source == "curated"
    assert result.models == list(REGISTRY["unsloth"].models)


def test_local_provider_requiring_key_is_fetched_once_key_is_set(monkeypatch):
    monkeypatch.setenv("UNSLOTH_API_KEY", "sk-unsloth-test")
    calls = _install_fetch(monkeypatch, ["unsloth/Qwen3.6-27B-GGUF"])

    result = config.get_model_catalog("unsloth", refresh=True)

    assert calls == ["unsloth"]
    assert result.source == "live"
    assert result.models == ["unsloth/Qwen3.6-27B-GGUF"]


# ---------------------------------------------------------------------------
# Resolving the model to use (no default models)
# ---------------------------------------------------------------------------


def test_resolve_model_cloud_without_a_choice_is_empty(monkeypatch):
    _forbid_fetch(monkeypatch)

    for provider in REGISTRY.values():
        if provider.is_local:
            continue
        assert config.resolve_model(provider) == "", provider.id


def test_resolve_model_cloud_returns_the_chosen_model_without_network(monkeypatch):
    _forbid_fetch(monkeypatch)
    config.set_provider_model("openai", "gpt-chosen")
    provider = config.get_provider_config("openai")

    assert config.resolve_model(provider) == "gpt-chosen"
    assert config.resolve_model(provider, config.load_config()) == "gpt-chosen"


def test_resolve_model_reads_the_given_cfg(monkeypatch):
    _forbid_fetch(monkeypatch)
    cfg = config.load_config()
    cfg["provider_models"] = {"openai": "in-memory-model"}

    provider = config.get_provider_config("openai")

    assert config.resolve_model(provider, cfg) == "in-memory-model"
    assert config.resolve_model(provider) == ""  # nothing was saved


def test_resolve_model_local_picks_first_live_model(monkeypatch):
    _install_fetch(monkeypatch, ["qwen3:8b", "llama3.1:8b"])

    provider = config.get_provider_config("ollama")

    assert config.resolve_model(provider) == "qwen3:8b"


def test_resolve_model_local_prefers_the_chosen_model_without_network(monkeypatch):
    _forbid_fetch(monkeypatch)
    config.set_provider_model("ollama", "llama3.1:8b")

    assert config.resolve_model(config.get_provider_config("ollama")) == "llama3.1:8b"


def test_resolve_model_local_uses_the_cached_listing_when_offline(monkeypatch):
    _install_fetch(monkeypatch, ["qwen3:8b", "llama3.1:8b"])
    config.get_model_catalog("ollama", refresh=True)
    cache = config.load_model_cache()
    cache["providers"]["ollama"]["fetched_at"] = (
        time.time() - catalog.CATALOG_TTL_SECONDS - 1
    )
    config.save_model_cache(cache)
    calls = _install_fetch(monkeypatch, error="connection refused")

    assert config.resolve_model(config.get_provider_config("ollama")) == "qwen3:8b"
    assert calls == ["ollama"]


def test_resolve_model_local_offline_with_only_suggestions_is_empty(monkeypatch):
    """The suggested tuple is never used as a hidden default."""
    _install_fetch(monkeypatch, error="connection refused")
    provider = config.get_provider_config("ollama")

    assert config.get_model_catalog("ollama").models == list(provider.models)
    assert config.resolve_model(provider) == ""


def test_resolve_model_key_requiring_local_without_a_key_is_empty(monkeypatch):
    calls = _install_fetch(monkeypatch, ["unsloth/Qwen3.6-27B-GGUF"])

    assert config.resolve_model(config.get_provider_config("unsloth")) == ""
    assert calls == []


def test_require_model_raises_with_the_fix_it_message(monkeypatch):
    _forbid_fetch(monkeypatch)
    provider = config.get_provider_config("openrouter")

    with pytest.raises(config.ModelNotChosenError) as excinfo:
        config.require_model(provider)

    assert isinstance(excinfo.value, ValueError)
    message = str(excinfo.value)
    assert message.startswith("No model chosen for OpenRouter")
    assert "kiwimatecoder config model set <model> --provider openrouter" in message
    # Never the setup wizard, which would ask for the API key again.
    assert "kiwimatecoder setup" not in message
    assert "config model set <model> --provider openrouter" in message
    assert "--model" in message
    assert message == config.no_model_message(provider)


def test_require_model_honours_the_override_and_the_chosen_model(monkeypatch):
    _forbid_fetch(monkeypatch)
    provider = config.get_provider_config("openrouter")

    assert config.require_model(provider, override="vendor/override") == (
        "vendor/override"
    )

    config.set_provider_model("openrouter", "vendor/chosen")
    assert config.require_model(provider) == "vendor/chosen"
    assert config.require_model(provider, override=" vendor/override ") == (
        "vendor/override"
    )
    # A blank override falls through to the chosen model.
    assert config.require_model(provider, override="   ") == "vendor/chosen"


def test_require_model_uses_the_live_local_model(monkeypatch):
    _install_fetch(monkeypatch, ["qwen3:8b"])

    assert config.require_model(config.get_provider_config("ollama")) == "qwen3:8b"


def test_require_model_local_server_offline_raises(monkeypatch):
    _install_fetch(monkeypatch, error="connection refused")

    with pytest.raises(ValueError, match="^No model chosen for Ollama"):
        config.require_model(config.get_provider_config("ollama"))


# ---------------------------------------------------------------------------
# Chosen models (provider_models) and the selected_model pin
# ---------------------------------------------------------------------------


def test_provider_models_get_set_and_forget():
    assert config.load_config()["provider_models"] == {}
    assert config.get_provider_model("openai") == ""

    config.set_provider_model("openai", "  gpt-chosen  ")

    assert config.get_provider_model("openai") == "gpt-chosen"
    stored = json.loads(config.CONFIG_FILE.read_text())
    assert stored["provider_models"] == {"openai": "gpt-chosen"}
    assert stored["version"] == config.CONFIG_VERSION == 3

    config.set_provider_model("openai", "")
    assert config.get_provider_model("openai") == ""

    config.set_provider_model("openai", "again")
    config.set_provider_model("openai", None)
    assert config.load_config()["provider_models"] == {}


def test_provider_models_are_kept_per_provider():
    config.add_provider("custom", "Custom", "https://custom.example/v1")
    config.set_provider_model("openai", "gpt-chosen")
    config.set_provider_model("kiwimate", "kiwimate-small-1-0")
    config.set_provider_model("custom", "custom-model")

    cfg = config.load_config()

    assert cfg["provider_models"] == {
        "openai": "gpt-chosen",
        "kiwimate": "kiwimate-small-1-0",
        "custom": "custom-model",
    }
    assert config.get_provider_model("deepseek", cfg) == ""


def test_set_provider_model_rejects_unknown_provider():
    with pytest.raises(KeyError):
        config.set_provider_model("does-not-exist", "m")

    assert config.load_config()["provider_models"] == {}


def test_get_provider_model_tolerates_a_malformed_section():
    assert config.get_provider_model("openai", {"provider_models": ["junk"]}) == ""


def test_stored_provider_models_are_read_normalized_and_bad_entries_reported():
    config.CONFIG_FILE.write_text(
        json.dumps(
            {
                "version": config.CONFIG_VERSION,
                "provider_models": {
                    "openai": "  gpt-chosen ",
                    "deepseek": "   ",
                    "mistral": 5,
                },
            }
        )
    )

    # Readers see only the usable entries...
    assert config.get_provider_model("openai") == "gpt-chosen"
    assert config.get_provider_model("deepseek") == ""
    assert config.get_provider_model("mistral") == ""
    # ...while `config validate` still sees (and reports) the bad ones.
    issues = {issue["key"]: issue["level"] for issue in config.validate_config()}
    assert issues["provider_models.deepseek"] == "error"
    assert issues["provider_models.mistral"] == "error"
    assert "provider_models.openai" not in issues

    # The next write keeps the good entries and drops the bad ones.
    config.set_provider_model("anthropic", "claude-chosen")
    stored = json.loads(config.CONFIG_FILE.read_text())["provider_models"]
    assert stored == {"openai": "gpt-chosen", "anthropic": "claude-chosen"}


def test_non_object_provider_models_is_reported_and_tolerated():
    config.CONFIG_FILE.write_text(
        json.dumps({"version": config.CONFIG_VERSION, "provider_models": "nope"})
    )

    assert config.get_provider_model("openai") == ""
    issues = {issue["key"]: issue["level"] for issue in config.validate_config()}
    assert issues["provider_models"] == "error"

    config.set_provider_model("openai", "gpt-chosen")
    assert config.get_provider_model("openai") == "gpt-chosen"


def test_selected_model_pin_applies_to_the_primary_only():
    config.set_active_providers(["openai", "deepseek"])
    config.set_provider_model("openai", "gpt-chosen")
    config.set_provider_model("deepseek", "ds-chosen")
    _pin_selected_model("pinned-model")

    cfg = config.load_config()

    assert config.get_provider_model("openai", cfg) == "pinned-model"
    assert config.get_provider_model("deepseek", cfg) == "ds-chosen"
    assert config.get_provider_model("openrouter", cfg) == ""
    # The pin does not overwrite the stored choice.
    assert cfg["provider_models"]["openai"] == "gpt-chosen"


def test_set_provider_model_clears_the_pin_only_for_the_primary():
    config.set_active_providers(["openai", "deepseek"])
    _pin_selected_model("pinned-model")

    config.set_provider_model("deepseek", "ds-chosen")
    assert config.load_config()["selected_model"] == "pinned-model"

    config.set_provider_model("openai", "gpt-chosen")
    cfg = config.load_config()
    assert cfg["selected_model"] is None
    assert config.get_provider_model("openai", cfg) == "gpt-chosen"
    assert config.get_provider_model("deepseek", cfg) == "ds-chosen"


def test_set_selected_model_stores_the_primary_providers_choice():
    config.set_selected_provider("openai")
    _pin_selected_model("stale-pin")

    config.set_selected_model("gpt-chosen")

    cfg = config.load_config()
    assert cfg["selected_model"] is None
    assert cfg["provider_models"] == {"openai": "gpt-chosen"}
    assert config.get_provider_model("openai", cfg) == "gpt-chosen"

    config.set_selected_model(None)
    assert config.load_config()["provider_models"] == {}


# ---------------------------------------------------------------------------
# Custom providers store their model as a choice
# ---------------------------------------------------------------------------


def test_add_provider_stores_the_model_in_provider_models(monkeypatch):
    _forbid_fetch(monkeypatch)
    provider = config.add_provider(
        "custom", "Custom", "https://custom.example/v1", " custom-model "
    )

    cfg = config.load_config()
    assert cfg["provider_models"] == {"custom": "custom-model"}
    assert "default_model" not in cfg["providers"]["custom"]
    assert config.resolve_model(provider) == "custom-model"


def test_add_provider_without_a_model(monkeypatch):
    _forbid_fetch(monkeypatch)
    provider = config.add_provider(
        "custom", "Custom", "https://custom.example/v1", key_env="CUSTOM_KEY"
    )

    assert provider.id == "custom"
    assert provider.key_env == "CUSTOM_KEY"
    cfg = config.load_config()
    assert "custom" in cfg["providers"]
    assert "custom" not in cfg["provider_models"]
    assert config.resolve_model(provider) == ""
    assert "custom" in {p.id for p in config.list_provider_configs()}


def test_add_provider_rejects_a_blank_model():
    with pytest.raises(ValueError):
        config.add_provider("custom", "Custom", "https://custom.example/v1", "   ")

    assert "custom" not in config.load_config()["providers"]


def test_add_provider_cannot_replace_the_builtin_kiwimate():
    with pytest.raises(ValueError):
        config.add_provider("kiwimate", "Mine", "https://mine.example/v1", "m")


def test_update_provider_changes_the_chosen_model():
    config.add_provider("custom", "Custom", "https://custom.example/v1", "one")

    provider = config.update_provider("custom", model=" two ")

    assert provider.name == "Custom"
    cfg = config.load_config()
    assert cfg["provider_models"]["custom"] == "two"
    assert "default_model" not in cfg["providers"]["custom"]

    with pytest.raises(ValueError):
        config.update_provider("custom", model="   ")
    assert config.get_provider_model("custom") == "two"


def test_update_provider_model_clears_a_stale_pin_on_the_primary():
    config.add_provider("custom", "Custom", "https://custom.example/v1", "one")
    config.set_active_providers(["custom"])
    cfg = config.load_config()
    cfg["selected_model"] = "profile-pin"
    config.save_config(cfg)
    assert config.get_provider_model("custom") == "profile-pin"

    config.update_provider("custom", model="two")

    assert config.load_config()["selected_model"] is None
    assert config.get_provider_model("custom") == "two"


def test_update_provider_model_keeps_the_pin_of_another_primary():
    config.add_provider("custom", "Custom", "https://custom.example/v1", "one")
    config.set_active_providers(["openai", "custom"])
    cfg = config.load_config()
    cfg["selected_model"] = "openai-pin"
    config.save_config(cfg)

    config.update_provider("custom", model="two")

    assert config.get_provider_model("openai") == "openai-pin"
    assert config.get_provider_model("custom") == "two"


def test_update_provider_sets_a_model_for_a_provider_added_without_one():
    config.add_provider("custom", "Custom", "https://custom.example/v1")

    config.update_provider("custom", model="later-model")

    assert config.get_provider_model("custom") == "later-model"


def test_update_provider_leaves_the_model_alone_when_not_given():
    config.add_provider("custom", "Custom", "https://custom.example/v1", "one")

    config.update_provider("custom", name="Renamed")

    assert config.get_provider_config("custom").name == "Renamed"
    assert config.get_provider_model("custom") == "one"


def test_remove_provider_drops_its_chosen_model():
    config.add_provider("custom", "Custom", "https://custom.example/v1", "m")
    config.set_provider_model("openai", "gpt-chosen")

    config.remove_provider("custom")

    assert config.load_config()["provider_models"] == {"openai": "gpt-chosen"}
