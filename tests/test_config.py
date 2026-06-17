import json

import pytest

from kiwimatecoder import config


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


def test_set_key_unknown_provider_raises():
    with pytest.raises(KeyError):
        config.set_key("nope", "x")


def test_legacy_shims():
    config.save_api_key("shim-key")
    assert config.load_api_key() == "shim-key"
    assert config.get_key("openrouter") == "shim-key"
