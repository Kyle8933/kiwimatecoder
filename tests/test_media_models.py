"""Per-provider image/video models: config, catalog, commands, tools."""

from __future__ import annotations

import io

import httpx
import pytest
from rich.console import Console

from kiwimatecoder import catalog, config
from kiwimatecoder import media as media_module
from kiwimatecoder.commands import dispatch
from kiwimatecoder.providers import REGISTRY


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for provider in REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def _output(console: Console) -> str:
    return console.file.getvalue()


# ---------------------------------------------------------------------------
# Config storage
# ---------------------------------------------------------------------------


def test_no_media_model_is_chosen_by_default():
    settings = config.get_media()
    assert settings["model"] == ""
    assert settings["video_model"] == ""
    assert settings["models"] == {}
    assert config.get_media_model("openai", "image") == ""


def test_set_media_model_is_per_provider_and_kind():
    config.set_media_model("openai", "image", "gpt-image-1")
    config.set_media_model("openrouter", "video", "vendor/video-2")

    assert config.get_media_model("openai", "image") == "gpt-image-1"
    assert config.get_media_model("openai", "video") == ""
    assert config.get_media_model("openrouter", "video") == "vendor/video-2"
    stored = config.load_config()["media"]
    assert stored["models"] == {
        "openai": {"image": "gpt-image-1"},
        "openrouter": {"video": "vendor/video-2"},
    }
    assert "model" not in stored
    assert "video_model" not in stored


def test_switching_provider_switches_the_effective_models():
    config.set_media_model("openai", "image", "model-a")
    config.set_media_model("openrouter", "image", "model-b")

    assert config.set_media(provider="openrouter")["model"] == "model-b"
    assert config.get_media()["model"] == "model-b"
    assert config.set_media(provider="openai")["model"] == "model-a"


def test_set_media_with_provider_and_model_in_one_call_uses_the_new_provider():
    settings = config.set_media(provider="openrouter", model="model-b")

    assert settings["model"] == "model-b"
    assert config.get_media_model("openrouter", "image") == "model-b"
    assert config.get_media_model("openai", "image") == ""


def test_set_media_model_forgets_and_prunes():
    config.set_media_model("openai", "image", "model-a")
    config.set_media_model("openai", "image", None)

    assert config.get_media_model("openai", "image") == ""
    assert config.load_config()["media"]["models"] == {}


def test_set_media_model_validates_kind_and_provider():
    with pytest.raises(ValueError, match="kind"):
        config.set_media_model("openai", "audio", "x")
    with pytest.raises(ValueError, match="Unknown provider"):
        config.set_media_model("nope", "image", "x")


def test_legacy_keys_apply_to_the_stored_provider_and_are_dropped_on_write():
    cfg = config.load_config()
    cfg["media"] = {
        "enabled": True,
        "provider": "openrouter",
        "model": "legacy-image",
        "video_model": "legacy-video",
    }
    config.save_config(cfg)

    assert config.get_media_model("openrouter", "image") == "legacy-image"
    assert config.get_media_model("openrouter", "video") == "legacy-video"
    assert config.get_media_model("openai", "image") == ""

    config.set_media(size="512x512")

    stored = config.load_config()["media"]
    assert stored["models"] == {
        "openrouter": {"image": "legacy-image", "video": "legacy-video"}
    }
    assert "model" not in stored
    assert "video_model" not in stored


def test_malformed_media_models_are_ignored_not_raised():
    cfg = config.load_config()
    cfg["media"]["models"] = {
        "openai": "not-an-object",
        "": {"image": "x"},
        "openrouter": {"image": 3, "video": " vid "},
    }
    config.save_config(cfg)

    assert config.get_media()["models"] == {"openrouter": {"video": "vid"}}
    cfg["media"]["models"] = ["nope"]
    config.save_config(cfg)
    assert config.get_media()["models"] == {}


def test_validate_config_flags_bad_media_models():
    cfg = config._empty_config()
    cfg["media"]["models"] = {"openai": "x", "openrouter": {"audio": "y", "image": 3}}

    keys = {i["key"] for i in config.validate_config(cfg) if i["level"] == "error"}

    assert {"media.models.openai", "media.models.openrouter.audio",
            "media.models.openrouter.image"} <= keys

    cfg["media"]["models"] = []
    keys = {i["key"] for i in config.validate_config(cfg) if i["level"] == "error"}
    assert "media.models" in keys


def test_remove_provider_drops_its_media_models_and_cached_catalogs():
    config.add_provider("local", "Local", "http://localhost:1234/v1")
    config.set_media_model("local", "image", "img-1")
    cache = config.load_model_cache()
    cache["providers"]["local:image"] = {"fetched_at": 1.0, "models": ["img-1"]}
    cache["providers"]["local:video"] = {"fetched_at": 1.0, "models": ["vid-1"]}
    config.save_model_cache(cache)

    config.remove_provider("local")

    assert "local" not in config.get_media()["models"]
    assert not [k for k in config.load_model_cache()["providers"] if k.startswith("local")]
