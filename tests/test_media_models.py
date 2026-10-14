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


# ---------------------------------------------------------------------------
# Catalog kinds
# ---------------------------------------------------------------------------

_LISTING = {
    "data": [
        {"id": "gpt-4o", "created": 5},
        {"id": "gpt-image-1", "created": 4},
        {"id": "sora-2", "created": 3},
        {"id": "text-embedding-3", "created": 2},
    ]
}


def test_is_media_model_prefers_modalities_over_names():
    both = {"architecture": {"output_modalities": ["image", "text"]}}
    text_only = {"architecture": {"output_modalities": ["text"]}}

    assert catalog.is_media_model(both, "vendor/gemini-pic", "image")
    assert not catalog.is_media_model(both, "vendor/gemini-pic", "video")
    assert not catalog.is_media_model(text_only, "vendor/image-describer", "image")
    assert catalog.is_media_model({}, "gpt-image-1", "image")
    assert catalog.is_media_model({}, "google/veo-3", "video")
    assert not catalog.is_media_model({}, "gpt-4o", "image")


def test_parse_models_response_by_kind():
    assert [m.id for m in catalog.parse_models_response(_LISTING)] == ["gpt-4o"]
    assert [m.id for m in catalog.parse_models_response(_LISTING, kind="image")] == [
        "gpt-image-1"
    ]
    assert [m.id for m in catalog.parse_models_response(_LISTING, kind="video")] == [
        "sora-2"
    ]


def _listing_client(monkeypatch, seen: list[str]):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=_LISTING, request=request)

    transport = httpx.MockTransport(handler)
    real = catalog.httpx.Client

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real(*args, **kwargs)

    monkeypatch.setattr(catalog.httpx, "Client", factory)


def test_media_catalogs_are_cached_separately_from_chat(monkeypatch):
    config.set_key("openai", "sk-test")
    seen: list[str] = []
    _listing_client(monkeypatch, seen)

    images = config.get_model_catalog("openai", refresh=True, kind="image")
    videos = config.get_model_catalog("openai", refresh=True, kind="video")
    chat = config.get_model_catalog("openai", refresh=True)

    assert images.models == ["gpt-image-1"] and images.source == "live"
    assert videos.models == ["sora-2"]
    assert chat.models == ["gpt-4o"]
    assert set(config.load_model_cache()["providers"]) >= {
        "openai", "openai:image", "openai:video"
    }
    # A second image lookup is served from the cache: no new request.
    before = len(seen)
    again = config.get_model_catalog("openai", refresh=True, kind="image")
    assert again.source == "cache" and len(seen) == before


def test_media_catalog_without_a_key_or_cache_is_empty_not_an_error():
    result = config.get_model_catalog("openai", refresh=True, kind="image")

    assert result.models == []
    assert result.source == "curated"


def test_search_model_catalog_by_kind(monkeypatch):
    config.set_key("openai", "sk-test")
    _listing_client(monkeypatch, [])

    assert config.search_model_catalog("openai", "sora", refresh=True, kind="video") == [
        "sora-2"
    ]
    assert config.search_model_catalog("openai", "sora", refresh=True, kind="image") == []


def test_media_catalog_with_no_matching_models_reports_the_error(monkeypatch):
    config.set_key("openai", "sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "gpt-4o"}]}, request=request)

    real = catalog.httpx.Client
    monkeypatch.setattr(
        catalog.httpx,
        "Client",
        lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(handler)}),
    )

    result = config.get_model_catalog("openai", force=True, kind="video")

    assert result.models == []
    assert "no usable video models" in (result.error or "")


# ---------------------------------------------------------------------------
# Model filters on media catalogs
# ---------------------------------------------------------------------------


def test_chat_allow_list_does_not_leak_into_media_catalogs(monkeypatch):
    config.set_key("openai", "sk-test")
    _listing_client(monkeypatch, [])
    config.set_model_filter("openai", "allow", ["gpt-4o"])

    assert config.list_visible_models("openai", refresh=True, kind="image") == []
    assert config.search_model_catalog("openai", "gpt", refresh=True, kind="image") == []
    assert config.list_visible_models("openai", refresh=True) == ["gpt-4o"]


def test_allow_list_intersects_the_media_catalog(monkeypatch):
    config.set_key("openai", "sk-test")
    _listing_client(monkeypatch, [])
    config.set_model_filter("openai", "allow", ["gpt-4o", "gpt-image-1"])

    assert config.list_visible_models("openai", refresh=True, kind="image") == [
        "gpt-image-1"
    ]
    assert config.list_visible_models("openai", refresh=True, kind="video") == []


def test_deny_list_hides_a_media_model(monkeypatch):
    config.set_key("openai", "sk-test")
    _listing_client(monkeypatch, [])
    config.set_model_filter("openai", "deny", ["gpt-image-1"])

    assert config.list_visible_models("openai", refresh=True, kind="image") == []
    assert config.list_visible_models("openai", refresh=True, kind="video") == [
        "sora-2"
    ]


def test_get_model_catalog_rejects_an_unknown_kind():
    with pytest.raises(ValueError):
        config.get_model_catalog("openai", kind="bogus")


# ---------------------------------------------------------------------------
# media.py
# ---------------------------------------------------------------------------


def test_generate_image_without_a_model_names_the_fix(session):
    config.set_media(enabled=True)
    config.set_key("openai", "sk-test")

    with pytest.raises(media_module.MediaError) as exc:
        media_module.generate_image("a cat", session=session)

    assert "/image model" in str(exc.value)


def test_generate_video_without_a_model_names_the_fix(session):
    config.set_media(enabled=True)
    config.set_key("openai", "sk-test")

    with pytest.raises(media_module.MediaError) as exc:
        media_module.generate_video("a boat", session=session)

    assert "/video model" in str(exc.value)


def test_generate_image_uses_the_model_of_the_requested_provider(session, monkeypatch):
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen["payload"] = _json.loads(request.content)
        return httpx.Response(
            200, json={"data": [{"b64_json": "iVBORw0KGgo="}]}, request=request
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        media_module, "_client", lambda timeout: httpx.Client(timeout=timeout, transport=transport)
    )
    config.set_media(enabled=True)  # media.provider stays openai
    config.set_key("openrouter", "sk-or")
    config.set_media_model("openai", "image", "openai-model")
    config.set_media_model("openrouter", "image", "or-model")

    try:
        media_module.generate_image("a cat", session=session, provider_id="openrouter")
    except media_module.MediaError:
        pass  # the stub body is not a valid PNG; only the request matters

    assert seen["payload"]["model"] == "or-model"


def test_readiness_reports_a_missing_image_model():
    config.set_media(enabled=True)
    config.set_key("openai", "sk-test")

    assert media_module.readiness() == [media_module.NO_IMAGE_MODEL_MESSAGE]
    config.set_media_model("openai", "image", "gpt-image-1")
    assert media_module.readiness() == []


def test_format_models_lists_each_provider():
    config.set_media_model("openai", "image", "gpt-image-1")
    config.set_media_model("openrouter", "video", "vendor/video-2")

    lines = media_module.format_models(config.get_media())

    assert lines == [
        "openai: image gpt-image-1, video none chosen",
        "openrouter: image none chosen, video vendor/video-2",
    ]
    assert media_module.format_models(
        {"models": {}}
    ) == []


# ---------------------------------------------------------------------------
# /image model and /video model
# ---------------------------------------------------------------------------


class _Selector:
    """A selector that picks a fixed value (or calls ``choice(prompt)``) and records prompts."""

    def __init__(self, choice):
        self.choice = choice
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.choice(prompt) if callable(self.choice) else self.choice


def _install_listing(monkeypatch):
    seen: list[str] = []
    _listing_client(monkeypatch, seen)
    return seen


def test_image_model_without_arguments_shows_the_current_choice(session):
    console = _console()
    dispatch("/image model", session, console)
    assert "none chosen" in _output(console)

    config.set_media_model("openai", "image", "gpt-image-1")
    console = _console()
    dispatch("/image model", session, console)
    assert "gpt-image-1" in _output(console)


def test_image_model_sets_by_name_and_does_not_require_media_to_be_enabled(session):
    console = _console()

    dispatch("/image model my-image-model", session, console)

    assert config.get_media()["enabled"] is False
    assert config.get_media_model("openai", "image") == "my-image-model"
    assert "my-image-model" in _output(console)


def test_video_model_sets_the_video_model_only(session):
    dispatch("/video model vendor/video-2", session, _console())

    assert config.get_media_model("openai", "video") == "vendor/video-2"
    assert config.get_media_model("openai", "image") == ""


def test_image_model_list_opens_a_picker_and_saves_the_choice(session, monkeypatch):
    config.set_key("openai", "sk-test")
    _install_listing(monkeypatch)
    selector = _Selector("gpt-image-1")

    dispatch("/image model list", session, _console(), selector)

    assert [o.value for o in selector.prompts[0].options] == ["gpt-image-1"]
    assert config.get_media_model("openai", "image") == "gpt-image-1"


def test_video_model_list_without_a_selector_prints_a_table(session, monkeypatch):
    config.set_key("openai", "sk-test")
    _install_listing(monkeypatch)
    console = _console()

    dispatch("/video model list", session, console)

    assert "sora-2" in _output(console)
    assert config.get_media_model("openai", "video") == ""


def test_image_model_search_filters_the_full_catalog(session, monkeypatch):
    config.set_key("openai", "sk-test")
    _install_listing(monkeypatch)
    selector = _Selector("gpt-image-1")

    dispatch("/image model search image", session, _console(), selector)

    assert [o.value for o in selector.prompts[0].options] == ["gpt-image-1"]
    console = _console()
    dispatch("/image model search zzz", session, console)
    assert "No" in _output(console) and "zzz" in _output(console)


def test_image_model_refresh_forces_a_new_fetch(session, monkeypatch):
    config.set_key("openai", "sk-test")
    seen = _install_listing(monkeypatch)

    dispatch("/image model list", session, _console())
    first = len(seen)
    dispatch("/image model refresh", session, _console())

    assert len(seen) == first + 1


def test_image_without_a_model_opens_the_picker_then_generates(session, monkeypatch):
    config.set_key("openai", "sk-test")
    config.set_media(enabled=True)
    _install_listing(monkeypatch)
    calls = []

    def fake(prompt, *, session, size=None, model=None, provider_id=None):
        calls.append(model)
        raise media_module.MediaError("stop here")

    monkeypatch.setattr(media_module, "generate_image", fake)
    selector = _Selector("gpt-image-1")

    dispatch("/image a cat", session, _console(), selector)

    assert config.get_media_model("openai", "image") == "gpt-image-1"
    assert calls == [None]  # the saved model is picked up by generate_image


def test_image_without_a_model_and_without_a_selector_names_the_fix(session, monkeypatch):
    config.set_media(enabled=True)
    monkeypatch.setattr(
        media_module,
        "generate_image",
        lambda *a, **k: pytest.fail("must not generate without a model"),
    )
    console = _console()

    dispatch("/image a cat", session, console)

    assert "/image model" in _output(console)


def test_video_without_a_model_and_without_a_selector_names_the_fix(session):
    config.set_media(enabled=True)
    console = _console()

    dispatch("/video a boat", session, console)

    assert "/video model" in _output(console)


def test_per_call_model_override_is_not_saved(session, monkeypatch):
    config.set_media(enabled=True)
    calls = []

    def fake(prompt, *, session, size=None, model=None, provider_id=None):
        calls.append(model)
        raise media_module.MediaError("stop here")

    monkeypatch.setattr(media_module, "generate_image", fake)

    dispatch("/image --model one-off a cat", session, _console())

    assert calls == ["one-off"]
    assert config.get_media_model("openai", "image") == ""


def test_a_prompt_starting_with_model_needs_a_leading_double_dash(session, monkeypatch):
    config.set_media(enabled=True)
    config.set_media_model("openai", "image", "gpt-image-1")
    prompts = []

    def fake(prompt, *, session, size=None, model=None, provider_id=None):
        prompts.append(prompt)
        raise media_module.MediaError("stop here")

    monkeypatch.setattr(media_module, "generate_image", fake)

    dispatch("/image -- model railway at dusk", session, _console())

    assert prompts == ["model railway at dusk"]


def test_image_model_with_several_words_is_not_saved_as_an_id(session):
    console = _console()

    dispatch("/image model a red fox", session, console)

    assert config.get_media_model("openai", "image") == ""
    assert "Usage: /image model [list|refresh|search <term>|<id>]" in _output(console)


def _help_descriptions(width: int, session) -> str:
    """The text of every help Description cell, rewrapped onto one line."""
    console = Console(file=io.StringIO(), force_terminal=False, width=width)
    dispatch("/help", session, console)
    cells = []
    for line in _output(console).splitlines():
        parts = [part.strip() for part in line.split("│")]
        if len(parts) >= 4:  # │ command │ description │
            cells.append(parts[2])
    return " ".join(" ".join(cells).split())


def test_help_keeps_the_media_model_descriptions_visible_at_100_columns(session):
    text = _help_descriptions(100, session)

    assert (
        "Choose the image model (remembered per provider). A prompt starting "
        "with the word 'model' needs `--` first, e.g. `/image -- model train`."
    ) in text
    assert "Choose the video model (remembered per provider)." in text
    assert (
        "Show readiness and recent generated files, or choose the media provider."
    ) in text


# ---------------------------------------------------------------------------
# /media provider and /config media
# ---------------------------------------------------------------------------


def test_media_provider_without_a_selector_prints_a_table(session):
    config.set_media_model("openai", "image", "gpt-image-1")
    console = _console()

    dispatch("/media provider", session, console)

    output = _output(console)
    assert "openai" in output and "openrouter" in output
    assert "gpt-image-1" in output


def test_media_provider_by_id_switches_and_offers_both_model_pickers(session, monkeypatch):
    config.set_key("openrouter", "sk-or")
    _install_listing(monkeypatch)
    selector = _Selector(lambda prompt: prompt.options[0].value)

    dispatch("/media provider openrouter", session, _console(), selector)

    assert config.get_media()["provider"] == "openrouter"
    assert config.get_media_model("openrouter", "image") == "gpt-image-1"
    assert config.get_media_model("openrouter", "video") == "sora-2"
    assert len(selector.prompts) == 2


def test_media_provider_skips_pickers_for_kinds_that_already_have_a_model(session):
    config.set_media_model("openrouter", "image", "keep-me")
    config.set_media_model("openrouter", "video", "keep-me-too")
    selector = _Selector(None)

    dispatch("/media provider openrouter", session, _console(), selector)

    assert selector.prompts == []
    assert config.get_media()["model"] == "keep-me"


def test_media_provider_without_a_selector_prints_the_hints(session):
    console = _console()

    dispatch("/media provider openrouter", session, console)

    output = _output(console)
    assert config.get_media()["provider"] == "openrouter"
    assert "/image model" in output and "/video model" in output


def test_media_provider_rejects_an_unknown_id(session):
    console = _console()

    dispatch("/media provider nope", session, console)

    assert config.get_media()["provider"] == "openai"
    assert "nope" in _output(console)


def test_media_status_lists_models_per_provider(session):
    config.set_media_model("openai", "image", "gpt-image-1")
    config.set_media_model("openrouter", "video", "vendor/video-2")
    console = _console()

    dispatch("/media", session, console)

    output = _output(console)
    assert "openai: image gpt-image-1" in output
    assert "openrouter: image none chosen, video vendor/video-2" in output


def test_config_media_model_takes_an_optional_provider(session):
    dispatch("/config media model gpt-image-1", session, _console())
    dispatch("/config media model or-image openrouter", session, _console())
    dispatch("/config media video-model vendor/video-2 openrouter", session, _console())

    assert config.get_media_model("openai", "image") == "gpt-image-1"
    assert config.get_media_model("openrouter", "image") == "or-image"
    assert config.get_media_model("openrouter", "video") == "vendor/video-2"

    dispatch("/config media video-model none openrouter", session, _console())
    assert config.get_media_model("openrouter", "video") == ""


def test_config_media_status_says_none_chosen_for_the_image_model(session):
    console = _console()

    dispatch("/config media show", session, console)

    assert "image model none chosen" in _output(console)


# ---------------------------------------------------------------------------
# CLI and tools
# ---------------------------------------------------------------------------


def test_cli_media_model_and_video_model_take_a_provider():
    from typer.testing import CliRunner

    from kiwimatecoder.main import app

    runner = CliRunner()
    assert runner.invoke(app, ["config", "media", "model", "m1"]).exit_code == 0
    result = runner.invoke(
        app, ["config", "media", "model", "m2", "--provider", "openrouter"]
    )
    assert result.exit_code == 0
    assert runner.invoke(
        app, ["config", "media", "video-model", "v1", "--provider", "openrouter"]
    ).exit_code == 0

    assert config.get_media_model("openai", "image") == "m1"
    assert config.get_media_model("openrouter", "image") == "m2"
    assert config.get_media_model("openrouter", "video") == "v1"

    assert runner.invoke(
        app, ["config", "media", "video-model", "none", "--provider", "openrouter"]
    ).exit_code == 0
    assert config.get_media_model("openrouter", "video") == ""

    bad = runner.invoke(app, ["config", "media", "model", "x", "--provider", "nope"])
    assert bad.exit_code == 1


def test_cli_media_show_lists_models_per_provider():
    from typer.testing import CliRunner

    from kiwimatecoder.main import app

    config.set_media_model("openrouter", "image", "or-image")

    result = CliRunner().invoke(app, ["config", "media", "show"])
    output = " ".join(result.output.split())

    assert "Image model: none chosen" in output
    assert "openrouter: image or-image" in output


def test_tool_previews_say_none_chosen(session):
    from kiwimatecoder.tools.image import generate_image_preview, generate_video_preview

    assert "(none chosen)" in generate_image_preview({"prompt": "a cat"}, session)
    assert "(none chosen)" in generate_video_preview({"prompt": "a boat"}, session)
    config.set_media_model("openai", "image", "gpt-image-1")
    assert "gpt-image-1" in generate_image_preview({"prompt": "a cat"}, session)


def test_generate_image_tool_without_a_model_returns_guidance(session):
    from kiwimatecoder.tools.image import generate_image_tool

    config.set_media(enabled=True)
    config.set_key("openai", "sk-test")

    result = generate_image_tool.func({"prompt": "a cat"}, session)

    assert "/image model" in result.content
