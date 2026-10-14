# Media Providers and Models Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/image`, `/video` and `/media` choose providers and models the way `/model` and `/provider` do for LLMs: per-provider remembered models, no built-in default, a live list you can refresh and search, and interactive pickers.

**Architecture:** Models move from two global strings (`media.model`, `media.video_model`) to `media.models = {provider_id: {"image": id, "video": id}}`, with the old keys read as a legacy fallback. The existing catalog code (`get_model_catalog`, cache, TTL, backoff, search, allow/deny filter) gains a `kind` of `chat | image | video` instead of a second catalog module. Slash commands, the `config media` CLI and the `generate_image` / `generate_video` tools all resolve models through one helper, `config.get_media_model(provider, kind)`.

**Tech Stack:** Python 3.14, httpx (mocked with `httpx.MockTransport` in tests), rich, typer, pytest.

**Spec:** `docs/superpowers/specs/2026-10-14-media-providers-models-design.md`

## Global Constraints

- `get_media()` and `set_media()` keep their current signatures; `get_media()["model"]` and `["video_model"]` stay and hold the effective model for `media.provider`.
- No built-in default model: `MEDIA_DEFAULTS["model"]` and the default config template's `media.model` become `""`.
- Legacy `media.model` / `media.video_model` stay readable and are dropped on the next media write.
- Existing chat catalog behaviour and cache keys are unchanged; media cache keys are `<provider_id>:image` and `<provider_id>:video`.
- `media.provider` stays a single provider shared by images and video. No fallback roster.
- A failed `/image model` or `/video model` fetch never raises; it falls back to cache, then to an empty list.
- All user-visible text going through rich is `escape()`d, as in the surrounding code.
- `--model <id>` on `/image` and `/video` stays a per-call override and is never persisted.

## Review Focus

Failure modes the spec implies but no happy-path test covers. Each has a test in the owning task.

1. A hand-edited `media.models` with wrong types (a string where an object belongs, an empty provider id, a number as a model) must not crash `get_media()` (Task 1).
2. A user with the old `media.model: "gpt-image-1"` and `media.provider: "openrouter"` keeps that model for openrouter after upgrade, not for openai (Task 1).
3. Removing or renaming a custom provider must not leave its media models or its `:image` / `:video` cache entries behind (Task 1).
4. A provider whose `/models` reply has no modality metadata and no matching ids yields "no usable" media models; the user must still be able to type an id (Task 2, Task 4).
5. `/image model railroad ...`: a prompt starting with the word `model` is read as the subcommand. The prompt must be reworded or prefixed with `--` (documented in the help text, tested in Task 4).

## File Structure

- `kiwimatecoder/config.py` — media models storage, `get_media_model`, `set_media_model`, validation, rename/remove cleanup, per-kind catalog cache.
- `kiwimatecoder/catalog.py` — media model filter, `kind` through parse and fetch.
- `kiwimatecoder/media.py` — resolve models per provider and kind, readiness, messages, `format_models`.
- `kiwimatecoder/commands.py` — `/image model`, `/video model`, `/media provider`, no-model pickers, `/config media` forms, help and completion.
- `kiwimatecoder/main.py` — `config media` CLI.
- `kiwimatecoder/tools/image.py` — approval previews.
- `tests/test_media_models.py` — new: all tests for this feature.
- Existing tests updated where they relied on the removed `gpt-image-1` default: `tests/test_media.py`, `tests/test_media_video.py`, `tests/test_config_menu.py`.
- `README.md`, `docs/kiwimatecoder.1` (regenerated) — docs.

Run tests with `venv/bin/python -m pytest`. Start by creating a branch: `git checkout -b media-models-parity`.

---

### Task 1: Per-provider media models in config

**Files:**
- Modify: `kiwimatecoder/config.py` (config template ~137; `MEDIA_DEFAULTS` ~2759; `get_media` ~2800; `set_media` ~2843; `validate_config` media block ~4566; `_rename_shadowed_custom_providers` ~508; `remove_provider` ~708; `clear_model_cache` ~4892)
- Create: `tests/test_media_models.py`
- Modify: `tests/test_media.py`, `tests/test_media_video.py`, `tests/test_config_menu.py` (default-model assertions, see Step 7)

**Interfaces:**
- Produces:
  - `MEDIA_KINDS: tuple[str, ...] = ("image", "video")`
  - `get_media_model(provider_id: str, kind: str, cfg: dict | None = None) -> str` (`""` when none chosen)
  - `set_media_model(provider_id: str, kind: str, model: str | None, cfg: dict | None = None) -> dict` (returns `get_media(cfg)`; `None` or `""` forgets the choice; `ValueError` for unknown kind or provider)
  - `get_media()` now also returns `"models": dict[str, dict[str, str]]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_media_models.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_media_models.py -v`
Expected: FAIL (`AttributeError: module 'kiwimatecoder.config' has no attribute 'get_media_model'` and similar).

- [ ] **Step 3: Implement storage in `config.py`**

Default config template (~line 137): replace the `"media"` block with

```python
        "media": {
            "enabled": False,
            "provider": "openai",
            "models": {},
            "size": "1024x1024",
            "output_dir": ".kiwimatecoder/media",
            "video_duration": 5,
            "video_size": "",
        },
```

`MEDIA_DEFAULTS` (~2759): set `"model": ""`, keep `"video_model": ""`, and add `"models": {}`. Then add below `MEDIA_DEFAULTS`:

```python
MEDIA_KINDS = ("image", "video")
_LEGACY_MEDIA_KEYS = {"image": "model", "video": "video_model"}


def _normalized_media_models(raw: object) -> dict[str, dict[str, str]]:
    """Keep only ``provider id -> {kind -> non-empty model id}`` string pairs.

    Stored raw so :func:`validate_config` can report bad entries; every reader
    and writer goes through here instead.
    """
    if not isinstance(raw, dict):
        return {}
    models: dict[str, dict[str, str]] = {}
    for provider_id, kinds in raw.items():
        if not isinstance(provider_id, str) or not provider_id.strip():
            continue
        if not isinstance(kinds, dict):
            continue
        chosen = {
            kind: value.strip()
            for kind in MEDIA_KINDS
            if isinstance(value := kinds.get(kind), str) and value.strip()
        }
        if chosen:
            models[provider_id.strip()] = chosen
    return models


def _media_models_with_legacy(
    stored: dict[str, Any], provider_id: str
) -> dict[str, dict[str, str]]:
    """``media.models`` plus the old global keys, applied to ``provider_id``."""
    models = _normalized_media_models(stored.get("models"))
    for kind, legacy_key in _LEGACY_MEDIA_KEYS.items():
        legacy = stored.get(legacy_key)
        if isinstance(legacy, str) and legacy.strip():
            models.setdefault(provider_id, {}).setdefault(kind, legacy.strip())
    return models
```

In `get_media`, delete the `model = ...` block and the `video_model = ...` block, and after the `provider` block add:

```python
    models = _media_models_with_legacy(stored, effective["provider"])
    effective["models"] = models
    chosen = models.get(effective["provider"], {})
    effective["model"] = chosen.get("image", "")
    effective["video_model"] = chosen.get("video", "")
```

In `set_media`: replace the `model is not None` and `video_model is not None` blocks with

```python
    if model is not None:
        cleaned_model = str(model).strip()
        if not cleaned_model:
            raise ValueError("media model must be a non-empty string.")
        current["models"].setdefault(current["provider"], {})["image"] = cleaned_model
    if video_model is not None:
        cleaned_video = str(video_model).strip()
        kinds = current["models"].setdefault(current["provider"], {})
        if cleaned_video:
            kinds["video"] = cleaned_video
        else:
            kinds.pop("video", None)
```

(they must stay after the `provider` block so a combined call uses the new provider), and replace the last three lines (`cfg["media"] = current` / `save_config(cfg)` / `return get_media(cfg)`) with `_save_media(cfg, current)` / `return get_media(cfg)`. Add:

```python
def _save_media(cfg: dict[str, Any], current: dict[str, Any]) -> None:
    """Persist ``current``; the effective model keys are derived, never stored."""
    current["models"] = {
        provider_id: kinds for provider_id, kinds in current["models"].items() if kinds
    }
    cfg["media"] = {
        key: value for key, value in current.items() if key not in {"model", "video_model"}
    }
    save_config(cfg)


def get_media_model(
    provider_id: str, kind: str, cfg: dict[str, Any] | None = None
) -> str:
    """Return the ``kind`` model chosen for ``provider_id``, or ``""``."""
    cfg = cfg or load_config()
    return get_media(cfg)["models"].get(provider_id, {}).get(kind, "")


def set_media_model(
    provider_id: str,
    kind: str,
    model: str | None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Remember the ``kind`` model for ``provider_id`` (None or "" forgets it)."""
    if kind not in MEDIA_KINDS:
        raise ValueError(f"media model kind must be one of: {', '.join(MEDIA_KINDS)}.")
    cfg = cfg or load_config()
    try:
        get_provider_config(provider_id, cfg)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    current = get_media(cfg)
    kinds = current["models"].setdefault(provider_id, {})
    cleaned = (model or "").strip()
    if cleaned:
        kinds[kind] = cleaned
    else:
        kinds.pop(kind, None)
    _save_media(cfg, current)
    return get_media(cfg)
```

`validate_config` media block: after the `video_model` check add

```python
            if "models" in media:
                raw_models = media["models"]
                if not isinstance(raw_models, dict):
                    add("error", "media.models", "'models' must be an object of provider ids.")
                else:
                    for pid, kinds in raw_models.items():
                        if not isinstance(kinds, dict):
                            add(
                                "error",
                                f"media.models.{pid}",
                                "must be an object with 'image' and/or 'video'.",
                            )
                            continue
                        for kind, value in kinds.items():
                            if kind not in MEDIA_KINDS:
                                add(
                                    "error",
                                    f"media.models.{pid}.{kind}",
                                    "kind must be 'image' or 'video'.",
                                )
                            elif not isinstance(value, str) or not value.strip():
                                add(
                                    "error",
                                    f"media.models.{pid}.{kind}",
                                    "model must be a non-empty string.",
                                )
```

`_rename_shadowed_custom_providers`: inside the `for provider_id in shadowed:` loop, right after the `provider_models` rename, add

```python
        media = cfg.get("media")
        if isinstance(media, dict) and isinstance(media.get("models"), dict):
            if provider_id in media["models"]:
                media["models"][new_id] = media["models"].pop(provider_id)
```

`remove_provider`: after `cfg["provider_models"] = models` add

```python
    media = cfg.get("media")
    if isinstance(media, dict) and isinstance(media.get("models"), dict):
        media["models"].pop(provider_id, None)
```

`clear_model_cache`: replace `cache["providers"].pop(provider_id, None)` with

```python
        for key in (provider_id, *(f"{provider_id}:{kind}" for kind in MEDIA_KINDS)):
            cache["providers"].pop(key, None)
```

- [ ] **Step 4: Run the new tests**

Run: `venv/bin/python -m pytest tests/test_media_models.py -v`
Expected: PASS.

- [ ] **Step 5: Update tests that relied on the removed default**

In `tests/test_media.py`, make `_enable_media` choose a model:

```python
def _enable_media(**kwargs) -> None:
    config.set_media(enabled=True, **kwargs)
    config.set_media_model("openai", "image", "gpt-image-1")
    config.set_key("openai", "sk-test")
```

In `tests/test_media_video.py` the same helper (if it has its own) gets the same `set_media_model` line. In `tests/test_config_menu.py` (~lines 389 and 396) change `== "gpt-image-1"` to `== ""`, and line ~326 area if a label asserts the old image model text: make it `"Image model — none chosen"` (see Task 5 Step 3 for the label change). In `tests/test_media_video.py` ~861 (`assert "gpt-image-1" in output`) call `config.set_media_model("openai", "image", "gpt-image-1")` before dispatching. Anywhere else a test calls `config.set_media(enabled=True)` and then generates an image, add the `set_media_model` line.

- [ ] **Step 6: Run the whole suite**

Run: `venv/bin/python -m pytest -q`
Expected: PASS. Fix any remaining test that assumed the default and note it here.

- [ ] **Step 7: Commit**

```bash
git add kiwimatecoder/config.py tests/
git commit -m "feat: store media models per provider and kind"
```

---

### Task 2: Catalog kinds (chat, image, video)

**Files:**
- Modify: `kiwimatecoder/catalog.py` (`parse_models_response` ~216, `fetch_models` ~261)
- Modify: `kiwimatecoder/config.py` (`get_model_catalog` ~4930, `list_visible_models`, `search_model_catalog`)
- Test: `tests/test_media_models.py`

**Interfaces:**
- Consumes: `config.MEDIA_KINDS` (Task 1).
- Produces:
  - `catalog.MEDIA_NAME_MARKERS: dict[str, tuple[str, ...]]`
  - `catalog.is_media_model(entry: dict, model_id: str, kind: str) -> bool`
  - `catalog.parse_models_response(payload, *, require_tools=True, kind="chat")`
  - `catalog.fetch_models(provider, api_key=None, *, timeout, transport=None, kind="chat")`
  - `config.get_model_catalog(provider_id, *, refresh=False, force=False, keep=(), timeout=None, limit=..., cfg=None, kind="chat") -> ModelCatalog`
  - `config.list_visible_models(..., kind="chat")` and `config.search_model_catalog(..., kind="chat")`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_media_models.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_media_models.py -k "media_model or by_kind or catalog" -v`
Expected: FAIL (`AttributeError: ... 'is_media_model'`, `TypeError ... unexpected keyword 'kind'`).

- [ ] **Step 3: Implement in `catalog.py`**

After `NON_CHAT_MARKERS` add:

```python
# Name fragments that mark an id as a model of the given media kind; used only
# when the provider's listing has no modality metadata.
MEDIA_NAME_MARKERS: dict[str, tuple[str, ...]] = {
    "image": ("image", "imagen", "dall-e", "flux", "diffusion"),
    "video": ("video", "veo", "sora"),
}
```

After `is_chat_model` add:

```python
def is_media_model(entry: dict[str, Any], model_id: str, kind: str) -> bool:
    """Return whether a listed model produces ``kind`` ("image" or "video").

    ``architecture.output_modalities`` wins when the provider reports it
    (OpenRouter does); otherwise the id is matched against
    :data:`MEDIA_NAME_MARKERS`.
    """
    lowered = model_id.lower()
    if not lowered:
        return False
    architecture = entry.get("architecture")
    if isinstance(architecture, dict):
        outputs = architecture.get("output_modalities")
        if isinstance(outputs, list) and outputs:
            return kind in outputs
    return any(marker in lowered for marker in MEDIA_NAME_MARKERS[kind])
```

`parse_models_response(payload, *, require_tools=True, kind="chat")`: replace the `if not is_chat_model(entry, model_id, require_tools=require_tools): continue` with

```python
        if kind == "chat":
            usable = is_chat_model(entry, model_id, require_tools=require_tools)
        else:
            usable = is_media_model(entry, model_id, kind)
        if not usable:
            continue
```

Update the docstring line "Extract usable chat models" to "Extract usable models of ``kind`` (chat by default)".

`fetch_models(..., transport=None, kind: str = "chat")`: change the parse call to `parse_models_response(payload, require_tools=provider.supports_tools, kind=kind)` and the error to `f"{provider.name} listed no usable {kind} models"`.

- [ ] **Step 4: Implement in `config.py`**

`get_model_catalog`: add parameter `kind: str = "chat"` (after `cfg`). At the top:

```python
    if kind != "chat" and kind not in MEDIA_KINDS:
        raise ValueError(f"Unknown model kind '{kind}'.")
    cache_key = provider_id if kind == "chat" else f"{provider_id}:{kind}"
```

Replace `curated = _curated_models(provider)` with `curated = _curated_models(provider) if kind == "chat" else []`. Replace both `cache["providers"].get(provider_id)` and `cache["providers"][provider_id] = ...` uses (three places: the `entry =` lookup, the `except CatalogFetchError` write, the success write) with `cache_key`. Replace the fetch call with

```python
            remote = catalog.fetch_models(
                provider,
                get_key(provider_id),
                timeout=timeout if timeout is not None else catalog.FETCH_TIMEOUT,
                **({} if kind == "chat" else {"kind": kind}),
            )
```

Extend the docstring: "``kind`` selects the chat catalog or the image / video one; media catalogs have no curated list."

`list_visible_models` and `search_model_catalog`: add `kind: str = "chat"` as the last keyword parameter and pass `kind=kind` to their `get_model_catalog` calls.

- [ ] **Step 5: Run the tests**

Run: `venv/bin/python -m pytest tests/test_media_models.py tests/test_catalog.py -v`
Expected: PASS (existing catalog tests unchanged).

- [ ] **Step 6: Run the whole suite and commit**

Run: `venv/bin/python -m pytest -q` — expected PASS.

```bash
git add kiwimatecoder/catalog.py kiwimatecoder/config.py tests/test_media_models.py
git commit -m "feat: model catalogs for image and video kinds"
```

---

### Task 3: Resolve media models per provider and kind

**Files:**
- Modify: `kiwimatecoder/media.py` (messages ~73; `readiness` ~336; `generate_image` ~376-400; `generate_video` ~752-785)
- Test: `tests/test_media_models.py`

**Interfaces:**
- Consumes: `config.get_media_model(provider_id, kind, cfg)` (Task 1).
- Produces:
  - `media.NO_IMAGE_MODEL_MESSAGE: str`, `media.NO_VIDEO_MODEL_MESSAGE: str` (updated text)
  - `media.format_models(settings: dict) -> list[str]` — plain-text lines, one per provider with any chosen model, e.g. `"openai: image gpt-image-1, video none"`; `[]` when none.
  - `generate_image` / `generate_video` read the model for the resolved provider (the `provider_id` argument now decides which provider's model is used).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_media_models.py`:

```python
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
```

The `session` fixture comes from `tests/conftest.py`.

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_media_models.py -k "without_a_model or requested_provider or readiness or format_models" -v`
Expected: FAIL (`NO_IMAGE_MODEL_MESSAGE`/`format_models` missing; image generation sends the old global model).

- [ ] **Step 3: Implement in `media.py`**

Replace `NO_VIDEO_MODEL_MESSAGE` with:

```python
NO_IMAGE_MODEL_MESSAGE = (
    "No image model chosen. Pick one with `/image model` "
    "(or `kiwimatecoder config media model <id>`)."
)
NO_VIDEO_MODEL_MESSAGE = (
    "No video model chosen. Pick one with `/video model` "
    "(or `kiwimatecoder config media video-model <id>`)."
)
```

Add after `format_bytes`:

```python
def format_models(settings: dict[str, Any]) -> list[str]:
    """One plain-text line per provider that has a chosen image or video model."""
    lines: list[str] = []
    for provider_id, kinds in sorted(settings.get("models", {}).items()):
        image = kinds.get("image") or "none chosen"
        video = kinds.get("video") or "none chosen"
        lines.append(f"{provider_id}: image {image}, video {video}")
    return lines
```

`readiness`: replace the final `if video and not settings["video_model"]:` block with

```python
    kind = "video" if video else "image"
    if not config.get_media_model(settings["provider"], kind, cfg):
        problems.append(NO_VIDEO_MODEL_MESSAGE if video else NO_IMAGE_MODEL_MESSAGE)
```

`generate_image`: delete `del resolved_provider` and replace the `chosen_model = str(model or settings["model"]).strip()` line with

```python
    chosen_model = str(
        model or config.get_media_model(resolved_provider, "image", cfg)
    ).strip()
    if not chosen_model:
        raise MediaError(NO_IMAGE_MODEL_MESSAGE)
```

(keep whatever follows it, such as an existing empty-model check, only if it is not now redundant.) `generate_video`: delete `del resolved_provider`; replace `chosen_model = str(model or settings["video_model"]).strip()` with `chosen_model = str(model or config.get_media_model(resolved_provider, "video", cfg)).strip()` (the existing `if not chosen_model: raise MediaError(NO_VIDEO_MODEL_MESSAGE)` stays).

- [ ] **Step 4: Run tests and fix messages in older tests**

Run: `venv/bin/python -m pytest tests/test_media_models.py tests/test_media.py tests/test_media_video.py -v`
Expected: PASS. If an older test asserts the old message text (`/config media video-model`), change it to assert `/video model`.

- [ ] **Step 5: Commit**

```bash
git add kiwimatecoder/media.py tests/
git commit -m "feat: resolve media models per provider and kind"
```

---

### Task 4: `/image model`, `/video model`, and model pickers

**Files:**
- Modify: `kiwimatecoder/commands.py` (`_report_catalog` ~395; `_provider_catalog` ~445; `model_selection_prompt` ~469; `dispatch` ~242-300; `_image` ~5516; `_video` ~5572; `_COMMANDS` ~5773; help ~5932; completion tables ~6020 and ~6171)
- Test: `tests/test_media_models.py`

**Interfaces:**
- Consumes: `config.get_media_model/set_media_model`, `config.get_model_catalog(kind=...)`, `config.search_model_catalog(kind=...)`, `media.NO_IMAGE_MODEL_MESSAGE`, `media.NO_VIDEO_MODEL_MESSAGE`.
- Produces:
  - `_provider_catalog(provider, current, console, *, force=False, verbose=False, kind="chat")`
  - `model_selection_prompt(provider, current="", console=None, *, force=False, title=None, kind="chat")`
  - `_media_model(kind: str, arg: str, console: Console, selector: CommandSelector | None) -> None`
  - `_pick_media_model(kind: str, provider: ProviderConfig, current: str, console: Console, selector: CommandSelector) -> str` (returns the chosen id after saving it, or `""`)
  - `_image(arg, session, console, selector=None)`, `_video(...)`, `_media(...)` accept a `selector`; `dispatch` passes it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_media_models.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_media_models.py -k "image_model or video_model or without_a_model or override or double_dash" -v`
Expected: FAIL.

- [ ] **Step 3: Generalise the catalog helpers in `commands.py`**

`_report_catalog`: add keyword parameter `command: str = "/model"` and change the string `"Pick another with /model."` to `f"Pick another with {command}."`.

`_provider_catalog`: add `kind: str = "chat"`:

```python
def _provider_catalog(
    provider: ProviderConfig,
    current: str,
    console: Console,
    *,
    force: bool = False,
    verbose: bool = False,
    kind: str = "chat",
) -> ModelCatalog:
    """Resolve ``provider``'s catalog (``current`` pinned first) and report changes."""
    noun = "models" if kind == "chat" else f"{kind} models"
    with console.status(f"Checking {escape(provider.name)} for new {noun}…"):
        catalog = get_model_catalog(
            provider.id, refresh=True, force=force, keep=(current,), kind=kind
        )
    command = "/model" if kind == "chat" else f"/{kind} model"
    _report_catalog(catalog, provider, current, console, verbose=verbose, command=command)
    return catalog
```

`model_selection_prompt`: add `kind: str = "chat"`, pass `kind=kind` to `_provider_catalog`, and for non-chat kinds use

```python
    noun = "model" if kind == "chat" else f"{kind} model"
    empty = (
        f"No models are visible for {provider.id}. "
        "Use /config models clear or /model <name>."
        if kind == "chat"
        else f"No {kind} models are listed for {provider.id}. "
        f"Type one by name with /{kind} model <id>."
    )
```

and use `f"Choose a {noun} for {provider.name}"` as the default title, `f"Choose a {noun} from {provider.name} ({provider.id}) — {_catalog_status(catalog)}."` in the text, and `empty_message=empty`. Keep chat strings byte-for-byte identical for `kind == "chat"` (existing tests assert them).

- [ ] **Step 4: Add the media model handlers**

Add near `_image` (before it):

```python
def _media_provider_config(console: Console) -> ProviderConfig | None:
    """The provider media is currently set to use, or None (after saying why)."""
    provider_id = get_media()["provider"]
    try:
        return get_provider_config(provider_id)
    except KeyError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return None


def _save_media_model(
    kind: str, provider: ProviderConfig, model: str, console: Console
) -> bool:
    try:
        set_media_model(provider.id, kind, model)
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return False
    console.print(
        f"{kind.capitalize()} model for {escape(provider.name)} set to "
        f"[cyan]{escape(model)}[/cyan]."
    )
    return True


def _pick_media_model(
    kind: str,
    provider: ProviderConfig,
    current: str,
    console: Console,
    selector: CommandSelector,
) -> str:
    """Offer ``provider``'s ``kind`` models, save the pick, and return it ("" if none)."""
    prompt = model_selection_prompt(provider, current, console, kind=kind)
    if not prompt.options:
        console.print(f"[yellow]{escape(prompt.empty_message)}[/yellow]")
        return ""
    chosen = _run_selector(selector, prompt)
    if not chosen:
        return ""
    return chosen if _save_media_model(kind, provider, chosen, console) else ""


def _print_media_models(
    kind: str,
    provider: ProviderConfig,
    models: Sequence[str],
    current: str,
    console: Console,
    title: str,
) -> None:
    if not models:
        console.print(
            f"[yellow]{escape(title)}: no {kind} models listed.[/yellow] Type one "
            f"by name with /{kind} model <id>."
        )
        return
    table = Table(title=escape(title), show_header=True)
    table.add_column("model", style="cyan")
    table.add_column("")
    for model in models:
        table.add_row(escape(model), "active" if model == current else "")
    console.print(table)


def _media_model(
    kind: str, arg: str, console: Console, selector: CommandSelector | None
) -> None:
    """``/image model`` and ``/video model``: show, list, refresh, search, or set."""
    provider = _media_provider_config(console)
    if provider is None:
        return
    current = get_media_model(provider.id, kind)
    parts = arg.strip().split(maxsplit=1)
    action = parts[0].lower() if parts else ""
    rest = parts[1].strip() if len(parts) > 1 else ""

    if not action:
        console.print(
            f"Current {kind} model for {escape(provider.name)}: "
            f"[cyan]{escape(current or '(none chosen)')}[/cyan]"
        )
        return

    if action in _MODEL_REFRESH_WORDS or action in _MODEL_LIST_WORDS:
        if action in _MODEL_LIST_WORDS and selector is not None:
            _pick_media_model(kind, provider, current, console, selector)
            return
        catalog = _provider_catalog(
            provider,
            current,
            console,
            force=action in _MODEL_REFRESH_WORDS,
            verbose=action in _MODEL_REFRESH_WORDS,
            kind=kind,
        )
        visible = apply_model_filter(provider.id, catalog.models)
        _print_media_models(
            kind,
            provider,
            visible,
            current,
            console,
            f"{provider.name} {kind} models ({_catalog_status(catalog)})",
        )
        return

    if action in _MODEL_SEARCH_WORDS:
        if not rest:
            console.print(f"[yellow]Usage: /{kind} model search <term>[/yellow]")
            return
        with console.status(f"Searching {escape(provider.name)} {kind} models…"):
            matches = search_model_catalog(
                provider.id, rest, refresh=True, keep=(current,), kind=kind
            )
        if not matches:
            console.print(
                f"[yellow]No {kind} models match '{escape(rest)}' for "
                f"{escape(provider.name)}.[/yellow] Type an id with "
                f"/{kind} model <id>."
            )
            return
        if selector is not None:
            prompt = SelectionPrompt(
                title=f"Search: {rest}",
                text=(
                    f"{len(matches)} match(es) from {provider.name} "
                    f"({provider.id}).\nCurrent {kind} model: {current or '(none chosen)'}"
                ),
                options=tuple(CommandOption(model, model) for model in matches),
                selected=current if current in matches else None,
            )
            chosen = _run_selector(selector, prompt)
            if chosen:
                _save_media_model(kind, provider, chosen, console)
            return
        _print_media_models(
            kind, provider, matches, current, console, f"{len(matches)} match(es) for '{rest}'"
        )
        return

    _save_media_model(kind, provider, arg.strip(), console)
```

Add `get_media_model`, `set_media_model`, `search_model_catalog` (if not already imported) and `apply_model_filter` to the `from kiwimatecoder.config import (...)` block at the top (it already imports `get_media`, `set_media`; keep alphabetical order).

- [ ] **Step 5: Wire `/image`, `/video` and dispatch**

Change the signatures to `def _image(arg: str, session: Session, console: Console, selector: CommandSelector | None = None) -> str:` and the same for `_video` and `_media`. In `_image`, make the `model` subcommand the first thing:

```python
    words = arg.split(None, 1)
    if words and words[0].lower() == "model":
        _media_model("image", words[1] if len(words) > 1 else "", console, selector)
        return CommandResult.CONTINUE
```

In `_image`, change the usage block's "Now:" text to show `settings['model'] or 'no model chosen'`, and after the usage check replace `model = options.get("model") or settings["model"]` with

```python
    model = options.get("model") or settings["model"]
    if not model:
        provider = _media_provider_config(console)
        if provider is None:
            return CommandResult.CONTINUE
        if selector is None:
            console.print(f"[yellow]{escape(media_module.NO_IMAGE_MODEL_MESSAGE)}[/yellow]")
            return CommandResult.CONTINUE
        if not _pick_media_model("image", provider, "", console, selector):
            return CommandResult.CONTINUE
        model = get_media_model(provider.id, "image")
```

(the `generate_image` call still passes `model=options.get("model")`; the saved model is picked up inside `media.generate_image`.) In `_video`, put the same `words[0].lower() == "model"` check first (before the enabled check, using `"video"`), and replace the cheap-failure block

```python
    if not resume_id and not (options.get("model") or settings["video_model"]):
        console.print(f"[yellow]{escape(media_module.NO_VIDEO_MODEL_MESSAGE)}[/yellow]")
        return CommandResult.CONTINUE
```

with

```python
    if not resume_id and not (options.get("model") or settings["video_model"]):
        provider = _media_provider_config(console)
        if provider is None:
            return CommandResult.CONTINUE
        if selector is None or not _pick_media_model(
            "video", provider, "", console, selector
        ):
            if selector is None:
                console.print(
                    f"[yellow]{escape(media_module.NO_VIDEO_MODEL_MESSAGE)}[/yellow]"
                )
            return CommandResult.CONTINUE
```

and fix the later `model = options.get("model") or settings["video_model"]` (used in the "Requesting a video" message) to `options.get("model") or get_media_model(settings["provider"], "video")`.

In `dispatch`, just before the final `return handler(arg, session, console)` add:

```python
    selector_handler = _SELECTOR_COMMANDS.get(name)
    if selector_handler is not None:
        return selector_handler(arg, session, console, selector)
```

and define right after `_COMMANDS`:

```python
_SELECTOR_COMMANDS: dict[
    str, Callable[[str, Session, Console, CommandSelector | None], str]
] = {"image": _image, "video": _video, "media": _media}
```

Help text (~5932): change the three entries to

```python
            (
                "/image [--size WxH] [--model id] <prompt>  |  /image model [list|refresh|search <term>|<id>]",
                "Generate an image and attach it to your next message, or choose "
                "the image model (remembered per provider). A prompt starting "
                "with the word 'model' needs `--` first: /image -- model train.",
            ),
            (
                "/video [--duration s] [--size 720p] [--image path] <prompt>  |  /video model [list|refresh|search <term>|<id>]",
                "Generate a short video (billed per second; takes minutes), "
                "choose the video model, or `/video resume <job-id>` to fetch a "
                "job that was still rendering.",
            ),
```

Completion tables (~6020): add `"model": "Choose the video model (list, refresh, search, or an id)."` to `_VIDEO_ACTION_DESCRIPTIONS` and create `_IMAGE_ACTION_DESCRIPTIONS = {"model": "Choose the image model (list, refresh, search, or an id)."}`. Find where `_VIDEO_ACTION_DESCRIPTIONS` is selected (~line 6171: `choices = _VIDEO_ACTION_DESCRIPTIONS`) and add the matching `image` branch beside it.

- [ ] **Step 6: Run the tests**

Run: `venv/bin/python -m pytest tests/test_media_models.py -v` then `venv/bin/python -m pytest -q`
Expected: PASS. Fix any older test that asserted help text or the usage line of `/image` (they live in `tests/test_media*.py`, `tests/test_slash_command_coverage.py`, `tests/test_slash_markup.py`).

- [ ] **Step 7: Commit**

```bash
git add kiwimatecoder/commands.py tests/
git commit -m "feat: /image model and /video model with catalog pickers"
```

---

### Task 5: `/media provider`, status lines, config menu and `/config media`

**Files:**
- Modify: `kiwimatecoder/commands.py` (`_media_status_line` ~2782; `_config_media` ~2804; config menu `media` section ~3965; `_media` ~5719; `_media_provider_choices` ~3556; help ~5942; `_MEDIA_ACTION_DESCRIPTIONS` ~6024)
- Test: `tests/test_media_models.py`

**Interfaces:**
- Consumes: `media.format_models`, `config.set_media_model`, `_pick_media_model` (Task 4).
- Produces:
  - `_media_provider(arg: str, console: Console, selector: CommandSelector | None) -> None`
  - `/config media model <id> [provider]` and `/config media video-model <id|none> [provider]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_media_models.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_media_models.py -k "media_provider or media_status or config_media" -v`
Expected: FAIL.

- [ ] **Step 3: Implement status lines**

`_media_status_line`: use `image_model = settings["model"] or "none chosen"` and print `image model [cyan]{escape(image_model)}[/cyan]`. In `/media` (`_media`), after `console.print(_media_status_line(settings))` add

```python
    for line in media_module.format_models(settings):
        console.print(f"  [dim]{escape(line)}[/dim]")
```

and do the same in `_config_media` for the `show` action.

- [ ] **Step 4: Implement `/media provider`**

Change `_media`'s usage check to accept `provider` first:

```python
    words = arg.split()
    if words and words[0].lower() == "provider":
        _media_provider(" ".join(words[1:]), console, selector)
        return CommandResult.CONTINUE
```

(placed before the existing `list` validation; update that validation's usage string to `/media \\[list \\[count]|provider \\[id]]`). Add:

```python
def _media_provider_label(provider: ProviderConfig, settings: dict[str, Any]) -> str:
    kinds = settings["models"].get(provider.id, {})
    image = kinds.get("image") or "none chosen"
    video = kinds.get("video") or "none chosen"
    return f"{provider.name} ({provider.id}) — image {image}, video {video}"


def _media_provider(
    arg: str, console: Console, selector: CommandSelector | None
) -> None:
    """``/media provider [id]``: show the providers or switch the media provider."""
    settings = get_media()
    provider_id = arg.strip()
    if not provider_id:
        if selector is not None:
            prompt = SelectionPrompt(
                title="Media provider",
                text=f"Current media provider: {settings['provider']}",
                options=tuple(
                    CommandOption(p.id, _media_provider_label(p, settings))
                    for p in list_provider_configs()
                ),
                selected=settings["provider"],
            )
            provider_id = _run_selector(selector, prompt) or ""
            if not provider_id:
                return
        else:
            table = Table(title="Media providers", show_header=True)
            table.add_column("id", style="cyan")
            table.add_column("image model")
            table.add_column("video model")
            for p in list_provider_configs():
                kinds = settings["models"].get(p.id, {})
                marker = " (current)" if p.id == settings["provider"] else ""
                table.add_row(
                    escape(p.id + marker),
                    escape(kinds.get("image") or "—"),
                    escape(kinds.get("video") or "—"),
                )
            console.print(table)
            return
    try:
        settings = set_media(provider=provider_id)
        provider = get_provider_config(provider_id)
    except (ValueError, KeyError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return
    console.print(f"Media provider set to [cyan]{escape(provider.id)}[/cyan].")
    for kind in MEDIA_KINDS:
        if get_media_model(provider.id, kind):
            continue
        if selector is not None:
            _pick_media_model(kind, provider, "", console, selector)
        else:
            console.print(
                f"[yellow]No {kind} model chosen for {escape(provider.name)} — "
                f"choose one with /{kind} model.[/yellow]"
            )
```

Import `MEDIA_KINDS` from `kiwimatecoder.config` in the config import block. `_media_provider_choices()` (~3556) stays as is (the config menu `provider` entry).

- [ ] **Step 5: Implement `/config media` forms and menu labels**

In `_config_media`, replace the `"model"` and `"video-model"` handling. Before the generic `hints` branch add:

```python
    if action in {"model", "video-model", "video_model"}:
        kind = "image" if action == "model" else "video"
        if not rest:
            hint = "<id>" if kind == "image" else "<id|none>"
            console.print(f"[yellow]Usage: /config media {action} {hint} [provider][/yellow]")
            return
        provider_id = get_media()["provider"]
        words = list(rest)
        if len(words) >= 2:
            try:
                get_provider_config(words[-1])
            except KeyError:
                pass  # not a provider id: the whole tail is the model id
            else:
                provider_id = words.pop()
        value = " ".join(words)
        cleared = kind == "video" and value.lower() in {"none", "default", "clear", "-"}
        try:
            set_media_model(provider_id, kind, "" if cleared else value)
        except ValueError as exc:
            console.print(f"[red]{escape(str(exc))}[/red]")
            return
        console.print(_media_status_line(get_media()))
        return
```

Remove the `"model"`/`"video-model"` entries from the generic `hints` dict handling only if they become unreachable (they do: delete the two `if action == "model"` / `elif action == "video-model"` branches in that block and their `hints` entries; keep the rest). In the `enable` branch change `if not get_media()["video_model"]:` hint text to point at `/video model`, and add the same hint for a missing image model: `/image model`. Update `_MEDIA_USAGE` to `model <id> [provider]|... video-model <id|none> [provider]`.

Config menu (~3985, ~3999): change `lambda _s: get_media()["model"]` to `lambda _s: _preview(get_media()["model"], "none chosen")`. The `_text(... "model")` and `video-model` entries keep their keys and route through `_config_media`, so no other change is needed.

Completion: add `"provider": "Choose the media provider (shows each provider's image and video model)."` to `_MEDIA_ACTION_DESCRIPTIONS`. Help entry for `/media`: `"/media [list [count]|provider [id]]"` with description "Show readiness and recent generated files, or choose the media provider."

- [ ] **Step 6: Run the tests and commit**

Run: `venv/bin/python -m pytest tests/test_media_models.py tests/test_config_menu.py tests/test_config_markup.py -v` then `venv/bin/python -m pytest -q`
Expected: PASS. In `tests/test_config_menu.py` the label assertion for the image model becomes `"Image model — none chosen"`.

```bash
git add kiwimatecoder/commands.py tests/
git commit -m "feat: /media provider and per-provider models in /config media"
```

---

### Task 6: CLI, tools, docs

**Files:**
- Modify: `kiwimatecoder/main.py` (`_print_media` ~2517, `media_model` ~2568, `media_video_model` ~2609)
- Modify: `kiwimatecoder/tools/image.py` (`generate_image_preview` ~208, `generate_video_preview` ~213)
- Modify: `README.md` (~line 609 and the media section), regenerate `docs/kiwimatecoder.1`
- Test: `tests/test_media_models.py`

**Interfaces:**
- Consumes: `config.set_media_model`, `config.get_media_model`, `media.format_models`.
- Produces: CLI `config media model <id> [--provider ID]`, `config media video-model <id|none> [--provider ID]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_media_models.py`:

```python
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

    assert "Image model: none chosen" in result.output
    assert "openrouter: image or-image" in result.output


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
```

If the CLI `app` object or the tool's callable have different names than `app` / `generate_image_tool.func` (check `main.py` and `tools/image.py`), use the real ones; the assertions stay the same.

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_media_models.py -k "cli_ or tool" -v`
Expected: FAIL.

- [ ] **Step 3: Implement the CLI**

`_print_media`: use `Image model: {escape(settings['model'] or 'none chosen')}` and, after the final `Output dir` line, print one `console.print(escape(line))` for each `media_module.format_models(settings)` line (import `from kiwimatecoder import media as media_module` locally as `media_show` already does).

```python
@media_app.command("model")
def media_model(
    model: Annotated[str, typer.Argument(help="Image model id")],
    provider: Annotated[
        str | None, typer.Option("--provider", help="Provider id (default: the media provider)")
    ] = None,
) -> None:
    """Set the model used for image generation (remembered per provider)."""
    try:
        provider_id = provider or get_media()["provider"]
        set_media_model(provider_id, "image", model)
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]{_check()} Image model for {escape(provider_id)} set to[/green] "
        f"[cyan]{escape(str(model))}[/cyan]."
    )
```

and `media_video_model` the same shape with kind `"video"`, `cleared = model.strip().lower() in {"none", "clear", "default", "-"}`, and the existing "Video model cleared." / "Video model set to" messages. Import `set_media_model` in the `from kiwimatecoder.config import (...)` block at the top of `main.py` beside `set_media`.

- [ ] **Step 4: Implement tool previews**

In `generate_image_preview`: `model = str(args.get("model") or settings["model"]).strip() or "(none chosen)"`. In `generate_video_preview` the video line already prints `model or '(none chosen)'`; make sure the model resolves with `args.get("model") or settings["video_model"]` (it does). `settings["model"]` is the effective model for `media.provider`, which is the provider the tools use, so no further change.

- [ ] **Step 5: Docs**

In `README.md`, change the `/config media model gpt-image-1` example (~line 609) to `/image model` and describe: models are chosen per provider, `/image model`, `/video model` (list, refresh, search, or an id), `/media provider`, no default model, and that a prompt starting with the word `model` needs `--` first. Then:

Run: `venv/bin/python scripts/generate_man.py` (check the script's `--help` for its output path argument; it must rewrite `docs/kiwimatecoder.1`).

- [ ] **Step 6: Run everything**

Run: `venv/bin/python -m pytest -q`
Expected: PASS (including `tests/test_packaging_assets.py`, which checks the man page).

- [ ] **Step 7: Commit**

```bash
git add kiwimatecoder/main.py kiwimatecoder/tools/image.py README.md docs/kiwimatecoder.1 tests/
git commit -m "feat: per-provider media models in the CLI, tool previews and docs"
```

---

## Self-Review

**Spec coverage.** Config storage, legacy fallback, no default, validation (Task 1). Catalog kinds, cache keys, media filter, curated-empty fallback (Task 2). Model resolution, readiness, no-model errors (Task 3). `/image model`, `/video model`, pickers, per-call `--model` override (Task 4). `/media provider`, `/media` and `config media show` per-provider listing, `/config media … [provider]` (Task 5). CLI `--provider`, tool previews, docs (Task 6). The allow/deny model filter applies to media lists through `apply_model_filter` in Task 4.

**Known deviations from the spec text, to confirm in review.**
- With a selector and no id, `/media provider` opens a picker whose labels carry each provider's models; without a selector it prints the table the spec describes.
- `/image model <id>` always treats the first word `model` as the subcommand, so a prompt starting with `model` needs `--` first. The spec didn't cover this collision.
- OpenRouter video models: the plan reuses `GET /models` as the spec says. If OpenRouter keeps video models on a separate endpoint, they will not be listed; typing an id works.

**Type consistency.** `get_media_model(provider_id, kind, cfg)`, `set_media_model(provider_id, kind, model, cfg)`, `MEDIA_KINDS`, `kind="chat"|"image"|"video"`, `_pick_media_model(kind, provider, current, console, selector)` and `media.format_models(settings)` are used with the same signatures in every task.
