"""Configuration storage for KiwiMateCoder.

Configuration lives in ``~/.kiwimatecoder/config.json`` with this shape::

    {
        "keys": {"openrouter": "sk-...", "openai": "sk-..."},
        "providers": {"local": {"name": "...", "base_url": "..."}},
        "model_filters": {"openai": {"mode": "allow", "models": ["gpt-5"]}},
        "selected_provider": "openrouter",
        "active_providers": ["openrouter", "openai"],
        "selected_model": null,
        "default_mode": "ask"
    }

Live model catalogs are cached separately in
``~/.kiwimatecoder/model_cache.json`` so ``/model`` can offer what a provider
serves today without hitting the network on every invocation. That file holds no
secrets and can be deleted at any time.

The original releases stored a single OpenRouter key in a flat
``~/.kiwimatecoder/config`` file (``OPENROUTER_API_KEY=...``). That file is read
transparently when the JSON config is absent, so existing users keep working;
the legacy file is never deleted.

API keys can also come from environment variables (each provider's ``key_env``),
which take precedence over stored keys so a shell can override config per run.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kiwimatecoder import catalog
from kiwimatecoder.catalog import CatalogFetchError, ModelCatalog
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.providers import (
    DEFAULT_PROVIDER_ID,
    REGISTRY,
    ProviderConfig,
    UnknownProviderError,
)

CONFIG_DIR = Path.home() / ".kiwimatecoder"
CONFIG_FILE = CONFIG_DIR / "config.json"
LEGACY_CONFIG_FILE = CONFIG_DIR / "config"
MODEL_CACHE_NAME = "model_cache.json"

DEFAULT_MODE = "ask"
MODEL_CACHE_VERSION = 1


def ensure_config_dir() -> Path:
    """Ensure ~/.kiwimatecoder exists with owner-only permissions and return it."""
    CONFIG_DIR.mkdir(exist_ok=True)
    try:
        os.chmod(CONFIG_DIR, 0o700)
    except OSError:
        pass
    return CONFIG_DIR


_ensure_config_dir = ensure_config_dir


def _empty_config() -> dict[str, Any]:
    return {
        "keys": {},
        "providers": {},
        "model_filters": {},
        "selected_provider": DEFAULT_PROVIDER_ID,
        "active_providers": [DEFAULT_PROVIDER_ID],
        "selected_model": None,
        "default_mode": DEFAULT_MODE,
    }


def _read_legacy_key() -> str | None:
    """Read the OpenRouter key from the legacy flat config file, if present."""
    if LEGACY_CONFIG_FILE.exists():
        for line in LEGACY_CONFIG_FILE.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def load_config() -> dict[str, Any]:
    """Load configuration, migrating from the legacy format when needed.

    The returned dict always has the full set of keys (with defaults filled in).
    Migration is non-destructive: the legacy file is left in place.
    """
    cfg: dict[str, Any] = _empty_config()
    stored: dict[str, Any] = {}
    if CONFIG_FILE.exists():
        try:
            stored = json.loads(CONFIG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            stored = {}
        if not isinstance(stored, dict):
            stored = {}
        if stored:
            cfg.update(
                {
                    k: v
                    for k, v in stored.items()
                    if v is not None or k == "selected_model"
                }
            )
            cfg["keys"] = dict(stored.get("keys") or {})
            cfg["providers"] = dict(stored.get("providers") or {})
            cfg["model_filters"] = dict(stored.get("model_filters") or {})
    else:
        legacy_key = _read_legacy_key()
        if legacy_key:
            cfg["keys"]["openrouter"] = legacy_key
    # Guarantee structural defaults even if the stored file was partial.
    cfg.setdefault("keys", {})
    cfg.setdefault("providers", {})
    cfg.setdefault("model_filters", {})
    cfg.setdefault("selected_provider", DEFAULT_PROVIDER_ID)
    cfg.setdefault("active_providers", [DEFAULT_PROVIDER_ID])
    cfg.setdefault("selected_model", None)
    cfg.setdefault("default_mode", DEFAULT_MODE)
    # Active-provider roster. Configs written before this feature lack the key;
    # migrate by seeding it from the single selected provider. An explicitly
    # stored empty list, a non-list, or a list of junk is seeded the same way.
    cfg["active_providers"] = _normalized_active_providers(
        stored, str(cfg.get("selected_provider") or DEFAULT_PROVIDER_ID)
    )
    return cfg


def _normalized_active_providers(stored: dict[str, Any], selected: str) -> list[str]:
    """Return a usable roster from stored config, or seed from ``selected``."""
    raw = stored.get("active_providers") if stored else None
    if isinstance(raw, list):
        cleaned = [
            item.strip()
            for item in raw
            if isinstance(item, str) and item.strip()
        ]
        if cleaned:
            return cleaned
    return [selected or DEFAULT_PROVIDER_ID]


def save_config(cfg: dict[str, Any]) -> None:
    """Persist configuration to the JSON config file.

    The file (and its directory) are tightened to owner-only permissions since
    they may contain API keys.
    """
    ensure_config_dir()
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2) + "\n")
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except OSError:
        pass


def _default_key_env(provider_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in provider_id.upper())
    return f"{cleaned}_API_KEY"


def _provider_from_config(provider_id: str, data: object) -> ProviderConfig | None:
    if not isinstance(data, dict):
        return None
    try:
        name = str(data["name"]).strip()
        base_url = str(data["base_url"]).strip()
        default_model = str(data["default_model"]).strip()
    except (KeyError, TypeError):
        return None
    if not name or not base_url or not default_model:
        return None

    key_env = str(data.get("key_env") or _default_key_env(provider_id)).strip()
    compat = str(data.get("compat") or "openai").strip().lower()
    if compat not in {"openai", "anthropic"}:
        compat = "openai"
    extra_headers = data.get("extra_headers") or {}
    if not isinstance(extra_headers, dict):
        extra_headers = {}
    raw_models = data.get("models") or []
    if not isinstance(raw_models, list):
        raw_models = []
    models = tuple(
        dict.fromkeys(str(model).strip() for model in raw_models if str(model).strip())
    )

    return ProviderConfig(
        id=provider_id,
        name=name,
        base_url=base_url.rstrip("/"),
        default_model=default_model,
        key_env=key_env,
        compat=compat,
        extra_headers={str(k): str(v) for k, v in extra_headers.items()},
        models=models,
    )


def _known_provider_ids(cfg: dict[str, Any] | None = None) -> list[str]:
    cfg = cfg or load_config()
    custom_ids = sorted(str(pid) for pid in cfg.get("providers", {}))
    return sorted(set(REGISTRY) | set(custom_ids))


def get_provider_config(provider_id: str, cfg: dict[str, Any] | None = None) -> ProviderConfig:
    """Return a built-in or user-defined provider config."""
    if provider_id in REGISTRY:
        return REGISTRY[provider_id]

    cfg = cfg or load_config()
    provider = _provider_from_config(
        provider_id, (cfg.get("providers") or {}).get(provider_id)
    )
    if provider is not None:
        return provider

    raise UnknownProviderError(
        f"Unknown provider '{provider_id}'. "
        f"Known providers: {', '.join(_known_provider_ids(cfg))}"
    )


def list_provider_configs(cfg: dict[str, Any] | None = None) -> list[ProviderConfig]:
    """Return built-in providers plus valid user-defined providers."""
    cfg = cfg or load_config()
    providers = list(REGISTRY.values())
    for provider_id in sorted(cfg.get("providers", {})):
        provider = _provider_from_config(provider_id, cfg["providers"][provider_id])
        if provider is not None:
            providers.append(provider)
    return providers


def add_provider(
    provider_id: str,
    name: str,
    base_url: str,
    default_model: str,
    key_env: str | None = None,
    compat: str = "openai",
) -> ProviderConfig:
    """Persist a user-defined provider and return its config."""
    provider_id = provider_id.strip().lower()
    if not provider_id or any(ch.isspace() for ch in provider_id):
        raise ValueError("Provider id must be non-empty and contain no spaces.")
    if provider_id in REGISTRY:
        raise ValueError(
            f"'{provider_id}' is a built-in provider and cannot be replaced."
        )
    if not name.strip():
        raise ValueError("Provider name is required.")
    if not base_url.strip():
        raise ValueError("Provider base_url is required.")
    if not default_model.strip():
        raise ValueError("Provider default_model is required.")
    compat = compat.strip().lower()
    if compat not in {"openai", "anthropic"}:
        raise ValueError("Provider compat must be 'openai' or 'anthropic'.")

    cfg = load_config()
    cfg["providers"][provider_id] = {
        "name": name.strip(),
        "base_url": base_url.strip().rstrip("/"),
        "default_model": default_model.strip(),
        "key_env": (key_env or _default_key_env(provider_id)).strip(),
        "compat": compat,
    }
    save_config(cfg)
    return get_provider_config(provider_id, cfg)


def remove_provider(provider_id: str) -> None:
    """Remove a user-defined provider and any config tied to it."""
    if provider_id in REGISTRY:
        raise ValueError(f"'{provider_id}' is built in and cannot be removed.")

    cfg = load_config()
    if provider_id not in cfg["providers"]:
        raise ValueError(f"Unknown custom provider '{provider_id}'.")
    del cfg["providers"][provider_id]
    cfg["keys"].pop(provider_id, None)
    cfg["model_filters"].pop(provider_id, None)
    # Drop the provider from the active roster; the first remaining id becomes
    # the primary. selected_provider stays aligned with that roster.
    active = [
        pid for pid in (cfg.get("active_providers") or []) if pid != provider_id
    ]
    if not active:
        fallback = cfg.get("selected_provider")
        active = [
            DEFAULT_PROVIDER_ID
            if not fallback or fallback == provider_id
            else str(fallback)
        ]
    cfg["active_providers"] = active
    if cfg.get("selected_provider") != active[0]:
        cfg["selected_provider"] = active[0]
        cfg["selected_model"] = None
    save_config(cfg)
    clear_model_cache(provider_id)


def update_provider(
    provider_id: str,
    *,
    name: str | None = None,
    base_url: str | None = None,
    default_model: str | None = None,
    key_env: str | None = None,
    compat: str | None = None,
) -> ProviderConfig:
    """Update fields of a user-defined provider and return its config.

    Only fields given are changed; None leaves the current value alone.
    ``name``, ``base_url``, and ``default_model`` must stay non-empty when
    changed. ``compat`` must be 'openai' or 'anthropic'. Built-in providers
    cannot be edited (their config lives in the registry).
    """
    if provider_id in REGISTRY:
        raise ValueError(f"'{provider_id}' is built in and cannot be edited.")

    cfg = load_config()
    if provider_id not in cfg["providers"]:
        raise ValueError(f"Unknown custom provider '{provider_id}'.")

    data = dict(cfg["providers"][provider_id])
    if name is not None:
        if not name.strip():
            raise ValueError("Provider name is required.")
        data["name"] = name.strip()
    if base_url is not None:
        if not base_url.strip():
            raise ValueError("Provider base_url is required.")
        data["base_url"] = base_url.strip().rstrip("/")
    if default_model is not None:
        if not default_model.strip():
            raise ValueError("Provider default_model is required.")
        data["default_model"] = default_model.strip()
    if key_env is not None:
        if not key_env.strip():
            raise ValueError("Provider key_env is required.")
        data["key_env"] = key_env.strip()
    if compat is not None:
        compat = compat.strip().lower()
        if compat not in {"openai", "anthropic"}:
            raise ValueError("Provider compat must be 'openai' or 'anthropic'.")
        data["compat"] = compat

    cfg["providers"][provider_id] = data
    save_config(cfg)
    return get_provider_config(provider_id, cfg)


def get_key(provider_id: str) -> str | None:
    """Return the API key for a provider.

    Environment variable (the provider's ``key_env``) takes precedence over the
    stored key (even if the env var is set to the empty string, which clears any
    stored key for this process and forces the friendly "no key" path).
    """
    provider = get_provider_config(provider_id)
    env_key = os.environ.get(provider.key_env)
    if env_key is not None:
        return env_key or None  # exported empty string -> treat as "no key"
    return load_config()["keys"].get(provider_id)


def get_key_env_override(provider_id: str) -> str | None:
    """Return the env var currently masking the stored key for ``provider_id``.

    Returns the provider's ``key_env`` name when that variable is exported in
    this process (``get_key`` will prefer it over the stored key), else None.
    """
    provider = get_provider_config(provider_id)
    if provider.key_env in os.environ:
        return provider.key_env
    return None


def key_source(provider_id: str) -> dict[str, Any]:
    """Describe where a provider's active API key comes from.

    Returns one of::

        {"origin": "env", "env": "OPENAI_API_KEY", "value": "<key>"}
        {"origin": "stored", "env": None, "value": "<key>"}
        {"origin": "legacy", "env": None, "value": "<key>"}   # ~/.kiwimatecoder/config
        {"origin": "missing", "env": None, "value": None}

    ``value`` is the raw key; callers that display it should redact.
    """
    provider = get_provider_config(provider_id)
    env_key = os.environ.get(provider.key_env)
    if env_key is not None:
        return {"origin": "env", "env": provider.key_env, "value": env_key or None}
    stored = load_config()["keys"].get(provider_id)
    if stored:
        # A key read from the flat legacy file sits in the keys map too, so
        # distinguish it for messaging when the JSON config has never been written.
        if not CONFIG_FILE.exists() and LEGACY_CONFIG_FILE.exists():
            return {"origin": "legacy", "env": None, "value": stored}
        return {"origin": "stored", "env": None, "value": stored}
    return {"origin": "missing", "env": None, "value": None}


def _redact(value: str) -> str:
    if len(value) <= 4:
        return value
    return "…" + value[-4:]


def describe_key(provider_id: str) -> str:
    """Return a human-readable, secret-safe status for a provider's key."""
    source = key_source(provider_id)
    origin = source["origin"]
    value = source["value"]
    if origin == "missing":
        provider = get_provider_config(provider_id)
        if provider.is_local:
            return (
                "required by local server"
                if provider.requires_key
                else "not required (local server)"
            )
        return "missing"
    redacted = _redact(value)
    if origin == "env":
        return f"from environment {source['env']} ({redacted}) — overrides any stored key"
    if origin == "legacy":
        return f"stored in legacy {LEGACY_CONFIG_FILE} ({redacted})"
    return f"stored in {CONFIG_FILE} ({redacted})"


def set_key(provider_id: str, key: str) -> str | None:
    """Store an API key for a provider and persist the config.

    Returns a warning string when the provider's environment variable is set
    (that env var still takes precedence at runtime, so the stored re-set key
    would not take effect until it is unset); otherwise returns None.
    """
    # Validate the provider id eagerly.
    provider = get_provider_config(provider_id)
    cfg = load_config()
    cfg["keys"][provider_id] = key
    save_config(cfg)
    if provider.key_env in os.environ:
        return (
            f"Stored, but {provider.key_env} is set in the environment and "
            f"takes precedence. Run `unset {provider.key_env}` (or export it "
            f"empty) for this key to take effect."
        )
    return None


def remove_key(provider_id: str) -> bool:
    """Remove a stored API key. Returns True when a stored key existed."""
    get_provider_config(provider_id)
    cfg = load_config()
    existed = provider_id in cfg["keys"]
    cfg["keys"].pop(provider_id, None)
    save_config(cfg)
    return existed


def set_selected_provider(provider_id: str) -> None:
    """Persist the default (primary) provider and reset the active roster to it.

    Choosing a single provider explicitly makes it the only active one; the
    checklist (``/provider``) or :func:`set_active_providers` is how a user opts
    back into a multi-provider roster.
    """
    get_provider_config(provider_id)
    cfg = load_config()
    cfg["selected_provider"] = provider_id
    cfg["active_providers"] = [provider_id]
    save_config(cfg)


def set_selected_model(model: str | None) -> None:
    """Persist the default model (None falls back to the provider default)."""
    cfg = load_config()
    cfg["selected_model"] = model
    save_config(cfg)


def get_default_mode(cfg: dict[str, Any] | None = None) -> str:
    """Return the persisted startup permission mode, never a None."""
    cfg = cfg or load_config()
    return str(cfg.get("default_mode") or DEFAULT_MODE)


def set_default_mode(mode: str, cfg: dict[str, Any] | None = None) -> str:
    """Persist the startup permission mode and return the effective value.

    Raises ``ValueError`` for an unknown mode (aliases accepted), matching
    ``PermissionMode.from_str`` so callers can validate eagerly.
    """
    cfg = cfg or load_config()
    effective = PermissionMode.from_str(mode).value
    cfg["default_mode"] = effective
    save_config(cfg)
    return effective


def reset_default_mode(cfg: dict[str, Any] | None = None) -> str:
    """Clear a custom startup mode, returning the default mode."""
    cfg = cfg or load_config()
    cfg["default_mode"] = DEFAULT_MODE
    save_config(cfg)
    return DEFAULT_MODE


def get_selected_provider_id(cfg: dict[str, Any] | None = None) -> str:
    """Return the primary provider: first valid active-roster id.

    ``selected_provider`` on disk is a backward-compatible alias for that
    primary. Callers that used to read it directly should go through here so a
    drifted selected/roster pair cannot point the UI at one vendor and the
    agent at another.
    """
    return get_active_provider_ids(cfg)[0]


def get_active_provider_ids(cfg: dict[str, Any] | None = None) -> list[str]:
    """Return the ordered list of active provider ids, validated and de-duplicated.

    The first id is the primary provider (the one the session chats with by
    default); the rest are fallbacks tried in order when the primary fails.
    Invalid ids and duplicates are dropped. The roster is the source of truth;
    ``selected_provider`` is only used when the stored roster is empty.
    """
    if cfg is None:
        cfg = load_config()
    raw = cfg.get("active_providers")
    if not isinstance(raw, list):
        raw = []
    seen: set[str] = set()
    ids: list[str] = []
    for pid in raw:
        pid = str(pid).strip() if pid is not None else ""
        if not pid or pid in seen:
            continue
        try:
            get_provider_config(pid, cfg)
        except KeyError:
            continue
        seen.add(pid)
        ids.append(pid)
    if ids:
        return ids
    fallback = str(cfg.get("selected_provider") or DEFAULT_PROVIDER_ID)
    try:
        get_provider_config(fallback, cfg)
        return [fallback]
    except KeyError:
        return [DEFAULT_PROVIDER_ID]


def set_active_providers(provider_ids: Sequence[str]) -> list[str]:
    """Persist the active-provider roster and return the validated ordered ids.

    ``provider_ids`` must be a non-empty sequence of known provider ids (order
    is significant: the first is the primary). The single ``selected_provider``
    is kept in sync with the primary for backward compatibility.
    """
    if not provider_ids:
        raise ValueError("At least one active provider is required.")
    cleaned: list[str] = []
    seen: set[str] = set()
    for pid in provider_ids:
        pid = str(pid).strip()
        if not pid or pid in seen:
            continue
        get_provider_config(pid)  # raises UnknownProviderError (a KeyError)
        seen.add(pid)
        cleaned.append(pid)
    if not cleaned:
        raise ValueError("At least one active provider is required.")

    cfg = load_config()
    cfg["active_providers"] = cleaned
    cfg["selected_provider"] = cleaned[0]
    save_config(cfg)
    return cleaned


def get_model_filter(provider_id: str) -> dict[str, Any]:
    """Return the model visibility filter for a provider."""
    get_provider_config(provider_id)
    cfg = load_config()
    stored = (cfg.get("model_filters") or {}).get(provider_id) or {}
    mode = stored.get("mode") or "all"
    if mode not in {"all", "allow", "deny"}:
        mode = "all"
    models = [str(model) for model in stored.get("models", []) if str(model).strip()]
    return {"mode": mode, "models": models}


def set_model_filter(provider_id: str, mode: str, models: list[str] | None = None) -> None:
    """Persist model visibility for a provider.

    ``mode='allow'`` shows only the listed models. ``mode='deny'`` hides the
    listed models. ``mode='all'`` clears the filter.
    """
    get_provider_config(provider_id)
    mode = mode.strip().lower()
    if mode not in {"all", "allow", "deny"}:
        raise ValueError("Model filter mode must be all, allow, or deny.")
    unique_models = list(dict.fromkeys(model.strip() for model in (models or [])))
    unique_models = [model for model in unique_models if model]
    if mode in {"allow", "deny"} and not unique_models:
        raise ValueError(f"Model filter mode '{mode}' requires at least one model.")

    cfg = load_config()
    if mode == "all":
        cfg["model_filters"].pop(provider_id, None)
    else:
        cfg["model_filters"][provider_id] = {"mode": mode, "models": unique_models}
    save_config(cfg)


def apply_model_filter(provider_id: str, models: Sequence[str]) -> list[str]:
    """Apply the provider's allow/deny visibility filter to ``models``."""
    model_filter = get_model_filter(provider_id)
    mode = model_filter["mode"]
    filtered = model_filter["models"]
    if mode == "allow":
        return list(filtered)
    if mode == "deny":
        denied = set(filtered)
        return [model for model in models if model not in denied]
    return list(models)


# ---------------------------------------------------------------------------
# Live model catalogs
# ---------------------------------------------------------------------------


def _model_cache_file() -> Path:
    """Path of the model cache (resolved lazily so CONFIG_DIR stays patchable)."""
    return CONFIG_DIR / MODEL_CACHE_NAME


def load_model_cache() -> dict[str, Any]:
    """Load the cached provider catalogs, tolerating a missing/corrupt file."""
    path = _model_cache_file()
    if path.exists():
        try:
            stored = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            stored = None
        if isinstance(stored, dict) and stored.get("version") == MODEL_CACHE_VERSION:
            providers = stored.get("providers")
            if isinstance(providers, dict):
                return {"version": MODEL_CACHE_VERSION, "providers": dict(providers)}
    return {"version": MODEL_CACHE_VERSION, "providers": {}}


def save_model_cache(cache: dict[str, Any]) -> None:
    """Persist the model cache, ignoring write failures (it is only a cache)."""
    try:
        CONFIG_DIR.mkdir(exist_ok=True)
        _model_cache_file().write_text(json.dumps(cache, indent=2) + "\n")
    except OSError:
        pass


def clear_model_cache(provider_id: str | None = None) -> None:
    """Drop cached catalogs for one provider, or all of them."""
    cache = load_model_cache()
    if provider_id is None:
        cache["providers"] = {}
    else:
        cache["providers"].pop(provider_id, None)
    save_model_cache(cache)


def _curated_models(provider: ProviderConfig) -> list[str]:
    return list(dict.fromkeys(m for m in (provider.default_model, *provider.models) if m))


def _can_fetch_models(provider: ProviderConfig) -> bool:
    """Whether a live fetch is worth attempting for ``provider``.

    A provider you have no key for is one you cannot use, so we skip the request
    rather than firing off a doomed call every time the selector opens. Local
    servers are exempt — except key-enforcing ones like Unsloth, whose listing
    answers 401 until the key exists.
    """
    if get_key(provider.id):
        return True
    return provider.is_local and not provider.requires_key


def _should_auto_refresh(entry: dict[str, Any], now: float) -> bool:
    """Whether a cached entry is stale enough to refetch on its own."""
    if not entry:
        return True
    if now - float(entry.get("failed_at") or 0.0) < catalog.FAILURE_BACKOFF_SECONDS:
        return False
    if not entry.get("models"):
        return True
    return now - float(entry.get("fetched_at") or 0.0) >= catalog.CATALOG_TTL_SECONDS


def get_model_catalog(
    provider_id: str,
    *,
    refresh: bool = False,
    force: bool = False,
    keep: Sequence[str] = (),
    timeout: float | None = None,
    limit: int | None = catalog.MAX_CATALOG_MODELS,
    cfg: dict[str, Any] | None = None,
) -> ModelCatalog:
    """Return the models offered for a provider, newest first.

    ``refresh`` asks the provider for its live listing when the cached one is
    missing or older than :data:`catalog.CATALOG_TTL_SECONDS`; ``force`` always
    asks. A successful fetch replaces the catalog wholesale, so models the
    provider has retired stop being offered. Any failure (offline, bad key,
    unsupported endpoint) falls back to the cached catalog and then to the
    curated one, with the error reported on the result instead of raised.

    The **full** live listing is always cached and compared for
    additions/removals; ``limit`` only trims the list this call returns (used by
    the interactive selector to stay snappy). Pass ``limit=None`` to get
    everything, e.g. for a search that must see the whole catalog.
    """
    provider = get_provider_config(provider_id, cfg)
    curated = _curated_models(provider)

    cache = load_model_cache()
    entry = cache["providers"].get(provider_id)
    entry = dict(entry) if isinstance(entry, dict) else {}
    cached_models = [
        str(model).strip() for model in entry.get("models") or [] if str(model).strip()
    ]
    fetched_at = float(entry.get("fetched_at") or 0.0) or None

    error: str | None = None
    now = time.time()
    want_live = force or (refresh and _should_auto_refresh(entry, now))
    if want_live and not _can_fetch_models(provider):
        # Only worth saying out loud when the user asked for a refresh; the
        # automatic path stays silent for providers they have not set up.
        if force:
            error = f"no API key for {provider.name} ({provider.key_env})"
        want_live = False
    if want_live:
        try:
            remote = catalog.fetch_models(
                provider,
                get_key(provider_id),
                timeout=timeout if timeout is not None else catalog.FETCH_TIMEOUT,
            )
        except CatalogFetchError as exc:
            error = str(exc)
            entry["failed_at"] = now
            cache["providers"][provider_id] = entry
            save_model_cache(cache)
        else:
            models = catalog.merge_catalog(provider, remote, keep=keep, limit=None)
            # Compare against what we previously believed: the last live fetch
            # if there was one, otherwise the curated tuple we shipped.
            baseline = cached_models or curated
            added = [model for model in models if model not in baseline]
            removed = [model for model in baseline if model not in models]
            cache["providers"][provider_id] = {"fetched_at": now, "models": models}
            save_model_cache(cache)
            return ModelCatalog(
                provider_id=provider_id,
                models=(
                    models if limit is None else models[: max(limit, 0)]
                ),
                source="live",
                fetched_at=now,
                added=added,
                removed=removed,
            )

    if cached_models:
        return ModelCatalog(
            provider_id=provider_id,
            models=(
                cached_models if limit is None else cached_models[: max(limit, 0)]
            ),
            source="cache",
            fetched_at=fetched_at,
            error=error,
        )
    return ModelCatalog(
        provider_id=provider_id, models=curated, source="curated", error=error
    )


def list_visible_models(
    provider_id: str,
    *,
    refresh: bool = False,
    force: bool = False,
    keep: Sequence[str] = (),
) -> list[str]:
    """Return models that should be offered in model-selection UI.

    Reads the cached catalog by default — callers such as tab completion must
    never block on the network. Pass ``refresh=True`` to let a stale catalog be
    refetched, or ``force=True`` to refetch unconditionally.
    """
    model_catalog = get_model_catalog(
        provider_id, refresh=refresh, force=force, keep=keep
    )
    return apply_model_filter(provider_id, model_catalog.models)


def search_model_catalog(
    provider_id: str,
    query: str,
    *,
    refresh: bool = False,
    force: bool = False,
    keep: Sequence[str] = (),
) -> list[str]:
    """Return models for ``provider_id`` whose id matches ``query``.

    Unlike :func:`list_visible_models`, the search sees the **full** catalog
    (no :data:`catalog.MAX_CATALOG_MODELS` cap) so a model ranked low by
    release date is still findable. The provider's allow/deny filter still
    applies, and ``keep`` names are pinned before searching so the current
    model survives a refresh.
    """
    model_catalog = get_model_catalog(
        provider_id, refresh=refresh, force=force, keep=keep, limit=None
    )
    visible = apply_model_filter(provider_id, model_catalog.models)
    return catalog.search_models(visible, query)


def resolve_default_model(provider: ProviderConfig, *, timeout: float = 2.0) -> str:
    """Return the model to start with for ``provider`` when none was chosen.

    Cloud providers ship a curated ``default_model``. Local providers serve
    whatever is currently loaded, so the model is resolved from the server's
    catalog (via the TTL cache, falling back to the curated tuple when the
    server is offline). A short timeout keeps provider switching snappy against
    a hung local server.
    """
    if provider.default_model:
        return provider.default_model
    models = get_model_catalog(provider.id, refresh=True, timeout=timeout).models
    return models[0] if models else ""


# ---------------------------------------------------------------------------
# Backward-compatible shims for the original single-key API.
# ---------------------------------------------------------------------------


def save_api_key(key: str) -> None:
    """Legacy shim: store the OpenRouter key."""
    set_key(DEFAULT_PROVIDER_ID, key)


def load_api_key() -> str | None:
    """Legacy shim: return the OpenRouter key (env var or stored)."""
    return get_key(DEFAULT_PROVIDER_ID)
