"""Configuration storage for KiwiMateCoder.

Configuration lives in ``~/.kiwimatecoder/config.json`` with this shape::

    {
        "keys": {"openrouter": "sk-...", "openai": "sk-..."},
        "providers": {"local": {"name": "...", "base_url": "..."}},
        "model_filters": {"openai": {"mode": "allow", "models": ["gpt-5"]}},
        "selected_provider": "openrouter",
        "selected_model": null,
        "default_mode": "ask"
    }

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
from pathlib import Path

from kiwimatecoder.providers import (
    DEFAULT_PROVIDER_ID,
    REGISTRY,
    ProviderConfig,
    UnknownProviderError,
)

CONFIG_DIR = Path.home() / ".kiwimatecoder"
CONFIG_FILE = CONFIG_DIR / "config.json"
LEGACY_CONFIG_FILE = CONFIG_DIR / "config"

DEFAULT_MODE = "ask"


def _empty_config() -> dict:
    return {
        "keys": {},
        "providers": {},
        "model_filters": {},
        "selected_provider": DEFAULT_PROVIDER_ID,
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


def load_config() -> dict:
    """Load configuration, migrating from the legacy format when needed.

    The returned dict always has the full set of keys (with defaults filled in).
    Migration is non-destructive: the legacy file is left in place.
    """
    cfg = _empty_config()
    if CONFIG_FILE.exists():
        try:
            stored = json.loads(CONFIG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            stored = {}
        if isinstance(stored, dict):
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
    cfg.setdefault("selected_model", None)
    cfg.setdefault("default_mode", DEFAULT_MODE)
    return cfg


def save_config(cfg: dict) -> None:
    """Persist configuration to the JSON config file.

    The file (and its directory) are tightened to owner-only permissions since
    they may contain API keys.
    """
    CONFIG_DIR.mkdir(exist_ok=True)
    try:
        os.chmod(CONFIG_DIR, 0o700)
    except OSError:
        pass
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2) + "\n")
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except OSError:
        pass


def _default_key_env(provider_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in provider_id.upper())
    return f"{cleaned}_API_KEY"


def _provider_from_config(provider_id: str, data: dict) -> ProviderConfig | None:
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


def _known_provider_ids(cfg: dict | None = None) -> list[str]:
    cfg = cfg or load_config()
    custom_ids = sorted(str(pid) for pid in cfg.get("providers", {}))
    return sorted(set(REGISTRY) | set(custom_ids))


def get_provider_config(provider_id: str, cfg: dict | None = None) -> ProviderConfig:
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


def list_provider_configs(cfg: dict | None = None) -> list[ProviderConfig]:
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
    if cfg.get("selected_provider") == provider_id:
        cfg["selected_provider"] = DEFAULT_PROVIDER_ID
        cfg["selected_model"] = None
    save_config(cfg)


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


def set_key(provider_id: str, key: str) -> None:
    """Store an API key for a provider and persist the config."""
    # Validate the provider id eagerly.
    get_provider_config(provider_id)
    cfg = load_config()
    cfg["keys"][provider_id] = key
    save_config(cfg)


def remove_key(provider_id: str) -> bool:
    """Remove a stored API key. Returns True when a stored key existed."""
    get_provider_config(provider_id)
    cfg = load_config()
    existed = provider_id in cfg["keys"]
    cfg["keys"].pop(provider_id, None)
    save_config(cfg)
    return existed


def set_selected_provider(provider_id: str) -> None:
    """Persist the default provider."""
    get_provider_config(provider_id)
    cfg = load_config()
    cfg["selected_provider"] = provider_id
    save_config(cfg)


def set_selected_model(model: str | None) -> None:
    """Persist the default model (None falls back to the provider default)."""
    cfg = load_config()
    cfg["selected_model"] = model
    save_config(cfg)


def get_selected_provider_id(cfg: dict | None = None) -> str:
    """Return the configured selected_provider if valid in the registry, else DEFAULT_PROVIDER_ID.

    Used by the bare launch (forgiving fallback) and ``config check`` (accurate display
    instead of advertising a stale/unknown value from the on-disk JSON).
    """
    if cfg is None:
        cfg = load_config()
    pid = cfg.get("selected_provider") or DEFAULT_PROVIDER_ID
    try:
        get_provider_config(pid, cfg)
        return pid
    except KeyError:
        return DEFAULT_PROVIDER_ID


def get_model_filter(provider_id: str) -> dict:
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


def list_visible_models(provider_id: str) -> list[str]:
    """Return models that should be offered in model-selection UI."""
    provider = get_provider_config(provider_id)
    model_filter = get_model_filter(provider_id)
    mode = model_filter["mode"]
    models = model_filter["models"]
    catalog = list(dict.fromkeys((provider.default_model, *provider.models)))
    if mode == "allow":
        return models
    if mode == "deny":
        denied = set(models)
        return [model for model in catalog if model not in denied]
    return catalog


# ---------------------------------------------------------------------------
# Backward-compatible shims for the original single-key API.
# ---------------------------------------------------------------------------


def save_api_key(key: str) -> None:
    """Legacy shim: store the OpenRouter key."""
    set_key(DEFAULT_PROVIDER_ID, key)


def load_api_key() -> str | None:
    """Legacy shim: return the OpenRouter key (env var or stored)."""
    return get_key(DEFAULT_PROVIDER_ID)
