"""Configuration storage for KiwiMateCoder.

Configuration lives in ``~/.kiwimatecoder/config.json`` with this shape::

    {
        "keys": {"openrouter": "sk-...", "openai": "sk-..."},
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
    get_provider,
)

CONFIG_DIR = Path.home() / ".kiwimatecoder"
CONFIG_FILE = CONFIG_DIR / "config.json"
LEGACY_CONFIG_FILE = CONFIG_DIR / "config"

DEFAULT_MODE = "ask"


def _empty_config() -> dict:
    return {
        "keys": {},
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
            cfg.update({k: v for k, v in stored.items() if v is not None or k == "selected_model"})
            cfg["keys"] = dict(stored.get("keys") or {})
    else:
        legacy_key = _read_legacy_key()
        if legacy_key:
            cfg["keys"]["openrouter"] = legacy_key
    # Guarantee structural defaults even if the stored file was partial.
    cfg.setdefault("keys", {})
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


def get_key(provider_id: str) -> str | None:
    """Return the API key for a provider.

    Environment variable (the provider's ``key_env``) takes precedence over the
    stored key (even if the env var is set to the empty string, which clears any
    stored key for this process and forces the friendly "no key" path).
    """
    provider = get_provider(provider_id)
    env_key = os.environ.get(provider.key_env)
    if env_key is not None:
        return env_key or None  # exported empty string -> treat as "no key"
    return load_config()["keys"].get(provider_id)


def set_key(provider_id: str, key: str) -> None:
    """Store an API key for a provider and persist the config."""
    # Validate the provider id eagerly.
    get_provider(provider_id)
    cfg = load_config()
    cfg["keys"][provider_id] = key
    save_config(cfg)


def set_selected_provider(provider_id: str) -> None:
    """Persist the default provider."""
    get_provider(provider_id)
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
        get_provider(pid)
        return pid
    except KeyError:
        return DEFAULT_PROVIDER_ID


# ---------------------------------------------------------------------------
# Backward-compatible shims for the original single-key API.
# ---------------------------------------------------------------------------


def save_api_key(key: str) -> None:
    """Legacy shim: store the OpenRouter key."""
    set_key(DEFAULT_PROVIDER_ID, key)


def load_api_key() -> str | None:
    """Legacy shim: return the OpenRouter key (env var or stored)."""
    return get_key(DEFAULT_PROVIDER_ID)
