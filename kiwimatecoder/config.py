"""Configuration storage for KiwiMateCoder.

Configuration lives in ``~/.kiwimatecoder/config.json`` with this shape::

    {
        "keys": {"openrouter": "sk-...", "openai": "sk-..."},
        "providers": {"local": {"name": "...", "base_url": "..."}},
        "model_filters": {"openai": {"mode": "allow", "models": ["gpt-5"]}},
        "selected_provider": "openrouter",
        "active_providers": ["openrouter", "openai"],
        "selected_model": null,
        "default_mode": "ask",
        "hooks": {"post_tool": ["echo ran $KIWI_TOOL_NAME"]},
        "mcp_servers": {
            "files": {"command": "npx", "args": ["-y", "server-filesystem"]},
            "remote": {"url": "https://host/mcp", "headers": {"Authorization": "Bearer ..."}}
        }
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
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kiwimatecoder import catalog
from kiwimatecoder import network as network_settings
from kiwimatecoder.catalog import CatalogFetchError, ModelCatalog
from kiwimatecoder.events import POST_TOOL, PRE_TOOL, SESSION_END, SESSION_START
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
PROJECT_CONFIG_NAME = ".kiwimatecoder.json"
PROJECT_CONFIG_ENV = "KIWIMATECODER_PROJECT_CONFIG"

DEFAULT_MODE = "ask"
CONFIG_VERSION = 2
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
        "version": CONFIG_VERSION,
        "keys": {},
        "providers": {},
        "model_filters": {},
        "tool_permissions": {},
        "sampling": {},
        "system_prompt": None,
        "output_style": "default",
        "selected_provider": DEFAULT_PROVIDER_ID,
        "active_providers": [DEFAULT_PROVIDER_ID],
        "selected_model": None,
        "default_mode": DEFAULT_MODE,
        "trusted_workspace": False,
        "verify_command": "",
        "budget": {},
        "hooks": {},
        "plugins": {},
        "mcp_servers": {},
        "profiles": {},
        "model_routing": {},
        "prompt_cache": False,
        "compact_at_tokens": 64000,
        "context_window": 128000,
        "ui": {},
        "web": {
            "max_chars": 50000,
            "timeout": 20.0,
            "allow_local": False,
            "search_provider": "duckduckgo",
            "search_api_key": "",
        },
        "network": {
            "proxy": "",
            "ca_bundle": "",
            "offline": False,
        },
        "memory": {
            "enabled": True,
            "max_bytes": 16384,
        },
        "vision": {
            "max_image_bytes": 5000000,
            "max_images_per_turn": 4,
        },
        "lsp": {
            "enabled": False,
            "timeout": 10.0,
            "diagnostics_after_edits": True,
            "servers": {},
        },
        "index": {
            "enabled": True,
            "max_files": 5000,
            "max_file_bytes": 262144,
            "embeddings": {"provider": "", "model": "", "batch_size": 32},
        },
        "subagents": {
            "enabled": True,
            "max_steps": 20,
            "model": "",
        },
        "browser": {
            "enabled": False,
            "headless": True,
            "timeout_ms": 15000,
        },
        "shell": {
            "persistent": True,
            "timeout": 120,
            "max_jobs": 8,
        },
        "sandbox": {
            "enabled": False,
            "network": True,
            "extra_writable": [],
        },
        "remote": {
            "enabled": False,
            "host": "",
            "user": "",
            "port": 22,
            "identity": "",
            "workspace": "",
            "devcontainer": "auto",
        },
        "acp": {
            "permission_timeout": 300,
        },
    }


# Top-level keys a stored config may contain. Anything else is a warning in
# :func:`validate_config` (forward compatibility: newer files stay loadable).
_KNOWN_CONFIG_KEYS = frozenset(_empty_config()) | {"version", "command_rules"}


def _read_legacy_key() -> str | None:
    """Read the OpenRouter key from the legacy flat config file, if present."""
    if LEGACY_CONFIG_FILE.exists():
        for line in LEGACY_CONFIG_FILE.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def project_config_path(project_root: Path | str | None = None) -> Path | None:
    """Return the project config file for ``project_root`` when it exists.

    ``$KIWIMATECODER_PROJECT_CONFIG`` overrides the location (useful for
    testing and monorepos). Otherwise the file is ``.kiwimatecoder.json`` in
    ``project_root``, defaulting to the current working directory.
    """
    override = os.environ.get(PROJECT_CONFIG_ENV)
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None
    root = Path(project_root) if project_root is not None else Path.cwd()
    candidate = root / PROJECT_CONFIG_NAME
    return candidate if candidate.is_file() else None


def load_project_config(project_root: Path | str | None = None) -> dict[str, Any]:
    """Load the project-level config overlay, tolerating absence/corruption.

    Project config may not carry API keys; that section is ignored if present.
    """
    path = project_config_path(project_root)
    if path is None:
        return {}
    try:
        stored = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(stored, dict):
        return {}
    stored.pop("keys", None)
    return stored


def _apply_project_overlay(cfg: dict[str, Any], project: dict[str, Any]) -> None:
    """Deep-merge the project config onto ``cfg`` (project values win)."""
    for key, value in project.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            merged = dict(cfg[key])
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict) and isinstance(merged.get(sub_key), dict):
                    merged[sub_key] = {**merged[sub_key], **sub_value}
                else:
                    merged[sub_key] = sub_value
            cfg[key] = merged
        else:
            cfg[key] = value


def load_config(project_root: Path | str | None = None) -> dict[str, Any]:
    """Load configuration, migrating from the legacy format when needed.

    The returned dict always has the full set of keys (with defaults filled in).
    Migration is non-destructive: the legacy file is left in place.

    A project-level ``.kiwimatecoder.json`` (see :func:`project_config_path`)
    is layered on top of the global config, so a repository can pin its
    provider, model, mode, sampling, and tool-permission policies. API keys are
    never read from project files.
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
    cfg.setdefault("tool_permissions", {})
    cfg.setdefault("sampling", {})
    cfg.setdefault("system_prompt", None)
    cfg.setdefault("output_style", "default")
    cfg.setdefault("selected_provider", DEFAULT_PROVIDER_ID)
    cfg.setdefault("active_providers", [DEFAULT_PROVIDER_ID])
    cfg.setdefault("selected_model", None)
    cfg.setdefault("default_mode", DEFAULT_MODE)
    cfg.setdefault("trusted_workspace", False)
    cfg.setdefault("verify_command", "")
    cfg.setdefault("budget", {})
    cfg.setdefault("hooks", {})
    cfg.setdefault("plugins", {})
    cfg.setdefault("mcp_servers", {})
    cfg.setdefault("profiles", {})
    cfg.setdefault("model_routing", {})
    cfg.setdefault("prompt_cache", False)
    cfg.setdefault("compact_at_tokens", 64000)
    cfg.setdefault("context_window", 128000)
    cfg.setdefault("ui", {})
    cfg.setdefault("web", {})
    cfg.setdefault("network", {})
    cfg.setdefault("memory", {})
    cfg.setdefault("vision", {})
    cfg.setdefault("lsp", {})
    cfg.setdefault("index", {})
    cfg.setdefault("shell", {})
    cfg.setdefault("remote", {})
    cfg.setdefault("acp", {})
    # Active-provider roster. Configs written before this feature lack the key;
    # migrate by seeding it from the single selected provider. An explicitly
    # stored empty list, a non-list, or a list of junk is seeded the same way.
    cfg["active_providers"] = _normalized_active_providers(
        stored, str(cfg.get("selected_provider") or DEFAULT_PROVIDER_ID)
    )
    # Project overlay: values win over global config, secrets excluded.
    project = load_project_config(project_root)
    if project:
        _apply_project_overlay(cfg, project)
        if "selected_provider" in project and "active_providers" not in project:
            cfg["active_providers"] = [str(project["selected_provider"])]
        cfg["active_providers"] = _normalized_active_providers(
            {"active_providers": cfg.get("active_providers")},
            str(cfg.get("selected_provider") or DEFAULT_PROVIDER_ID),
        )
    cfg["version"] = CONFIG_VERSION
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
    cfg.setdefault("version", CONFIG_VERSION)
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
    key_header = str(data.get("key_header") or "Authorization").strip()
    if not key_header:
        key_header = "Authorization"
    raw_prefix = data.get("key_prefix")
    key_prefix = "Bearer " if raw_prefix is None else str(raw_prefix)
    api_version = str(data.get("api_version") or "").strip()

    return ProviderConfig(
        id=provider_id,
        name=name,
        base_url=base_url.rstrip("/"),
        default_model=default_model,
        key_env=key_env,
        compat=compat,
        extra_headers={str(k): str(v) for k, v in extra_headers.items()},
        models=models,
        key_header=key_header,
        key_prefix=key_prefix,
        api_version=api_version,
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
    key_header: str | None = None,
    key_prefix: str | None = None,
    api_version: str | None = None,
) -> ProviderConfig:
    """Persist a user-defined provider and return its config.

    ``key_header``/``key_prefix`` follow the OpenAI auth scheme (defaults:
    ``Authorization``/``Bearer ``); Azure-style endpoints use
    ``key_header="api-key"``, ``key_prefix=""``, and an ``api_version`` that is
    appended as ``?api-version=``.
    """
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

    data: dict[str, Any] = {
        "name": name.strip(),
        "base_url": base_url.strip().rstrip("/"),
        "default_model": default_model.strip(),
        "key_env": (key_env or _default_key_env(provider_id)).strip(),
        "compat": compat,
    }
    if key_header is not None:
        if not key_header.strip():
            raise ValueError("Provider key_header is required.")
        data["key_header"] = key_header.strip()
    if key_prefix is not None:
        data["key_prefix"] = str(key_prefix)
    if api_version is not None:
        data["api_version"] = str(api_version).strip()

    cfg = load_config()
    cfg["providers"][provider_id] = data
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
    key_header: str | None = None,
    key_prefix: str | None = None,
    api_version: str | None = None,
) -> ProviderConfig:
    """Update fields of a user-defined provider and return its config.

    Only fields given are changed; None leaves the current value alone.
    ``name``, ``base_url``, and ``default_model`` must stay non-empty when
    changed. ``compat`` must be 'openai' or 'anthropic'. ``key_prefix`` may be
    cleared with an empty string, and ``api_version`` with an empty string.
    Built-in providers cannot be edited (their config lives in the registry).
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
    if key_header is not None:
        if not key_header.strip():
            raise ValueError("Provider key_header is required.")
        data["key_header"] = key_header.strip()
    if key_prefix is not None:
        data["key_prefix"] = str(key_prefix)
    if api_version is not None:
        data["api_version"] = str(api_version).strip()

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
    back into a multi-provider roster. Switching the primary drops the saved
    model so a vendor-specific id cannot leak onto the next session.
    """
    set_active_providers([provider_id])


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
    is kept in sync with the primary for backward compatibility. Changing the
    primary clears ``selected_model`` so the next session uses the new
    provider's default until a model is chosen again.
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
    previous_primary = get_selected_provider_id(cfg)
    cfg["active_providers"] = cleaned
    cfg["selected_provider"] = cleaned[0]
    if cleaned[0] != previous_primary:
        cfg["selected_model"] = None
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
# Tool permissions, sampling, and prompt customization
# ---------------------------------------------------------------------------

SAMPLING_KEYS = ("temperature", "top_p", "max_tokens", "reasoning_effort")
REASONING_EFFORTS = ("minimal", "low", "medium", "high")
OUTPUT_STYLES = ("default", "concise", "explanatory", "code")


def get_always_allowed_tools(cfg: dict[str, Any] | None = None) -> list[str]:
    """Return tools the user permanently approved with "always"."""
    cfg = cfg or load_config()
    perms = cfg.get("tool_permissions") or {}
    raw = perms.get("always_allow") if isinstance(perms, dict) else None
    if not isinstance(raw, list):
        return []
    return list(dict.fromkeys(str(name) for name in raw if str(name).strip()))


def persist_always_allowed_tool(tool_name: str) -> list[str]:
    """Persist a tool approval, returning the updated allowlist."""
    name = tool_name.strip()
    cfg = load_config()
    current = get_always_allowed_tools(cfg)
    if not name:
        return current
    allowed = list(dict.fromkeys([*current, name]))
    perms = dict(cfg.get("tool_permissions") or {})
    perms["always_allow"] = allowed
    cfg["tool_permissions"] = perms
    save_config(cfg)
    return allowed


def remove_always_allowed_tool(tool_name: str) -> bool:
    """Drop a persisted tool approval. Returns whether it existed."""
    cfg = load_config()
    current = get_always_allowed_tools(cfg)
    if tool_name not in current:
        return False
    perms = dict(cfg.get("tool_permissions") or {})
    perms["always_allow"] = [name for name in current if name != tool_name]
    cfg["tool_permissions"] = perms
    save_config(cfg)
    return True


def clear_always_allowed_tools() -> int:
    """Remove every persisted tool approval, returning how many were cleared."""
    cfg = load_config()
    current = get_always_allowed_tools(cfg)
    perms = dict(cfg.get("tool_permissions") or {})
    perms["always_allow"] = []
    cfg["tool_permissions"] = perms
    save_config(cfg)
    return len(current)


def get_command_rules(cfg: dict[str, Any] | None = None) -> dict[str, list[str]]:
    """Return the run_bash allow/deny regex rules."""
    cfg = cfg or load_config()
    stored = cfg.get("command_rules") or {}
    if not isinstance(stored, dict):
        return {"allow": [], "deny": []}
    result: dict[str, list[str]] = {}
    for kind in ("allow", "deny"):
        raw = stored.get(kind)
        patterns = raw if isinstance(raw, list) else []
        result[kind] = list(
            dict.fromkeys(str(p) for p in patterns if str(p).strip())
        )
    return result


def _validate_command_pattern(pattern: str) -> str:
    import re

    cleaned = pattern.strip()
    if not cleaned:
        raise ValueError("Command rule pattern is required.")
    try:
        re.compile(cleaned)
    except re.error as exc:
        raise ValueError(f"Invalid regular expression: {exc}") from exc
    return cleaned


def set_command_rules(
    allow: list[str] | None = None,
    deny: list[str] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Replace the command rules for the given kinds and return the result."""
    cfg = cfg or load_config()
    current = get_command_rules(cfg)
    if allow is not None:
        current["allow"] = list(
            dict.fromkeys(_validate_command_pattern(p) for p in allow)
        )
    if deny is not None:
        current["deny"] = list(
            dict.fromkeys(_validate_command_pattern(p) for p in deny)
        )
    cfg["command_rules"] = current
    save_config(cfg)
    return current


def add_command_rule(
    kind: str, pattern: str, cfg: dict[str, Any] | None = None
) -> dict[str, list[str]]:
    """Add one allow/deny pattern for run_bash commands."""
    kind = kind.strip().lower()
    if kind not in {"allow", "deny"}:
        raise ValueError("Command rule kind must be 'allow' or 'deny'.")
    cleaned = _validate_command_pattern(pattern)
    cfg = cfg or load_config()
    current = get_command_rules(cfg)
    current[kind] = list(dict.fromkeys([*current[kind], cleaned]))
    cfg["command_rules"] = current
    save_config(cfg)
    return current


def remove_command_rule(kind: str, pattern: str) -> bool:
    """Remove one pattern; returns whether it existed."""
    kind = kind.strip().lower()
    if kind not in {"allow", "deny"}:
        raise ValueError("Command rule kind must be 'allow' or 'deny'.")
    cleaned = pattern.strip()
    cfg = load_config()
    current = get_command_rules(cfg)
    if cleaned not in current[kind]:
        return False
    current[kind] = [p for p in current[kind] if p != cleaned]
    cfg["command_rules"] = current
    save_config(cfg)
    return True


def clear_command_rules() -> dict[str, list[str]]:
    """Remove every command rule."""
    cfg = load_config()
    cfg["command_rules"] = {"allow": [], "deny": []}
    save_config(cfg)
    return {"allow": [], "deny": []}


# ---------------------------------------------------------------------------
# Lifecycle hook commands
# ---------------------------------------------------------------------------

HOOK_EVENTS = (SESSION_START, SESSION_END, PRE_TOOL, POST_TOOL)


def _normalized_hooks(stored: Any) -> dict[str, list[str]]:
    """Return per-event hook command lists, dropping junk and duplicates."""
    hooks: dict[str, list[str]] = {event: [] for event in HOOK_EVENTS}
    if not isinstance(stored, dict):
        return hooks
    for event in HOOK_EVENTS:
        raw = stored.get(event)
        if not isinstance(raw, list):
            continue
        hooks[event] = list(
            dict.fromkeys(str(command) for command in raw if str(command).strip())
        )
    return hooks


def _validate_hook_event(event: str) -> str:
    cleaned = str(event).strip().lower()
    if cleaned not in HOOK_EVENTS:
        raise ValueError(
            f"Unknown hook event '{event}'. Choose: {', '.join(HOOK_EVENTS)}."
        )
    return cleaned


def get_hooks(cfg: dict[str, Any] | None = None) -> dict[str, list[str]]:
    """Return configured hook commands for every lifecycle event.

    The result always contains all four event keys (empty lists when unset).
    """
    cfg = cfg or load_config()
    return _normalized_hooks(cfg.get("hooks"))


def add_hook(
    event: str, command: str, cfg: dict[str, Any] | None = None
) -> dict[str, list[str]]:
    """Append a shell command to one lifecycle event and persist it."""
    event = _validate_hook_event(event)
    cleaned = str(command).strip()
    if not cleaned:
        raise ValueError("Hook command is required.")
    cfg = cfg or load_config()
    hooks = _normalized_hooks(cfg.get("hooks"))
    hooks[event] = list(dict.fromkeys([*hooks[event], cleaned]))
    cfg["hooks"] = hooks
    save_config(cfg)
    return hooks


def remove_hook(event: str, index: int, cfg: dict[str, Any] | None = None) -> bool:
    """Remove one hook command by its position; returns whether it existed."""
    event = _validate_hook_event(event)
    cfg = cfg or load_config()
    hooks = _normalized_hooks(cfg.get("hooks"))
    if not isinstance(index, int) or index < 0 or index >= len(hooks[event]):
        return False
    hooks[event].pop(index)
    cfg["hooks"] = hooks
    save_config(cfg)
    return True


def clear_hooks() -> None:
    """Remove every configured hook command."""
    cfg = load_config()
    cfg["hooks"] = {}
    save_config(cfg)


# ---------------------------------------------------------------------------
# Plugins
# ---------------------------------------------------------------------------


def get_plugins_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return plugin loader settings, always fully populated.

    ``allow_project`` gates the workspace's ``.kiwimatecoder/plugins``
    directory (off by default so cloning a repository cannot execute code).
    ``disabled`` lists plugin names to skip entirely.
    """
    cfg = cfg or load_config()
    stored = cfg.get("plugins") or {}
    if not isinstance(stored, dict):
        stored = {}
    raw_disabled = stored.get("disabled")
    disabled: list[str] = []
    if isinstance(raw_disabled, list):
        disabled = list(
            dict.fromkeys(
                str(name).strip() for name in raw_disabled if str(name).strip()
            )
        )
    return {
        "allow_project": bool(stored.get("allow_project", False)),
        "disabled": disabled,
    }


def set_plugins_config(
    *,
    allow_project: bool | None = None,
    disabled: list[str] | None = None,
) -> dict[str, Any]:
    """Update plugin settings and persist them.

    Omitted arguments keep their current value. ``disabled`` must be a list of
    non-empty plugin names (duplicates are collapsed).
    """
    cfg = load_config()
    current = get_plugins_config(cfg)
    if allow_project is not None:
        current["allow_project"] = bool(allow_project)
    if disabled is not None:
        if not isinstance(disabled, list):
            raise ValueError("Plugin 'disabled' must be a list of plugin names.")
        cleaned: list[str] = []
        for name in disabled:
            text = str(name).strip()
            if not text:
                raise ValueError("Plugin names in 'disabled' must be non-empty.")
            if text not in cleaned:
                cleaned.append(text)
        current["disabled"] = cleaned
    cfg["plugins"] = current
    save_config(cfg)
    return current


# ---------------------------------------------------------------------------
# MCP servers
# ---------------------------------------------------------------------------

MCP_NAME_RE = r"[a-z0-9][a-z0-9_-]*"
_MCP_NAME_PATTERN = re.compile(rf"^{MCP_NAME_RE}$")


def _validate_mcp_name(name: object) -> str:
    cleaned = str(name).strip()
    if not _MCP_NAME_PATTERN.match(cleaned):
        raise ValueError(
            f"Invalid MCP server name '{name}': use lowercase letters, digits, "
            "'-', or '_', starting with a letter or digit."
        )
    return cleaned


def _string_map(value: object, field: str, server: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            f"MCP server '{server}': '{field}' must be a mapping of strings."
        )
    result: dict[str, str] = {}
    for key, item in value.items():
        text_key = str(key).strip()
        if not text_key:
            raise ValueError(
                f"MCP server '{server}': '{field}' keys must be non-empty."
            )
        result[text_key] = str(item)
    return result


def _normalize_mcp_spec(name: object, spec: object) -> dict[str, Any]:
    """Validate and normalize one MCP server spec, raising ``ValueError``."""
    server = _validate_mcp_name(name)
    if not isinstance(spec, dict):
        raise ValueError(f"MCP server '{server}': spec must be an object.")
    command = spec.get("command")
    url = spec.get("url")
    if bool(command) == bool(url):
        raise ValueError(
            f"MCP server '{server}': set exactly one of 'command' (stdio) "
            "or 'url' (HTTP)."
        )
    normalized: dict[str, Any] = {}
    if command is not None:
        if not isinstance(command, str) or not command.strip():
            raise ValueError(
                f"MCP server '{server}': 'command' must be a non-empty string."
            )
        raw_args = spec.get("args") or []
        if not isinstance(raw_args, list):
            raise ValueError(
                f"MCP server '{server}': 'args' must be a list of strings."
            )
        normalized["command"] = command.strip()
        normalized["args"] = [str(arg) for arg in raw_args]
        normalized["env"] = _string_map(spec.get("env"), "env", server)
    else:
        if not isinstance(url, str) or not url.strip():
            raise ValueError(
                f"MCP server '{server}': 'url' must be a non-empty string."
            )
        cleaned_url = url.strip()
        if not cleaned_url.startswith(("http://", "https://")):
            raise ValueError(
                f"MCP server '{server}': 'url' must start with http:// or https://."
            )
        normalized["url"] = cleaned_url
        normalized["headers"] = _string_map(spec.get("headers"), "headers", server)
    normalized["disabled"] = bool(spec.get("disabled", False))
    return normalized


def get_mcp_servers(cfg: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Return validated MCP server specs, dropping malformed entries."""
    cfg = cfg or load_config()
    stored = cfg.get("mcp_servers") or {}
    if not isinstance(stored, dict):
        return {}
    servers: dict[str, dict[str, Any]] = {}
    for name, spec in stored.items():
        try:
            servers[_validate_mcp_name(name)] = _normalize_mcp_spec(name, spec)
        except ValueError:
            continue
    return servers


def set_mcp_server(name: str, spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Add or replace one MCP server spec, persist it, and return the section."""
    cfg = load_config()
    servers = get_mcp_servers(cfg)
    servers[_validate_mcp_name(name)] = _normalize_mcp_spec(name, spec)
    cfg["mcp_servers"] = servers
    save_config(cfg)
    return servers


def remove_mcp_server(name: str) -> bool:
    """Remove one MCP server spec. Returns whether it existed."""
    server = str(name).strip()
    cfg = load_config()
    servers = get_mcp_servers(cfg)
    if server not in servers:
        return False
    del servers[server]
    cfg["mcp_servers"] = servers
    save_config(cfg)
    return True


def set_mcp_servers(servers: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Replace the whole MCP server section with validated, normalized specs."""
    if not isinstance(servers, dict):
        raise ValueError("MCP servers must be a mapping of name -> spec.")
    normalized: dict[str, dict[str, Any]] = {}
    for name, spec in servers.items():
        normalized[_validate_mcp_name(name)] = _normalize_mcp_spec(name, spec)
    cfg = load_config()
    cfg["mcp_servers"] = normalized
    save_config(cfg)
    return normalized


def get_trusted_workspace(cfg: dict[str, Any] | None = None) -> bool:
    """Whether read-only tools may read outside the workspace root."""
    cfg = cfg or load_config()
    return bool(cfg.get("trusted_workspace", False))


def set_trusted_workspace(enabled: bool) -> bool:
    """Enable/disable trusted-workspace reads for new sessions."""
    cfg = load_config()
    cfg["trusted_workspace"] = bool(enabled)
    save_config(cfg)
    return bool(enabled)


def get_verify_command(cfg: dict[str, Any] | None = None) -> str:
    """Return the command run automatically after successful file edits."""
    cfg = cfg or load_config()
    return str(cfg.get("verify_command") or "").strip()


def set_verify_command(command: str | None) -> str:
    """Persist the auto-verify command (empty clears it)."""
    cfg = load_config()
    cfg["verify_command"] = (command or "").strip()
    save_config(cfg)
    return cfg["verify_command"]


def get_budget(cfg: dict[str, Any] | None = None) -> dict[str, float]:
    """Return the session budget: max_tokens and/or max_cost_usd."""
    cfg = cfg or load_config()
    stored = cfg.get("budget") or {}
    if not isinstance(stored, dict):
        return {}
    budget: dict[str, float] = {}
    try:
        if stored.get("max_tokens") is not None:
            budget["max_tokens"] = int(stored["max_tokens"])
    except (TypeError, ValueError):
        pass
    try:
        if stored.get("max_cost_usd") is not None:
            budget["max_cost_usd"] = float(stored["max_cost_usd"])
    except (TypeError, ValueError):
        pass
    return budget


_UNSET: Any = object()


def clear_budget() -> None:
    """Remove every budget limit."""
    cfg = load_config()
    cfg["budget"] = {}
    save_config(cfg)


def set_budget(
    max_tokens: int | str | None | Any = _UNSET,
    max_cost_usd: float | str | None | Any = _UNSET,
) -> dict[str, float]:
    """Set/clear budget limits.

    Omitted arguments keep their current value; passing None clears that limit.
    Use :func:`clear_budget` to remove every limit at once.
    """
    cfg = load_config()
    current = dict(cfg.get("budget") or {})
    if max_tokens is not _UNSET:
        if max_tokens is None:
            current.pop("max_tokens", None)
        else:
            tokens = int(max_tokens)
            if tokens < 1:
                raise ValueError("max_tokens budget must be at least 1.")
            current["max_tokens"] = tokens
    if max_cost_usd is not _UNSET:
        if max_cost_usd is None:
            current.pop("max_cost_usd", None)
        else:
            cost = float(max_cost_usd)
            if cost <= 0:
                raise ValueError("max_cost_usd budget must be positive.")
            current["max_cost_usd"] = cost
    cfg["budget"] = current
    save_config(cfg)
    return get_budget(cfg)


SUBAGENT_MAX_STEPS_MIN = 1
SUBAGENT_MAX_STEPS_MAX = 100
SUBAGENT_MAX_STEPS_DEFAULT = 20


def get_subagents(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return normalized subagent settings, always fully populated.

    Malformed stored values fall back to the defaults so a hand-edited config
    can never break the task tool; ``validate_config`` reports exactly what
    would be ignored.
    """
    cfg = cfg or load_config()
    stored = cfg.get("subagents") or {}
    if not isinstance(stored, dict):
        stored = {}
    enabled = stored.get("enabled")
    if not isinstance(enabled, bool):
        enabled = True
    try:
        max_steps = int(stored.get("max_steps", SUBAGENT_MAX_STEPS_DEFAULT))
    except (TypeError, ValueError):
        max_steps = SUBAGENT_MAX_STEPS_DEFAULT
    if not SUBAGENT_MAX_STEPS_MIN <= max_steps <= SUBAGENT_MAX_STEPS_MAX:
        max_steps = SUBAGENT_MAX_STEPS_DEFAULT
    model = stored.get("model")
    return {
        "enabled": enabled,
        "max_steps": max_steps,
        "model": model.strip() if isinstance(model, str) else "",
    }


def set_subagents(
    enabled: bool | None = None,
    max_steps: int | str | None = None,
    model: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update subagent settings; omitted arguments keep their current value.

    Raises ``ValueError`` for a non-boolean ``enabled``, an out-of-range
    ``max_steps`` (1-100), or a non-string ``model`` so callers validate eagerly.
    """
    cfg = cfg or load_config()
    current = get_subagents(cfg)
    if enabled is not None:
        if not isinstance(enabled, bool):
            raise ValueError("subagents enabled must be true or false.")
        current["enabled"] = enabled
    if max_steps is not None:
        try:
            steps = int(max_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError("subagents max_steps must be an integer.") from exc
        if not SUBAGENT_MAX_STEPS_MIN <= steps <= SUBAGENT_MAX_STEPS_MAX:
            raise ValueError(
                f"subagents max_steps must be between {SUBAGENT_MAX_STEPS_MIN} "
                f"and {SUBAGENT_MAX_STEPS_MAX}."
            )
        current["max_steps"] = steps
    if model is not None:
        if not isinstance(model, str):
            raise ValueError("subagents model must be a string.")
        current["model"] = model.strip()
    cfg["subagents"] = current
    save_config(cfg)
    return get_subagents(cfg)


# ---------------------------------------------------------------------------
# Optional browser automation (Playwright)
# ---------------------------------------------------------------------------

BROWSER_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "headless": True,
    "timeout_ms": 15000,
}
BROWSER_TIMEOUT_MIN = 1_000
BROWSER_TIMEOUT_MAX = 120_000


def get_browser(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return normalized browser-automation settings, always fully populated.

    Malformed stored values fall back to :data:`BROWSER_DEFAULTS` so a
    hand-edited config can never crash the tool; ``validate_config`` reports
    exactly what would be ignored.
    """
    cfg = cfg or load_config()
    stored = cfg.get("browser") or {}
    if not isinstance(stored, dict):
        stored = {}
    effective = dict(BROWSER_DEFAULTS)
    enabled = stored.get("enabled")
    if isinstance(enabled, bool):
        effective["enabled"] = enabled
    headless = stored.get("headless")
    if isinstance(headless, bool):
        effective["headless"] = headless
    effective["timeout_ms"] = _bounded_int(
        stored.get("timeout_ms"),
        int(BROWSER_DEFAULTS["timeout_ms"]),
        BROWSER_TIMEOUT_MIN,
        BROWSER_TIMEOUT_MAX,
    )
    return effective


def set_browser(
    enabled: bool | None = None,
    headless: bool | None = None,
    timeout_ms: int | str | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update browser settings; omitted arguments keep their current value.

    Raises ``ValueError`` for a non-boolean flag or a timeout outside
    1-120 seconds so callers can validate eagerly.
    """
    cfg = cfg or load_config()
    current = get_browser(cfg)
    if enabled is not None:
        if not isinstance(enabled, bool):
            raise ValueError("browser enabled must be true or false.")
        current["enabled"] = enabled
    if headless is not None:
        if not isinstance(headless, bool):
            raise ValueError("browser headless must be true or false.")
        current["headless"] = headless
    if timeout_ms is not None:
        try:
            value = int(timeout_ms)
        except (TypeError, ValueError) as exc:
            raise ValueError("browser timeout_ms must be an integer.") from exc
        if not BROWSER_TIMEOUT_MIN <= value <= BROWSER_TIMEOUT_MAX:
            raise ValueError(
                "browser timeout_ms must be between "
                f"{BROWSER_TIMEOUT_MIN} and {BROWSER_TIMEOUT_MAX}."
            )
        current["timeout_ms"] = value
    cfg["browser"] = current
    save_config(cfg)
    return get_browser(cfg)


# ---------------------------------------------------------------------------
# Persistent shell and background processes
# ---------------------------------------------------------------------------

SHELL_DEFAULTS: dict[str, Any] = {
    "persistent": True,
    "timeout": 120,
    "max_jobs": 8,
}
SHELL_TIMEOUT_MIN = 1
SHELL_TIMEOUT_MAX = 3600
SHELL_MAX_JOBS_MIN = 1
SHELL_MAX_JOBS_MAX = 100


def get_shell_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return normalized persistent-shell settings, always fully populated.

    Malformed stored values fall back to :data:`SHELL_DEFAULTS` so a
    hand-edited config can never crash the shell tools; ``validate_config``
    reports exactly what would be ignored.
    """
    cfg = cfg or load_config()
    stored = cfg.get("shell") or {}
    if not isinstance(stored, dict):
        stored = {}
    effective = dict(SHELL_DEFAULTS)
    persistent = stored.get("persistent")
    if isinstance(persistent, bool):
        effective["persistent"] = persistent
    effective["timeout"] = _bounded_int(
        stored.get("timeout"),
        int(SHELL_DEFAULTS["timeout"]),
        SHELL_TIMEOUT_MIN,
        SHELL_TIMEOUT_MAX,
    )
    effective["max_jobs"] = _bounded_int(
        stored.get("max_jobs"),
        int(SHELL_DEFAULTS["max_jobs"]),
        SHELL_MAX_JOBS_MIN,
        SHELL_MAX_JOBS_MAX,
    )
    return effective


def set_shell_config(
    persistent: bool | None = None,
    timeout: int | str | None = None,
    max_jobs: int | str | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update persistent-shell settings; omitted arguments keep their value.

    Raises ``ValueError`` for a non-boolean ``persistent`` or a ``timeout`` /
    ``max_jobs`` outside the supported range, so callers validate eagerly.
    """
    cfg = cfg or load_config()
    current = get_shell_config(cfg)
    if persistent is not None:
        if not isinstance(persistent, bool):
            raise ValueError("shell persistent must be true or false.")
        current["persistent"] = persistent
    if timeout is not None:
        try:
            value = int(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("shell timeout must be an integer.") from exc
        if not SHELL_TIMEOUT_MIN <= value <= SHELL_TIMEOUT_MAX:
            raise ValueError(
                f"shell timeout must be between {SHELL_TIMEOUT_MIN} and "
                f"{SHELL_TIMEOUT_MAX} seconds."
            )
        current["timeout"] = value
    if max_jobs is not None:
        try:
            jobs = int(max_jobs)
        except (TypeError, ValueError) as exc:
            raise ValueError("shell max_jobs must be an integer.") from exc
        if not SHELL_MAX_JOBS_MIN <= jobs <= SHELL_MAX_JOBS_MAX:
            raise ValueError(
                f"shell max_jobs must be between {SHELL_MAX_JOBS_MIN} and "
                f"{SHELL_MAX_JOBS_MAX}."
            )
        current["max_jobs"] = jobs
    cfg["shell"] = current
    save_config(cfg)
    return current


SANDBOX_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "network": True,
    "extra_writable": [],
}


def get_sandbox(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return normalized OS-sandbox settings, always fully populated.

    Malformed stored values fall back to :data:`SANDBOX_DEFAULTS` so a
    hand-edited config can never crash run_bash; ``validate_config`` reports
    exactly what would be ignored. The default is **off**, with network access
    allowed when a sandbox is enabled.
    """
    cfg = cfg or load_config()
    stored = cfg.get("sandbox") or {}
    if not isinstance(stored, dict):
        stored = {}
    effective: dict[str, Any] = {
        "enabled": SANDBOX_DEFAULTS["enabled"],
        "network": SANDBOX_DEFAULTS["network"],
        "extra_writable": [],
    }
    enabled = stored.get("enabled")
    if isinstance(enabled, bool):
        effective["enabled"] = enabled
    network = stored.get("network")
    if isinstance(network, bool):
        effective["network"] = network
    writable = stored.get("extra_writable")
    if isinstance(writable, list):
        effective["extra_writable"] = [
            str(item) for item in writable if str(item).strip()
        ]
    return effective


def set_sandbox(
    enabled: bool | None = None,
    network: bool | None = None,
    extra_writable: Sequence[str] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update sandbox settings; omitted arguments keep their value.

    ``extra_writable`` replaces the list. Raises ``ValueError`` for non-boolean
    flags or an empty/non-string path, so callers validate eagerly.
    """
    cfg = cfg or load_config()
    current = get_sandbox(cfg)
    if enabled is not None:
        if not isinstance(enabled, bool):
            raise ValueError("sandbox enabled must be true or false.")
        current["enabled"] = enabled
    if network is not None:
        if not isinstance(network, bool):
            raise ValueError("sandbox network must be true or false.")
        current["network"] = network
    if extra_writable is not None:
        paths: list[str] = []
        for item in extra_writable:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    "sandbox extra_writable paths must be non-empty strings."
                )
            paths.append(item.strip())
        current["extra_writable"] = paths
    cfg["sandbox"] = current
    save_config(cfg)
    return current


# ---------------------------------------------------------------------------
# Remote (SSH) and devcontainer execution
# ---------------------------------------------------------------------------

REMOTE_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "host": "",
    "user": "",
    "port": 22,
    "identity": "",
    "workspace": "",
    "devcontainer": "auto",
}
REMOTE_SPECIAL_DEVCONTAINERS = ("auto", "off")
REMOTE_PORT_MIN = 1
REMOTE_PORT_MAX = 65535


def get_remote(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return normalized remote-execution settings, always fully populated.

    Malformed stored values fall back to :data:`REMOTE_DEFAULTS` so a
    hand-edited config can never crash a shell command; ``validate_config``
    reports exactly what would be ignored. An ``identity`` path is kept as
    stored (existence is validated on write and reported by ``validate_config``)
    and any non-empty ``devcontainer`` string other than ``"auto"``/``"off"``
    is treated as a container name.
    """
    cfg = cfg or load_config()
    stored = cfg.get("remote") or {}
    if not isinstance(stored, dict):
        stored = {}
    effective = dict(REMOTE_DEFAULTS)
    enabled = stored.get("enabled")
    if isinstance(enabled, bool):
        effective["enabled"] = enabled
    for key in ("host", "user", "identity", "workspace"):
        value = stored.get(key)
        if isinstance(value, str):
            effective[key] = value.strip()
    effective["port"] = _bounded_int(
        stored.get("port"),
        int(REMOTE_DEFAULTS["port"]),
        REMOTE_PORT_MIN,
        REMOTE_PORT_MAX,
    )
    devcontainer = stored.get("devcontainer")
    if isinstance(devcontainer, str) and devcontainer.strip():
        effective["devcontainer"] = devcontainer.strip()
    return effective


def set_remote(
    enabled: bool | None = None,
    host: str | None = None,
    user: str | None = None,
    port: int | str | None = None,
    identity: str | None = None,
    workspace: str | None = None,
    devcontainer: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update remote-execution settings; omitted arguments keep their value.

    Raises ``ValueError`` for a non-boolean ``enabled``, a non-string field, a
    ``port`` outside 1-65535, an ``identity`` that is neither empty nor an
    existing file, an empty ``devcontainer``, or enabling remote execution with
    no host and no explicit container name. Pass an empty string to clear
    ``host``/``user``/``identity``/``workspace``.
    """
    cfg = cfg or load_config()
    current = get_remote(cfg)
    if enabled is not None:
        if not isinstance(enabled, bool):
            raise ValueError("remote enabled must be true or false.")
        current["enabled"] = enabled
    for key, text in (("host", host), ("user", user), ("workspace", workspace)):
        if text is not None:
            if not isinstance(text, str):
                raise ValueError(f"remote {key} must be a string.")
            current[key] = text.strip()
    if port is not None:
        try:
            port_number = int(port)
        except (TypeError, ValueError) as exc:
            raise ValueError("remote port must be an integer.") from exc
        if not REMOTE_PORT_MIN <= port_number <= REMOTE_PORT_MAX:
            raise ValueError(
                f"remote port must be between {REMOTE_PORT_MIN} and {REMOTE_PORT_MAX}."
            )
        current["port"] = port_number
    if identity is not None:
        if not isinstance(identity, str):
            raise ValueError("remote identity must be a path string.")
        cleaned = identity.strip()
        if cleaned:
            path = Path(cleaned).expanduser()
            if not path.is_file():
                raise ValueError(f"remote identity file does not exist: {cleaned}")
            cleaned = str(path)
        current["identity"] = cleaned
    if devcontainer is not None:
        if not isinstance(devcontainer, str) or not devcontainer.strip():
            raise ValueError(
                "remote devcontainer must be 'auto', 'off', or a container name."
            )
        current["devcontainer"] = devcontainer.strip()
    if current["enabled"] and not current["host"] and (
        current["devcontainer"] in REMOTE_SPECIAL_DEVCONTAINERS
    ):
        raise ValueError(
            "remote host is required when remote execution is enabled "
            "(set a host or name a devcontainer)."
        )
    cfg["remote"] = current
    save_config(cfg)
    return current


# ---------------------------------------------------------------------------
# Agent Client Protocol (editor integration)
# ---------------------------------------------------------------------------

ACP_DEFAULTS: dict[str, Any] = {"permission_timeout": 300}
ACP_TIMEOUT_MIN = 1
ACP_TIMEOUT_MAX = 3600


def get_acp(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return normalized ACP server settings, always fully populated.

    ``permission_timeout`` is how long a ``session/request_permission`` request
    may wait before the action is denied (seconds). Malformed stored values
    fall back to :data:`ACP_DEFAULTS` so a hand-edited config can never wedge
    the editor; ``validate_config`` reports exactly what would be ignored.
    """
    cfg = cfg or load_config()
    stored = cfg.get("acp") or {}
    if not isinstance(stored, dict):
        stored = {}
    return {
        "permission_timeout": _bounded_int(
            stored.get("permission_timeout"),
            int(ACP_DEFAULTS["permission_timeout"]),
            ACP_TIMEOUT_MIN,
            ACP_TIMEOUT_MAX,
        )
    }


def set_acp(
    permission_timeout: int | str | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update ACP server settings; omitted arguments keep their current value.

    Raises ``ValueError`` for a non-integer or out-of-range timeout so callers
    can validate eagerly.
    """
    cfg = cfg or load_config()
    current = get_acp(cfg)
    if permission_timeout is not None:
        try:
            value = int(permission_timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("acp permission_timeout must be an integer.") from exc
        if not ACP_TIMEOUT_MIN <= value <= ACP_TIMEOUT_MAX:
            raise ValueError(
                "acp permission_timeout must be between "
                f"{ACP_TIMEOUT_MIN} and {ACP_TIMEOUT_MAX} seconds."
            )
        current["permission_timeout"] = value
    cfg["acp"] = current
    save_config(cfg)
    return get_acp(cfg)


def get_compact_at_tokens(cfg: dict[str, Any] | None = None) -> int:
    """Token budget above which history is trimmed before a request."""
    cfg = cfg or load_config()
    try:
        value = int(cfg.get("compact_at_tokens", 64000))
    except (TypeError, ValueError):
        return 64000
    return value if value > 0 else 64000


def set_compact_at_tokens(tokens: int | str | None) -> int:
    """Persist the auto-compaction threshold."""
    cfg = load_config()
    if tokens is None:
        cfg["compact_at_tokens"] = 64000
    else:
        value = int(tokens)
        if value < 1000:
            raise ValueError("compact_at_tokens must be at least 1000.")
        cfg["compact_at_tokens"] = value
    save_config(cfg)
    return get_compact_at_tokens(cfg)


def get_context_window(cfg: dict[str, Any] | None = None) -> int:
    """Approximate model context window used for the context gauge."""
    cfg = cfg or load_config()
    try:
        value = int(cfg.get("context_window", 128000))
    except (TypeError, ValueError):
        return 128000
    return value if value > 0 else 128000


def set_context_window(tokens: int | str | None) -> int:
    """Persist the context-window size used for the gauge."""
    cfg = load_config()
    if tokens is None:
        cfg["context_window"] = 128000
    else:
        value = int(tokens)
        if value < 1000:
            raise ValueError("context_window must be at least 1000.")
        cfg["context_window"] = value
    save_config(cfg)
    return get_context_window(cfg)


def get_sampling(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return validated sampling parameters (only explicitly set keys)."""
    cfg = cfg or load_config()
    stored = cfg.get("sampling") or {}
    if not isinstance(stored, dict):
        return {}
    clean: dict[str, Any] = {}
    for key in SAMPLING_KEYS:
        value = stored.get(key)
        if value is None:
            continue
        try:
            if key == "temperature":
                clean[key] = float(value)
            elif key == "top_p":
                clean[key] = float(value)
            elif key == "max_tokens":
                clean[key] = int(value)
            else:
                effort = str(value).strip().lower()
                if effort:
                    clean[key] = effort
        except (TypeError, ValueError):
            continue
    return clean


def _coerce_sampling(key: str, value: Any) -> Any:
    if key in ("temperature", "top_p"):
        number = float(value)
        limit = 2.0 if key == "temperature" else 1.0
        if not 0.0 <= number <= limit:
            raise ValueError(f"{key} must be between 0 and {limit:g}.")
        return number
    if key == "max_tokens":
        tokens = int(value)
        if tokens < 1:
            raise ValueError("max_tokens must be at least 1.")
        return tokens
    effort = str(value).strip().lower()
    if effort not in REASONING_EFFORTS:
        raise ValueError(
            f"reasoning_effort must be one of: {', '.join(REASONING_EFFORTS)}."
        )
    return effort


def set_sampling(updates: dict[str, Any]) -> dict[str, Any]:
    """Set/clear sampling parameters (a value of None or "" clears a key)."""
    cfg = load_config()
    current = dict(cfg.get("sampling") or {})
    for key, value in updates.items():
        if key not in SAMPLING_KEYS:
            raise ValueError(
                f"Unknown sampling parameter '{key}'. "
                f"Choose: {', '.join(SAMPLING_KEYS)}."
            )
        if value is None or str(value).strip() == "":
            current.pop(key, None)
        else:
            current[key] = _coerce_sampling(key, value)
    cfg["sampling"] = current
    save_config(cfg)
    return get_sampling(cfg)


def reset_sampling() -> None:
    """Clear every sampling parameter."""
    cfg = load_config()
    cfg["sampling"] = {}
    save_config(cfg)


def get_system_prompt(cfg: dict[str, Any] | None = None) -> str | None:
    """Return the user's custom system-prompt addition, if any."""
    cfg = cfg or load_config()
    value = cfg.get("system_prompt")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def set_system_prompt(prompt: str | None) -> None:
    """Persist a custom system-prompt addition (None/empty clears it)."""
    cfg = load_config()
    text = (prompt or "").strip()
    cfg["system_prompt"] = text or None
    save_config(cfg)


def get_output_style(cfg: dict[str, Any] | None = None) -> str:
    """Return the configured output style, falling back to 'default'."""
    cfg = cfg or load_config()
    style = str(cfg.get("output_style") or "default").strip().lower()
    return style if style in OUTPUT_STYLES else "default"


def set_output_style(style: str) -> str:
    """Persist the output style and return the effective value."""
    cleaned = style.strip().lower()
    if cleaned not in OUTPUT_STYLES:
        raise ValueError(f"Unknown output style '{style}'. Choose: {', '.join(OUTPUT_STYLES)}.")
    cfg = load_config()
    cfg["output_style"] = cleaned
    save_config(cfg)
    return cleaned


# ---------------------------------------------------------------------------
# Web fetch/search
# ---------------------------------------------------------------------------

WEB_DEFAULTS: dict[str, Any] = {
    "max_chars": 50000,
    "timeout": 20.0,
    "allow_local": False,
    "search_provider": "duckduckgo",
    "search_api_key": "",
}
WEB_SEARCH_PROVIDERS = ("duckduckgo",)
WEB_MAX_CHARS_MIN = 1_000
WEB_MAX_CHARS_MAX = 2_000_000
WEB_TIMEOUT_MIN = 1.0
WEB_TIMEOUT_MAX = 120.0


def get_web(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return normalized web-fetch/search settings, always fully populated."""
    cfg = cfg or load_config()
    stored = cfg.get("web") or {}
    if not isinstance(stored, dict):
        stored = {}
    effective = dict(WEB_DEFAULTS)
    try:
        max_chars = int(stored.get("max_chars", WEB_DEFAULTS["max_chars"]))
    except (TypeError, ValueError):
        max_chars = int(WEB_DEFAULTS["max_chars"])
    if WEB_MAX_CHARS_MIN <= max_chars <= WEB_MAX_CHARS_MAX:
        effective["max_chars"] = max_chars
    try:
        timeout = float(stored.get("timeout", WEB_DEFAULTS["timeout"]))
    except (TypeError, ValueError):
        timeout = float(WEB_DEFAULTS["timeout"])
    if WEB_TIMEOUT_MIN <= timeout <= WEB_TIMEOUT_MAX:
        effective["timeout"] = timeout
    allow_local = stored.get("allow_local")
    if isinstance(allow_local, bool):
        effective["allow_local"] = allow_local
    provider = str(stored.get("search_provider") or "").strip().lower()
    if provider in WEB_SEARCH_PROVIDERS:
        effective["search_provider"] = provider
    api_key = stored.get("search_api_key")
    effective["search_api_key"] = str(api_key) if api_key else ""
    return effective


def _validate_web_max_chars(value: Any) -> int:
    try:
        cleaned = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("web max_chars must be an integer.") from exc
    if not WEB_MAX_CHARS_MIN <= cleaned <= WEB_MAX_CHARS_MAX:
        raise ValueError(
            f"web max_chars must be between {WEB_MAX_CHARS_MIN} "
            f"and {WEB_MAX_CHARS_MAX}."
        )
    return cleaned


def _validate_web_timeout(value: Any) -> float:
    try:
        cleaned = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("web timeout must be a number of seconds.") from exc
    if not WEB_TIMEOUT_MIN <= cleaned <= WEB_TIMEOUT_MAX:
        raise ValueError(
            f"web timeout must be between {WEB_TIMEOUT_MIN:g} "
            f"and {WEB_TIMEOUT_MAX:g} seconds."
        )
    return cleaned


def set_web(
    max_chars: int | str | None = None,
    timeout: float | str | None = None,
    allow_local: bool | None = None,
    search_provider: str | None = None,
    search_api_key: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update web settings; omitted arguments keep their current value.

    Raises ``ValueError`` for out-of-range numbers or an unsupported search
    provider, so callers can validate eagerly.
    """
    cfg = cfg or load_config()
    current = get_web(cfg)
    if max_chars is not None:
        current["max_chars"] = _validate_web_max_chars(max_chars)
    if timeout is not None:
        current["timeout"] = _validate_web_timeout(timeout)
    if allow_local is not None:
        current["allow_local"] = bool(allow_local)
    if search_provider is not None:
        provider = str(search_provider).strip().lower()
        if provider not in WEB_SEARCH_PROVIDERS:
            raise ValueError(
                f"Unknown search provider '{search_provider}'. "
                f"Choose: {', '.join(WEB_SEARCH_PROVIDERS)}."
            )
        current["search_provider"] = provider
    if search_api_key is not None:
        current["search_api_key"] = str(search_api_key).strip()
    cfg["web"] = current
    save_config(cfg)
    return current


# ---------------------------------------------------------------------------
# Network transport (proxy, custom CA, offline mode)
# ---------------------------------------------------------------------------


def get_network(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return validated proxy/CA/offline settings, always fully populated.

    Malformed stored values fall back to :data:`network.NETWORK_DEFAULTS`
    (same tolerance as the other getters); ``validate_config`` reports exactly
    what would be ignored.
    """
    cfg = cfg or load_config()
    return network_settings.normalize(cfg.get("network"))


def set_network(
    proxy: str | None = None,
    ca_bundle: str | None = None,
    offline: bool | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update network transport settings; omitted arguments keep their value.

    Raises ``ValueError`` for a proxy that is not an http(s) URL, a CA bundle
    path that does not exist, or a non-boolean ``offline``, so callers can
    validate eagerly. Pass an empty string to clear ``proxy``/``ca_bundle``.
    """
    cfg = cfg or load_config()
    current = get_network(cfg)
    if proxy is not None:
        current["proxy"] = network_settings.validate_proxy(proxy)
    if ca_bundle is not None:
        current["ca_bundle"] = network_settings.validate_ca_bundle(ca_bundle)
    if offline is not None:
        if not isinstance(offline, bool):
            raise ValueError("network offline must be true or false.")
        current["offline"] = offline
    cfg["network"] = current
    save_config(cfg)
    return current


def get_network_options(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return ``httpx`` client keyword arguments for the active network config."""
    return network_settings.httpx_options(get_network(cfg))


def offline_enabled(cfg: dict[str, Any] | None = None) -> bool:
    """Whether outbound cloud requests are blocked by offline mode."""
    return bool(get_network(cfg)["offline"])


# ---------------------------------------------------------------------------
# Persistent memory
# ---------------------------------------------------------------------------

MEMORY_DEFAULTS: dict[str, Any] = {"enabled": True, "max_bytes": 16384}
MEMORY_MAX_BYTES_MIN = 1024
MEMORY_MAX_BYTES_MAX = 1024 * 1024


def get_memory(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return normalized persistent-memory settings, always fully populated."""
    cfg = cfg or load_config()
    stored = cfg.get("memory") or {}
    if not isinstance(stored, dict):
        stored = {}
    effective = dict(MEMORY_DEFAULTS)
    enabled = stored.get("enabled")
    if isinstance(enabled, bool):
        effective["enabled"] = enabled
    try:
        max_bytes = int(stored.get("max_bytes", MEMORY_DEFAULTS["max_bytes"]))
    except (TypeError, ValueError):
        max_bytes = int(MEMORY_DEFAULTS["max_bytes"])
    if MEMORY_MAX_BYTES_MIN <= max_bytes <= MEMORY_MAX_BYTES_MAX:
        effective["max_bytes"] = max_bytes
    return effective


def set_memory(
    enabled: bool | None = None,
    max_bytes: int | str | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update persistent-memory settings; omitted arguments are unchanged.

    Raises ``ValueError`` for a non-boolean ``enabled`` or a ``max_bytes``
    outside 1 KB - 1 MB.
    """
    cfg = cfg or load_config()
    current = get_memory(cfg)
    if enabled is not None:
        if not isinstance(enabled, bool):
            raise ValueError("memory enabled must be true or false.")
        current["enabled"] = enabled
    if max_bytes is not None:
        try:
            value = int(max_bytes)
        except (TypeError, ValueError) as exc:
            raise ValueError("memory max_bytes must be an integer.") from exc
        if not MEMORY_MAX_BYTES_MIN <= value <= MEMORY_MAX_BYTES_MAX:
            raise ValueError(
                f"memory max_bytes must be between {MEMORY_MAX_BYTES_MIN} "
                f"and {MEMORY_MAX_BYTES_MAX}."
            )
        current["max_bytes"] = value
    cfg["memory"] = current
    save_config(cfg)
    return current


# ---------------------------------------------------------------------------
# Vision (image attachments)
# ---------------------------------------------------------------------------

VISION_DEFAULTS: dict[str, Any] = {
    "max_image_bytes": 5_000_000,
    "max_images_per_turn": 4,
}
VISION_MAX_IMAGE_BYTES_MIN = 1_024
VISION_MAX_IMAGE_BYTES_MAX = 50_000_000
VISION_MAX_IMAGES_MIN = 1
VISION_MAX_IMAGES_MAX = 100


def get_vision(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return normalized vision settings, always fully populated."""
    cfg = cfg or load_config()
    stored = cfg.get("vision") or {}
    if not isinstance(stored, dict):
        stored = {}
    effective = dict(VISION_DEFAULTS)
    try:
        max_bytes = int(
            stored.get("max_image_bytes", VISION_DEFAULTS["max_image_bytes"])
        )
    except (TypeError, ValueError):
        max_bytes = int(VISION_DEFAULTS["max_image_bytes"])
    if VISION_MAX_IMAGE_BYTES_MIN <= max_bytes <= VISION_MAX_IMAGE_BYTES_MAX:
        effective["max_image_bytes"] = max_bytes
    try:
        max_images = int(
            stored.get("max_images_per_turn", VISION_DEFAULTS["max_images_per_turn"])
        )
    except (TypeError, ValueError):
        max_images = int(VISION_DEFAULTS["max_images_per_turn"])
    if VISION_MAX_IMAGES_MIN <= max_images <= VISION_MAX_IMAGES_MAX:
        effective["max_images_per_turn"] = max_images
    return effective


def set_vision(
    max_image_bytes: int | str | None = None,
    max_images_per_turn: int | str | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update vision limits; omitted arguments keep their current value.

    Raises ``ValueError`` for a non-integer value or one outside the supported
    range, so callers can validate eagerly.
    """
    cfg = cfg or load_config()
    current = get_vision(cfg)
    if max_image_bytes is not None:
        try:
            value = int(max_image_bytes)
        except (TypeError, ValueError) as exc:
            raise ValueError("vision max_image_bytes must be an integer.") from exc
        if not VISION_MAX_IMAGE_BYTES_MIN <= value <= VISION_MAX_IMAGE_BYTES_MAX:
            raise ValueError(
                "vision max_image_bytes must be between "
                f"{VISION_MAX_IMAGE_BYTES_MIN} and {VISION_MAX_IMAGE_BYTES_MAX}."
            )
        current["max_image_bytes"] = value
    if max_images_per_turn is not None:
        try:
            count = int(max_images_per_turn)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "vision max_images_per_turn must be an integer."
            ) from exc
        if not VISION_MAX_IMAGES_MIN <= count <= VISION_MAX_IMAGES_MAX:
            raise ValueError(
                "vision max_images_per_turn must be between "
                f"{VISION_MAX_IMAGES_MIN} and {VISION_MAX_IMAGES_MAX}."
            )
        current["max_images_per_turn"] = count
    cfg["vision"] = current
    save_config(cfg)
    return current


# ---------------------------------------------------------------------------
# Language server (LSP) diagnostics
# ---------------------------------------------------------------------------

LSP_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "timeout": 10.0,
    "diagnostics_after_edits": True,
    "servers": {},
}
LSP_TIMEOUT_MIN = 1.0
LSP_TIMEOUT_MAX = 60.0


def _normalize_lsp_server(name: object, spec: object) -> dict[str, Any]:
    """Validate one LSP server override, raising ``ValueError``.

    A spec is ``{"command": str, "args": [str], "extensions": [str]}``.
    Extensions are normalized to lowercase with a leading dot; a user spec may
    omit them to inherit the built-in preset's extensions.
    """
    server = str(name).strip()
    if not server:
        raise ValueError("LSP server name must be non-empty.")
    if not isinstance(spec, dict):
        raise ValueError(f"LSP server '{server}': spec must be an object.")
    command = spec.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError(
            f"LSP server '{server}': 'command' must be a non-empty string."
        )
    raw_args = spec.get("args") or []
    if not isinstance(raw_args, list):
        raise ValueError(f"LSP server '{server}': 'args' must be a list of strings.")
    raw_extensions = spec.get("extensions") or []
    if not isinstance(raw_extensions, list):
        raise ValueError(
            f"LSP server '{server}': 'extensions' must be a list of strings."
        )
    extensions: list[str] = []
    for raw in raw_extensions:
        extension = str(raw).strip().lower()
        if not extension:
            raise ValueError(
                f"LSP server '{server}': extension entries must be non-empty."
            )
        if not extension.startswith("."):
            extension = "." + extension
        if extension not in extensions:
            extensions.append(extension)
    return {
        "command": command.strip(),
        "args": [str(arg) for arg in raw_args],
        "extensions": extensions,
    }


def get_lsp(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return normalized LSP settings, always fully populated.

    ``servers`` holds only the user's overrides/additions; built-in presets are
    merged by the LSP manager. Malformed entries are dropped.
    """
    cfg = cfg or load_config()
    stored = cfg.get("lsp") or {}
    if not isinstance(stored, dict):
        stored = {}
    effective = dict(LSP_DEFAULTS)
    enabled = stored.get("enabled")
    if isinstance(enabled, bool):
        effective["enabled"] = enabled
    after_edits = stored.get("diagnostics_after_edits")
    if isinstance(after_edits, bool):
        effective["diagnostics_after_edits"] = after_edits
    try:
        timeout = float(stored.get("timeout", LSP_DEFAULTS["timeout"]))
    except (TypeError, ValueError):
        timeout = float(LSP_DEFAULTS["timeout"])
    if LSP_TIMEOUT_MIN <= timeout <= LSP_TIMEOUT_MAX:
        effective["timeout"] = timeout
    servers: dict[str, dict[str, Any]] = {}
    raw_servers = stored.get("servers")
    if isinstance(raw_servers, dict):
        for name, spec in raw_servers.items():
            try:
                servers[str(name)] = _normalize_lsp_server(name, spec)
            except ValueError:
                continue
    effective["servers"] = servers
    return effective


def set_lsp(
    enabled: bool | None = None,
    timeout: float | str | None = None,
    diagnostics_after_edits: bool | None = None,
    servers: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update LSP settings; omitted arguments keep their current value.

    ``servers`` replaces the whole override map and must be a mapping of
    name -> ``{"command", "args", "extensions"}``. Raises ``ValueError`` for an
    out-of-range timeout or a malformed spec so callers can validate eagerly.
    """
    cfg = cfg or load_config()
    current = get_lsp(cfg)
    if enabled is not None:
        if not isinstance(enabled, bool):
            raise ValueError("lsp enabled must be true or false.")
        current["enabled"] = enabled
    if timeout is not None:
        try:
            value = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("lsp timeout must be a number of seconds.") from exc
        if not LSP_TIMEOUT_MIN <= value <= LSP_TIMEOUT_MAX:
            raise ValueError(
                f"lsp timeout must be between {LSP_TIMEOUT_MIN:g} "
                f"and {LSP_TIMEOUT_MAX:g} seconds."
            )
        current["timeout"] = value
    if diagnostics_after_edits is not None:
        if not isinstance(diagnostics_after_edits, bool):
            raise ValueError("lsp diagnostics_after_edits must be true or false.")
        current["diagnostics_after_edits"] = diagnostics_after_edits
    if servers is not None:
        if not isinstance(servers, dict):
            raise ValueError("LSP servers must be a mapping of name -> spec.")
        normalized: dict[str, dict[str, Any]] = {}
        for name, spec in servers.items():
            normalized[str(name).strip()] = _normalize_lsp_server(name, spec)
        current["servers"] = normalized
    cfg["lsp"] = current
    save_config(cfg)
    return get_lsp(cfg)


# ---------------------------------------------------------------------------
# Codebase index / semantic search
# ---------------------------------------------------------------------------

INDEX_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "max_files": 5000,
    "max_file_bytes": 262144,
    "embeddings": {"provider": "", "model": "", "batch_size": 32},
}
INDEX_MAX_FILES_MIN = 1
INDEX_MAX_FILES_MAX = 100_000
INDEX_MAX_FILE_BYTES_MIN = 1_024
INDEX_MAX_FILE_BYTES_MAX = 50_000_000
INDEX_BATCH_SIZE_MIN = 1
INDEX_BATCH_SIZE_MAX = 256


def _bounded_int(value: object, default: int, minimum: int, maximum: int) -> int:
    """Coerce ``value`` to an int inside ``[minimum, maximum]``, else default."""
    try:
        number = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return default
    return number if minimum <= number <= maximum else default


def get_index(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return normalized codebase-index settings, always fully populated.

    Malformed stored values fall back to :data:`INDEX_DEFAULTS` so a
    hand-edited config can never crash semantic search; ``validate_config``
    reports exactly what would be ignored.
    """
    cfg = cfg or load_config()
    stored = cfg.get("index") or {}
    if not isinstance(stored, dict):
        stored = {}
    effective: dict[str, Any] = {
        "enabled": INDEX_DEFAULTS["enabled"],
        "max_files": INDEX_DEFAULTS["max_files"],
        "max_file_bytes": INDEX_DEFAULTS["max_file_bytes"],
        "embeddings": dict(INDEX_DEFAULTS["embeddings"]),
    }
    enabled = stored.get("enabled")
    if isinstance(enabled, bool):
        effective["enabled"] = enabled
    effective["max_files"] = _bounded_int(
        stored.get("max_files"),
        int(INDEX_DEFAULTS["max_files"]),
        INDEX_MAX_FILES_MIN,
        INDEX_MAX_FILES_MAX,
    )
    effective["max_file_bytes"] = _bounded_int(
        stored.get("max_file_bytes"),
        int(INDEX_DEFAULTS["max_file_bytes"]),
        INDEX_MAX_FILE_BYTES_MIN,
        INDEX_MAX_FILE_BYTES_MAX,
    )
    embeddings = stored.get("embeddings") or {}
    if isinstance(embeddings, dict):
        provider = embeddings.get("provider")
        effective["embeddings"]["provider"] = (
            provider.strip() if isinstance(provider, str) else ""
        )
        model = embeddings.get("model")
        effective["embeddings"]["model"] = (
            model.strip() if isinstance(model, str) else ""
        )
        effective["embeddings"]["batch_size"] = _bounded_int(
            embeddings.get("batch_size"),
            int(INDEX_DEFAULTS["embeddings"]["batch_size"]),
            INDEX_BATCH_SIZE_MIN,
            INDEX_BATCH_SIZE_MAX,
        )
    return effective


def set_index(
    enabled: bool | None = None,
    max_files: int | str | None = None,
    max_file_bytes: int | str | None = None,
    embed_provider: str | None = None,
    embed_model: str | None = None,
    embed_batch_size: int | str | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update codebase-index settings; omitted arguments keep their value.

    Raises ``ValueError`` for an out-of-range cap, an unknown embedding
    provider, or a non-boolean ``enabled`` so callers can validate eagerly.
    Pass an empty provider/model to turn embeddings back off.
    """
    cfg = cfg or load_config()
    current = get_index(cfg)
    if enabled is not None:
        if not isinstance(enabled, bool):
            raise ValueError("index enabled must be true or false.")
        current["enabled"] = enabled
    if max_files is not None:
        try:
            value = int(max_files)
        except (TypeError, ValueError) as exc:
            raise ValueError("index max_files must be an integer.") from exc
        if not INDEX_MAX_FILES_MIN <= value <= INDEX_MAX_FILES_MAX:
            raise ValueError(
                f"index max_files must be between {INDEX_MAX_FILES_MIN} and "
                f"{INDEX_MAX_FILES_MAX}."
            )
        current["max_files"] = value
    if max_file_bytes is not None:
        try:
            size = int(max_file_bytes)
        except (TypeError, ValueError) as exc:
            raise ValueError("index max_file_bytes must be an integer.") from exc
        if not INDEX_MAX_FILE_BYTES_MIN <= size <= INDEX_MAX_FILE_BYTES_MAX:
            raise ValueError(
                "index max_file_bytes must be between "
                f"{INDEX_MAX_FILE_BYTES_MIN} and {INDEX_MAX_FILE_BYTES_MAX}."
            )
        current["max_file_bytes"] = size
    if embed_provider is not None:
        provider_id = str(embed_provider).strip()
        if provider_id:
            try:
                get_provider_config(provider_id, cfg)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
        current["embeddings"]["provider"] = provider_id
    if embed_model is not None:
        current["embeddings"]["model"] = str(embed_model).strip()
    if embed_batch_size is not None:
        try:
            batch = int(embed_batch_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "index embeddings batch_size must be an integer."
            ) from exc
        if not INDEX_BATCH_SIZE_MIN <= batch <= INDEX_BATCH_SIZE_MAX:
            raise ValueError(
                "index embeddings batch_size must be between "
                f"{INDEX_BATCH_SIZE_MIN} and {INDEX_BATCH_SIZE_MAX}."
            )
        current["embeddings"]["batch_size"] = batch
    cfg["index"] = current
    save_config(cfg)
    return current


# ---------------------------------------------------------------------------
# UI preferences (color, theme, output mode, ASCII)
# ---------------------------------------------------------------------------


def get_ui(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return normalized UI preferences, always fully populated.

    Stored values that are not recognized fall back to :data:`ui.UI_DEFAULTS`
    (same tolerance as the other section getters), so a hand-edited config can
    never crash the REPL.
    """
    from kiwimatecoder.ui import COLOR_MODES, OUTPUT_MODES, THEMES, UI_DEFAULTS

    cfg = cfg or load_config()
    stored = cfg.get("ui") or {}
    if not isinstance(stored, dict):
        stored = {}
    effective = dict(UI_DEFAULTS)
    color = str(stored.get("color") or "").strip().lower()
    if color in COLOR_MODES:
        effective["color"] = color
    output_mode = str(stored.get("output_mode") or "").strip().lower()
    if output_mode in OUTPUT_MODES:
        effective["output_mode"] = output_mode
    ascii_value = stored.get("ascii")
    if isinstance(ascii_value, bool):
        effective["ascii"] = ascii_value
    theme = str(stored.get("theme") or "").strip().lower()
    if theme in THEMES:
        effective["theme"] = theme
    return effective


def set_ui(
    color: str | None = None,
    output_mode: str | None = None,
    ascii: bool | None = None,
    theme: str | None = None,
) -> dict[str, Any]:
    """Update UI preferences and persist them; omitted values are unchanged."""
    from kiwimatecoder.ui import COLOR_MODES, OUTPUT_MODES, THEMES

    cfg = load_config()
    current = get_ui(cfg)
    if color is not None:
        cleaned = str(color).strip().lower()
        if cleaned not in COLOR_MODES:
            raise ValueError(
                f"Unknown color mode '{color}'. Choose: {', '.join(COLOR_MODES)}."
            )
        current["color"] = cleaned
    if output_mode is not None:
        cleaned = str(output_mode).strip().lower()
        if cleaned not in OUTPUT_MODES:
            raise ValueError(
                f"Unknown output mode '{output_mode}'. "
                f"Choose: {', '.join(OUTPUT_MODES)}."
            )
        current["output_mode"] = cleaned
    if ascii is not None:
        if not isinstance(ascii, bool):
            raise ValueError("UI ascii must be true or false.")
        current["ascii"] = ascii
    if theme is not None:
        cleaned = str(theme).strip().lower()
        if cleaned not in THEMES:
            raise ValueError(f"Unknown theme '{theme}'. Choose: {', '.join(THEMES)}.")
        current["theme"] = cleaned
    cfg["ui"] = current
    save_config(cfg)
    return current


# ---------------------------------------------------------------------------
# Per-turn model routing
# ---------------------------------------------------------------------------

MODEL_ROUTING_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "simple_model": "",
    "simple_max_chars": 200,
    "exclude_keywords": [
        "refactor",
        "implement",
        "debug",
        "explain",
        "why",
        "design",
        "architecture",
        "test",
        "migrate",
        "review",
    ],
}


def get_model_routing(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return routing settings, always fully populated and normalized."""
    cfg = cfg or load_config()
    stored = cfg.get("model_routing") or {}
    if not isinstance(stored, dict):
        stored = {}
    enabled = bool(stored.get("enabled", MODEL_ROUTING_DEFAULTS["enabled"]))
    simple_model = str(stored.get("simple_model") or "").strip()
    try:
        max_chars = int(
            stored.get("simple_max_chars", MODEL_ROUTING_DEFAULTS["simple_max_chars"])
        )
    except (TypeError, ValueError):
        max_chars = int(MODEL_ROUTING_DEFAULTS["simple_max_chars"])
    if max_chars < 1:
        max_chars = int(MODEL_ROUTING_DEFAULTS["simple_max_chars"])
    raw_keywords = stored.get(
        "exclude_keywords", MODEL_ROUTING_DEFAULTS["exclude_keywords"]
    )
    if not isinstance(raw_keywords, list):
        raw_keywords = MODEL_ROUTING_DEFAULTS["exclude_keywords"]
    keywords = list(
        dict.fromkeys(str(keyword).strip() for keyword in raw_keywords if str(keyword).strip())
    )
    return {
        "enabled": enabled,
        "simple_model": simple_model,
        "simple_max_chars": max_chars,
        "exclude_keywords": keywords,
    }


def set_model_routing(
    enabled: bool | None = None,
    simple_model: str | None = None,
    simple_max_chars: int | str | None = None,
    exclude_keywords: list[str] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update routing settings; omitted arguments keep their current value.

    ``simple_max_chars`` must be a positive integer and ``exclude_keywords`` a
    list of non-empty strings (duplicates collapse).
    """
    cfg = cfg or load_config()
    current = get_model_routing(cfg)
    if enabled is not None:
        current["enabled"] = bool(enabled)
    if simple_model is not None:
        current["simple_model"] = str(simple_model).strip()
    if simple_max_chars is not None:
        try:
            value = int(simple_max_chars)
        except (TypeError, ValueError) as exc:
            raise ValueError("simple_max_chars must be a positive integer.") from exc
        if value < 1:
            raise ValueError("simple_max_chars must be a positive integer.")
        current["simple_max_chars"] = value
    if exclude_keywords is not None:
        if not isinstance(exclude_keywords, list):
            raise ValueError("exclude_keywords must be a list of non-empty strings.")
        cleaned: list[str] = []
        for keyword in exclude_keywords:
            text = str(keyword).strip()
            if not text:
                raise ValueError("exclude_keywords entries must be non-empty.")
            if text not in cleaned:
                cleaned.append(text)
        current["exclude_keywords"] = cleaned
    cfg["model_routing"] = current
    save_config(cfg)
    return current


# ---------------------------------------------------------------------------
# Prompt caching
# ---------------------------------------------------------------------------


def get_prompt_cache(cfg: dict[str, Any] | None = None) -> bool:
    """Whether native Anthropic requests should carry prompt-cache hints."""
    cfg = cfg or load_config()
    return bool(cfg.get("prompt_cache", False))


def set_prompt_cache(enabled: bool) -> bool:
    """Enable/disable Anthropic prompt caching and return the effective value."""
    cfg = load_config()
    cfg["prompt_cache"] = bool(enabled)
    save_config(cfg)
    return bool(enabled)


# ---------------------------------------------------------------------------
# Profiles (named presets)
# ---------------------------------------------------------------------------

PROFILE_KEYS = (
    "provider",
    "model",
    "mode",
    "sampling",
    "output_style",
    "system_prompt",
    "verify_command",
    "budget",
    "trusted_workspace",
    "command_rules",
    "always_allowed",
)


def _profile_name(name: object) -> str:
    cleaned = str(name).strip()
    if not cleaned:
        raise ValueError("Profile name is required.")
    return cleaned


def _normalize_sampling(raw: object) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Profile sampling must be an object.")
    clean: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in SAMPLING_KEYS:
            raise ValueError(
                f"Unknown sampling parameter '{key}'. "
                f"Choose: {', '.join(SAMPLING_KEYS)}."
            )
        if value is None or str(value).strip() == "":
            continue
        clean[key] = _coerce_sampling(key, value)
    return clean


def _normalize_budget(raw: object) -> dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Profile budget must be an object.")
    unknown = set(raw) - {"max_tokens", "max_cost_usd"}
    if unknown:
        raise ValueError(
            "Unknown budget key(s): "
            + ", ".join(sorted(str(key) for key in unknown))
            + ". Choose: max_tokens, max_cost_usd."
        )
    budget: dict[str, float] = {}
    if raw.get("max_tokens") is not None:
        tokens = int(raw["max_tokens"])
        if tokens < 1:
            raise ValueError("max_tokens budget must be at least 1.")
        budget["max_tokens"] = tokens
    if raw.get("max_cost_usd") is not None:
        cost = float(raw["max_cost_usd"])
        if cost <= 0:
            raise ValueError("max_cost_usd budget must be positive.")
        budget["max_cost_usd"] = cost
    return budget


def _normalize_command_rules(raw: object) -> dict[str, list[str]]:
    if raw is None:
        return {"allow": [], "deny": []}
    if not isinstance(raw, dict):
        raise ValueError("Profile command_rules must be an object.")
    unknown = set(raw) - {"allow", "deny"}
    if unknown:
        raise ValueError(
            "Unknown command_rules key(s): "
            + ", ".join(sorted(str(key) for key in unknown))
            + ". Choose: allow, deny."
        )
    rules: dict[str, list[str]] = {}
    for kind in ("allow", "deny"):
        values = raw.get(kind) or []
        if not isinstance(values, list):
            raise ValueError(f"Profile command_rules.{kind} must be a list.")
        rules[kind] = list(
            dict.fromkeys(_validate_command_pattern(str(pattern)) for pattern in values)
        )
    return rules


def _normalize_always_allowed(raw: object) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("Profile always_allowed must be a list of tool names.")
    names: list[str] = []
    for name in raw:
        text = str(name).strip()
        if not text:
            raise ValueError("Profile always_allowed entries must be non-empty.")
        if text not in names:
            names.append(text)
    return names


def _normalize_profile(
    name: object, values: object, cfg: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate and normalize one profile's values, raising ``ValueError``."""
    profile_name = _profile_name(name)
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError(f"Profile '{profile_name}' must be an object.")
    unknown = set(values) - set(PROFILE_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown profile key(s): {', '.join(sorted(str(k) for k in unknown))}. "
            f"Choose: {', '.join(PROFILE_KEYS)}."
        )

    normalized: dict[str, Any] = {}
    if "provider" in values:
        provider_id = str(values["provider"]).strip()
        try:
            get_provider_config(provider_id, cfg)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        normalized["provider"] = provider_id
    if "model" in values:
        normalized["model"] = str(values["model"] or "").strip() or None
    if "mode" in values:
        normalized["mode"] = PermissionMode.from_str(str(values["mode"])).value
    if "sampling" in values:
        normalized["sampling"] = _normalize_sampling(values["sampling"])
    if "output_style" in values:
        style = str(values["output_style"] or "").strip().lower()
        if style not in OUTPUT_STYLES:
            raise ValueError(
                f"Unknown output style '{values['output_style']}'. "
                f"Choose: {', '.join(OUTPUT_STYLES)}."
            )
        normalized["output_style"] = style
    if "system_prompt" in values:
        normalized["system_prompt"] = str(values["system_prompt"] or "").strip() or None
    if "verify_command" in values:
        normalized["verify_command"] = str(values["verify_command"] or "").strip()
    if "budget" in values:
        normalized["budget"] = _normalize_budget(values["budget"])
    if "trusted_workspace" in values:
        normalized["trusted_workspace"] = bool(values["trusted_workspace"])
    if "command_rules" in values:
        normalized["command_rules"] = _normalize_command_rules(values["command_rules"])
    if "always_allowed" in values:
        normalized["always_allowed"] = _normalize_always_allowed(values["always_allowed"])
    return normalized


def _capture_profile_values(cfg: dict[str, Any]) -> dict[str, Any]:
    """Snapshot the current effective global settings as profile values."""
    captured: dict[str, Any] = {
        "provider": get_selected_provider_id(cfg),
        "mode": get_default_mode(cfg),
    }
    model = cfg.get("selected_model")
    if model:
        captured["model"] = str(model)
    sampling = get_sampling(cfg)
    if sampling:
        captured["sampling"] = sampling
    style = get_output_style(cfg)
    if style != "default":
        captured["output_style"] = style
    prompt = get_system_prompt(cfg)
    if prompt:
        captured["system_prompt"] = prompt
    verify = get_verify_command(cfg)
    if verify:
        captured["verify_command"] = verify
    budget = get_budget(cfg)
    if budget:
        captured["budget"] = budget
    if get_trusted_workspace(cfg):
        captured["trusted_workspace"] = True
    rules = get_command_rules(cfg)
    if rules["allow"] or rules["deny"]:
        captured["command_rules"] = rules
    allowed = get_always_allowed_tools(cfg)
    if allowed:
        captured["always_allowed"] = allowed
    return captured


def get_profiles(cfg: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Return saved profiles, dropping entries that no longer validate."""
    cfg = cfg or load_config()
    stored = cfg.get("profiles") or {}
    if not isinstance(stored, dict):
        return {}
    profiles: dict[str, dict[str, Any]] = {}
    for name, values in stored.items():
        try:
            profiles[_profile_name(name)] = _normalize_profile(name, values, cfg)
        except (ValueError, KeyError, TypeError):
            continue
    return profiles


def get_profile(name: str, cfg: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Return one validated profile, or None when it does not exist/is invalid."""
    cleaned = str(name).strip()
    if not cleaned:
        return None
    return get_profiles(cfg).get(cleaned)


def save_profile(
    name: str, values: dict[str, Any] | None = None, cfg: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Save a named profile, capturing current settings when ``values`` is None."""
    cfg = cfg or load_config()
    if values is None:
        values = _capture_profile_values(cfg)
    profile = _normalize_profile(name, values, cfg)
    stored = dict(cfg.get("profiles") or {})
    stored[_profile_name(name)] = profile
    cfg["profiles"] = stored
    save_config(cfg)
    return profile


def _apply_profile_values(cfg: dict[str, Any], profile: dict[str, Any]) -> None:
    if "provider" in profile:
        provider_id = str(profile["provider"])
        cfg["selected_provider"] = provider_id
        # A profile pins a single primary provider; clear per-provider state
        # that would otherwise point at the previous vendor.
        cfg["active_providers"] = [provider_id]
    if "model" in profile:
        cfg["selected_model"] = profile["model"]
    if "mode" in profile:
        cfg["default_mode"] = profile["mode"]
    if "sampling" in profile:
        cfg["sampling"] = dict(profile["sampling"])
    if "output_style" in profile:
        cfg["output_style"] = profile["output_style"]
    if "system_prompt" in profile:
        cfg["system_prompt"] = profile["system_prompt"]
    if "verify_command" in profile:
        cfg["verify_command"] = profile["verify_command"]
    if "budget" in profile:
        cfg["budget"] = dict(profile["budget"])
    if "trusted_workspace" in profile:
        cfg["trusted_workspace"] = profile["trusted_workspace"]
    if "command_rules" in profile:
        cfg["command_rules"] = {
            "allow": list(profile["command_rules"]["allow"]),
            "deny": list(profile["command_rules"]["deny"]),
        }
    if "always_allowed" in profile:
        perms = dict(cfg.get("tool_permissions") or {})
        perms["always_allow"] = list(profile["always_allowed"])
        cfg["tool_permissions"] = perms


def apply_profile(name: str, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write a profile's values into the global config and persist them.

    Raises ``ValueError`` when the profile does not exist or fails validation.
    """
    cfg = cfg or load_config()
    profile = get_profile(name, cfg)
    if profile is None:
        raise ValueError(f"Unknown profile '{name}'.")
    _apply_profile_values(cfg, profile)
    save_config(cfg)
    return profile


def remove_profile(name: str) -> bool:
    """Remove a saved profile. Returns whether it existed."""
    cleaned = str(name).strip()
    if not cleaned:
        return False
    cfg = load_config()
    stored = dict(cfg.get("profiles") or {})
    if cleaned not in stored:
        return False
    del stored[cleaned]
    cfg["profiles"] = stored
    save_config(cfg)
    return True


def rename_profile(old: str, new: str) -> bool:
    """Rename a saved profile. Returns whether the old name existed."""
    cfg = load_config()
    stored = dict(cfg.get("profiles") or {})
    old_clean = str(old).strip()
    if old_clean not in stored:
        return False
    new_clean = _profile_name(new)
    stored[new_clean] = _normalize_profile(new_clean, stored.pop(old_clean), cfg)
    cfg["profiles"] = stored
    save_config(cfg)
    return True


# ---------------------------------------------------------------------------
# Config schema validation
# ---------------------------------------------------------------------------


def validate_config(cfg: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Validate a config and return issues as level/key/message dicts.

    The getters elsewhere in this module tolerate bad stored data silently
    (dropping it); validation surfaces exactly what would be dropped. Unknown
    top-level keys are warnings so configs written by newer releases still
    pass; every other structural problem is an error.
    """
    cfg = cfg if cfg is not None else load_config()
    issues: list[dict[str, str]] = []

    def add(level: str, key: str, message: str) -> None:
        issues.append({"level": level, "key": key, "message": message})

    if not isinstance(cfg, dict):
        add("error", "", "Configuration must be a JSON object.")
        return issues

    for key in cfg:
        if key not in _KNOWN_CONFIG_KEYS:
            add("warning", str(key), f"Unknown top-level key '{key}' is ignored.")

    keys = cfg.get("keys")
    if not isinstance(keys, dict):
        add("error", "keys", "'keys' must be an object mapping provider -> key.")
    else:
        for provider_id, value in keys.items():
            if not isinstance(value, str):
                add("error", f"keys.{provider_id}", "API key must be a string.")

    providers = cfg.get("providers")
    if not isinstance(providers, dict):
        add("error", "providers", "'providers' must be an object.")
    else:
        for provider_id, data in providers.items():
            if _provider_from_config(str(provider_id), data) is None:
                add(
                    "error",
                    f"providers.{provider_id}",
                    "Provider needs non-empty name, base_url, and default_model.",
                )

    filters = cfg.get("model_filters")
    if not isinstance(filters, dict):
        add("error", "model_filters", "'model_filters' must be an object.")
    else:
        for provider_id, data in filters.items():
            if not isinstance(data, dict):
                add("error", f"model_filters.{provider_id}", "Filter must be an object.")
                continue
            mode = data.get("mode") or "all"
            if mode not in {"all", "allow", "deny"}:
                add(
                    "error",
                    f"model_filters.{provider_id}.mode",
                    "Mode must be all, allow, or deny.",
                )
            models = data.get("models") or []
            if not isinstance(models, list):
                add(
                    "error",
                    f"model_filters.{provider_id}.models",
                    "'models' must be a list.",
                )
            elif mode in {"allow", "deny"} and not [
                model for model in models if str(model).strip()
            ]:
                add(
                    "error",
                    f"model_filters.{provider_id}.models",
                    f"Mode '{mode}' requires at least one model.",
                )

    sampling = cfg.get("sampling")
    if not isinstance(sampling, dict):
        add("error", "sampling", "'sampling' must be an object.")
    else:
        for key, value in sampling.items():
            if key not in SAMPLING_KEYS:
                add("error", f"sampling.{key}", "Unknown sampling parameter.")
                continue
            if value is None or str(value).strip() == "":
                continue
            try:
                _coerce_sampling(key, value)
            except (TypeError, ValueError) as exc:
                add("error", f"sampling.{key}", str(exc))

    if "budget" in cfg:
        try:
            _normalize_budget(cfg.get("budget"))
        except (TypeError, ValueError) as exc:
            add("error", "budget", str(exc))

    hooks = cfg.get("hooks")
    if not isinstance(hooks, dict):
        add("error", "hooks", "'hooks' must be an object.")
    else:
        for event, commands in hooks.items():
            if event not in HOOK_EVENTS:
                add("error", f"hooks.{event}", "Unknown hook event.")
                continue
            if not isinstance(commands, list):
                add("error", f"hooks.{event}", "Hook commands must be a list of strings.")
                continue
            for index, command in enumerate(commands):
                if not str(command).strip():
                    add(
                        "error",
                        f"hooks.{event}[{index}]",
                        "Hook command must be non-empty.",
                    )

    rules = cfg.get("command_rules")
    if rules is not None:
        if not isinstance(rules, dict):
            add("error", "command_rules", "'command_rules' must be an object.")
        else:
            for kind, patterns in rules.items():
                if kind not in {"allow", "deny"}:
                    add(
                        "warning",
                        f"command_rules.{kind}",
                        "Unknown rule kind; expected allow or deny.",
                    )
                    continue
                if not isinstance(patterns, list):
                    add("error", f"command_rules.{kind}", "Patterns must be a list.")
                    continue
                for index, pattern in enumerate(patterns):
                    try:
                        _validate_command_pattern(str(pattern))
                    except ValueError as exc:
                        add("error", f"command_rules.{kind}[{index}]", str(exc))

    routing = cfg.get("model_routing")
    if routing is not None:
        if not isinstance(routing, dict):
            add("error", "model_routing", "'model_routing' must be an object.")
        else:
            if not isinstance(routing.get("enabled", False), bool):
                add("error", "model_routing.enabled", "'enabled' must be a boolean.")
            if "simple_model" in routing and not isinstance(
                routing["simple_model"], str
            ):
                add(
                    "error",
                    "model_routing.simple_model",
                    "'simple_model' must be a string.",
                )
            if "simple_max_chars" in routing:
                try:
                    if int(routing["simple_max_chars"]) < 1:
                        raise ValueError
                except (TypeError, ValueError):
                    add(
                        "error",
                        "model_routing.simple_max_chars",
                        "Must be a positive integer.",
                    )
            keywords = routing.get("exclude_keywords")
            if keywords is not None:
                if not isinstance(keywords, list):
                    add(
                        "error",
                        "model_routing.exclude_keywords",
                        "Must be a list of non-empty strings.",
                    )
                else:
                    for index, keyword in enumerate(keywords):
                        if not str(keyword).strip():
                            add(
                                "error",
                                f"model_routing.exclude_keywords[{index}]",
                                "Keyword must be non-empty.",
                            )

    if "prompt_cache" in cfg and not isinstance(cfg["prompt_cache"], bool):
        add("error", "prompt_cache", "'prompt_cache' must be true or false.")

    subagents = cfg.get("subagents")
    if subagents is not None:
        if not isinstance(subagents, dict):
            add("error", "subagents", "'subagents' must be an object.")
        else:
            if "enabled" in subagents and not isinstance(subagents["enabled"], bool):
                add("error", "subagents.enabled", "'enabled' must be true or false.")
            if "max_steps" in subagents:
                try:
                    steps = int(subagents["max_steps"])
                except (TypeError, ValueError):
                    steps = -1
                if not SUBAGENT_MAX_STEPS_MIN <= steps <= SUBAGENT_MAX_STEPS_MAX:
                    add(
                        "error",
                        "subagents.max_steps",
                        f"Must be between {SUBAGENT_MAX_STEPS_MIN} and "
                        f"{SUBAGENT_MAX_STEPS_MAX}.",
                    )
            if "model" in subagents and not isinstance(subagents["model"], str):
                add("error", "subagents.model", "'model' must be a string.")

    browser = cfg.get("browser")
    if browser is not None:
        if not isinstance(browser, dict):
            add("error", "browser", "'browser' must be an object.")
        else:
            if "enabled" in browser and not isinstance(browser["enabled"], bool):
                add("error", "browser.enabled", "'enabled' must be true or false.")
            if "headless" in browser and not isinstance(browser["headless"], bool):
                add("error", "browser.headless", "'headless' must be true or false.")
            if "timeout_ms" in browser:
                try:
                    timeout_ms = int(browser["timeout_ms"])
                except (TypeError, ValueError):
                    timeout_ms = -1
                if not BROWSER_TIMEOUT_MIN <= timeout_ms <= BROWSER_TIMEOUT_MAX:
                    add(
                        "error",
                        "browser.timeout_ms",
                        "Must be between "
                        f"{BROWSER_TIMEOUT_MIN} and {BROWSER_TIMEOUT_MAX}.",
                    )

    shell_section = cfg.get("shell")
    if shell_section is not None:
        if not isinstance(shell_section, dict):
            add("error", "shell", "'shell' must be an object.")
        else:
            if "persistent" in shell_section and not isinstance(
                shell_section["persistent"], bool
            ):
                add("error", "shell.persistent", "'persistent' must be true or false.")
            if "timeout" in shell_section:
                try:
                    shell_timeout = int(shell_section["timeout"])
                except (TypeError, ValueError):
                    shell_timeout = -1
                if not SHELL_TIMEOUT_MIN <= shell_timeout <= SHELL_TIMEOUT_MAX:
                    add(
                        "error",
                        "shell.timeout",
                        "Must be between "
                        f"{SHELL_TIMEOUT_MIN} and {SHELL_TIMEOUT_MAX} seconds.",
                    )
            if "max_jobs" in shell_section:
                try:
                    shell_jobs = int(shell_section["max_jobs"])
                except (TypeError, ValueError):
                    shell_jobs = -1
                if not SHELL_MAX_JOBS_MIN <= shell_jobs <= SHELL_MAX_JOBS_MAX:
                    add(
                        "error",
                        "shell.max_jobs",
                        "Must be between "
                        f"{SHELL_MAX_JOBS_MIN} and {SHELL_MAX_JOBS_MAX}.",
                    )

    sandbox_section = cfg.get("sandbox")
    if sandbox_section is not None:
        if not isinstance(sandbox_section, dict):
            add("error", "sandbox", "'sandbox' must be an object.")
        else:
            if "enabled" in sandbox_section and not isinstance(
                sandbox_section["enabled"], bool
            ):
                add("error", "sandbox.enabled", "'enabled' must be true or false.")
            if "network" in sandbox_section and not isinstance(
                sandbox_section["network"], bool
            ):
                add("error", "sandbox.network", "'network' must be true or false.")
            if "extra_writable" in sandbox_section:
                writable = sandbox_section["extra_writable"]
                if not isinstance(writable, list):
                    add(
                        "error",
                        "sandbox.extra_writable",
                        "'extra_writable' must be a list of paths.",
                    )
                else:
                    for index, path in enumerate(writable):
                        if not isinstance(path, str) or not path.strip():
                            add(
                                "error",
                                f"sandbox.extra_writable[{index}]",
                                "Path must be a non-empty string.",
                            )

    remote_section = cfg.get("remote")
    if remote_section is not None:
        if not isinstance(remote_section, dict):
            add("error", "remote", "'remote' must be an object.")
        else:
            if "enabled" in remote_section and not isinstance(
                remote_section["enabled"], bool
            ):
                add("error", "remote.enabled", "'enabled' must be true or false.")
            for field in ("host", "user", "workspace"):
                if field in remote_section and not isinstance(
                    remote_section[field], str
                ):
                    add("error", f"remote.{field}", f"'{field}' must be a string.")
            if "port" in remote_section:
                try:
                    remote_port = int(remote_section["port"])
                except (TypeError, ValueError):
                    remote_port = -1
                if not REMOTE_PORT_MIN <= remote_port <= REMOTE_PORT_MAX:
                    add(
                        "error",
                        "remote.port",
                        "Must be between "
                        f"{REMOTE_PORT_MIN} and {REMOTE_PORT_MAX}.",
                    )
            if "identity" in remote_section:
                identity = remote_section["identity"]
                if not isinstance(identity, str):
                    add("error", "remote.identity", "'identity' must be a string.")
                elif identity.strip() and not Path(identity).expanduser().is_file():
                    add(
                        "error",
                        "remote.identity",
                        f"Identity file does not exist: {identity}",
                    )
            if "devcontainer" in remote_section:
                devcontainer = remote_section["devcontainer"]
                if not isinstance(devcontainer, str) or not devcontainer.strip():
                    add(
                        "error",
                        "remote.devcontainer",
                        "'devcontainer' must be 'auto', 'off', or a container name.",
                    )
            if (
                remote_section.get("enabled") is True
                and not str(remote_section.get("host") or "").strip()
                and str(remote_section.get("devcontainer") or "auto").strip()
                in REMOTE_SPECIAL_DEVCONTAINERS
            ):
                add(
                    "error",
                    "remote.host",
                    "A host is required when remote execution is enabled.",
                )

    acp_section = cfg.get("acp")
    if acp_section is not None:
        if not isinstance(acp_section, dict):
            add("error", "acp", "'acp' must be an object.")
        else:
            if "permission_timeout" in acp_section:
                try:
                    acp_timeout = int(acp_section["permission_timeout"])
                except (TypeError, ValueError):
                    acp_timeout = -1
                if not ACP_TIMEOUT_MIN <= acp_timeout <= ACP_TIMEOUT_MAX:
                    add(
                        "error",
                        "acp.permission_timeout",
                        "Must be between "
                        f"{ACP_TIMEOUT_MIN} and {ACP_TIMEOUT_MAX} seconds.",
                    )

    profiles = cfg.get("profiles")
    if not isinstance(profiles, dict):
        add("error", "profiles", "'profiles' must be an object.")
    else:
        for name, values in profiles.items():
            try:
                _normalize_profile(name, values, cfg)
            except (ValueError, KeyError, TypeError) as exc:
                add("error", f"profiles.{name}", str(exc))

    servers = cfg.get("mcp_servers")
    if not isinstance(servers, dict):
        add("error", "mcp_servers", "'mcp_servers' must be an object.")
    else:
        for name, spec in servers.items():
            try:
                _normalize_mcp_spec(name, spec)
            except ValueError as exc:
                add("error", f"mcp_servers.{name}", str(exc))

    plugins = cfg.get("plugins")
    if not isinstance(plugins, dict):
        add("error", "plugins", "'plugins' must be an object.")
    else:
        if not isinstance(plugins.get("allow_project", False), bool):
            add("error", "plugins.allow_project", "'allow_project' must be a boolean.")
        disabled = plugins.get("disabled")
        if disabled is not None:
            if not isinstance(disabled, list):
                add("error", "plugins.disabled", "'disabled' must be a list.")
            else:
                for index, name in enumerate(disabled):
                    if not str(name).strip():
                        add(
                            "error",
                            f"plugins.disabled[{index}]",
                            "Plugin name must be non-empty.",
                        )

    default_mode = cfg.get("default_mode")
    if default_mode is not None:
        try:
            PermissionMode.from_str(str(default_mode))
        except ValueError:
            add("error", "default_mode", f"Unknown permission mode '{default_mode}'.")

    if not isinstance(cfg.get("trusted_workspace", False), bool):
        add("error", "trusted_workspace", "'trusted_workspace' must be true or false.")

    style = cfg.get("output_style")
    if style is not None and str(style).strip().lower() not in OUTPUT_STYLES:
        add("error", "output_style", f"Unknown output style '{style}'.")

    ui = cfg.get("ui")
    if not isinstance(ui, dict):
        add("error", "ui", "'ui' must be an object.")
    else:
        from kiwimatecoder.ui import COLOR_MODES, OUTPUT_MODES, THEMES

        color = ui.get("color")
        if color is not None and str(color).strip().lower() not in COLOR_MODES:
            add("error", "ui.color", f"Unknown color mode '{color}'.")
        output_mode = ui.get("output_mode")
        if (
            output_mode is not None
            and str(output_mode).strip().lower() not in OUTPUT_MODES
        ):
            add("error", "ui.output_mode", f"Unknown output mode '{output_mode}'.")
        if "ascii" in ui and not isinstance(ui["ascii"], bool):
            add("error", "ui.ascii", "'ascii' must be true or false.")
        theme = ui.get("theme")
        if theme is not None and str(theme).strip().lower() not in THEMES:
            add("error", "ui.theme", f"Unknown theme '{theme}'.")

    web = cfg.get("web")
    if web is not None:
        if not isinstance(web, dict):
            add("error", "web", "'web' must be an object.")
        else:
            if "max_chars" in web:
                try:
                    _validate_web_max_chars(web["max_chars"])
                except ValueError as exc:
                    add("error", "web.max_chars", str(exc))
            if "timeout" in web:
                try:
                    _validate_web_timeout(web["timeout"])
                except ValueError as exc:
                    add("error", "web.timeout", str(exc))
            if "allow_local" in web and not isinstance(web["allow_local"], bool):
                add("error", "web.allow_local", "'allow_local' must be true or false.")
            provider = web.get("search_provider")
            if (
                provider is not None
                and str(provider).strip().lower() not in WEB_SEARCH_PROVIDERS
            ):
                add(
                    "error",
                    "web.search_provider",
                    f"Unknown search provider '{provider}'.",
                )
            if "search_api_key" in web and not isinstance(web["search_api_key"], str):
                add("error", "web.search_api_key", "'search_api_key' must be a string.")

    network = cfg.get("network")
    if network is not None:
        if not isinstance(network, dict):
            add("error", "network", "'network' must be an object.")
        else:
            if "proxy" in network:
                try:
                    network_settings.validate_proxy(network["proxy"])
                except ValueError as exc:
                    add("error", "network.proxy", str(exc))
            if "ca_bundle" in network:
                try:
                    network_settings.validate_ca_bundle(network["ca_bundle"])
                except ValueError as exc:
                    add("error", "network.ca_bundle", str(exc))
            if "offline" in network and not isinstance(network["offline"], bool):
                add("error", "network.offline", "'offline' must be true or false.")

    memory = cfg.get("memory")
    if memory is not None:
        if not isinstance(memory, dict):
            add("error", "memory", "'memory' must be an object.")
        else:
            if "enabled" in memory and not isinstance(memory["enabled"], bool):
                add("error", "memory.enabled", "'enabled' must be true or false.")
            if "max_bytes" in memory:
                try:
                    value = int(memory["max_bytes"])
                except (TypeError, ValueError):
                    value = -1
                if not MEMORY_MAX_BYTES_MIN <= value <= MEMORY_MAX_BYTES_MAX:
                    add(
                        "error",
                        "memory.max_bytes",
                        "Must be between "
                        f"{MEMORY_MAX_BYTES_MIN} and {MEMORY_MAX_BYTES_MAX}.",
                    )

    vision = cfg.get("vision")
    if vision is not None:
        if not isinstance(vision, dict):
            add("error", "vision", "'vision' must be an object.")
        else:
            if "max_image_bytes" in vision:
                try:
                    image_bytes = int(vision["max_image_bytes"])
                except (TypeError, ValueError):
                    image_bytes = -1
                if not (
                    VISION_MAX_IMAGE_BYTES_MIN
                    <= image_bytes
                    <= VISION_MAX_IMAGE_BYTES_MAX
                ):
                    add(
                        "error",
                        "vision.max_image_bytes",
                        "Must be between "
                        f"{VISION_MAX_IMAGE_BYTES_MIN} and "
                        f"{VISION_MAX_IMAGE_BYTES_MAX}.",
                    )
            if "max_images_per_turn" in vision:
                try:
                    image_count = int(vision["max_images_per_turn"])
                except (TypeError, ValueError):
                    image_count = -1
                if not VISION_MAX_IMAGES_MIN <= image_count <= VISION_MAX_IMAGES_MAX:
                    add(
                        "error",
                        "vision.max_images_per_turn",
                        "Must be between "
                        f"{VISION_MAX_IMAGES_MIN} and {VISION_MAX_IMAGES_MAX}.",
                    )

    lsp = cfg.get("lsp")
    if lsp is not None:
        if not isinstance(lsp, dict):
            add("error", "lsp", "'lsp' must be an object.")
        else:
            if "enabled" in lsp and not isinstance(lsp["enabled"], bool):
                add("error", "lsp.enabled", "'enabled' must be true or false.")
            if "diagnostics_after_edits" in lsp and not isinstance(
                lsp["diagnostics_after_edits"], bool
            ):
                add(
                    "error",
                    "lsp.diagnostics_after_edits",
                    "'diagnostics_after_edits' must be true or false.",
                )
            if "timeout" in lsp:
                try:
                    timeout = float(lsp["timeout"])
                except (TypeError, ValueError):
                    timeout = -1.0
                if not LSP_TIMEOUT_MIN <= timeout <= LSP_TIMEOUT_MAX:
                    add(
                        "error",
                        "lsp.timeout",
                        "Must be between "
                        f"{LSP_TIMEOUT_MIN:g} and {LSP_TIMEOUT_MAX:g} seconds.",
                    )
            servers = lsp.get("servers")
            if servers is not None:
                if not isinstance(servers, dict):
                    add("error", "lsp.servers", "'servers' must be an object.")
                else:
                    for name, spec in servers.items():
                        try:
                            _normalize_lsp_server(name, spec)
                        except ValueError as exc:
                            add("error", f"lsp.servers.{name}", str(exc))

    index_section = cfg.get("index")
    if index_section is not None:
        if not isinstance(index_section, dict):
            add("error", "index", "'index' must be an object.")
        else:
            if "enabled" in index_section and not isinstance(
                index_section["enabled"], bool
            ):
                add("error", "index.enabled", "'enabled' must be true or false.")
            if "max_files" in index_section:
                try:
                    max_files = int(index_section["max_files"])
                except (TypeError, ValueError):
                    max_files = -1
                if not INDEX_MAX_FILES_MIN <= max_files <= INDEX_MAX_FILES_MAX:
                    add(
                        "error",
                        "index.max_files",
                        "Must be between "
                        f"{INDEX_MAX_FILES_MIN} and {INDEX_MAX_FILES_MAX}.",
                    )
            if "max_file_bytes" in index_section:
                try:
                    max_bytes = int(index_section["max_file_bytes"])
                except (TypeError, ValueError):
                    max_bytes = -1
                if not (
                    INDEX_MAX_FILE_BYTES_MIN
                    <= max_bytes
                    <= INDEX_MAX_FILE_BYTES_MAX
                ):
                    add(
                        "error",
                        "index.max_file_bytes",
                        "Must be between "
                        f"{INDEX_MAX_FILE_BYTES_MIN} and "
                        f"{INDEX_MAX_FILE_BYTES_MAX}.",
                    )
            embeddings = index_section.get("embeddings")
            if embeddings is not None:
                if not isinstance(embeddings, dict):
                    add(
                        "error",
                        "index.embeddings",
                        "'embeddings' must be an object.",
                    )
                else:
                    provider = embeddings.get("provider")
                    if provider is not None and str(provider).strip():
                        try:
                            get_provider_config(str(provider).strip(), cfg)
                        except KeyError:
                            add(
                                "error",
                                "index.embeddings.provider",
                                f"Unknown provider '{provider}'.",
                            )
                    if "model" in embeddings and not isinstance(
                        embeddings["model"], str
                    ):
                        add(
                            "error",
                            "index.embeddings.model",
                            "'model' must be a string.",
                        )
                    if "batch_size" in embeddings:
                        try:
                            batch = int(embeddings["batch_size"])
                        except (TypeError, ValueError):
                            batch = -1
                        if not INDEX_BATCH_SIZE_MIN <= batch <= INDEX_BATCH_SIZE_MAX:
                            add(
                                "error",
                                "index.embeddings.batch_size",
                                "Must be between "
                                f"{INDEX_BATCH_SIZE_MIN} and "
                                f"{INDEX_BATCH_SIZE_MAX}.",
                            )

    prompt = cfg.get("system_prompt")
    if prompt is not None and not isinstance(prompt, str):
        add("warning", "system_prompt", "'system_prompt' should be a string or null.")

    verify = cfg.get("verify_command")
    if verify is not None and not isinstance(verify, str):
        add("warning", "verify_command", "'verify_command' should be a string.")

    active = cfg.get("active_providers")
    if active is not None:
        if not isinstance(active, list):
            add("error", "active_providers", "'active_providers' must be a list.")
        else:
            for index, provider_id in enumerate(active):
                try:
                    get_provider_config(str(provider_id), cfg)
                except KeyError:
                    add(
                        "error",
                        f"active_providers[{index}]",
                        f"Unknown provider '{provider_id}'.",
                    )

    selected = cfg.get("selected_provider")
    if selected is not None:
        try:
            get_provider_config(str(selected), cfg)
        except KeyError:
            add("warning", "selected_provider", f"Unknown provider '{selected}'.")

    return issues


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
