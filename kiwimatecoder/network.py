"""Proxy, custom CA bundle, and offline-mode settings.

The ``network`` config section is the single place users configure how
outbound HTTP is routed:

* ``proxy`` routes chat, catalog, web, and embedding requests through an
  HTTP(S) proxy.
* ``ca_bundle`` points at a PEM file for a private or enterprise CA instead of
  the system trust store.
* ``offline`` blocks cloud requests before they are made. Local providers
  (Ollama, LM Studio, Unsloth, and custom ``localhost``/``*.local`` endpoints)
  keep working, so an air-gapped machine can still use a model server on the
  same host.

Every network entry point funnels through :func:`httpx_options` (and
:func:`offline_enabled`) so proxy, CA, and offline handling stay consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

NETWORK_DEFAULTS: dict[str, Any] = {"proxy": "", "ca_bundle": "", "offline": False}

OFFLINE_MESSAGE = (
    "offline mode is enabled (network.offline is on); turn it off with "
    "`/config network offline off` or `config network offline off`"
)


def validate_proxy(value: object) -> str:
    """Return a validated proxy URL, or "" to clear the setting."""
    cleaned = str(value or "").strip()
    if not cleaned:
        return ""
    parsed = urlsplit(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(
            "network proxy must be an http:// or https:// URL "
            "(or empty to clear it)."
        )
    return cleaned


def validate_ca_bundle(value: object) -> str:
    """Return a validated CA bundle path, or "" to clear the setting."""
    cleaned = str(value or "").strip()
    if not cleaned:
        return ""
    path = Path(cleaned).expanduser()
    if not path.is_file():
        raise ValueError(f"network ca_bundle file does not exist: {cleaned}")
    return str(path)


def normalize(stored: object) -> dict[str, Any]:
    """Return the network section as settings, dropping malformed values.

    Tolerant on purpose: a hand-edited config should never crash a request.
    :func:`kiwimatecoder.config.validate_config` reports what would be dropped.
    """
    if not isinstance(stored, dict):
        stored = {}
    try:
        proxy = validate_proxy(stored.get("proxy"))
    except ValueError:
        proxy = ""
    try:
        ca_bundle = validate_ca_bundle(stored.get("ca_bundle"))
    except ValueError:
        ca_bundle = ""
    offline = stored.get("offline")
    return {
        "proxy": proxy,
        "ca_bundle": ca_bundle,
        "offline": offline if isinstance(offline, bool) else False,
    }


def httpx_options(settings: dict[str, Any]) -> dict[str, Any]:
    """Map network settings onto ``httpx`` client keyword arguments.

    ``verify`` is ``True`` (httpx's system trust store) unless a CA bundle is
    configured, and ``proxy`` stays ``None`` when no proxy is set.
    """
    proxy = str(settings.get("proxy") or "").strip() or None
    ca_bundle = str(settings.get("ca_bundle") or "").strip()
    return {"proxy": proxy, "verify": ca_bundle or True}


def current_settings() -> dict[str, Any]:
    """Load the network section from the active config."""
    from kiwimatecoder import config

    return config.get_network()


def current_options() -> dict[str, Any]:
    """``httpx`` client keyword arguments for the active config."""
    return httpx_options(current_settings())


def offline_enabled() -> bool:
    """Whether offline mode is on for the active config."""
    return bool(current_settings()["offline"])


def offline_message(what: str) -> str:
    """A user-facing refusal message naming the blocked request."""
    return f"{OFFLINE_MESSAGE}; refusing {what}."
