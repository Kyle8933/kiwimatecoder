"""Live model catalogs fetched from provider APIs.

Model ids drift fast: providers ship new ones every few weeks and retire old
ones without warning. The curated tuples in :mod:`kiwimatecoder.providers` are
only a shipped-at-build-time starting point, so this module asks each provider
what it actually serves today via ``GET {base_url}/models`` and turns the answer
into the list ``/model`` offers.

Two rules shape the result:

* **New models are added.** Anything the provider lists is offered, ordered
  newest-first so freshly released ids surface at the top.
* **Deprecated models are removed.** Only ids present in the live listing
  survive, so a retired model disappears from ``/model`` on the next refresh
  even if it is still hard-coded in the curated catalog.

Listings are filtered to models this CLI can actually drive: text chat models
that support tool calling. Embedding, audio, image, and moderation endpoints are
dropped by name, and provider metadata (OpenRouter's ``supported_parameters`` /
``architecture``, Mistral's ``capabilities``) is honored when present.

Nothing here touches disk; :mod:`kiwimatecoder.config` owns the cache that keeps
``/model`` from hitting the network on every invocation.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import httpx

from kiwimatecoder.providers import ProviderConfig

# A catalog older than this is refetched the next time the models are shown.
CATALOG_TTL_SECONDS = 24 * 60 * 60
# After a failed fetch, don't retry automatically for this long (an explicit
# `/model refresh` always retries). Keeps `/model` snappy when offline.
FAILURE_BACKOFF_SECONDS = 30 * 60
# Model listings are small; a short timeout keeps a hung provider from blocking
# the selector.
FETCH_TIMEOUT = 8.0
# Gateways such as OpenRouter serve hundreds of models. Offer the newest slice
# in the selector — any id can still be set by name with `/model <id>`.
MAX_CATALOG_MODELS = 60
ANTHROPIC_VERSION = "2023-06-01"

# Substrings that mark an id as something other than a text chat model. Matched
# case-insensitively against the model id.
NON_CHAT_MARKERS = (
    "embed",
    "rerank",
    "moderation",
    "guard",
    "whisper",
    "audio",
    "speech",
    "transcribe",
    "voice",
    "tts",
    "dall-e",
    "image",
    "imagen",
    "video",
    "veo",
    "sora",
    "diffusion",
    "flux",
    "clip",
    "ocr",
)


class CatalogFetchError(RuntimeError):
    """Raised when a provider's model listing cannot be retrieved or parsed."""


@dataclass(frozen=True)
class RemoteModel:
    """One model as reported by a provider's listing endpoint."""

    id: str
    created: float = 0.0  # epoch seconds; 0 when the provider reports no date


@dataclass(frozen=True)
class ModelCatalog:
    """The model list offered for a provider, plus how it was obtained.

    ``source`` is ``"live"`` (just fetched), ``"cache"`` (a previous fetch), or
    ``"curated"`` (the tuple shipped in the registry). ``added`` and ``removed``
    are only populated for a live fetch and describe the change against the
    previously known catalog — ``removed`` is the list of deprecated ids that
    just dropped out of ``/model``.
    """

    provider_id: str
    models: list[str]
    source: str
    fetched_at: float | None = None
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    error: str | None = None


def models_url(provider: ProviderConfig) -> str:
    """Return the listing endpoint for ``provider``."""
    return f"{provider.base_url.rstrip('/')}/models"


def request_headers(provider: ProviderConfig, api_key: str | None) -> dict[str, str]:
    """Build auth headers for a listing request.

    Anthropic's native API wants ``x-api-key`` plus a version header; every
    OpenAI-compatible provider takes a bearer token. Provider ``extra_headers``
    are applied last so a custom provider can override anything.
    """
    headers = {"Accept": "application/json"}
    if provider.compat == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
        if api_key:
            headers["x-api-key"] = api_key
    elif api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers.update(provider.extra_headers)
    return headers


def request_params(provider: ProviderConfig) -> dict[str, str | int]:
    """Return query params for the listing request."""
    if provider.compat == "anthropic":
        # Anthropic paginates at 20 by default; 1000 is the documented maximum
        # and comfortably covers the whole catalog in one request.
        return {"limit": 1000}
    return {}


def _parse_timestamp(value: object) -> float:
    """Coerce a provider timestamp (epoch seconds or ISO 8601) to a float."""
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def normalize_model_id(raw: object) -> str:
    """Normalize a listed id to the form the chat endpoint expects.

    Google's OpenAI-compatible listing returns ``models/gemini-...`` while its
    chat endpoint expects the bare id, so the prefix is stripped.
    """
    model_id = str(raw or "").strip()
    if model_id.startswith("models/"):
        model_id = model_id[len("models/") :]
    return model_id


def is_chat_model(entry: dict, model_id: str) -> bool:
    """Return whether a listed model is a text chat model this CLI can drive.

    Provider metadata wins when present; otherwise the id is matched against
    :data:`NON_CHAT_MARKERS`. Tool calling is required — the agent loop is built
    on it — so a provider that reports ``supported_parameters`` without
    ``tools`` (OpenRouter does this for every entry) is filtered out.
    """
    lowered = model_id.lower()
    if not lowered or any(marker in lowered for marker in NON_CHAT_MARKERS):
        return False

    capabilities = entry.get("capabilities")
    if isinstance(capabilities, dict):
        chat = capabilities.get("completion_chat")
        if chat is False:
            return False
        if capabilities.get("function_calling") is False:
            return False

    architecture = entry.get("architecture")
    if isinstance(architecture, dict):
        outputs = architecture.get("output_modalities")
        if isinstance(outputs, list) and outputs and "text" not in outputs:
            return False
        inputs = architecture.get("input_modalities")
        if isinstance(inputs, list) and inputs and "text" not in inputs:
            return False

    supported = entry.get("supported_parameters")
    if isinstance(supported, list) and supported and "tools" not in supported:
        return False

    return True


def parse_models_response(payload: object) -> list[RemoteModel]:
    """Extract usable chat models from a provider's listing payload.

    Accepts the OpenAI shape (``{"data": [...]}``), a bare list, or a
    ``{"models": [...]}`` variant, and tolerates entries that are plain strings.
    Unusable entries are skipped rather than raising, so one odd record cannot
    break the whole catalog.
    """
    if isinstance(payload, dict):
        entries = payload.get("data")
        if not isinstance(entries, list):
            entries = payload.get("models")
    else:
        entries = payload
    if not isinstance(entries, list):
        return []

    models: dict[str, RemoteModel] = {}
    for raw_entry in entries:
        if isinstance(raw_entry, str):
            entry: dict = {"id": raw_entry}
        elif isinstance(raw_entry, dict):
            entry = raw_entry
        else:
            continue

        model_id = normalize_model_id(entry.get("id") or entry.get("name"))
        if not model_id or model_id in models:
            continue
        if not is_chat_model(entry, model_id):
            continue

        created = _parse_timestamp(
            entry.get("created")
            if entry.get("created") is not None
            else entry.get("created_at")
        )
        models[model_id] = RemoteModel(id=model_id, created=created)

    return list(models.values())


def fetch_models(
    provider: ProviderConfig,
    api_key: str | None = None,
    *,
    timeout: float = FETCH_TIMEOUT,
    transport: httpx.BaseTransport | None = None,
) -> list[RemoteModel]:
    """Fetch and parse ``provider``'s live model listing.

    Raises :class:`CatalogFetchError` for any transport, status, payload, or
    "nothing usable came back" failure so callers have a single thing to catch.
    ``transport`` is an injection point for tests.
    """
    url = models_url(provider)
    try:
        with httpx.Client(
            timeout=timeout, transport=transport, follow_redirects=True
        ) as client:
            response = client.get(
                url,
                headers=request_headers(provider, api_key),
                params=request_params(provider),
            )
    except httpx.HTTPError as exc:
        raise CatalogFetchError(f"{provider.name}: {exc}") from exc

    if response.status_code != 200:
        body = response.text[:200].replace("\n", " ").strip()
        raise CatalogFetchError(
            f"{provider.name} returned HTTP {response.status_code} for {url}"
            + (f": {body}" if body else "")
        )

    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise CatalogFetchError(
            f"{provider.name} returned a non-JSON model list from {url}"
        ) from exc

    models = parse_models_response(payload)
    if not models:
        raise CatalogFetchError(f"{provider.name} listed no usable chat models")
    return models


def summarize_ids(model_ids: Sequence[str], limit: int = 8) -> str:
    """Join model ids for display, trimming long lists to ``limit`` entries.

    A first refresh can add dozens of models at once; printing all of them
    buries the part the user cares about.
    """
    shown = ", ".join(model_ids[:limit])
    extra = len(model_ids) - limit
    return f"{shown}, +{extra} more" if extra > 0 else shown


def merge_catalog(
    provider: ProviderConfig,
    remote: Iterable[RemoteModel],
    *,
    keep: Sequence[str] = (),
    limit: int | None = MAX_CATALOG_MODELS,
) -> list[str]:
    """Turn a live listing into the catalog ``/model`` should offer.

    The provider default comes first (it is the recommended pick), then any
    caller-supplied ``keep`` ids such as the session's current model, then the
    rest newest-first. Only ids the provider still lists survive — that is how
    deprecated models leave the catalog — so ``keep`` can pin a model but never
    resurrect a retired one.
    """
    ordered = [model.id for model in sorted(remote, key=lambda m: -m.created)]
    live = set(ordered)

    pinned = list(
        dict.fromkeys(
            model
            for model in (provider.default_model, *keep)
            if model and model in live
        )
    )
    rest = [model for model in ordered if model not in set(pinned)]
    catalog = pinned + rest
    if limit is not None and limit > 0:
        catalog = catalog[:limit]
    return catalog
