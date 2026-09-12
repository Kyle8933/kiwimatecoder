"""Optional OpenAI-compatible embeddings for semantic search.

Embeddings are an enhancement, never a requirement: any failure raises
:class:`EmbeddingError`, which callers turn into a lexical-only fallback so a
bad key or an offline provider can never break a search.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import httpx

from kiwimatecoder import config
from kiwimatecoder.providers import ProviderConfig

DEFAULT_TIMEOUT = 30.0
MAX_BATCH_SIZE = 256
MAX_ERROR_DETAIL = 200


class EmbeddingError(Exception):
    """A user-facing embedding failure with a ready-to-show message."""


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors (0.0 when either is empty)."""
    size = min(len(left), len(right))
    if size == 0:
        return 0.0
    dot = sum(float(left[i]) * float(right[i]) for i in range(size))
    norm_left = math.sqrt(sum(float(left[i]) ** 2 for i in range(size)))
    norm_right = math.sqrt(sum(float(right[i]) ** 2 for i in range(size)))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


def _provider_and_key(provider_id: str) -> tuple[ProviderConfig, str | None]:
    try:
        provider = config.get_provider_config(provider_id)
    except KeyError as exc:
        raise EmbeddingError(str(exc)) from exc
    if provider.compat != "openai":
        raise EmbeddingError(
            f"Provider '{provider_id}' is not OpenAI-compatible; semantic "
            "embeddings need a provider that serves POST /embeddings."
        )
    api_key = config.get_key(provider_id)
    if provider.needs_key and not api_key:
        raise EmbeddingError(
            f"No API key for {provider.name} ({provider.key_env}) to embed with."
        )
    return provider, api_key


def _vector_batch(payload: object, expected: int) -> list[list[float]]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or len(data) != expected:
        raise EmbeddingError(
            "Embedding provider returned an unexpected number of vectors."
        )
    ordered = sorted(
        (item for item in data if isinstance(item, dict)),
        key=lambda item: int(item.get("index") or 0),
    )
    vectors: list[list[float]] = []
    for item in ordered:
        vector = item.get("embedding")
        if not isinstance(vector, list):
            raise EmbeddingError("Embedding provider returned a malformed vector.")
        try:
            vectors.append([float(value) for value in vector])
        except (TypeError, ValueError) as exc:
            raise EmbeddingError(
                "Embedding provider returned a malformed vector."
            ) from exc
    return vectors


def embed_texts(
    texts: Sequence[str],
    provider_id: str,
    model: str,
    *,
    batch_size: int | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    transport: httpx.BaseTransport | None = None,
) -> list[list[float]]:
    """Embed ``texts`` with an OpenAI-compatible provider.

    Requests are issued in bounded batches and results come back in input
    order. Raises :class:`EmbeddingError` for a missing provider/model/key, an
    unsupported provider, an HTTP error, or a malformed response.
    """
    if not texts:
        return []
    provider_id = str(provider_id or "").strip()
    model = str(model or "").strip()
    if not provider_id or not model:
        raise EmbeddingError("Embeddings need both a provider and a model.")
    provider, api_key = _provider_and_key(provider_id)

    if batch_size is None:
        batch_size = int(config.get_index()["embeddings"]["batch_size"])
    size = max(1, min(int(batch_size), MAX_BATCH_SIZE))

    headers = {"Content-Type": "application/json", **provider.extra_headers}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = f"{provider.base_url.rstrip('/')}/embeddings"

    vectors: list[list[float]] = []
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            for start in range(0, len(texts), size):
                batch = [str(text) for text in texts[start : start + size]]
                response = client.post(
                    url, json={"model": model, "input": batch}, headers=headers
                )
                if response.status_code >= 400:
                    detail = response.text.strip()
                    if len(detail) > MAX_ERROR_DETAIL:
                        detail = detail[: MAX_ERROR_DETAIL - 3] + "..."
                    message = f"Embedding request failed with HTTP {response.status_code}"
                    raise EmbeddingError(message + (f": {detail}" if detail else "."))
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise EmbeddingError(
                        "Embedding provider returned invalid JSON."
                    ) from exc
                vectors.extend(_vector_batch(payload, len(batch)))
    except EmbeddingError:
        raise
    except httpx.HTTPError as exc:
        raise EmbeddingError(
            f"Embedding request failed: {exc.__class__.__name__}."
        ) from exc
    if len(vectors) != len(texts):
        raise EmbeddingError(
            "Embedding provider returned an unexpected number of vectors."
        )
    return vectors
