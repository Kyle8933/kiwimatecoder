"""Image generation through OpenAI-compatible ``/images/generations`` APIs.

Generation is opt-in (``media.enabled``) and approval-gated when driven by the
``generate_image`` tool, because a call costs money. Requests reuse the model
provider registry for auth (``key_header``/``key_prefix``/``extra_headers``),
API versioning (``api_version``), and network transport (proxy/CA/offline) so
image endpoints behave exactly like chat endpoints.

Responses may carry either ``data[0].b64_json`` (gpt-image and DALL·E with
``response_format: b64_json``) or a ``data[0].url`` that is downloaded with a
bounded size. The decoded PNG is written atomically under the workspace at
``<output_dir>/<YYYYmmdd_HHMMSS>-<slug>.png``; callers attach the result to
``session.pending_images`` best-effort via :func:`attach_result`.
"""

from __future__ import annotations

import base64
import datetime
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from kiwimatecoder import config, images, network
from kiwimatecoder.redaction import redact
from kiwimatecoder.web import is_local_address

if TYPE_CHECKING:
    from kiwimatecoder.providers import ProviderConfig
    from kiwimatecoder.session import Session

REQUEST_TIMEOUT = 120.0
MAX_IMAGE_BYTES = 20 * 1024 * 1024
_SLUG_MAX = 40
_DISABLED_MESSAGE = (
    "Image generation is disabled. Enable it with `/config media enable on` "
    "(or `kiwimatecoder config media enable on`)."
)


class MediaError(RuntimeError):
    """Raised when an image cannot be generated; the message is user-facing."""


@dataclass(frozen=True)
class MediaResult:
    """One generated image written to disk."""

    path: Path
    revised_prompt: str | None
    model: str
    bytes: int


def _client(timeout: float) -> httpx.Client:
    """Build an httpx client honoring proxy/CA settings (tests patch this)."""
    return httpx.Client(timeout=timeout, **network.current_options())


def _headers(provider: ProviderConfig, key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if key:
        if provider.compat == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
            headers["x-api-key"] = key
        else:
            headers[provider.key_header] = f"{provider.key_prefix}{key}"
    headers.update(provider.extra_headers)
    return headers


def _payload(model: str, prompt: str, size: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "n": 1,
    }
    # DALL·E defaults to a URL response; gpt-image models always return b64.
    if "dall-e" in model.lower():
        payload["response_format"] = "b64_json"
    return payload


def _slug(prompt: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")
    cleaned = cleaned[:_SLUG_MAX].strip("-")
    return cleaned or "image"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".kiwi.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _write_image(workspace_root: Path | str, output_dir: str, prompt: str, data: bytes) -> Path:
    root = Path(workspace_root).resolve()
    target_dir = (root / output_dir).resolve()
    if target_dir != root and not target_dir.is_relative_to(root):
        raise MediaError(
            f"Media output directory '{output_dir}' is outside the workspace root."
        )
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MediaError(f"Could not create media directory '{target_dir}': {exc}") from exc
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slug(prompt)
    path = target_dir / f"{stamp}-{slug}.png"
    counter = 1
    while path.exists():
        counter += 1
        path = target_dir / f"{stamp}-{slug}-{counter}.png"
    try:
        _atomic_write_bytes(path, data)
    except OSError as exc:
        raise MediaError(f"Could not save the generated image to '{path}': {exc}") from exc
    return path


def _download_image(url: str) -> bytes:
    """Download an image URL with a bounded size and local/offline guards."""
    if config.offline_enabled():
        raise MediaError(network.offline_message("image download"))
    if is_local_address(url) and not config.get_web()["allow_local"]:
        raise MediaError(
            "Refusing to download the image from a local or private address. "
            "Enable it with `/config web allow-local on`."
        )
    try:
        with _client(REQUEST_TIMEOUT) as client:
            with client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise MediaError(
                        f"Image download returned HTTP {response.status_code}."
                    )
                data = bytearray()
                for chunk in response.iter_bytes():
                    data.extend(chunk)
                    if len(data) > MAX_IMAGE_BYTES:
                        raise MediaError(
                            "Image download exceeded the "
                            f"{MAX_IMAGE_BYTES:,}-byte limit."
                        )
    except httpx.HTTPError as exc:
        raise MediaError(
            f"Image download failed: {exc.__class__.__name__}."
        ) from exc
    return bytes(data)


def _decode_b64(payload: str) -> bytes:
    try:
        return base64.b64decode(payload, validate=False)
    except (ValueError, TypeError) as exc:
        raise MediaError("The provider returned an invalid base64 image.") from exc


def _first_image(response: httpx.Response, provider_name: str) -> dict[str, Any]:
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise MediaError(
            f"{provider_name} returned a non-JSON image response."
        ) from exc
    if not isinstance(body, dict):
        raise MediaError(f"{provider_name} returned an unexpected image response.")
    data = body.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        error = body.get("error")
        if error:
            raise MediaError(f"{provider_name} image error: {redact(str(error))[:500]}")
        raise MediaError(f"{provider_name} returned no image data.")
    return data[0]


def generate_image(
    prompt: str,
    *,
    session: Session,
    size: str | None = None,
    model: str | None = None,
    provider_id: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> MediaResult:
    """Generate one image and return where it was saved.

    Raises :class:`MediaError` with a user-facing message for every failure;
    raw httpx errors are never leaked. The caller decides whether to attach the
    result to the conversation (see :func:`attach_result`).
    """
    text = str(prompt or "").strip()
    if not text:
        raise MediaError("A prompt is required to generate an image.")

    settings = config.get_media(cfg)
    if not settings["enabled"]:
        raise MediaError(_DISABLED_MESSAGE)

    resolved_provider = str(provider_id or settings["provider"]).strip()
    try:
        provider = config.get_provider_config(resolved_provider, cfg)
    except KeyError as exc:
        raise MediaError(str(exc) or f"Unknown provider '{resolved_provider}'.") from exc

    if config.offline_enabled(cfg):
        raise MediaError(network.offline_message("image generation"))

    key = config.get_key(resolved_provider)
    if not key and provider.needs_key:
        raise MediaError(
            f"No API key for {provider.name}. Set one with "
            f"`config key set {provider.id} <KEY>` or the "
            f"{provider.key_env} environment variable."
        )

    chosen_model = str(model or settings["model"]).strip()
    if not chosen_model:
        raise MediaError("A model is required to generate an image.")
    chosen_size = str(size or settings["size"]).strip()
    if not config.MEDIA_SIZE_RE.match(chosen_size):
        raise MediaError("Image size must look like WxH, e.g. 1024x1024.")

    url = provider.versioned_url(
        f"{provider.base_url.rstrip('/')}/images/generations"
    )
    payload = _payload(chosen_model, text, chosen_size)
    try:
        with _client(REQUEST_TIMEOUT) as client:
            response = client.post(url, json=payload, headers=_headers(provider, key))
    except httpx.HTTPError as exc:
        raise MediaError(
            f"{provider.name} image request failed: {exc.__class__.__name__}."
        ) from exc

    if response.status_code != 200:
        body = response.text[:500]
        raise MediaError(
            f"{provider.name} image generation returned HTTP "
            f"{response.status_code}: {redact(body)}"
        )

    item = _first_image(response, provider.name)
    revised = item.get("revised_prompt")
    revised_prompt = str(revised) if revised else None

    b64_payload = item.get("b64_json")
    if isinstance(b64_payload, str) and b64_payload:
        data = _decode_b64(b64_payload)
    else:
        image_url = item.get("url")
        if not isinstance(image_url, str) or not image_url:
            raise MediaError(
                "The provider response contained neither b64_json nor url."
            )
        data = _download_image(image_url)
    if not data:
        raise MediaError("The provider returned an empty image.")

    path = _write_image(
        session.workspace_root, str(settings["output_dir"]), text, data
    )
    return MediaResult(
        path=path,
        revised_prompt=revised_prompt,
        model=chosen_model,
        bytes=len(data),
    )


def attach_result(result: MediaResult, session: Session) -> str:
    """Best-effort attach a generated image; returns a note ("" on success).

    Honors the vision size/count limits. A failure to attach never fails the
    generation itself -- the caller has the path either way.
    """
    try:
        vision = config.get_vision()
        limit = int(vision["max_images_per_turn"])
        if len(session.pending_images) >= limit:
            return (
                f"Not attached: {limit} image(s) already pending. Send them "
                "first, then generate again."
            )
        entry = images.encode_image(result.path, int(vision["max_image_bytes"]))
    except (ValueError, OSError) as exc:
        return f"Not attached: {exc}"
    session.pending_images.append(entry)
    return ""
