"""Image and video generation through provider media APIs.

Generation is opt-in (``media.enabled``) and approval-gated when driven by the
``generate_image`` / ``generate_video`` tools, because a call costs money.
Requests reuse the model provider registry for auth
(``key_header``/``key_prefix``/``extra_headers``), API versioning
(``api_version``), and network transport (proxy/CA/offline) so media endpoints
behave exactly like chat endpoints.

Images
    ``POST <base>/images/generations`` (OpenAI-compatible) or ``POST
    <base>/images`` (OpenRouter). Responses carry either ``data[0].b64_json`` or
    a ``data[0].url`` that is downloaded with a bounded size. The decoded image
    is written atomically under the workspace at
    ``<output_dir>/<YYYYmmdd_HHMMSS>-<slug>.png``; callers attach the result to
    ``session.pending_images`` best-effort via :func:`attach_result`.

Video
    Video generation is an asynchronous job: ``POST <base>/videos`` returns a
    job id, ``GET <base>/videos/<id>`` is polled until it completes, and the
    finished file is streamed to ``<output_dir>/<stamp>-<slug>.mp4`` with a
    bounded size. A job that is still running when we stop waiting (timeout or
    Ctrl+C) keeps running -- and billing -- on the provider, so its id is
    surfaced and :func:`resume_video` can pick it up again.

Two wire dialects are spoken, chosen from the provider's host: OpenRouter's
(``duration``/``resolution``/``frame_images``, ``unsigned_urls`` on
completion) and the OpenAI Videos API's (``seconds``/``size``, multipart
form, ``/videos/<id>/content``).
"""

from __future__ import annotations

import base64
import datetime
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx

from kiwimatecoder import config, images, network
from kiwimatecoder.redaction import redact
from kiwimatecoder.web import is_local_address

if TYPE_CHECKING:
    from kiwimatecoder.providers import ProviderConfig
    from kiwimatecoder.session import Session

REQUEST_TIMEOUT = 120.0
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VIDEO_BYTES = 500 * 1024 * 1024
VIDEO_POLL_INTERVAL = 10.0
VIDEO_MAX_WAIT = 15 * 60.0
# Consecutive failed status polls tolerated before giving up on a job.
_VIDEO_POLL_RETRIES = 3
_SLUG_MAX = 40
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_VIDEO_DONE = {"completed", "succeeded"}
_VIDEO_FAILED = {"failed", "cancelled", "canceled", "expired", "error"}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_SUFFIXES = (".mp4", ".webm", ".mov")
_DISABLED_MESSAGE = (
    "Media generation is disabled. Enable it with `/config media enable on` "
    "(or `kiwimatecoder config media enable on`)."
)
NO_VIDEO_MODEL_MESSAGE = (
    "No video model chosen. Pick one with `/config media video-model <id>` "
    "(for example an id from your provider's video model list)."
)


class MediaError(RuntimeError):
    """Raised when media cannot be generated; the message is user-facing."""


class VideoJobError(MediaError):
    """A video job exists on the provider but we have no file for it yet.

    Raised when we stopped waiting (timeout, lost connection) or could not fetch
    the finished file. Carries ``job_id`` so the caller can offer
    :func:`resume_video`.
    """

    def __init__(self, message: str, job_id: str) -> None:
        super().__init__(message)
        self.job_id = job_id


@dataclass(frozen=True)
class MediaResult:
    """One generated image written to disk."""

    path: Path
    revised_prompt: str | None
    model: str
    bytes: int


@dataclass(frozen=True)
class VideoResult:
    """One generated video written to disk."""

    path: Path
    job_id: str
    model: str
    bytes: int
    cost: float | None = None


@dataclass(frozen=True)
class VideoProgress:
    """A progress tick passed to the ``progress`` callback while polling."""

    job_id: str
    status: str
    elapsed: float
    percent: int | None = None


ProgressCallback = Callable[[VideoProgress], None]


def format_bytes(count: int) -> str:
    """Human-readable size, e.g. ``1.4 MB``."""
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{count} B"  # pragma: no cover - loop always returns


def _client(timeout: float) -> httpx.Client:
    """Build an httpx client honoring proxy/CA settings (tests patch this)."""
    return httpx.Client(timeout=timeout, **network.current_options())


def _sleep(seconds: float) -> None:
    """Indirection so tests can poll without waiting."""
    time.sleep(seconds)


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _is_openrouter(provider: ProviderConfig) -> bool:
    host = _host(provider.base_url)
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


def _headers(
    provider: ProviderConfig, key: str | None, *, json_body: bool = True
) -> dict[str, str]:
    headers = {"Content-Type": "application/json"} if json_body else {}
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


def _output_dir(workspace_root: Path | str, output_dir: str) -> Path:
    """Resolve the media directory, refusing anything outside the workspace."""
    root = Path(workspace_root).resolve()
    target_dir = (root / output_dir).resolve()
    if target_dir != root and not target_dir.is_relative_to(root):
        raise MediaError(
            f"Media output directory '{output_dir}' is outside the workspace root."
        )
    return target_dir


def _reserve_path(
    workspace_root: Path | str, output_dir: str, label: str, suffix: str
) -> Path:
    """Create the output directory and return a free ``<stamp>-<slug><suffix>``."""
    target_dir = _output_dir(workspace_root, output_dir)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MediaError(f"Could not create media directory '{target_dir}': {exc}") from exc
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slug(label)
    path = target_dir / f"{stamp}-{slug}{suffix}"
    counter = 1
    while path.exists():
        counter += 1
        path = target_dir / f"{stamp}-{slug}-{counter}{suffix}"
    return path


def _write_image(workspace_root: Path | str, output_dir: str, prompt: str, data: bytes) -> Path:
    path = _reserve_path(workspace_root, output_dir, prompt, ".png")
    try:
        _atomic_write_bytes(path, data)
    except OSError as exc:
        raise MediaError(f"Could not save the generated image to '{path}': {exc}") from exc
    return path


def _guard_download(url: str, what: str) -> None:
    """Offline and local-address guards shared by image and video downloads."""
    if config.offline_enabled():
        raise MediaError(network.offline_message(what))
    if is_local_address(url) and not config.get_web()["allow_local"]:
        raise MediaError(
            f"Refusing to download the {what.split()[0]} from a local or private "
            "address. Enable it with `/config web allow-local on`."
        )


def _download_image(url: str) -> bytes:
    """Download an image URL with a bounded size and local/offline guards."""
    _guard_download(url, "image download")
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


def _resolve_provider(
    provider_id: str | None, cfg: dict[str, Any] | None, what: str
) -> tuple[dict[str, Any], str, ProviderConfig, str | None]:
    """Common preflight: enabled, known provider, online, and an API key.

    Returns ``(settings, provider_id, provider, key)``. Every failure is a
    :class:`MediaError` whose message says how to fix it.
    """
    settings = config.get_media(cfg)
    if not settings["enabled"]:
        raise MediaError(_DISABLED_MESSAGE)

    resolved_provider = str(provider_id or settings["provider"]).strip()
    try:
        provider = config.get_provider_config(resolved_provider, cfg)
    except KeyError as exc:
        raise MediaError(str(exc) or f"Unknown provider '{resolved_provider}'.") from exc

    if config.offline_enabled(cfg):
        raise MediaError(network.offline_message(what))

    key = config.get_key(resolved_provider)
    if not key and provider.needs_key:
        raise MediaError(
            f"No API key for {provider.name}. Set one with "
            f"`config key set {provider.id} <KEY>` or the "
            f"{provider.key_env} environment variable."
        )
    return settings, resolved_provider, provider, key


def readiness(*, video: bool = False, cfg: dict[str, Any] | None = None) -> list[str]:
    """Why generation would fail right now; an empty list means ready."""
    problems: list[str] = []
    try:
        settings, *_ = _resolve_provider(
            None, cfg, "video generation" if video else "image generation"
        )
    except MediaError as exc:
        problems.append(str(exc))
        settings = config.get_media(cfg)
    if video and not settings["video_model"]:
        problems.append(NO_VIDEO_MODEL_MESSAGE)
    return problems


def list_outputs(
    session: Session, limit: int = 10, cfg: dict[str, Any] | None = None
) -> list[Path]:
    """Most recent generated images and videos, newest first."""
    settings = config.get_media(cfg)
    try:
        directory = _output_dir(session.workspace_root, str(settings["output_dir"]))
    except MediaError:
        return []
    if not directory.is_dir():
        return []
    found: list[tuple[float, Path]] = []
    try:
        for entry in directory.iterdir():
            if entry.suffix.lower() in IMAGE_SUFFIXES + VIDEO_SUFFIXES and entry.is_file():
                try:
                    found.append((entry.stat().st_mtime, entry))
                except OSError:
                    continue
    except OSError:
        return []
    found.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    return [path for _, path in found[: max(limit, 0)]]


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

    settings, resolved_provider, provider, key = _resolve_provider(
        provider_id, cfg, "image generation"
    )
    del resolved_provider

    chosen_model = str(model or settings["model"]).strip()
    if not chosen_model:
        raise MediaError("A model is required to generate an image.")
    chosen_size = str(size or settings["size"]).strip()
    if not config.MEDIA_SIZE_RE.match(chosen_size):
        raise MediaError("Image size must look like WxH, e.g. 1024x1024.")

    route = "images" if _is_openrouter(provider) else "images/generations"
    url = provider.versioned_url(f"{provider.base_url.rstrip('/')}/{route}")
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


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


def normalize_video_size(size: str) -> str:
    """Canonical casing for a video size: ``720p``/``1080p`` and ``2K``/``4K``."""
    text = size.strip()
    if "x" in text.lower():
        return text.lower()
    return text.lower() if text.lower().endswith("p") else text.upper()


def _video_payload(
    provider: ProviderConfig,
    model: str,
    prompt: str,
    duration: int,
    size: str,
    first_frame_url: str | None,
) -> dict[str, Any]:
    if _is_openrouter(provider):
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "duration": duration,
        }
        if size:
            payload["size" if "x" in size else "resolution"] = size
        if first_frame_url:
            payload["frame_images"] = [
                {
                    "type": "image_url",
                    "image_url": {"url": first_frame_url},
                    "frame_type": "first_frame",
                }
            ]
        return payload
    if first_frame_url:
        raise MediaError(
            f"{provider.name} video generation does not take a first-frame image "
            "here. Use an OpenRouter video model for image-to-video."
        )
    payload = {"model": model, "prompt": prompt, "seconds": str(duration)}
    if size:
        if "x" not in size:
            raise MediaError(
                f"{provider.name} needs an exact video size such as 1280x720, "
                f"not '{size}'."
            )
        payload["size"] = size
    return payload


def _json_object(response: httpx.Response, provider_name: str) -> dict[str, Any]:
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise MediaError(f"{provider_name} returned a non-JSON video response.") from exc
    if not isinstance(body, dict):
        raise MediaError(f"{provider_name} returned an unexpected video response.")
    return body


def _error_text(body: dict[str, Any]) -> str:
    error = body.get("error")
    if isinstance(error, dict):
        error = error.get("message") or error
    return redact(str(error or "no reason given"))[:500]


def _poll_url(provider: ProviderConfig, job_id: str, advertised: object) -> str:
    """Where to poll: the provider's ``polling_url`` only if it stays on-host."""
    base = provider.base_url.rstrip("/")
    fallback = provider.versioned_url(f"{base}/videos/{job_id}")
    if isinstance(advertised, str) and advertised:
        if _host(advertised) == _host(provider.base_url):
            return advertised
    return fallback


def _download_video(
    url: str,
    dest: Path,
    headers: dict[str, str],
    *,
    send_auth: bool,
) -> tuple[int, bytes]:
    """Stream a video to ``dest`` (atomically) with a size bound.

    Returns ``(bytes written, first 16 bytes)``; the head is used to pick the
    file extension without re-reading a large file.
    """
    _guard_download(url, "video download")
    tmp = dest.with_name(dest.name + ".kiwi.tmp")
    total = 0
    head = b""
    try:
        with _client(REQUEST_TIMEOUT) as client:
            with client.stream("GET", url, headers=headers if send_auth else {}) as response:
                if response.status_code != 200:
                    raise MediaError(
                        f"Video download returned HTTP {response.status_code}."
                    )
                with tmp.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        if len(head) < 16:
                            head = (head + chunk)[:16]
                        total += len(chunk)
                        if total > MAX_VIDEO_BYTES:
                            raise MediaError(
                                "Video download exceeded the "
                                f"{MAX_VIDEO_BYTES:,}-byte limit."
                            )
                        handle.write(chunk)
        if total == 0:
            raise MediaError("The provider returned an empty video.")
        if not _looks_like_video(head):
            raise MediaError("The provider returned a file that is not a video.")
        os.replace(tmp, dest)
    except httpx.HTTPError as exc:
        raise MediaError(f"Video download failed: {exc.__class__.__name__}.") from exc
    except OSError as exc:
        raise MediaError(f"Could not save the generated video to '{dest}': {exc}") from exc
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    return total, head


def _looks_like_video(head: bytes) -> bool:
    """MP4/MOV (``ftyp`` box) or WebM/Matroska (EBML header)."""
    return head[4:8] == b"ftyp" or head[:4] == b"\x1a\x45\xdf\xa3"


def _video_suffix(head: bytes) -> str:
    return ".webm" if head[:4] == b"\x1a\x45\xdf\xa3" else ".mp4"


def _await_video(
    provider: ProviderConfig,
    key: str | None,
    job_id: str,
    poll_url: str,
    *,
    progress: ProgressCallback | None,
    poll_interval: float,
    max_wait: float,
    started: float,
) -> dict[str, Any]:
    """Poll a job until it completes; returns the final status body."""
    headers = _headers(provider, key, json_body=False)
    failures = 0
    while True:
        status = "unknown"
        percent: int | None = None
        body: dict[str, Any] | None = None
        try:
            with _client(REQUEST_TIMEOUT) as client:
                response = client.get(poll_url, headers=headers)
        except httpx.HTTPError:
            response = None
        if response is not None and response.status_code in (401, 403, 404):
            raise MediaError(
                f"{provider.name} video status check for job {job_id} returned "
                f"HTTP {response.status_code}: {redact(response.text[:300])}"
            )
        if response is not None and response.status_code < 400:
            body = _json_object(response, provider.name)

        if body is None:
            # Network blip or a 5xx/429: the job is fine, keep polling a while.
            failures += 1
            if failures > _VIDEO_POLL_RETRIES:
                raise VideoJobError(
                    f"Lost contact with {provider.name} while the video was "
                    f"rendering. Job {job_id} may still finish; resume it with "
                    f"`/video resume {job_id}`.",
                    job_id,
                )
        else:
            failures = 0
            status = str(body.get("status") or "unknown").lower()
            raw_progress = body.get("progress")
            if isinstance(raw_progress, (int, float)) and not isinstance(raw_progress, bool):
                percent = max(0, min(100, int(raw_progress)))
            if status in _VIDEO_DONE:
                return body
            if status in _VIDEO_FAILED:
                raise MediaError(
                    f"{provider.name} video generation {status}: {_error_text(body)}"
                )

        elapsed = time.monotonic() - started
        if progress is not None:
            try:
                progress(VideoProgress(job_id, status, elapsed, percent))
            except Exception:  # noqa: BLE001 - a display glitch must not kill the job
                pass
        if elapsed >= max_wait:
            raise VideoJobError(
                f"Still rendering after {int(elapsed // 60)} min. The job keeps "
                f"running (and billing) on {provider.name}; resume it later with "
                f"`/video resume {job_id}`.",
                job_id,
            )
        _sleep(poll_interval)


def _finish_video(
    provider: ProviderConfig,
    key: str | None,
    settings: dict[str, Any],
    session: Session,
    job_id: str,
    advertised_poll_url: object,
    label: str,
    model: str,
    *,
    progress: ProgressCallback | None,
    poll_interval: float | None,
    max_wait: float | None,
) -> VideoResult:
    interval = VIDEO_POLL_INTERVAL if poll_interval is None else poll_interval
    limit = VIDEO_MAX_WAIT if max_wait is None else max_wait
    final = _await_video(
        provider,
        key,
        job_id,
        _poll_url(provider, job_id, advertised_poll_url),
        progress=progress,
        poll_interval=interval,
        max_wait=limit,
        started=time.monotonic(),
    )

    base = provider.base_url.rstrip("/")
    content_url = provider.versioned_url(f"{base}/videos/{job_id}/content")
    urls = final.get("unsigned_urls")
    download_url = content_url
    if isinstance(urls, list) and urls and isinstance(urls[0], str) and urls[0]:
        download_url = urls[0]
    # Only the provider's own host ever sees the API key.
    send_auth = _host(download_url) == _host(provider.base_url)

    dest = _reserve_path(
        session.workspace_root, str(settings["output_dir"]), label, ".mp4"
    )
    try:
        size, head = _download_video(
            download_url,
            dest,
            _headers(provider, key, json_body=False),
            send_auth=send_auth,
        )
    except MediaError as exc:
        raise VideoJobError(
            f"{exc} Job {job_id} finished on {provider.name}; once the problem "
            f"is fixed, fetch it with `/video resume {job_id}`.",
            job_id,
        ) from exc
    if _video_suffix(head) != dest.suffix:
        renamed = dest.with_suffix(_video_suffix(head))
        if not renamed.exists():
            os.replace(dest, renamed)
            dest = renamed

    cost: float | None = None
    usage = final.get("usage")
    if isinstance(usage, dict):
        raw_cost = usage.get("cost")
        if isinstance(raw_cost, (int, float)) and not isinstance(raw_cost, bool):
            cost = float(raw_cost)
    return VideoResult(path=dest, job_id=job_id, model=model, bytes=size, cost=cost)


def generate_video(
    prompt: str,
    *,
    session: Session,
    model: str | None = None,
    duration: int | None = None,
    size: str | None = None,
    first_frame: Path | str | None = None,
    provider_id: str | None = None,
    cfg: dict[str, Any] | None = None,
    progress: ProgressCallback | None = None,
    poll_interval: float | None = None,
    max_wait: float | None = None,
) -> VideoResult:
    """Submit a video job, wait for it, and save the result.

    ``first_frame`` is an already-validated workspace image to animate
    (image-to-video). Raises :class:`MediaError` for every failure and
    :class:`VideoJobError` -- carrying the job id -- when the job is still
    running after ``max_wait`` seconds or the file could not be fetched yet.
    """
    text = str(prompt or "").strip()
    if not text:
        raise MediaError("A prompt is required to generate a video.")

    settings, resolved_provider, provider, key = _resolve_provider(
        provider_id, cfg, "video generation"
    )
    del resolved_provider

    chosen_model = str(model or settings["video_model"]).strip()
    if not chosen_model:
        raise MediaError(NO_VIDEO_MODEL_MESSAGE)
    seconds = settings["video_duration"] if duration is None else duration
    if not isinstance(seconds, int) or isinstance(seconds, bool) or not (
        config.VIDEO_DURATION_MIN <= seconds <= config.VIDEO_DURATION_MAX
    ):
        raise MediaError(
            "Video duration must be whole seconds between "
            f"{config.VIDEO_DURATION_MIN} and {config.VIDEO_DURATION_MAX}."
        )
    chosen_size = str(size or settings["video_size"]).strip()
    if chosen_size and not config.VIDEO_SIZE_RE.match(chosen_size):
        raise MediaError("Video size must be WxH (e.g. 1280x720) or a tier like 720p.")
    chosen_size = normalize_video_size(chosen_size) if chosen_size else ""

    frame_url: str | None = None
    if first_frame is not None:
        try:
            vision = config.get_vision()
            entry = images.encode_image(Path(first_frame), int(vision["max_image_bytes"]))
        except (ValueError, OSError) as exc:
            raise MediaError(f"Cannot use the first-frame image: {exc}") from exc
        frame_url = f"data:{entry['media_type']};base64,{entry['data']}"

    payload = _video_payload(provider, chosen_model, text, seconds, chosen_size, frame_url)
    url = provider.versioned_url(f"{provider.base_url.rstrip('/')}/videos")
    started = time.monotonic()
    try:
        with _client(REQUEST_TIMEOUT) as client:
            if _is_openrouter(provider):
                response = client.post(url, json=payload, headers=_headers(provider, key))
            else:
                # The OpenAI Videos API takes a multipart form, not JSON.
                response = client.post(
                    url,
                    files={name: (None, str(value)) for name, value in payload.items()},
                    headers=_headers(provider, key, json_body=False),
                )
    except httpx.HTTPError as exc:
        raise MediaError(
            f"{provider.name} video request failed: {exc.__class__.__name__}."
        ) from exc
    if response.status_code not in (200, 201, 202):
        raise MediaError(
            f"{provider.name} video generation returned HTTP "
            f"{response.status_code}: {redact(response.text[:500])}"
        )
    body = _json_object(response, provider.name)
    job_id = str(body.get("id") or "").strip()
    if not _JOB_ID_RE.match(job_id):
        raise MediaError(f"{provider.name} did not return a usable video job id.")
    status = str(body.get("status") or "pending").lower()
    if status in _VIDEO_FAILED:
        raise MediaError(f"{provider.name} video generation {status}: {_error_text(body)}")

    if progress is not None:
        try:
            progress(VideoProgress(job_id, "submitted", time.monotonic() - started, None))
        except Exception:  # noqa: BLE001
            pass
    return _finish_video(
        provider,
        key,
        settings,
        session,
        job_id,
        body.get("polling_url"),
        f"video {text}",
        chosen_model,
        progress=progress,
        poll_interval=poll_interval,
        max_wait=max_wait,
    )


def resume_video(
    job_id: str,
    *,
    session: Session,
    provider_id: str | None = None,
    cfg: dict[str, Any] | None = None,
    progress: ProgressCallback | None = None,
    poll_interval: float | None = None,
    max_wait: float | None = None,
) -> VideoResult:
    """Pick up a video job that was still running when we stopped waiting."""
    cleaned = str(job_id or "").strip()
    if not _JOB_ID_RE.match(cleaned):
        raise MediaError("A valid video job id is required to resume a job.")
    settings, resolved_provider, provider, key = _resolve_provider(
        provider_id, cfg, "video generation"
    )
    del resolved_provider
    return _finish_video(
        provider,
        key,
        settings,
        session,
        cleaned,
        None,
        f"video {cleaned}",
        str(settings["video_model"]) or "resumed job",
        progress=progress,
        poll_interval=poll_interval,
        max_wait=max_wait,
    )
