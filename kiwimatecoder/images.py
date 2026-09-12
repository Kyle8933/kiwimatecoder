"""Image attachment helpers for vision-capable models.

Images are validated (extension, magic bytes, size), base64-encoded, and
attached to user messages using OpenAI's multimodal content-part format.
:func:`kiwimatecoder.client.format_anthropic_messages` converts those parts to
native Anthropic base64 image blocks for Anthropic-compatible providers.
"""

from __future__ import annotations

import base64
import struct
from pathlib import Path
from typing import Any

# Extension -> MIME type. Kept deliberately small: every entry must be a type
# the major providers accept natively.
IMAGE_MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

SUPPORTED_SUFFIXES = tuple(sorted(IMAGE_MEDIA_TYPES))


def detect_media_type(path: str | Path) -> str | None:
    """Return the MIME type for a supported image suffix, else None."""
    return IMAGE_MEDIA_TYPES.get(Path(path).suffix.lower())


def _sniff_media_type(data: bytes) -> str | None:
    """Identify an image from its leading bytes (magic numbers)."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def encode_image(path: str | Path, max_bytes: int) -> dict[str, str]:
    """Validate and encode an image for a pending-image entry.

    Returns ``{"media_type", "data", "name"}`` where ``data`` is base64 text.
    Raises ``ValueError`` with a caller-facing message for unsupported types,
    missing files, oversized payloads, or bytes that do not match the
    extension.
    """
    image_path = Path(path)
    media_type = detect_media_type(image_path)
    if media_type is None:
        raise ValueError(
            f"Unsupported image type '{image_path.suffix or image_path.name}'. "
            f"Supported: {', '.join(SUPPORTED_SUFFIXES)}."
        )
    if not image_path.is_file():
        raise ValueError(f"Image not found: {image_path}")
    try:
        limit = int(max_bytes)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_image_bytes must be an integer.") from exc
    try:
        data = image_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Could not read image '{image_path}': {exc}") from exc
    if limit > 0 and len(data) > limit:
        raise ValueError(
            f"Image '{image_path.name}' is {len(data):,} bytes, over the "
            f"{limit:,}-byte limit. Raise it with /config vision max-bytes."
        )
    sniffed = _sniff_media_type(data[:16])
    if sniffed is None:
        raise ValueError(
            f"'{image_path.name}' does not look like a valid {media_type} image."
        )
    if sniffed != media_type:
        raise ValueError(
            f"'{image_path.name}' has extension {image_path.suffix or '?'} "
            f"but appears to be {sniffed}."
        )
    return {
        "media_type": media_type,
        "data": base64.b64encode(data).decode("ascii"),
        "name": image_path.name,
    }


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    index = 2
    length = len(data)
    while index + 4 <= length:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        if marker in (0xD9, 0xDA):  # end of image / start of scan: no headers
            return None
        segment_length = int.from_bytes(data[index + 2 : index + 4], "big")
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            if index + 9 > length:
                return None
            height = int.from_bytes(data[index + 5 : index + 7], "big")
            width = int.from_bytes(data[index + 7 : index + 9], "big")
            return (width, height) if width and height else None
        if segment_length <= 0:
            return None
        index += 2 + segment_length
    return None


def _webp_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 30:
        return None
    chunk = data[12:16]
    if chunk == b"VP8X":
        return (
            int.from_bytes(data[24:27], "little") + 1,
            int.from_bytes(data[27:30], "little") + 1,
        )
    if chunk == b"VP8 ":
        if data[23:26] != b"\x9d\x01\x2a":
            return None
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return (width, height) if width and height else None
    if chunk == b"VP8L":
        if len(data) < 25 or data[20] != 0x2F:
            return None
        bits = int.from_bytes(data[21:25], "little")
        return ((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    return None


def image_dimensions(data: bytes) -> tuple[int, int] | None:
    """Best-effort width/height from image headers (no Pillow required)."""
    try:
        if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
            return struct.unpack(">II", data[16:24])
        if data.startswith((b"GIF87a", b"GIF89a")) and len(data) >= 10:
            width, height = struct.unpack("<HH", data[6:10])
            return (width, height) if width and height else None
        if data.startswith(b"\xff\xd8\xff"):
            return _jpeg_dimensions(data)
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return _webp_dimensions(data)
    except (struct.error, IndexError):
        return None
    return None


def image_message(images: list[dict[str, Any]], note: str | None = None) -> dict[str, Any]:
    """Build an OpenAI-format user message carrying ``images`` as content parts.

    Text parts name each image, followed by one ``image_url`` data-URL part per
    image. The optional ``note`` is prepended as the first text part.
    """
    parts: list[dict[str, Any]] = []
    if note:
        parts.append({"type": "text", "text": str(note)})
    image_parts: list[dict[str, Any]] = []
    for index, image in enumerate(images, 1):
        media_type = str(image.get("media_type") or "image/png")
        data = str(image.get("data") or "")
        name = str(image.get("name") or f"image {index}")
        parts.append({"type": "text", "text": f"Attached image: {name}"})
        image_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"},
            }
        )
    return {"role": "user", "content": [*parts, *image_parts]}
