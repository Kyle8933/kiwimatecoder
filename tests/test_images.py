from __future__ import annotations

import base64
import json

import pytest

from kiwimatecoder import config, images, tools
from kiwimatecoder.pricing import estimate_messages_tokens
from kiwimatecoder.tools.image import _view_image

# 1x1 images, small enough to inline.
PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
GIF_1PX = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")


@pytest.fixture(autouse=True)
def isolate_config_home(tmp_path, monkeypatch):
    home = tmp_path / "config-home"
    monkeypatch.setattr(config, "CONFIG_DIR", home)
    monkeypatch.setattr(config, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", home / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)


# ---------------------------------------------------------------------------
# detect_media_type / encode_image
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("shot.png", "image/png"),
        ("shot.PNG", "image/png"),
        ("photo.jpg", "image/jpeg"),
        ("photo.jpeg", "image/jpeg"),
        ("anim.gif", "image/gif"),
        ("modern.webp", "image/webp"),
        ("notes.txt", None),
        ("archive.zip", None),
        ("noext", None),
    ],
)
def test_detect_media_type(name, expected):
    assert images.detect_media_type(name) == expected


def test_encode_image_roundtrip(tmp_path):
    path = tmp_path / "pixel.png"
    path.write_bytes(PNG_1PX)

    entry = images.encode_image(path, max_bytes=1_000_000)

    assert entry["media_type"] == "image/png"
    assert entry["name"] == "pixel.png"
    assert base64.b64decode(entry["data"]) == PNG_1PX


def test_encode_image_rejects_oversized(tmp_path):
    path = tmp_path / "big.png"
    path.write_bytes(PNG_1PX + b"\x00" * 64)

    with pytest.raises(ValueError, match="over the"):
        images.encode_image(path, max_bytes=len(PNG_1PX))


def test_encode_image_rejects_bad_magic(tmp_path):
    path = tmp_path / "fake.png"
    path.write_bytes(b"not really an image")

    with pytest.raises(ValueError, match="does not look like a valid"):
        images.encode_image(path, max_bytes=1_000_000)


def test_encode_image_rejects_extension_mismatch(tmp_path):
    path = tmp_path / "wrong.png"
    path.write_bytes(GIF_1PX)

    with pytest.raises(ValueError, match="appears to be image/gif"):
        images.encode_image(path, max_bytes=1_000_000)


def test_encode_image_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "vector.svg"
    path.write_text("<svg/>")

    with pytest.raises(ValueError, match="Unsupported image type"):
        images.encode_image(path, max_bytes=1_000_000)


def test_encode_image_missing_file(tmp_path):
    with pytest.raises(ValueError, match="Image not found"):
        images.encode_image(tmp_path / "missing.png", max_bytes=1_000_000)


def test_image_dimensions_reads_headers():
    assert images.image_dimensions(PNG_1PX) == (1, 1)
    assert images.image_dimensions(GIF_1PX) == (1, 1)
    assert images.image_dimensions(b"garbage") is None


def test_image_message_shape():
    message = images.image_message(
        [
            {"media_type": "image/png", "data": "AAA", "name": "a.png"},
            {"media_type": "image/jpeg", "data": "BBB", "name": "b.jpg"},
        ],
        note="Images attached for analysis:",
    )

    assert message["role"] == "user"
    content = message["content"]
    assert content[0] == {"type": "text", "text": "Images attached for analysis:"}
    assert content[1] == {"type": "text", "text": "Attached image: a.png"}
    assert content[2] == {"type": "text", "text": "Attached image: b.jpg"}
    assert content[3] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAA"},
    }
    assert content[4] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,BBB"},
    }


def test_image_message_names_are_optional():
    message = images.image_message([{"media_type": "image/png", "data": "AAA"}])
    assert message["content"][0]["text"] == "Attached image: image 1"


# ---------------------------------------------------------------------------
# view_image tool
# ---------------------------------------------------------------------------


def test_view_image_is_read_only_and_registered():
    tool = tools.get_tool("view_image")
    assert tool is not None
    assert not tool.needs_approval


def test_view_image_attaches_pending_image(session):
    (session.workspace_root / "pixel.png").write_bytes(PNG_1PX)

    result = _view_image({"path": "pixel.png"}, session)

    assert result.ok
    assert result.content == "Attached pixel.png (1x1)"
    assert len(session.pending_images) == 1
    assert session.pending_images[0]["media_type"] == "image/png"


def test_view_image_rejects_missing_and_non_images(session):
    assert not _view_image({"path": "ghost.png"}, session).ok
    (session.workspace_root / "notes.txt").write_text("hi")
    assert not _view_image({"path": "notes.txt"}, session).ok
    assert session.pending_images == []


def test_view_image_respects_per_turn_limit(session):
    (session.workspace_root / "a.png").write_bytes(PNG_1PX)
    (session.workspace_root / "b.png").write_bytes(PNG_1PX)
    config.set_vision(max_images_per_turn=1)

    first = _view_image({"path": "a.png"}, session)
    second = _view_image({"path": "b.png"}, session)

    assert first.ok
    assert not second.ok
    assert "limit reached" in second.content
    assert len(session.pending_images) == 1


def test_view_image_stays_inside_workspace(session, tmp_path):
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(PNG_1PX)

    result = _view_image({"path": str(outside)}, session)

    assert not result.ok
    assert session.pending_images == []


# ---------------------------------------------------------------------------
# Session serialization and token estimate
# ---------------------------------------------------------------------------


def test_pending_images_are_not_serialized(session):
    session.pending_images.append({"media_type": "image/png", "data": "AAA"})

    data = session.to_dict()
    restored = type(session).from_dict(data)

    assert "pending_images" not in data
    assert restored.pending_images == []


def test_image_parts_use_flat_token_estimate():
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "hi"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + "A" * 1_000_000},
            },
        ],
    }

    assert estimate_messages_tokens([message]) < 10_000


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_get_vision_defaults():
    assert config.get_vision() == {
        "max_image_bytes": 5_000_000,
        "max_images_per_turn": 4,
    }


def test_set_vision_roundtrip():
    config.set_vision(max_image_bytes=2_000_000, max_images_per_turn=2)

    settings = config.get_vision()
    assert settings["max_image_bytes"] == 2_000_000
    assert settings["max_images_per_turn"] == 2
    assert config.load_config()["vision"]["max_images_per_turn"] == 2


@pytest.mark.parametrize(
    "updates",
    [
        {"max_image_bytes": "not-a-number"},
        {"max_image_bytes": 10},
        {"max_image_bytes": 999_999_999_999},
        {"max_images_per_turn": "many"},
        {"max_images_per_turn": 0},
        {"max_images_per_turn": 101},
    ],
)
def test_set_vision_rejects_invalid_values(updates):
    with pytest.raises(ValueError):
        config.set_vision(**updates)


def test_validate_flags_bad_vision_values():
    cfg = config.load_config()
    cfg["vision"] = {"max_image_bytes": "huge", "max_images_per_turn": 0}

    issues = config.validate_config(cfg)

    error_keys = {issue["key"] for issue in issues if issue["level"] == "error"}
    assert {"vision.max_image_bytes", "vision.max_images_per_turn"} <= error_keys
    assert config.validate_config() == []


def test_validate_flags_non_object_vision_section():
    cfg = config.load_config()
    cfg["vision"] = ["nope"]

    issues = config.validate_config(cfg)

    assert any(
        issue["level"] == "error" and issue["key"] == "vision" for issue in issues
    )


def test_empty_config_contains_vision_defaults():
    cfg = config.load_config()
    assert json.loads(json.dumps(cfg["vision"])) == {
        "max_image_bytes": 5_000_000,
        "max_images_per_turn": 4,
    }


def test_config_vision_slash_command_roundtrip(session):
    from io import StringIO

    from rich.console import Console

    from kiwimatecoder.commands import CommandResult, dispatch

    console = Console(file=StringIO(), force_terminal=False, width=120)

    assert (
        dispatch("/config vision max-bytes 2000000", session, console)
        == CommandResult.CONTINUE
    )
    assert config.get_vision()["max_image_bytes"] == 2_000_000

    assert (
        dispatch("/config vision max-images 2", session, console)
        == CommandResult.CONTINUE
    )
    assert config.get_vision()["max_images_per_turn"] == 2

    dispatch("/config vision", session, console)
    assert "2,000,000" in console.file.getvalue()
