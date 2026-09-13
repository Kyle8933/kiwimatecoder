"""Image generation: media client, tool, /image command, and config.

All HTTP goes through a mocked httpx transport; the config dir is a temp path
so nothing touches the real home directory.
"""

from __future__ import annotations

import base64
import io
import json

import httpx
import pytest
from rich.console import Console

from kiwimatecoder import config, media, tools
from kiwimatecoder import media as media_module
from kiwimatecoder.agent import Agent
from kiwimatecoder.commands import dispatch
from kiwimatecoder.providers import REGISTRY

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for provider in REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def _install_client(monkeypatch, handler) -> httpx.MockTransport:
    transport = httpx.MockTransport(handler)

    def factory(timeout: float) -> httpx.Client:
        return httpx.Client(timeout=timeout, transport=transport)

    monkeypatch.setattr(media_module, "_client", factory)
    return transport


def _forbid_client(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request to {request.url}")

    _install_client(monkeypatch, handler)


def _enable_media(**kwargs) -> None:
    config.set_media(enabled=True, **kwargs)
    config.set_key("openai", "sk-test")


def _b64(data: bytes = PNG_BYTES) -> str:
    return base64.b64encode(data).decode("ascii")


# ---------------------------------------------------------------------------
# generate_image
# ---------------------------------------------------------------------------


def test_generate_image_b64_response_writes_png(session, monkeypatch):
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "data": [
                    {"b64_json": _b64(), "revised_prompt": "A red square"}
                ]
            },
            request=request,
        )

    _install_client(monkeypatch, handler)
    _enable_media()

    result = media.generate_image("a red square", session=session)

    assert result.path.is_file()
    assert result.path.read_bytes() == PNG_BYTES
    assert result.bytes == len(PNG_BYTES)
    assert result.revised_prompt == "A red square"
    assert result.model == "gpt-image-1"
    assert result.path.parent == (
        session.workspace_root / ".kiwimatecoder" / "media"
    )
    assert result.path.name.endswith("-a-red-square.png")
    assert str(seen["url"]).endswith("/images/generations")
    assert seen["auth"] == "Bearer sk-test"
    payload = seen["payload"]
    assert isinstance(payload, dict)
    assert payload["prompt"] == "a red square"
    assert payload["n"] == 1
    assert "response_format" not in payload  # gpt-image returns b64 by default


def test_dall_e_requests_b64_response_format():
    payload = media._payload("dall-e-3", "a cat", "512x512")
    assert payload["response_format"] == "b64_json"


def test_generate_image_url_response_downloads(session, monkeypatch):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "POST":
            return httpx.Response(
                200,
                json={"data": [{"url": "https://images.example/test.png"}]},
                request=request,
            )
        return httpx.Response(200, content=PNG_BYTES, request=request)

    _install_client(monkeypatch, handler)
    _enable_media()

    result = media.generate_image("cat", session=session)

    assert calls == ["POST", "GET"]
    assert result.path.read_bytes() == PNG_BYTES


def test_generate_image_url_download_size_guard(session, monkeypatch):
    monkeypatch.setattr(media_module, "MAX_IMAGE_BYTES", 8)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200,
                json={"data": [{"url": "https://images.example/big.png"}]},
                request=request,
            )
        return httpx.Response(200, content=b"x" * 64, request=request)

    _install_client(monkeypatch, handler)
    _enable_media()

    with pytest.raises(media.MediaError, match="limit"):
        media.generate_image("cat", session=session)


def test_generate_image_refuses_offline(session, monkeypatch):
    _forbid_client(monkeypatch)
    _enable_media()
    config.set_network(offline=True)

    with pytest.raises(media.MediaError, match="offline"):
        media.generate_image("cat", session=session)


def test_generate_image_missing_key(session, monkeypatch):
    _forbid_client(monkeypatch)
    config.set_media(enabled=True)

    with pytest.raises(media.MediaError, match="No API key"):
        media.generate_image("cat", session=session)


def test_generate_image_disabled(session, monkeypatch):
    _forbid_client(monkeypatch)

    with pytest.raises(media.MediaError, match="disabled"):
        media.generate_image("cat", session=session)


def test_generate_image_empty_prompt(session):
    with pytest.raises(media.MediaError, match="prompt"):
        media.generate_image("   ", session=session)


def test_generate_image_provider_http_error_is_wrapped(session, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad key", request=request)

    _install_client(monkeypatch, handler)
    _enable_media()

    with pytest.raises(media.MediaError, match="HTTP 401"):
        media.generate_image("cat", session=session)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def test_generate_image_tool_is_approval_gated():
    tool = tools.get_tool("generate_image")
    assert tool is not None
    assert tool.runs is True
    assert tool.needs_approval is True


def test_generate_image_tool_preview_shows_details(session):
    config.set_media(provider="openai", model="gpt-image-1", size="512x512")
    preview = tools.preview("generate_image", {"prompt": "a cat"}, session)

    assert preview is not None
    assert "openai" in preview
    assert "gpt-image-1" in preview
    assert "512x512" in preview
    assert "a cat" in preview
    assert ".kiwimatecoder/media" in preview


def test_generate_image_tool_disabled_returns_guidance(session, monkeypatch):
    _forbid_client(monkeypatch)

    result = tools.dispatch("generate_image", {"prompt": "cat"}, session)

    assert not result.ok
    assert "disabled" in result.content
    assert "/config media enable on" in result.content


def test_generate_image_tool_saves_and_attaches(session, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"data": [{"b64_json": _b64()}]}, request=request
        )

    _install_client(monkeypatch, handler)
    _enable_media()

    result = tools.dispatch("generate_image", {"prompt": "a cat"}, session)

    assert result.ok
    assert "Generated image saved" in result.content
    assert len(session.pending_images) == 1
    assert session.pending_images[0]["media_type"] == "image/png"


def test_agent_summary_for_generate_image(session):
    agent = Agent(session, _console(), confirm=lambda summary, preview: True)
    summary = agent._format_call_summary("generate_image", {"prompt": "a red fox"})
    assert "image-gen" in summary
    assert "a red fox" in summary


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_media_config_roundtrip_and_validation():
    settings = config.set_media(
        enabled=True,
        provider="openai",
        model="gpt-image-1",
        size="256x256",
        output_dir="art/generated",
    )
    assert settings["enabled"] is True
    assert settings["provider"] == "openai"
    assert settings["size"] == "256x256"
    assert config.get_media()["output_dir"] == "art/generated"

    with pytest.raises(ValueError, match="Unknown provider"):
        config.set_media(provider="nope")
    with pytest.raises(ValueError, match="WxH"):
        config.set_media(size="huge")
    with pytest.raises(ValueError, match="relative"):
        config.set_media(output_dir="/absolute/path")
    with pytest.raises(ValueError, match="relative"):
        config.set_media(output_dir="../escape")
    with pytest.raises(ValueError, match="true or false"):
        config.set_media(enabled="yes")
    with pytest.raises(ValueError, match="non-empty"):
        config.set_media(model="  ")


def test_validate_config_flags_bad_media():
    cfg = config._empty_config()
    cfg["media"] = {
        "enabled": "yes",
        "provider": "nope",
        "model": "",
        "size": "big",
        "output_dir": "/tmp/out",
    }

    issues = config.validate_config(cfg)
    keys = {issue["key"] for issue in issues if issue["level"] == "error"}

    assert {
        "media.enabled",
        "media.provider",
        "media.model",
        "media.size",
        "media.output_dir",
    } <= keys


def test_validate_config_accepts_media_defaults():
    cfg = config._empty_config()
    assert config.validate_config(cfg) == []


# ---------------------------------------------------------------------------
# /image command
# ---------------------------------------------------------------------------


def test_image_command_generates_and_attaches(session, monkeypatch):
    calls: list[str] = []

    def fake(prompt, *, session, size=None, model=None, provider_id=None):
        calls.append(prompt)
        target = session.workspace_root / ".kiwimatecoder" / "media" / "gen.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(PNG_BYTES)
        return media.MediaResult(
            path=target,
            revised_prompt=None,
            model="gpt-image-1",
            bytes=len(PNG_BYTES),
        )

    monkeypatch.setattr(media_module, "generate_image", fake)
    config.set_media(enabled=True)
    console = _console()

    dispatch("/image a cat", session, console)

    assert calls == ["a cat"]
    assert len(session.pending_images) == 1
    output = console.file.getvalue()
    assert "gen.png" in output
    assert "attached to your next message" in output


def test_image_command_disabled_prints_guidance(session):
    console = _console()

    dispatch("/image a cat", session, console)

    output = console.file.getvalue()
    assert "disabled" in output
    assert session.pending_images == []
