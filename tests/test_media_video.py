"""Video generation, /video, /media, /image options, and media config.

All HTTP goes through a mocked httpx transport and polling never sleeps, so
nothing touches the network or the real home directory.
"""

from __future__ import annotations

import io
import json
import os
import time

import httpx
import pytest
from rich.console import Console

from kiwimatecoder import config, tools
from kiwimatecoder import media as media_module
from kiwimatecoder.commands import CommandResult, dispatch, slash_command_completions
from kiwimatecoder.providers import REGISTRY

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
MP4_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64
WEBM_BYTES = b"\x1a\x45\xdf\xa3" + b"\x00" * 64
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
# Paths are shown workspace-relative with the OS separator (``\`` on Windows).
MEDIA_DIR = os.path.join(".kiwimatecoder", "media")


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for provider in REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=140)


def _output(console: Console) -> str:
    return console.file.getvalue()


class FakeVideoApi:
    """A tiny async-video server: POST creates a job, GET polls, GET /content."""

    def __init__(
        self,
        monkeypatch,
        *,
        polls=("in_progress", "completed"),
        video: bytes = MP4_BYTES,
        openrouter: bool = True,
        final: dict | None = None,
        post_status: int = 202,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self.polls = list(polls)
        self.video = video
        self.openrouter = openrouter
        self.final = final or {}
        self.post_status = post_status
        self.poll_count = 0
        self.sleeps: list[float] = []
        monkeypatch.setattr(media_module, "_sleep", self.sleeps.append)
        transport = httpx.MockTransport(self)

        def factory(timeout: float) -> httpx.Client:
            return httpx.Client(timeout=timeout, transport=transport)

        monkeypatch.setattr(media_module, "_client", factory)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "POST":
            body = {"id": "job-1", "status": "pending"}
            if self.openrouter:
                body["polling_url"] = f"{OPENROUTER_BASE}/videos/job-1"
            return httpx.Response(self.post_status, json=body, request=request)
        if path.endswith("/content"):
            return httpx.Response(200, content=self.video, request=request)
        if path.endswith("/videos/job-1"):
            index = min(self.poll_count, len(self.polls) - 1)
            self.poll_count += 1
            status = self.polls[index]
            if status == "http-503":
                return httpx.Response(503, text="busy", request=request)
            body = {"id": "job-1", "status": status}
            if status == "completed":
                body.update(self.final)
            if status == "failed":
                body["error"] = "content policy"
            return httpx.Response(200, json=body, request=request)
        return httpx.Response(404, text="nope", request=request)

    def posts(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == "POST"]


def _enable_openrouter_video(model: str = "vendor/video-1", **kwargs) -> None:
    config.set_media(
        enabled=True, provider="openrouter", video_model=model, **kwargs
    )
    config.set_key("openrouter", "sk-or-test")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_video_settings_default_to_no_model():
    settings = config.get_media()
    assert settings["video_model"] == ""
    assert settings["video_duration"] == 5
    assert settings["video_size"] == ""


def test_video_settings_roundtrip_and_validation():
    settings = config.set_media(
        video_model=" vendor/video-1 ", video_duration="8s", video_size="720P"
    )
    assert settings["video_model"] == "vendor/video-1"
    assert settings["video_duration"] == 8
    assert settings["video_size"] == "720P"
    assert config.set_media(video_size="1280x720")["video_size"] == "1280x720"
    assert config.set_media(video_size="")["video_size"] == ""
    assert config.set_media(video_model="")["video_model"] == ""

    for bad in ("0", "61", "abc", "", "1.5", True):
        with pytest.raises(ValueError, match="duration"):
            config.set_media(video_duration=bad)
    for bad_size in ("huge", "720", "p720", "12345p"):
        with pytest.raises(ValueError, match="video size"):
            config.set_media(video_size=bad_size)


def test_malformed_video_settings_fall_back_to_defaults():
    cfg = config._empty_config()
    cfg["media"] = {
        "video_model": 5,
        "video_duration": "ten",
        "video_size": "wide",
    }
    settings = config.get_media(cfg)
    assert settings["video_model"] == ""
    assert settings["video_duration"] == 5
    assert settings["video_size"] == ""

    issues = {i["key"] for i in config.validate_config(cfg) if i["level"] == "error"}
    assert {"media.video_model", "media.video_duration", "media.video_size"} <= issues


# ---------------------------------------------------------------------------
# generate_video: OpenRouter dialect
# ---------------------------------------------------------------------------


def test_openrouter_video_end_to_end(session, monkeypatch):
    api = FakeVideoApi(
        monkeypatch,
        final={
            "unsigned_urls": [f"{OPENROUTER_BASE}/videos/job-1/content?index=0"],
            "usage": {"cost": 0.25},
        },
    )
    _enable_openrouter_video(video_size="720p")
    ticks: list[media_module.VideoProgress] = []

    result = media_module.generate_video(
        "a dog on a beach", session=session, duration=6, progress=ticks.append
    )

    assert result.path.is_file()
    assert result.path.read_bytes() == MP4_BYTES
    assert result.path.suffix == ".mp4"
    assert result.path.parent == session.workspace_root / ".kiwimatecoder" / "media"
    assert "a-dog-on-a-beach" in result.path.name
    assert result.job_id == "job-1"
    assert result.model == "vendor/video-1"
    assert result.cost == 0.25
    assert result.bytes == len(MP4_BYTES)
    assert [t.status for t in ticks][0] == "submitted"
    assert "in_progress" in [t.status for t in ticks]
    assert not list(result.path.parent.glob("*.kiwi.tmp"))

    payload = json.loads(api.posts()[0].content)
    assert payload == {
        "model": "vendor/video-1",
        "prompt": "a dog on a beach",
        "duration": 6,
        "resolution": "720p",
    }
    assert str(api.posts()[0].url) == f"{OPENROUTER_BASE}/videos"
    # Same host: every request is authenticated, including the download.
    assert all(r.headers["authorization"] == "Bearer sk-or-test" for r in api.requests)
    assert api.sleeps  # it waited between polls rather than hammering the API


def test_openrouter_exact_size_uses_size_field(session, monkeypatch):
    api = FakeVideoApi(monkeypatch, polls=("completed",))
    _enable_openrouter_video()

    media_module.generate_video("x", session=session, size="1280x720")

    assert json.loads(api.posts()[0].content)["size"] == "1280x720"
    assert "resolution" not in json.loads(api.posts()[0].content)


def test_video_first_frame_sent_as_frame_image(session, monkeypatch):
    api = FakeVideoApi(monkeypatch, polls=("completed",))
    _enable_openrouter_video()
    frame = session.workspace_root / "start.png"
    frame.write_bytes(PNG_BYTES)

    media_module.generate_video("it walks", session=session, first_frame=frame)

    entry = json.loads(api.posts()[0].content)["frame_images"][0]
    assert entry["frame_type"] == "first_frame"
    assert entry["image_url"]["url"].startswith("data:image/png;base64,")


def test_video_download_never_sends_key_to_other_hosts(session, monkeypatch):
    api = FakeVideoApi(
        monkeypatch,
        final={"unsigned_urls": ["https://cdn.example/out/job-1.mp4"]},
    )
    original = api.__call__

    def route(request: httpx.Request) -> httpx.Response:
        if request.url.host == "cdn.example":
            api.requests.append(request)
            return httpx.Response(200, content=MP4_BYTES, request=request)
        return original(request)

    transport = httpx.MockTransport(route)
    monkeypatch.setattr(
        media_module,
        "_client",
        lambda timeout: httpx.Client(timeout=timeout, transport=transport),
    )
    _enable_openrouter_video()

    result = media_module.generate_video("x", session=session)

    assert result.path.read_bytes() == MP4_BYTES
    cdn = [r for r in api.requests if r.url.host == "cdn.example"]
    assert len(cdn) == 1
    assert "authorization" not in cdn[0].headers


def test_offsite_polling_url_is_ignored(session, monkeypatch):
    api = FakeVideoApi(monkeypatch, polls=("completed",))
    original = api.__call__

    def route(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            api.requests.append(request)
            return httpx.Response(
                202,
                json={
                    "id": "job-1",
                    "status": "pending",
                    "polling_url": "https://evil.example/steal",
                },
                request=request,
            )
        return original(request)

    transport = httpx.MockTransport(route)
    monkeypatch.setattr(
        media_module,
        "_client",
        lambda timeout: httpx.Client(timeout=timeout, transport=transport),
    )
    _enable_openrouter_video()

    media_module.generate_video("x", session=session)

    assert all(r.url.host == "openrouter.ai" for r in api.requests)


def test_webm_download_gets_webm_extension(session, monkeypatch):
    FakeVideoApi(monkeypatch, polls=("completed",), video=WEBM_BYTES)
    _enable_openrouter_video()

    result = media_module.generate_video("x", session=session)

    assert result.path.suffix == ".webm"
    assert result.path.read_bytes() == WEBM_BYTES


# ---------------------------------------------------------------------------
# generate_video: OpenAI dialect
# ---------------------------------------------------------------------------


def test_openai_video_uses_multipart_and_content_endpoint(session, monkeypatch):
    api = FakeVideoApi(monkeypatch, openrouter=False, polls=("queued", "completed"))
    config.set_media(enabled=True, provider="openai", video_model="sora-2")
    config.set_key("openai", "sk-test")

    result = media_module.generate_video(
        "a paper boat", session=session, duration=8, size="1280x720"
    )

    post = api.posts()[0]
    assert post.headers["content-type"].startswith("multipart/form-data")
    body = post.content.decode()
    for field, value in (
        ("model", "sora-2"),
        ("prompt", "a paper boat"),
        ("seconds", "8"),
        ("size", "1280x720"),
    ):
        assert f'name="{field}"' in body
        assert value in body
    assert str(post.url) == "https://api.openai.com/v1/videos"
    paths = [r.url.path for r in api.requests if r.method == "GET"]
    assert paths[-1] == "/v1/videos/job-1/content"
    assert result.path.read_bytes() == MP4_BYTES


def test_openai_video_rejects_tier_sizes_and_first_frames(session, monkeypatch):
    FakeVideoApi(monkeypatch, openrouter=False)
    config.set_media(enabled=True, provider="openai", video_model="sora-2")
    config.set_key("openai", "sk-test")
    frame = session.workspace_root / "f.png"
    frame.write_bytes(PNG_BYTES)

    with pytest.raises(media_module.MediaError, match="exact video size"):
        media_module.generate_video("x", session=session, size="720p")
    with pytest.raises(media_module.MediaError, match="first-frame"):
        media_module.generate_video("x", session=session, first_frame=frame)


# ---------------------------------------------------------------------------
# generate_video: failures, waiting, resuming
# ---------------------------------------------------------------------------


def test_video_requires_a_model(session, monkeypatch):
    FakeVideoApi(monkeypatch)
    config.set_media(enabled=True, provider="openrouter")
    config.set_key("openrouter", "sk-or-test")

    with pytest.raises(media_module.MediaError, match="No video model chosen"):
        media_module.generate_video("x", session=session)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"duration": 0}, "duration"),
        ({"duration": 999}, "duration"),
        ({"size": "huge"}, "Video size"),
    ],
)
def test_video_validates_inputs_before_any_request(session, monkeypatch, kwargs, match):
    api = FakeVideoApi(monkeypatch)
    _enable_openrouter_video()

    with pytest.raises(media_module.MediaError, match=match):
        media_module.generate_video("x", session=session, **kwargs)
    assert api.requests == []


def test_video_disabled_offline_and_missing_key(session, monkeypatch):
    api = FakeVideoApi(monkeypatch)
    with pytest.raises(media_module.MediaError, match="disabled"):
        media_module.generate_video("x", session=session)

    config.set_media(enabled=True, provider="openrouter", video_model="m")
    with pytest.raises(media_module.MediaError, match="No API key"):
        media_module.generate_video("x", session=session)

    config.set_key("openrouter", "sk-or-test")
    config.set_network(offline=True)
    with pytest.raises(media_module.MediaError, match="offline"):
        media_module.generate_video("x", session=session)
    with pytest.raises(media_module.MediaError, match="prompt"):
        media_module.generate_video("  ", session=session)
    assert api.requests == []


def test_video_submit_error_is_wrapped_and_redacted(session, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, text="insufficient credits", request=request)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        media_module,
        "_client",
        lambda timeout: httpx.Client(timeout=timeout, transport=transport),
    )
    _enable_openrouter_video()

    with pytest.raises(media_module.MediaError, match="HTTP 402.*insufficient credits"):
        media_module.generate_video("x", session=session)


def test_video_failed_job_surfaces_reason(session, monkeypatch):
    FakeVideoApi(monkeypatch, polls=("in_progress", "failed"))
    _enable_openrouter_video()

    with pytest.raises(media_module.MediaError, match="failed: content policy") as info:
        media_module.generate_video("x", session=session)
    assert not isinstance(info.value, media_module.VideoJobError)


def test_video_timeout_keeps_job_id_for_resume(session, monkeypatch):
    api = FakeVideoApi(monkeypatch, polls=("in_progress",))
    _enable_openrouter_video()

    with pytest.raises(media_module.VideoJobError) as info:
        media_module.generate_video("x", session=session, max_wait=0)

    assert info.value.job_id == "job-1"
    assert "/video resume job-1" in str(info.value)

    # The job later finishes; resuming fetches it without a second POST.
    api.polls = ["completed"]
    api.poll_count = 0
    result = media_module.resume_video("job-1", session=session)
    assert result.path.read_bytes() == MP4_BYTES
    assert "video-job-1" in result.path.name
    assert len(api.posts()) == 1


def test_video_poll_tolerates_transient_errors(session, monkeypatch):
    api = FakeVideoApi(monkeypatch, polls=("http-503", "http-503", "completed"))
    _enable_openrouter_video()

    result = media_module.generate_video("x", session=session)

    assert result.path.is_file()
    assert api.poll_count == 3


def test_video_poll_gives_up_after_repeated_errors(session, monkeypatch):
    FakeVideoApi(monkeypatch, polls=("http-503",))
    _enable_openrouter_video()

    with pytest.raises(media_module.VideoJobError, match="Lost contact") as info:
        media_module.generate_video("x", session=session)
    assert info.value.job_id == "job-1"


def test_video_status_check_auth_failure_is_not_retried(session, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"id": "job-1"}, request=request)
        return httpx.Response(401, text="bad key", request=request)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(media_module, "_sleep", lambda _s: None)
    monkeypatch.setattr(
        media_module,
        "_client",
        lambda timeout: httpx.Client(timeout=timeout, transport=transport),
    )
    _enable_openrouter_video()

    with pytest.raises(media_module.MediaError, match="HTTP 401"):
        media_module.generate_video("x", session=session)


def test_video_download_refuses_non_video_and_cleans_up(session, monkeypatch):
    FakeVideoApi(monkeypatch, polls=("completed",), video=b'{"error": "oops"}' + b"\x00" * 8)
    _enable_openrouter_video()

    with pytest.raises(media_module.VideoJobError, match="not a video"):
        media_module.generate_video("x", session=session)

    media_dir = session.workspace_root / ".kiwimatecoder" / "media"
    assert list(media_dir.glob("*")) == []


def test_video_download_size_limit(session, monkeypatch):
    monkeypatch.setattr(media_module, "MAX_VIDEO_BYTES", 16)
    FakeVideoApi(monkeypatch, polls=("completed",))
    _enable_openrouter_video()

    with pytest.raises(media_module.VideoJobError, match="limit"):
        media_module.generate_video("x", session=session)

    media_dir = session.workspace_root / ".kiwimatecoder" / "media"
    assert list(media_dir.glob("*")) == []


def test_video_resume_rejects_unsafe_job_ids(session, monkeypatch):
    api = FakeVideoApi(monkeypatch)
    _enable_openrouter_video()

    for bad in ("", "../etc/passwd", "a b", "x" * 200):
        with pytest.raises(media_module.MediaError, match="job id"):
            media_module.resume_video(bad, session=session)
    assert api.requests == []


def test_video_progress_callback_errors_do_not_kill_the_job(session, monkeypatch):
    FakeVideoApi(monkeypatch)
    _enable_openrouter_video()

    def boom(_tick) -> None:
        raise RuntimeError("display glitch")

    result = media_module.generate_video("x", session=session, progress=boom)
    assert result.path.is_file()


# ---------------------------------------------------------------------------
# Images on OpenRouter, readiness, outputs
# ---------------------------------------------------------------------------


def test_openrouter_images_use_the_images_route(session, monkeypatch):
    import base64

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(
            200,
            json={"data": [{"b64_json": base64.b64encode(PNG_BYTES).decode()}]},
            request=request,
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        media_module,
        "_client",
        lambda timeout: httpx.Client(timeout=timeout, transport=transport),
    )
    config.set_media(enabled=True, provider="openrouter", model="vendor/img")
    config.set_key("openrouter", "sk-or-test")

    media_module.generate_image("a cat", session=session)

    assert seen == [f"{OPENROUTER_BASE}/images"]


def test_readiness_reports_what_is_missing():
    assert any("disabled" in p for p in media_module.readiness())

    config.set_media(enabled=True, provider="openrouter")
    assert any("No API key" in p for p in media_module.readiness())

    config.set_key("openrouter", "sk-or-test")
    assert media_module.readiness() == []
    assert media_module.readiness(video=True) == [media_module.NO_VIDEO_MODEL_MESSAGE]

    config.set_media(video_model="vendor/video-1")
    assert media_module.readiness(video=True) == []


def test_list_outputs_newest_first_and_filtered(session):
    media_dir = session.workspace_root / ".kiwimatecoder" / "media"
    media_dir.mkdir(parents=True)
    old, new, note = media_dir / "a.png", media_dir / "b.mp4", media_dir / "c.txt"
    for path in (old, new, note):
        path.write_bytes(b"x")
    now = time.time()
    os.utime(old, (now - 100, now - 100))
    os.utime(new, (now, now))

    assert media_module.list_outputs(session) == [new, old]
    assert media_module.list_outputs(session, limit=1) == [new]


def test_format_bytes():
    assert media_module.format_bytes(512) == "512 B"
    assert media_module.format_bytes(1536) == "1.5 KB"
    assert media_module.format_bytes(5 * 1024 * 1024) == "5.0 MB"


# ---------------------------------------------------------------------------
# generate_video tool
# ---------------------------------------------------------------------------


def test_generate_video_tool_is_approval_gated():
    tool = tools.get_tool("generate_video")
    assert tool is not None
    assert tool.runs is True
    assert tool.needs_approval is True


def test_generate_video_preview_warns_about_cost(session):
    config.set_media(
        provider="openrouter", video_model="vendor/video-1", video_duration=7
    )
    preview = tools.preview(
        "generate_video", {"prompt": "a fox", "first_frame": "start.png"}, session
    )

    assert preview is not None
    for expected in ("openrouter", "vendor/video-1", "7s", "a fox", "start.png", "billed per second"):
        assert expected in preview


def test_generate_video_preview_for_resume(session):
    preview = tools.preview("generate_video", {"resume_job_id": "job-9"}, session)
    assert preview is not None and "job-9" in preview


def test_generate_video_tool_disabled_returns_guidance(session, monkeypatch):
    api = FakeVideoApi(monkeypatch)

    result = tools.dispatch("generate_video", {"prompt": "x"}, session)

    assert not result.ok
    assert "/config media enable on" in result.content
    assert api.requests == []


def test_generate_video_tool_saves_file(session, monkeypatch):
    FakeVideoApi(monkeypatch, final={"usage": {"cost": 0.5}})
    _enable_openrouter_video()

    result = tools.dispatch(
        "generate_video", {"prompt": "a fox", "duration": 4}, session
    )

    assert result.ok, result.content
    assert f"Generated video saved to {MEDIA_DIR}{os.sep}" in result.content
    assert "job job-1" in result.content
    assert "$0.50" in result.content


def test_generate_video_tool_validates_arguments(session, monkeypatch):
    FakeVideoApi(monkeypatch)
    _enable_openrouter_video()

    assert not tools.dispatch("generate_video", {}, session).ok
    bad_duration = tools.dispatch(
        "generate_video", {"prompt": "x", "duration": "soon"}, session
    )
    assert not bad_duration.ok and "duration" in bad_duration.content
    missing_frame = tools.dispatch(
        "generate_video", {"prompt": "x", "first_frame": "nope.png"}, session
    )
    assert not missing_frame.ok and "not found" in missing_frame.content.lower()


def test_generate_video_tool_timeout_message_names_resume_id(session, monkeypatch):
    FakeVideoApi(monkeypatch, polls=("in_progress",))
    monkeypatch.setattr(media_module, "VIDEO_MAX_WAIT", 0)
    _enable_openrouter_video()

    result = tools.dispatch("generate_video", {"prompt": "x"}, session)

    assert not result.ok
    assert "job-1" in result.content


def test_generate_video_tool_resumes_a_job(session, monkeypatch):
    api = FakeVideoApi(monkeypatch, polls=("completed",))
    _enable_openrouter_video()

    result = tools.dispatch("generate_video", {"resume_job_id": "job-1"}, session)

    assert result.ok, result.content
    assert api.posts() == []


# ---------------------------------------------------------------------------
# /video
# ---------------------------------------------------------------------------


def test_video_command_disabled_prints_guidance(session):
    console = _console()
    dispatch("/video a fox", session, console)
    assert "disabled" in _output(console)


def test_video_command_without_prompt_shows_usage_and_settings(session):
    _enable_openrouter_video()
    console = _console()

    dispatch("/video", session, console)

    output = _output(console)
    assert "Usage: /video" in output
    assert "/video resume" in output
    assert "vendor/video-1" in output
    assert "billed per second" in output


def test_video_command_requires_a_model_before_any_request(session, monkeypatch):
    api = FakeVideoApi(monkeypatch)
    config.set_media(enabled=True, provider="openrouter")
    config.set_key("openrouter", "sk-or-test")
    console = _console()

    dispatch("/video a fox", session, console)

    assert "No video model chosen" in _output(console)
    assert api.requests == []


def test_video_command_generates_and_reports(session, monkeypatch):
    api = FakeVideoApi(monkeypatch, final={"usage": {"cost": 0.1}})
    _enable_openrouter_video()
    console = _console()

    dispatch("/video --duration 8 --size 1080p a fox's den, at dawn", session, console)

    payload = json.loads(api.posts()[0].content)
    assert payload["duration"] == 8
    assert payload["resolution"] == "1080p"
    assert payload["prompt"] == "a fox's den, at dawn"  # quotes survive
    output = _output(console)
    assert "Job job-1 submitted" in output
    assert "Saved" in output
    assert f"{MEDIA_DIR}{os.sep}" in output
    assert "$0.10" in output
    assert "billed per second" in output


def test_video_command_uses_first_frame_image(session, monkeypatch):
    api = FakeVideoApi(monkeypatch, polls=("completed",))
    _enable_openrouter_video()
    (session.workspace_root / "start.png").write_bytes(PNG_BYTES)
    console = _console()

    dispatch("/video --image start.png it begins to walk", session, console)

    assert "frame_images" in json.loads(api.posts()[0].content)
    assert "Saved" in _output(console)


@pytest.mark.parametrize(
    "line, expected",
    [
        ("/video --bogus 1 a fox", "Unknown option --bogus"),
        ("/video --duration", "needs a value"),
        ("/video --duration soon a fox", "whole seconds"),
        ("/video --image missing.png a fox", "not found"),
        ("/video resume", "Usage: /video resume"),
    ],
)
def test_video_command_input_errors(session, monkeypatch, line, expected):
    api = FakeVideoApi(monkeypatch)
    _enable_openrouter_video()
    console = _console()

    dispatch(line, session, console)

    assert expected in _output(console)
    assert api.requests == []


def test_video_command_ctrl_c_prints_resume_hint(session, monkeypatch):
    api = FakeVideoApi(monkeypatch, polls=("in_progress",))

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(media_module, "_sleep", interrupt)
    _enable_openrouter_video()
    console = _console()

    result = dispatch("/video a fox", session, console)

    assert result == CommandResult.CONTINUE
    output = _output(console)
    assert "Stopped waiting" in output
    assert "/video resume job-1" in output
    assert len(api.posts()) == 1


def test_video_command_resume(session, monkeypatch):
    api = FakeVideoApi(monkeypatch, polls=("completed",))
    _enable_openrouter_video()
    console = _console()

    dispatch("/video resume job-1", session, console)

    assert "Saved" in _output(console)
    assert api.posts() == []


def test_video_command_timeout_error_mentions_resume(session, monkeypatch):
    FakeVideoApi(monkeypatch, polls=("in_progress",))
    monkeypatch.setattr(media_module, "VIDEO_MAX_WAIT", 0)
    _enable_openrouter_video()
    console = _console()

    dispatch("/video a fox", session, console)

    assert "/video resume job-1" in _output(console)


# ---------------------------------------------------------------------------
# /image options
# ---------------------------------------------------------------------------


def _fake_generate(calls: list[dict]):
    def fake(prompt, *, session, size=None, model=None, provider_id=None):
        calls.append({"prompt": prompt, "size": size, "model": model})
        target = session.workspace_root / ".kiwimatecoder" / "media" / "gen.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(PNG_BYTES)
        return media_module.MediaResult(
            path=target, revised_prompt=None, model=model or "m", bytes=len(PNG_BYTES)
        )

    return fake


def test_image_command_accepts_size_and_model_options(session, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(media_module, "generate_image", _fake_generate(calls))
    config.set_media(enabled=True)
    console = _console()

    dispatch('/image --size 512x512 --model=gpt-image-1 a cat\'s "hat"', session, console)

    assert calls == [
        {"prompt": 'a cat\'s "hat"', "size": "512x512", "model": "gpt-image-1"}
    ]
    output = _output(console)
    assert os.path.join(MEDIA_DIR, "gen.png") in output  # workspace-relative
    assert "attached to your next message" in output
    assert len(session.pending_images) == 1


def test_image_command_double_dash_ends_options(session, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(media_module, "generate_image", _fake_generate(calls))
    config.set_media(enabled=True)

    dispatch("/image -- --a neon sign", session, _console())

    assert calls[0]["prompt"] == "--a neon sign"


def test_image_command_unknown_option_and_usage(session, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(media_module, "generate_image", _fake_generate(calls))
    config.set_media(enabled=True)

    console = _console()
    dispatch("/image --colour red a cat", session, console)
    assert "Unknown option --colour" in _output(console)

    console = _console()
    dispatch("/image", session, console)
    output = _output(console)
    assert "Usage: /image" in output
    assert "gpt-image-1" in output  # shows the current settings
    assert calls == []


def test_image_command_reports_ctrl_c(session, monkeypatch):
    def interrupted(prompt, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(media_module, "generate_image", interrupted)
    config.set_media(enabled=True)
    console = _console()

    dispatch("/image a cat", session, console)

    assert "Cancelled" in _output(console)


# ---------------------------------------------------------------------------
# /media and /config media
# ---------------------------------------------------------------------------


def test_media_command_shows_readiness_and_recent_files(session):
    media_dir = session.workspace_root / ".kiwimatecoder" / "media"
    media_dir.mkdir(parents=True)
    (media_dir / "20260101_000000-fox.png").write_bytes(PNG_BYTES)
    config.set_media(enabled=True, provider="openrouter")
    console = _console()

    dispatch("/media", session, console)

    output = _output(console)
    assert "Images: No API key" in output
    assert "Video: No API key" in output
    assert "20260101_000000-fox.png" in output
    assert "@<path>" in output


def test_media_command_ready_and_empty(session):
    _enable_openrouter_video()
    console = _console()

    dispatch("/media", session, console)

    output = _output(console)
    assert "Images: ready" in output
    assert "Video: ready" in output
    assert "No generated media" in output


def test_media_command_when_disabled_says_so_once(session):
    console = _console()

    dispatch("/media", session, console)

    output = _output(console)
    assert output.count("/config media enable on") == 1
    assert "Disabled" in output


def test_media_command_usage_errors(session):
    for line in ("/media nonsense", "/media list many"):
        console = _console()
        dispatch(line, session, console)
        assert "Usage: /media" in _output(console)


def test_config_media_video_options_via_slash(session):
    console = _console()

    dispatch("/config media video-model vendor/video-2", session, console)
    dispatch("/config media duration 12", session, console)
    dispatch("/config media video-size 1080p", session, console)
    dispatch("/config media output-dir art/gen", session, console)

    settings = config.get_media()
    assert settings["video_model"] == "vendor/video-2"
    assert settings["video_duration"] == 12
    assert settings["video_size"] == "1080p"
    assert settings["output_dir"] == "art/gen"
    assert "vendor/video-2" in _output(console)

    dispatch("/config media video-model none", session, console)
    dispatch("/config media video-size default", session, console)
    assert config.get_media()["video_model"] == ""
    assert config.get_media()["video_size"] == ""


def test_config_media_bad_values_report_errors(session):
    console = _console()

    dispatch("/config media duration 0", session, console)
    dispatch("/config media video-size huge", session, console)
    dispatch("/config media output-dir ../escape", session, console)
    dispatch("/config media duration", session, console)

    output = _output(console)
    assert "duration must be whole seconds" in output
    assert "video size must be" in output
    assert "relative path" in output
    assert "Usage: /config media duration" in output
    assert config.get_media()["video_duration"] == 5


def test_enabling_media_mentions_video_model(session):
    console = _console()

    dispatch("/config media enable on", session, console)

    output = _output(console)
    assert "/video" in output
    assert "video-model" in output


def test_new_commands_are_listed_in_help_and_completions(session):
    console = _console()
    dispatch("/help", session, console)
    output = _output(console)
    assert "/video" in output
    assert "/media" in output

    names = {name for name, _ in slash_command_completions("")}
    assert {"/image", "/video", "/media"} <= names
