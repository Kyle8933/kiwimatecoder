"""Image tools: attach an image to the conversation, or generate media.

``view_image`` attaches an existing workspace file (read-only). The
``generate_image`` and ``generate_video`` tools call the configured media
provider, which costs money, so they are approval-gated (``runs=True``) and
refuse with enable guidance while media generation is turned off.
"""

from __future__ import annotations

import base64
from typing import Any

from kiwimatecoder import config, images, media
from kiwimatecoder.redaction import redact
from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult
from kiwimatecoder.tools.paths import PathError, display_path, resolve_for_read


def _view_image(args: dict[str, Any], session: Session) -> ToolResult:
    path = str(args.get("path") or "").strip()
    if not path:
        return ToolResult.error("'path' is required")
    try:
        resolved = resolve_for_read(
            path,
            session.workspace_root,
            trusted=bool(getattr(session, "trusted_workspace", False)),
        )
    except PathError as exc:
        return ToolResult.error(str(exc))
    if not resolved.is_file():
        return ToolResult.error(f"File not found: {path}")

    vision = config.get_vision()
    limit = int(vision["max_images_per_turn"])
    if len(session.pending_images) >= limit:
        return ToolResult.error(
            f"Image limit reached ({limit} per turn). Send these images first, "
            "then attach more in the next message."
        )

    try:
        entry = images.encode_image(resolved, int(vision["max_image_bytes"]))
    except (ValueError, OSError) as exc:
        return ToolResult.error(str(exc))

    session.pending_images.append(entry)
    dimensions = images.image_dimensions(base64.b64decode(entry["data"]))
    label = f"{path} ({dimensions[0]}x{dimensions[1]})" if dimensions else path
    return ToolResult(content=f"Attached {label}")


view_image_tool = FunctionTool(
    name="view_image",
    description=(
        "Attach an image file (.png, .jpg, .jpeg, .gif, .webp) from the "
        "workspace so the model can see it. Read-only; the image is sent with "
        "the next request. Use it for screenshots, diagrams, and design mockups."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the image, relative to the workspace root.",
            },
        },
        "required": ["path"],
    },
    func=_view_image,
)


def _generate_image(args: dict[str, Any], session: Session) -> ToolResult:
    if not config.get_media()["enabled"]:
        return ToolResult.error(
            "Image generation is disabled. Enable it with "
            "`/config media enable on` (or `config media enable on`)."
        )
    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        return ToolResult.error("'prompt' is required")
    size = args.get("size")
    model = args.get("model")
    try:
        result = media.generate_image(
            prompt,
            session=session,
            size=str(size).strip() if size else None,
            model=str(model).strip() if model else None,
        )
    except media.MediaError as exc:
        return ToolResult.error(str(exc))
    note = media.attach_result(result, session)
    location = display_path(result.path, session.workspace_root)
    content = (
        f"Generated image saved to {location} ({result.bytes:,} bytes, "
        f"model {result.model})"
    )
    if note:
        content += f". {note}"
    return ToolResult(content=content)


def generate_image_preview(args: dict[str, Any], session: Session) -> str:
    """Render the approval preview: provider, model, size, and prompt."""
    del session
    settings = config.get_media()
    prompt = redact(str(args.get("prompt") or "")).strip()
    if len(prompt) > 300:
        prompt = prompt[:297] + "..."
    model = str(args.get("model") or settings["model"]).strip()
    size = str(args.get("size") or settings["size"]).strip()
    return (
        f"Provider: {settings['provider']}\n"
        f"Model: {model}\n"
        f"Size: {size}\n"
        f"Output: {settings['output_dir']}\n"
        f"Prompt: {prompt or '(missing)'}\n\n"
        "This calls a paid image API."
    )


generate_image_tool = FunctionTool(
    name="generate_image",
    description=(
        "Generate an image from a text prompt through the configured media "
        "provider and attach it to the conversation. Costs money and requires "
        "media generation to be enabled (`/config media enable on`). Use it "
        "for mockups, icons, diagrams, and illustrations."
    ),
    parameters={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "A detailed description of the image to generate.",
            },
            "size": {
                "type": "string",
                "description": "Image size as WxH (default: configured size).",
            },
            "model": {
                "type": "string",
                "description": "Image model override (default: configured model).",
            },
        },
        "required": ["prompt"],
    },
    func=_generate_image,
    runs=True,
)


def _generate_video(args: dict[str, Any], session: Session) -> ToolResult:
    if not config.get_media()["enabled"]:
        return ToolResult.error(
            "Media generation is disabled. Enable it with "
            "`/config media enable on` (or `config media enable on`)."
        )
    resume_id = str(args.get("resume_job_id") or "").strip()
    prompt = str(args.get("prompt") or "").strip()
    if not resume_id and not prompt:
        return ToolResult.error("'prompt' is required (or 'resume_job_id')")

    first_frame = None
    frame_arg = str(args.get("first_frame") or "").strip()
    if frame_arg and not resume_id:
        try:
            first_frame = resolve_for_read(
                frame_arg,
                session.workspace_root,
                trusted=bool(getattr(session, "trusted_workspace", False)),
            )
        except PathError as exc:
            return ToolResult.error(str(exc))
        if not first_frame.is_file():
            return ToolResult.error(f"File not found: {frame_arg}")

    duration = args.get("duration")
    size = args.get("size")
    model = args.get("model")
    try:
        if resume_id:
            result = media.resume_video(resume_id, session=session)
        else:
            result = media.generate_video(
                prompt,
                session=session,
                model=str(model).strip() if model else None,
                duration=int(duration) if duration not in (None, "") else None,
                size=str(size).strip() if size else None,
                first_frame=first_frame,
            )
    except (TypeError, ValueError):
        return ToolResult.error("'duration' must be a whole number of seconds")
    except media.MediaError as exc:
        return ToolResult.error(str(exc))
    location = display_path(result.path, session.workspace_root)
    content = (
        f"Generated video saved to {location} ({media.format_bytes(result.bytes)}, "
        f"model {result.model}, job {result.job_id}"
    )
    if result.cost is not None:
        content += f", cost ${result.cost:.2f}"
    return ToolResult(content=content + ")")


def generate_video_preview(args: dict[str, Any], session: Session) -> str:
    """Render the approval preview: provider, model, length, size, and prompt."""
    del session
    settings = config.get_media()
    resume_id = str(args.get("resume_job_id") or "").strip()
    if resume_id:
        return (
            f"Provider: {settings['provider']}\n"
            f"Resume video job: {resume_id}\n\n"
            "This fetches a video job that was already started (and billed)."
        )
    prompt = redact(str(args.get("prompt") or "")).strip()
    if len(prompt) > 300:
        prompt = prompt[:297] + "..."
    model = str(args.get("model") or settings["video_model"]).strip()
    duration = args.get("duration") or settings["video_duration"]
    size = str(args.get("size") or settings["video_size"] or "provider default").strip()
    lines = [
        f"Provider: {settings['provider']}",
        f"Model: {model or '(none chosen)'}",
        f"Length: {duration}s",
        f"Size: {size}",
    ]
    if args.get("first_frame"):
        lines.append(f"First frame: {args['first_frame']}")
    lines += [
        f"Output: {settings['output_dir']}",
        f"Prompt: {prompt or '(missing)'}",
        "",
        "This calls a paid video API. Video is billed per second of footage "
        "and a clip can cost anywhere from cents to several dollars; it may "
        "take a few minutes to render.",
    ]
    return "\n".join(lines)


generate_video_tool = FunctionTool(
    name="generate_video",
    description=(
        "Generate a short video from a text prompt (optionally animating a "
        "workspace image as the first frame) through the configured media "
        "provider and save it under the media output directory. Slow (minutes) "
        "and billed per second, so use it only when asked for a video. Requires "
        "media generation to be enabled and a video model to be chosen "
        "(`/config media video-model <id>`). If a job times out, call again "
        "with its `resume_job_id` instead of regenerating."
    ),
    parameters={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "A detailed description of the video to generate.",
            },
            "duration": {
                "type": "integer",
                "description": "Length in seconds (default: configured duration).",
            },
            "size": {
                "type": "string",
                "description": (
                    "WxH (e.g. 1280x720) or a tier such as 720p "
                    "(default: configured size, else the provider's default)."
                ),
            },
            "model": {
                "type": "string",
                "description": "Video model override (default: configured model).",
            },
            "first_frame": {
                "type": "string",
                "description": (
                    "Workspace path to an image to animate as the first frame "
                    "(image-to-video)."
                ),
            },
            "resume_job_id": {
                "type": "string",
                "description": (
                    "Job id from an earlier timed-out call; fetches that job "
                    "instead of starting (and paying for) a new one."
                ),
            },
        },
        "required": [],
    },
    func=_generate_video,
    runs=True,
)
