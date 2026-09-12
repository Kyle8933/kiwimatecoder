"""view_image tool: attach an image from the workspace to the conversation."""

from __future__ import annotations

import base64
from typing import Any

from kiwimatecoder import config, images
from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult
from kiwimatecoder.tools.paths import PathError, resolve_for_read


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
