"""Redacted, shareable session bundles (local files, or a secret GitHub gist).

Sharing is deliberately local-first: a bundle is a redacted JSON snapshot of a
conversation that can be committed, pasted into a chat, or uploaded to a GitHub
gist through the user's own ``gh`` login. There is no hosted service. Tool
output is excluded by default and every string passes through
:func:`kiwimatecoder.redaction.redact`, so API keys never leave the machine.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from kiwimatecoder import forge
from kiwimatecoder.redaction import redact
from kiwimatecoder.session import Session, save_session

SHARE_VERSION = 1
MAX_SHARE_BYTES = 256 * 1024
MAX_MESSAGE_CHARS = 16_000
MAX_TOOL_CHARS = 2048
OMITTED_TOOL_OUTPUT = "[tool output omitted]"
OMITTED_IMAGE = "[image omitted]"
SHARES_DIR_NAME = "shares"


class ShareError(Exception):
    """Raised when a share bundle cannot be read, validated, or imported."""


def _truncate(text: str, limit: int) -> str:
    """Clip ``text`` to ``limit`` chars, noting how much was dropped."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[truncated {len(text) - limit} chars]"


def _share_text(value: Any, limit: int = MAX_MESSAGE_CHARS) -> str:
    """Redact and clip one text value for a share bundle."""
    return _truncate(redact(str(value)), limit)


def _share_arguments(arguments: Any) -> str:
    """Keep a tool call's arguments, redacted and clipped (names stay visible)."""
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        raw = arguments
    else:
        raw = json.dumps(arguments, default=str, sort_keys=True)
    return _truncate(redact(raw), MAX_TOOL_CHARS)


def _share_block(block: Any, include_tool_output: bool) -> dict[str, Any]:
    """Convert one content block, dropping images and (by default) tool output."""
    if isinstance(block, str):
        return {"type": "text", "text": _share_text(block)}
    if not isinstance(block, dict):
        return {"type": "text", "text": _share_text(block)}
    block_type = str(block.get("type") or "text")
    if block_type == "tool_result":
        content = block.get("content")
        if isinstance(content, str):
            text = content
        else:
            text = "" if content is None else json.dumps(content, default=str)
        return {
            "type": "tool_result",
            "tool_use_id": str(block.get("tool_use_id") or ""),
            "content": (
                _truncate(redact(text), MAX_TOOL_CHARS)
                if include_tool_output
                else OMITTED_TOOL_OUTPUT
            ),
        }
    if block_type in {"image", "image_url", "input_image"}:
        return {"type": "text", "text": OMITTED_IMAGE}
    if block_type == "text":
        return {"type": "text", "text": _share_text(block.get("text"))}
    return {"type": "text", "text": _share_text(json.dumps(block, default=str))}


def _share_message(
    message: dict[str, Any], include_tool_output: bool
) -> dict[str, Any]:
    """Build one redacted message, preserving roles and tool call names."""
    role = str(message.get("role") or "unknown")
    shared: dict[str, Any] = {"role": role}
    content = message.get("content")
    if isinstance(content, list):
        shared["content"] = [
            _share_block(block, include_tool_output) for block in content
        ]
    elif role == "tool":
        shared["content"] = (
            _share_text("" if content is None else content, MAX_TOOL_CHARS)
            if include_tool_output
            else OMITTED_TOOL_OUTPUT
        )
    else:
        shared["content"] = _share_text("" if content is None else content)
    if message.get("tool_call_id"):
        shared["tool_call_id"] = str(message["tool_call_id"])
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        kept: list[dict[str, Any]] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                function = {}
            kept.append(
                {
                    "id": str(call.get("id") or ""),
                    "type": str(call.get("type") or "function"),
                    "function": {
                        "name": str(function.get("name") or ""),
                        "arguments": _share_arguments(function.get("arguments")),
                    },
                }
            )
        if kept:
            shared["tool_calls"] = kept
    return shared


def _content_text(message: dict[str, Any]) -> str:
    """Extract plain text from one message's content (for summary/slug)."""
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return "" if content is None else str(content)


def _summary(messages: list[dict[str, Any]]) -> str:
    """The first non-empty user message, redacted and clipped for display."""
    for message in messages:
        if str(message.get("role") or "") != "user":
            continue
        cleaned = redact(" ".join(_content_text(message).split()))
        if cleaned:
            return _truncate(cleaned, 200)
    return ""


def _slug(session: Session) -> str:
    """A filename-safe slug derived from the first user message."""
    for message in session.messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        words = re.findall(r"[a-z0-9]+", redact(_content_text(message)).lower())
        if words:
            return "-".join(words[:6])[:48].strip("-") or "session"
    return "session"


def _encoded_size(bundle: dict[str, Any]) -> int:
    return len(json.dumps(bundle, ensure_ascii=False, default=str).encode("utf-8"))


def _fit_bundle(bundle: dict[str, Any]) -> None:
    """Drop oldest messages until the serialized bundle fits the size cap."""
    messages: list[dict[str, Any]] = bundle["messages"]
    if _encoded_size(bundle) <= MAX_SHARE_BYTES:
        return
    dropped = 0
    # Set the real markers before measuring so the final size is within cap.
    bundle["truncated"] = True
    bundle["truncated_messages"] = 0
    while messages and _encoded_size(bundle) > MAX_SHARE_BYTES:
        messages.pop(0)
        dropped += 1
        bundle["truncated_messages"] = dropped
    # Never start with an orphaned tool result (some providers reject it).
    while messages and messages[0].get("role") == "tool":
        messages.pop(0)
        dropped += 1
        bundle["truncated_messages"] = dropped


def build_share(session: Session, *, include_tool_output: bool = False) -> dict[str, Any]:
    """Build a redacted share bundle for ``session``.

    Tool results are replaced with a placeholder unless ``include_tool_output``
    is set, in which case each result is redacted and clipped to 2 KB; tool
    call arguments are always redacted and clipped. The whole bundle is capped
    at ~256 KB: when the conversation is larger the oldest messages are dropped
    and ``truncated``/``truncated_messages`` record it.
    """
    messages = [
        _share_message(message, include_tool_output)
        for message in session.messages
        if isinstance(message, dict)
    ]
    bundle: dict[str, Any] = {
        "version": SHARE_VERSION,
        "created_at": datetime.datetime.now().isoformat(),
        "provider": session.provider_id,
        "model": session.model,
        "mode": session.mode.value,
        "summary": _summary(messages),
        "messages": messages,
        "todos": [dict(todo) for todo in session.todos if isinstance(todo, dict)],
        "stats": {
            "messages": len(messages),
            "prompt_tokens": session.prompt_tokens,
            "completion_tokens": session.completion_tokens,
            "total_tokens": session.total_tokens,
        },
        "truncated": False,
        "truncated_messages": 0,
    }
    _fit_bundle(bundle)
    return bundle


def default_share_path(session: Session) -> Path:
    """Default bundle path: ``<workspace>/.kiwimatecoder/shares/<ts>-<slug>.json``."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        session.workspace_root
        / ".kiwimatecoder"
        / SHARES_DIR_NAME
        / f"{stamp}-{_slug(session)}.share.json"
    )


def write_share(
    session: Session,
    dest: str | Path | None = None,
    include_tool_output: bool = False,
) -> Path:
    """Write a redacted bundle atomically and return the file path.

    ``dest`` may be a directory (the default share name is used inside it) or a
    file path; when omitted the bundle lands in the session workspace under
    ``.kiwimatecoder/shares/``.
    """
    bundle = build_share(session, include_tool_output=include_tool_output)
    if dest is None:
        target = default_share_path(session)
    else:
        target = Path(dest).expanduser()
        if target.is_dir():
            target = target / default_share_path(session).name
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(bundle, indent=2, ensure_ascii=False) + "\n"
    temp = target.with_name(f".{target.name}.tmp")
    temp.write_text(payload, encoding="utf-8")
    os.replace(temp, target)
    return target


def load_share(path: str | Path) -> dict[str, Any]:
    """Load and validate a share bundle, tolerating missing optional fields.

    Raises :class:`ShareError` for a missing file, unreadable data, a document
    that is not an object, an unsupported version, or a missing message list.
    """
    target = Path(path).expanduser()
    if not target.is_file():
        raise ShareError(f"Share file not found: {path}")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ShareError(f"Cannot read share file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ShareError(f"Share file {path} must be a JSON object.")
    version = data.get("version")
    if version != SHARE_VERSION:
        raise ShareError(
            f"Unsupported share version {version!r} (expected {SHARE_VERSION})."
        )
    messages = data.get("messages")
    if not isinstance(messages, list):
        raise ShareError(f"Share file {path} has no message list.")
    stats = data.get("stats")
    try:
        truncated_messages = int(data.get("truncated_messages") or 0)
    except (TypeError, ValueError):
        truncated_messages = 0
    return {
        "version": SHARE_VERSION,
        "created_at": str(data.get("created_at") or ""),
        "provider": str(data.get("provider") or ""),
        "model": str(data.get("model") or ""),
        "mode": str(data.get("mode") or "ask"),
        "summary": str(data.get("summary") or ""),
        "messages": [item for item in messages if isinstance(item, dict)],
        "todos": [item for item in (data.get("todos") or []) if isinstance(item, dict)],
        "stats": stats if isinstance(stats, dict) else {},
        "truncated": bool(data.get("truncated", False)),
        "truncated_messages": truncated_messages,
    }


def import_share(path: str | Path, name: str | None = None) -> Path:
    """Create a normal saved session from a share bundle and return its path.

    The reconstructed session can be resumed with ``/load`` or
    ``--resume``; a bundle without a provider falls back to the configured
    primary provider so it stays loadable.
    """
    bundle = load_share(path)
    provider_id = bundle["provider"]
    if not provider_id:
        from kiwimatecoder.config import get_selected_provider_id

        provider_id = get_selected_provider_id()
    session = Session.from_dict(
        {
            "provider_id": provider_id,
            "model": bundle["model"],
            "mode": bundle["mode"],
            "messages": bundle["messages"],
            "todos": bundle["todos"],
            "active_provider_ids": [provider_id],
        }
    )
    if not name:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"shared_{stamp}"
    return save_session(session, name)


def gist_args(path: str | Path) -> list[str]:
    """Return the exact ``gh gist create`` argv for a share file (secret gist)."""
    return [
        "gh",
        "gist",
        "create",
        "--public=false",
        "--desc",
        "kiwimatecoder session share",
        str(path),
    ]


def share_to_gist(
    session: Session,
    *,
    include_tool_output: bool = False,
    dest: str | Path | None = None,
) -> tuple[bool, str]:
    """Write a bundle and upload it as a secret gist via the user's ``gh`` CLI.

    Returns ``(ok, url-or-error)``; failures are messages, never exceptions, so
    callers can print them. ``gh`` must be installed and authenticated; no
    token is read or stored by KiwiMateCoder.
    """
    if not forge.cli_available("gh"):
        return (
            False,
            "gh is not installed or not on PATH. Install the GitHub CLI and "
            "run `gh auth login` first.",
        )
    path = write_share(session, dest=dest, include_tool_output=include_tool_output)
    try:
        proc = subprocess.run(
            gist_args(path),
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except FileNotFoundError:
        return False, "gh is not installed or not on PATH."
    except subprocess.TimeoutExpired:
        return False, "gh gist create timed out after 60s."
    output = (proc.stdout or "").strip()
    if proc.returncode != 0:
        detail = (proc.stderr or output or "gh gist create failed").strip()
        return False, detail
    url = output.splitlines()[0].strip() if output else ""
    return True, url or "Gist created."
