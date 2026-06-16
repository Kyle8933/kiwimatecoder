"""read_file tool: read a text file from the workspace."""

from __future__ import annotations

from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult
from kiwimatecoder.tools.paths import PathError, resolve_in_workspace

MAX_BYTES = 256 * 1024


def _is_binary(sample: bytes) -> bool:
    return b"\x00" in sample


def _read_file(args: dict, session: Session) -> ToolResult:
    path = args.get("path")
    if not path:
        return ToolResult.error("'path' is required")
    try:
        resolved = resolve_in_workspace(path, session.workspace_root)
    except PathError as exc:
        return ToolResult.error(str(exc))

    if not resolved.exists():
        return ToolResult.error(f"File not found: {path}")
    if resolved.is_dir():
        return ToolResult.error(f"'{path}' is a directory, use list_dir")

    data = resolved.read_bytes()
    if _is_binary(data[:1024]):
        return ToolResult.error(f"'{path}' appears to be a binary file")

    text = data.decode("utf-8", "replace")
    lines = text.splitlines()

    offset = int(args.get("offset", 0) or 0)
    limit = args.get("limit")
    if offset or limit is not None:
        end = offset + int(limit) if limit is not None else len(lines)
        lines = lines[offset:end]

    truncated = False
    body = "\n".join(lines)
    if len(body.encode("utf-8")) > MAX_BYTES:
        body = body.encode("utf-8")[:MAX_BYTES].decode("utf-8", "ignore")
        truncated = True

    numbered = "\n".join(
        f"{offset + i + 1}\t{line}" for i, line in enumerate(body.splitlines())
    )
    if truncated:
        numbered += "\n... [truncated: file exceeds 256KB]"
    return ToolResult(content=numbered or "[empty file]")


read_file_tool = FunctionTool(
    name="read_file",
    description=(
        "Read the contents of a text file in the workspace. Returns the file "
        "with 1-based line numbers. Use offset/limit for large files."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file, relative to the workspace root.",
            },
            "offset": {
                "type": "integer",
                "description": "0-based line number to start reading from.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read.",
            },
        },
        "required": ["path"],
    },
    func=_read_file,
)
