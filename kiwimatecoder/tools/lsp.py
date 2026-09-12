"""Read-only LSP tools: diagnostics, definitions, and references.

These tools never mutate the workspace (``writes=False, runs=False``). They
report a clear error when LSP is disabled or no configured server matches the
file, and they never start a server under an extension the presets/overrides do
not claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kiwimatecoder import config, lsp
from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult
from kiwimatecoder.tools.paths import PathError, display_path, resolve_in_workspace

MAX_DIAGNOSTIC_FILES = 5


def _disabled_error() -> ToolResult:
    return ToolResult.error(lsp.LSP_DISABLED_MESSAGE)


def format_diagnostic(path: str, diagnostic: dict[str, Any]) -> str:
    line = (
        f"{path}:{diagnostic['line']}:{diagnostic['character']} "
        f"{diagnostic['severity']} {diagnostic['message']}"
    )
    source = diagnostic.get("source")
    if source:
        line += f" ({source})"
    return line.rstrip()


def _format_location(location: dict[str, Any], session: Session) -> str:
    raw_path = str(location.get("path") or "")
    display = raw_path
    if raw_path:
        try:
            display = display_path(Path(raw_path), session.workspace_root)
        except (OSError, ValueError):
            display = raw_path
    return f"{display}:{location.get('line')}:{location.get('character')}"


def _parse_position(args: dict[str, Any]) -> tuple[int, int] | ToolResult:
    raw_line = args.get("line")
    raw_character = args.get("character")
    if raw_line is None or raw_character is None:
        return ToolResult.error("'line' and 'character' are required (1-based)")
    try:
        line = int(raw_line)
        character = int(raw_character)
    except (TypeError, ValueError):
        return ToolResult.error("'line' and 'character' must be integers (1-based)")
    if line < 1 or character < 1:
        return ToolResult.error("'line' and 'character' are 1-based and must be >= 1")
    return line, character


def _include_declaration(args: dict[str, Any]) -> bool:
    value = args.get("include_declaration")
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"false", "0", "no", "off"}


def _lsp_diagnostics(args: dict[str, Any], session: Session) -> ToolResult:
    if not config.get_lsp()["enabled"]:
        return _disabled_error()
    raw_path = str(args.get("path") or "").strip()
    if raw_path:
        targets = [raw_path]
    else:
        targets = list(session.touched_files)[-MAX_DIAGNOSTIC_FILES:]
        if not targets:
            return ToolResult.error(
                "No file given and none have been touched yet; pass a path."
            )

    manager = lsp.ensure_manager(root=session.workspace_root)
    lines: list[str] = []
    failures: list[str] = []
    for target in targets:
        try:
            resolved = resolve_in_workspace(target, session.workspace_root)
        except PathError as exc:
            failures.append(str(exc))
            continue
        display = display_path(resolved, session.workspace_root)
        try:
            diagnostics = manager.diagnostics_for(resolved)
        except lsp.LspError as exc:
            failures.append(f"{display}: {exc}")
            continue
        lines.extend(format_diagnostic(display, item) for item in diagnostics)

    if lines:
        return ToolResult(content="\n".join(lines))
    if failures:
        return ToolResult.error("; ".join(failures[:3]))
    return ToolResult(content="No diagnostics.")


def _lsp_definition(args: dict[str, Any], session: Session) -> ToolResult:
    if not config.get_lsp()["enabled"]:
        return _disabled_error()
    path = str(args.get("path") or "").strip()
    if not path:
        return ToolResult.error("'path' is required")
    position = _parse_position(args)
    if isinstance(position, ToolResult):
        return position
    line, character = position
    try:
        resolved = resolve_in_workspace(path, session.workspace_root)
    except PathError as exc:
        return ToolResult.error(str(exc))

    manager = lsp.ensure_manager(root=session.workspace_root)
    try:
        locations = manager.definition(resolved, line, character)
    except lsp.LspError as exc:
        return ToolResult.error(str(exc))
    if not locations:
        return ToolResult(
            content=f"No definition found at {path}:{line}:{character}."
        )
    return ToolResult(
        content="\n".join(_format_location(item, session) for item in locations)
    )


def _lsp_references(args: dict[str, Any], session: Session) -> ToolResult:
    if not config.get_lsp()["enabled"]:
        return _disabled_error()
    path = str(args.get("path") or "").strip()
    if not path:
        return ToolResult.error("'path' is required")
    position = _parse_position(args)
    if isinstance(position, ToolResult):
        return position
    line, character = position
    try:
        resolved = resolve_in_workspace(path, session.workspace_root)
    except PathError as exc:
        return ToolResult.error(str(exc))

    manager = lsp.ensure_manager(root=session.workspace_root)
    try:
        locations = manager.references(
            resolved,
            line,
            character,
            include_declaration=_include_declaration(args),
        )
    except lsp.LspError as exc:
        return ToolResult.error(str(exc))
    if not locations:
        return ToolResult(
            content=f"No references found at {path}:{line}:{character}."
        )
    return ToolResult(
        content="\n".join(_format_location(item, session) for item in locations)
    )


lsp_diagnostics_tool = FunctionTool(
    name="lsp_diagnostics",
    description=(
        "Get language-server diagnostics (errors, warnings) for a file, or for "
        "the most recently touched files when no path is given. Requires LSP to "
        "be enabled and a matching server installed. Use it after edits to "
        "check the code compiles/type-checks."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File to check, relative to the workspace root.",
            },
        },
    },
    func=_lsp_diagnostics,
    writes=False,
    runs=False,
)

lsp_definition_tool = FunctionTool(
    name="lsp_definition",
    description=(
        "Find where the symbol at a 1-based line/character is defined using the "
        "language server. Returns one location per line."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File to inspect, relative to the workspace root.",
            },
            "line": {
                "type": "integer",
                "description": "1-based line number of the symbol.",
            },
            "character": {
                "type": "integer",
                "description": "1-based column number of the symbol.",
            },
        },
        "required": ["path", "line", "character"],
    },
    func=_lsp_definition,
    writes=False,
    runs=False,
)

lsp_references_tool = FunctionTool(
    name="lsp_references",
    description=(
        "Find every reference to the symbol at a 1-based line/character using "
        "the language server. Returns one location per line."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File to inspect, relative to the workspace root.",
            },
            "line": {
                "type": "integer",
                "description": "1-based line number of the symbol.",
            },
            "character": {
                "type": "integer",
                "description": "1-based column number of the symbol.",
            },
            "include_declaration": {
                "type": "boolean",
                "description": "Include the declaration itself (default true).",
            },
        },
        "required": ["path", "line", "character"],
    },
    func=_lsp_references,
    writes=False,
    runs=False,
)
