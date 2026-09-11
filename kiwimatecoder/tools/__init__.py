"""Tool registry: schema export, dispatch, and approval previews.

A tool is "advertised" to the model via its JSON schema. In PLAN mode only
read-only tools are advertised (and the permission gate blocks the rest as a
second line of defense).

The mapping lives in :class:`~kiwimatecoder.tools.registry.ToolRegistry`, which
also records each tool's source (``"builtin"`` or a plugin label). The module
level ``TOOLS`` name is that registry and remains dict-compatible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from kiwimatecoder.tools.ask import ask_user_tool
from kiwimatecoder.tools.base import FunctionTool, ToolResult
from kiwimatecoder.tools.edit_file import edit_file_tool
from kiwimatecoder.tools.edit_file import preview as _edit_preview
from kiwimatecoder.tools.list_dir import list_dir_tool
from kiwimatecoder.tools.read_file import read_file_tool
from kiwimatecoder.tools.registry import (
    BUILTIN_SOURCE,
    PLUGIN_SOURCE,
    ToolRegistry,
)
from kiwimatecoder.tools.run_bash import preview as _bash_preview
from kiwimatecoder.tools.run_bash import run_bash_tool
from kiwimatecoder.tools.search import search_tool
from kiwimatecoder.tools.selection import select_hunks as select_hunks
from kiwimatecoder.tools.todo import update_todos_tool
from kiwimatecoder.tools.write_file import preview as _write_preview
from kiwimatecoder.tools.write_file import write_file_tool

if TYPE_CHECKING:
    from kiwimatecoder.session import Session

_ALL_TOOLS: list[FunctionTool] = [
    read_file_tool,
    list_dir_tool,
    search_tool,
    write_file_tool,
    edit_file_tool,
    run_bash_tool,
    update_todos_tool,
    ask_user_tool,
]

TOOLS = ToolRegistry()
for _tool in _ALL_TOOLS:
    TOOLS.register(_tool, source=BUILTIN_SOURCE)

# Preview functions used by the permission gate to render what an action will do.
_PREVIEWS: dict[str, Callable[[dict[str, Any], Any], str]] = {
    "write_file": _write_preview,
    "edit_file": _edit_preview,
    "run_bash": _bash_preview,
}


def register_tool(tool: FunctionTool, source: str = PLUGIN_SOURCE) -> FunctionTool:
    """Register a tool dynamically and return it.

    Raises ``ValueError`` when the name is already registered; pass through
    ``TOOLS.register(tool, replace=True)`` to deliberately replace one.
    """
    return TOOLS.register(tool, source=source)


def unregister_tool(name: str, force: bool = False) -> bool:
    """Remove a dynamically registered tool; built-ins need ``force=True``."""
    return TOOLS.unregister(name, force=force)


def tool_sources() -> dict[str, list[str]]:
    """Map each tool source to its sorted tool names."""
    return TOOLS.sources()


def read_only_tools() -> list[FunctionTool]:
    return TOOLS.read_only()


def tool_schemas(read_only: bool = False) -> list[dict[str, Any]]:
    """Return OpenAI tool schemas, optionally restricted to read-only tools."""
    return TOOLS.schemas(read_only=read_only)


def get_tool(name: str) -> FunctionTool | None:
    return TOOLS.get(name)


def preview(name: str, args: dict[str, Any], session: Session) -> str | None:
    """Return a human-readable preview of a tool action, or None if not previewable."""
    fn = _PREVIEWS.get(name)
    return fn(args, session) if fn else None


def dispatch(name: str, args: dict[str, Any], session: Session) -> ToolResult:
    """Execute a tool by name."""
    tool = TOOLS.get(name)
    if tool is None:
        return ToolResult.error(f"Unknown tool: {name}")
    return tool.execute(args, session)


__all__ = [
    "BUILTIN_SOURCE",
    "PLUGIN_SOURCE",
    "TOOLS",
    "FunctionTool",
    "ToolRegistry",
    "ToolResult",
    "dispatch",
    "get_tool",
    "preview",
    "read_only_tools",
    "register_tool",
    "select_hunks",
    "tool_schemas",
    "tool_sources",
    "unregister_tool",
]
