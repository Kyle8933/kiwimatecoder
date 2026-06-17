"""Tool registry: schema export, dispatch, and approval previews.

A tool is "advertised" to the model via its JSON schema. In PLAN mode only
read-only tools are advertised (and the permission gate blocks the rest as a
second line of defense).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from kiwimatecoder.tools.base import FunctionTool, ToolResult
from kiwimatecoder.tools.edit_file import edit_file_tool
from kiwimatecoder.tools.edit_file import preview as _edit_preview
from kiwimatecoder.tools.list_dir import list_dir_tool
from kiwimatecoder.tools.read_file import read_file_tool
from kiwimatecoder.tools.run_bash import preview as _bash_preview
from kiwimatecoder.tools.run_bash import run_bash_tool
from kiwimatecoder.tools.search import search_tool
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
]

TOOLS: dict[str, FunctionTool] = {t.name: t for t in _ALL_TOOLS}

# Preview functions used by the permission gate to render what an action will do.
_PREVIEWS: dict[str, Callable[[dict, "Session"], str]] = {
    "write_file": _write_preview,
    "edit_file": _edit_preview,
    "run_bash": _bash_preview,
}


def read_only_tools() -> list[FunctionTool]:
    return [t for t in _ALL_TOOLS if not t.needs_approval]


def tool_schemas(read_only: bool = False) -> list[dict]:
    """Return OpenAI tool schemas, optionally restricted to read-only tools."""
    tools = read_only_tools() if read_only else _ALL_TOOLS
    return [t.schema() for t in tools]


def get_tool(name: str) -> FunctionTool | None:
    return TOOLS.get(name)


def preview(name: str, args: dict, session: "Session") -> str | None:
    """Return a human-readable preview of a tool action, or None if not previewable."""
    fn = _PREVIEWS.get(name)
    return fn(args, session) if fn else None


def dispatch(name: str, args: dict, session: "Session") -> ToolResult:
    """Execute a tool by name."""
    tool = TOOLS.get(name)
    if tool is None:
        return ToolResult.error(f"Unknown tool: {name}")
    return tool.execute(args, session)
