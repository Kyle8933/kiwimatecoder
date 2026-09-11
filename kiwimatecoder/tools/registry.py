"""Extensible registry owning the name -> tool map and each tool's source.

The registry implements the mapping protocol so existing callers that treat
``TOOLS`` as a plain dict keep working (``TOOLS["read_file"]``,
``TOOLS.get("nope")``, iteration, ``.values()``). Tools registered at import
time are labelled ``"builtin"``; dynamically registered tools default to a
``"plugin"`` source and can be loaded from an external source in future.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from kiwimatecoder.tools.base import FunctionTool

BUILTIN_SOURCE = "builtin"
PLUGIN_SOURCE = "plugin"


class ToolRegistry(Mapping[str, FunctionTool]):
    """Ordered name -> :class:`FunctionTool` map with source tracking."""

    def __init__(self) -> None:
        self._tools: dict[str, FunctionTool] = {}
        self._sources: dict[str, str] = {}

    def register(
        self,
        tool: FunctionTool,
        source: str = BUILTIN_SOURCE,
        replace: bool = False,
    ) -> FunctionTool:
        """Register ``tool``, raising ``ValueError`` on a duplicate name.

        Pass ``replace=True`` to swap an existing tool with the same name.
        """
        name = getattr(tool, "name", "")
        if not name:
            raise ValueError("Tool name must be non-empty.")
        if name in self._tools and not replace:
            raise ValueError(f"Tool '{name}' is already registered.")
        self._tools[name] = tool
        self._sources[name] = str(source or BUILTIN_SOURCE)
        return tool

    def unregister(self, name: str, force: bool = False) -> bool:
        """Remove a tool, returning whether it was removed.

        Unknown names return ``False``. Built-in tools are protected unless
        ``force=True`` so a plugin cannot silently shadow core behaviour.
        """
        if name not in self._tools:
            return False
        if self._sources.get(name) == BUILTIN_SOURCE and not force:
            return False
        del self._tools[name]
        self._sources.pop(name, None)
        return True

    def __getitem__(self, name: str) -> FunctionTool:
        return self._tools[name]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def all(self) -> list[FunctionTool]:
        """Every registered tool in registration order."""
        return list(self._tools.values())

    def read_only(self) -> list[FunctionTool]:
        """Every read-only tool (no approval required) in registration order."""
        return [tool for tool in self._tools.values() if not tool.needs_approval]

    def schemas(self, read_only: bool = False) -> list[dict[str, Any]]:
        """OpenAI tool schemas, optionally restricted to read-only tools."""
        tools = self.read_only() if read_only else self.all()
        return [tool.schema() for tool in tools]

    def sources(self) -> dict[str, list[str]]:
        """Map each source label to its sorted tool names."""
        grouped: dict[str, list[str]] = {}
        for name, source in self._sources.items():
            grouped.setdefault(source, []).append(name)
        return {source: sorted(names) for source, names in grouped.items()}
