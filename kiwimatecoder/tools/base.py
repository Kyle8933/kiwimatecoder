"""Core types shared by all tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol


class SessionProtocol(Protocol):
    """Protocol representing the session interface required by tools."""

    workspace_root: Path

    def record_touched(self, path: str) -> None: ...


@dataclass
class ToolResult:
    """The outcome of running a tool."""

    content: str
    ok: bool = True

    @classmethod
    def error(cls, message: str) -> "ToolResult":
        return cls(content=f"Error: {message}", ok=False)


class Tool(Protocol):
    """Protocol every tool implements."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the function arguments
    writes: bool  # mutates the filesystem
    runs: bool  # executes commands

    def execute(self, args: dict[str, Any], session: Any) -> ToolResult: ...


@dataclass
class FunctionTool:
    """Concrete tool backed by a plain callable."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[[dict[str, Any], Any], ToolResult]
    writes: bool = False
    runs: bool = False

    def execute(self, args: dict[str, Any], session: Any) -> ToolResult:
        return self.func(args, session)

    def schema(self) -> dict[str, Any]:
        """Return the OpenAI tool/function schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @property
    def needs_approval(self) -> bool:
        return self.writes or self.runs
