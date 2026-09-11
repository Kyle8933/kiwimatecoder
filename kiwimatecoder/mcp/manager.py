"""MCP server lifecycle: connect, register tools, and tear down cleanly.

The manager reads the ``mcp_servers`` config section, connects each enabled
server, discovers its tools, and registers them in the shared tool registry as
``mcp__<server>__<tool>``. Servers that fail to start or initialize are
recorded and skipped; loading never raises, so a broken MCP entry cannot stop a
session from starting. A module-level slot lets the REPL install the session's
manager so slash commands can inspect and reload it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from kiwimatecoder import config, events, tools
from kiwimatecoder.mcp.client import (
    HttpMcpClient,
    McpClient,
    McpError,
    StdioMcpClient,
)
from kiwimatecoder.redaction import redact
from kiwimatecoder.tools.base import FunctionTool, ToolResult

MCP_SOURCE_PREFIX = "mcp:"


def _sanitize(identifier: str) -> str:
    """Reduce an arbitrary server/tool name to ``[A-Za-z0-9_]``."""
    cleaned = "".join(
        ch if (ch.isascii() and (ch.isalnum() or ch == "_")) else "_"
        for ch in identifier
    )
    return cleaned or "unnamed"


def mcp_tool_name(server: str, tool: str) -> str:
    """Return the registry name for a tool exposed by ``server``."""
    return f"mcp__{_sanitize(server)}__{_sanitize(tool)}"


@dataclass
class McpLoadResult:
    """Outcome of one :meth:`McpManager.load` pass."""

    servers: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)


class McpManager:
    """Owns live MCP clients and the tools registered on their behalf."""

    def __init__(
        self, console: Console | None = None, bus: events.EventBus | None = None
    ) -> None:
        self._console = console if console is not None else Console()
        self._bus = bus
        self._clients: dict[str, McpClient] = {}
        self._tools: dict[str, list[str]] = {}
        self._failures: dict[str, str] = {}
        self._resource_counts: dict[str, int] = {}

    @property
    def clients(self) -> dict[str, McpClient]:
        """A copy of the connected server -> client map."""
        return dict(self._clients)

    def client(self, name: str) -> McpClient | None:
        """Return the live client for ``name``, if connected."""
        return self._clients.get(name)

    def is_connected(self, name: str) -> bool:
        return name in self._clients

    def tools_for(self, name: str) -> list[str]:
        """Registered tool names contributed by ``name``."""
        return list(self._tools.get(name, []))

    def failure_for(self, name: str) -> str | None:
        """The recorded failure reason for ``name``, if it failed to load."""
        return self._failures.get(name)

    def resource_count(self, name: str) -> int | None:
        """Number of resources discovered on ``name`` (0 when unsupported)."""
        return self._resource_counts.get(name)

    def load(
        self, servers: dict[str, Any] | None = None
    ) -> McpLoadResult:
        """Connect enabled servers and register their tools. Never raises."""
        self.shutdown()
        if servers is None:
            servers = config.get_mcp_servers()
        result = McpLoadResult()
        for name in sorted(servers):
            spec = servers[name]
            if not isinstance(spec, dict) or spec.get("disabled"):
                continue
            client: McpClient | None = None
            try:
                client = self._build_client(name, spec)
                client.initialize()
                tool_defs = client.list_tools()
            except Exception as exc:
                if client is not None:
                    self._close_quietly(client)
                self._record_failure(name, exc, result)
                continue
            self._clients[name] = client
            try:
                registered = self._register_tools(name, client, tool_defs, result)
            except Exception as exc:
                self._unregister_tools(name)
                self._clients.pop(name, None)
                self._close_quietly(client)
                self._record_failure(name, exc, result)
                continue
            self._tools[name] = registered
            self._resource_counts[name] = self._count_resources(client)
            result.servers.append(name)
            result.tools.extend(registered)
        return result

    def reload(self) -> McpLoadResult:
        """Drop every connection and load the configured servers again."""
        return self.load()

    def shutdown(self) -> None:
        """Close every client, unregister its tools, and clear state."""
        for name in list(self._clients):
            client = self._clients.pop(name, None)
            if client is not None:
                self._close_quietly(client)
        for name in list(self._tools):
            self._unregister_tools(name)
        self._failures.clear()
        self._resource_counts.clear()

    def _build_client(self, name: str, spec: dict[str, Any]) -> McpClient:
        if spec.get("command"):
            return StdioMcpClient(
                str(spec["command"]),
                args=spec.get("args"),
                env=spec.get("env"),
            )
        return HttpMcpClient(str(spec["url"]), headers=spec.get("headers"))

    def _register_tools(
        self,
        server: str,
        client: McpClient,
        tool_defs: list[dict[str, Any]],
        result: McpLoadResult,
    ) -> list[str]:
        registered: list[str] = []
        for definition in tool_defs:
            tool_name = str(definition.get("name") or "").strip()
            if not tool_name:
                continue
            registry_name = mcp_tool_name(server, tool_name)
            parameters = definition.get("inputSchema")
            if not isinstance(parameters, dict):
                parameters = {"type": "object", "properties": {}}
            annotations = definition.get("annotations")
            read_only = (
                bool(annotations.get("readOnlyHint"))
                if isinstance(annotations, dict)
                else False
            )
            description = str(
                definition.get("description")
                or f"MCP tool '{tool_name}' from server '{server}'."
            )
            tool = FunctionTool(
                name=registry_name,
                description=description,
                parameters=parameters,
                func=self._make_caller(client, tool_name),
                writes=not read_only,
                runs=False,
            )
            try:
                tools.register_tool(tool, source=f"{MCP_SOURCE_PREFIX}{server}")
            except ValueError as exc:
                result.failed.append((server, f"{registry_name}: {exc}"))
                continue
            registered.append(registry_name)
        return registered

    def _make_caller(
        self, client: McpClient, tool_name: str
    ) -> Callable[[dict[str, Any], Any], ToolResult]:
        def call(arguments: dict[str, Any], session: Any) -> ToolResult:
            try:
                content = client.call_tool(tool_name, arguments)
            except McpError as exc:
                return ToolResult.error(str(exc))
            return ToolResult(content=content or "(no output)")

        return call

    def _count_resources(self, client: McpClient) -> int:
        capabilities = client.capabilities or {}
        if "resources" not in capabilities:
            return 0
        try:
            return len(client.list_resources())
        except Exception:
            return 0

    def _unregister_tools(self, name: str) -> None:
        for tool_name in self._tools.pop(name, []):
            try:
                tools.unregister_tool(tool_name)
            except Exception:
                pass

    def _record_failure(
        self, name: str, exc: BaseException, result: McpLoadResult
    ) -> None:
        reason = redact(str(exc)) or exc.__class__.__name__
        self._failures[name] = reason
        result.failed.append((name, reason))
        self._console.print(f"[dim]MCP server '{name}' failed: {reason}[/dim]")

    @staticmethod
    def _close_quietly(client: McpClient) -> None:
        try:
            client.close()
        except Exception:
            pass


_MANAGER: McpManager | None = None


def get_manager() -> McpManager | None:
    """Return the session's MCP manager, if the REPL installed one."""
    return _MANAGER


def set_manager(manager: McpManager | None) -> None:
    """Install (or clear) the session's MCP manager."""
    global _MANAGER
    _MANAGER = manager
