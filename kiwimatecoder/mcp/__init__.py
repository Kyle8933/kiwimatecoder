"""MCP (Model Context Protocol) client support.

Connects to configured MCP servers over stdio or streamable HTTP, discovers
their tools, and registers them as ``mcp__<server>__<tool>`` in the shared tool
registry. The ``mcp_servers`` config section holds one spec per server: a
``command`` (+ ``args``/``env``) for stdio, or a ``url`` (+ ``headers``) for
HTTP. Authentication is whatever headers or environment the user configures;
interactive OAuth is not implemented.
"""

from kiwimatecoder.mcp.client import (
    HttpMcpClient,
    McpClient,
    McpError,
    StdioMcpClient,
)
from kiwimatecoder.mcp.manager import (
    McpLoadResult,
    McpManager,
    get_manager,
    mcp_tool_name,
    set_manager,
)
from kiwimatecoder.mcp.protocol import (
    CLIENT_INFO,
    MCP_PROTOCOL_VERSION,
    flatten_content,
    parse_sse_messages,
)

__all__ = [
    "CLIENT_INFO",
    "HttpMcpClient",
    "MCP_PROTOCOL_VERSION",
    "McpClient",
    "McpError",
    "McpLoadResult",
    "McpManager",
    "StdioMcpClient",
    "flatten_content",
    "get_manager",
    "mcp_tool_name",
    "parse_sse_messages",
    "set_manager",
]
