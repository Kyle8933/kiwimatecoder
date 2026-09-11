"""Errors raised by the MCP client layer."""

from __future__ import annotations


class McpError(Exception):
    """Any MCP transport, protocol, or server-reported failure."""
