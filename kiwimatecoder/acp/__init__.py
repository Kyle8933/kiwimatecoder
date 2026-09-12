"""Agent Client Protocol (ACP) server for editor integration.

The public surface is small: :class:`~kiwimatecoder.acp.server.AcpServer`
speaks JSON-RPC 2.0 over newline-delimited JSON, :mod:`kiwimatecoder.acp.stdio`
wires that to real stdin/stdout, and :mod:`kiwimatecoder.acp.protocol` holds
the transport-agnostic codec. See the README's "Editor integration (ACP)"
section for the implemented subset.
"""

from kiwimatecoder.acp.protocol import decode_lines, encode
from kiwimatecoder.acp.server import AcpServer

__all__ = ["AcpServer", "decode_lines", "encode"]
