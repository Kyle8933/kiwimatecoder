"""Real stdio transport: run the ACP server on stdin/stdout.

Only protocol lines are ever written to stdout; diagnostics go to stderr via
the standard logging fallback. The blocking stdin read runs on a worker thread
so the server's event loop stays responsive for outstanding requests.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from kiwimatecoder.acp.server import AcpServer


def serve_stdio(workspace: str | Path | None = None) -> int:
    """Run the ACP server until stdin reaches EOF; 0 on a clean EOF."""
    server = AcpServer(workspace=workspace)

    async def read_line() -> str | None:
        line = await asyncio.to_thread(sys.stdin.readline)
        return line or None

    def write_line(line: str) -> None:
        sys.stdout.write(line)
        sys.stdout.flush()

    try:
        return asyncio.run(server.serve(read_line, write_line))
    except KeyboardInterrupt:
        return 130
