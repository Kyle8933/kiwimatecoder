"""MCP clients over stdio subprocesses and streamable HTTP.

Both transports share the JSON-RPC request/response plumbing in
:class:`JsonRpcClient`; a transport only has to implement ``_exchange``. The
protocol is intentionally minimal: one request in flight at a time, responses
matched by ``id``, and bounded waits so a hung server can never wedge a
session. Interactive OAuth is out of scope; HTTP auth is whatever headers the
user configures.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import httpx

from kiwimatecoder.mcp.errors import McpError
from kiwimatecoder.mcp.protocol import (
    CLIENT_INFO,
    MCP_PROTOCOL_VERSION,
    flatten_content,
    flatten_resource_contents,
    notification_message,
    parse_response,
    parse_sse_messages,
    request_message,
)

INITIALIZE_TIMEOUT = 10.0
CALL_TIMEOUT = 60.0
STDERR_TAIL_LINES = 100


class McpClient(Protocol):
    """The surface the manager needs from any MCP transport."""

    capabilities: dict[str, Any]
    server_info: dict[str, Any]

    def initialize(self) -> dict[str, Any]: ...
    def list_tools(self) -> list[dict[str, Any]]: ...
    def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...
    def list_resources(self) -> list[dict[str, Any]]: ...
    def read_resource(self, uri: str) -> str: ...
    def close(self) -> None: ...


class JsonRpcClient:
    """Shared initialize/tools/resources calls over an abstract transport."""

    def __init__(
        self,
        *,
        initialize_timeout: float = INITIALIZE_TIMEOUT,
        call_timeout: float = CALL_TIMEOUT,
    ) -> None:
        self.initialize_timeout = initialize_timeout
        self.call_timeout = call_timeout
        self.server_info: dict[str, Any] = {}
        self.capabilities: dict[str, Any] = {}
        self._next_id = 0

    def _exchange(
        self, message: dict[str, Any], timeout: float
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    def _request(
        self, method: str, params: dict[str, Any] | None, timeout: float
    ) -> Any:
        self._next_id += 1
        message = request_message(self._next_id, method, params)
        response = self._exchange(message, timeout)
        if response is None:
            raise McpError(f"{method}: no response from server")
        return parse_response(response, method)

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._exchange(notification_message(method, params), self.initialize_timeout)

    def initialize(self) -> dict[str, Any]:
        """Perform the MCP handshake and cache server info/capabilities."""
        result = self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            },
            self.initialize_timeout,
        )
        if isinstance(result, dict):
            server_info = result.get("serverInfo")
            capabilities = result.get("capabilities")
            self.server_info = server_info if isinstance(server_info, dict) else {}
            self.capabilities = capabilities if isinstance(capabilities, dict) else {}
        else:
            result = {}
        self._notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        result = self._request("tools/list", {}, self.call_timeout)
        raw = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(raw, list):
            return []
        return [tool for tool in raw if isinstance(tool, dict)]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool and return its flattened content.

        Raises :class:`McpError` when the server reports ``isError`` so the
        registered tool can surface a failed ``ToolResult``.
        """
        result = self._request(
            "tools/call", {"name": name, "arguments": dict(arguments)}, self.call_timeout
        )
        payload = result if isinstance(result, dict) else {}
        content = flatten_content(payload.get("content"))
        if payload.get("isError"):
            raise McpError(content or f"tool '{name}' reported an error")
        return content

    def list_resources(self) -> list[dict[str, Any]]:
        result = self._request("resources/list", {}, self.call_timeout)
        raw = result.get("resources") if isinstance(result, dict) else None
        if not isinstance(raw, list):
            return []
        return [resource for resource in raw if isinstance(resource, dict)]

    def read_resource(self, uri: str) -> str:
        result = self._request("resources/read", {"uri": uri}, self.call_timeout)
        payload = result if isinstance(result, dict) else {}
        return flatten_resource_contents(payload.get("contents"))

    def close(self) -> None:  # pragma: no cover - overridden by transports
        pass


class StdioMcpClient(JsonRpcClient):
    """Talk to an MCP server that speaks newline-delimited JSON-RPC on stdio.

    A background reader thread feeds stdout lines into a queue so requests can
    time out instead of blocking forever, and a second thread drains stderr
    into a bounded tail used for diagnostics (draining is what keeps a chatty
    server from filling its pipe buffer and deadlocking).
    """

    def __init__(
        self,
        command: str,
        args: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        *,
        initialize_timeout: float = INITIALIZE_TIMEOUT,
        call_timeout: float = CALL_TIMEOUT,
        stderr_limit: int = STDERR_TAIL_LINES,
    ) -> None:
        super().__init__(
            initialize_timeout=initialize_timeout, call_timeout=call_timeout
        )
        self.command = command
        self.args = [str(arg) for arg in (args or [])]
        self._stderr_tail: deque[str] = deque(maxlen=stderr_limit)
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._write_lock = threading.Lock()
        self._closed = False
        process_env = os.environ.copy()
        if env:
            process_env.update({str(key): str(value) for key, value in env.items()})
        try:
            self._process = subprocess.Popen(
                [command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                env=process_env,
                bufsize=1,
            )
        except OSError as exc:
            raise McpError(f"could not start '{command}': {exc}") from exc
        self._reader = threading.Thread(
            target=self._read_stdout, name=f"mcp-stdout-{command}", daemon=True
        )
        self._reader.start()
        self._stderr_reader = threading.Thread(
            target=self._read_stderr, name="mcp-stderr", daemon=True
        )
        self._stderr_reader.start()

    @property
    def running(self) -> bool:
        """Whether the server process is still alive."""
        return self._process.poll() is None

    def stderr_tail(self) -> str:
        """The most recent stderr lines, joined (bounded)."""
        return "\n".join(self._stderr_tail)

    def _read_stdout(self) -> None:
        stream = self._process.stdout
        if stream is None:
            self._lines.put(None)
            return
        try:
            for line in stream:
                self._lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self._lines.put(None)

    def _read_stderr(self) -> None:
        stream = self._process.stderr
        if stream is None:
            return
        try:
            for line in stream:
                self._stderr_tail.append(line.rstrip("\n"))
        except (OSError, ValueError):
            pass

    def _exit_message(self) -> str:
        message = "MCP server exited"
        if self._process.returncode is not None:
            message += f" (code {self._process.returncode})"
        tail = self.stderr_tail()
        if tail:
            message += f": {tail}"
        return message

    def _exchange(
        self, message: dict[str, Any], timeout: float
    ) -> dict[str, Any] | None:
        if self._closed:
            raise McpError("MCP client is closed")
        line = json.dumps(message) + "\n"
        with self._write_lock:
            stdin = self._process.stdin
            if stdin is None or self._process.poll() is not None:
                raise McpError(self._exit_message())
            try:
                stdin.write(line)
                stdin.flush()
            except (OSError, ValueError) as exc:
                raise McpError(f"could not write to MCP server: {exc}") from exc
            if "id" not in message:
                return None
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise McpError(
                        f"timed out waiting for MCP response after {timeout:g}s"
                    )
                try:
                    response_line = self._lines.get(timeout=remaining)
                except queue.Empty:
                    raise McpError(
                        f"timed out waiting for MCP response after {timeout:g}s"
                    ) from None
                if response_line is None:
                    raise McpError(self._exit_message())
                try:
                    response = json.loads(response_line)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(response, dict)
                    and response.get("id") == message["id"]
                    and ("result" in response or "error" in response)
                ):
                    return response

    def close(self) -> None:
        """Terminate the server process, escalating to kill on timeout."""
        if self._closed:
            return
        self._closed = True
        process = self._process
        try:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        except OSError:
            pass


class HttpMcpClient(JsonRpcClient):
    """POST JSON-RPC to a streamable-HTTP MCP endpoint.

    Responses may be a plain JSON body or an SSE stream; both are accepted.
    One :class:`httpx.Client` is reused for the life of the connection, with
    configured headers carrying any bearer/API auth.
    """

    def __init__(
        self,
        url: str,
        headers: Mapping[str, str] | None = None,
        *,
        initialize_timeout: float = INITIALIZE_TIMEOUT,
        call_timeout: float = CALL_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(
            initialize_timeout=initialize_timeout, call_timeout=call_timeout
        )
        self.url = url
        request_headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if headers:
            request_headers.update(
                {str(key): str(value) for key, value in headers.items()}
            )
        self._client = httpx.Client(
            headers=request_headers,
            timeout=httpx.Timeout(call_timeout, connect=initialize_timeout),
            transport=transport,
            follow_redirects=True,
        )

    def _exchange(
        self, message: dict[str, Any], timeout: float
    ) -> dict[str, Any] | None:
        try:
            response = self._client.post(self.url, json=message, timeout=timeout)
        except httpx.HTTPError as exc:
            raise McpError(f"HTTP request to {self.url} failed: {exc}") from exc
        if response.status_code >= 400:
            body = response.text[:200].replace("\n", " ").strip()
            raise McpError(
                f"HTTP {response.status_code} from {self.url}"
                + (f": {body}" if body else "")
            )
        if "id" not in message:
            return None
        return self._parse_body(response, message["id"])

    @staticmethod
    def _parse_body(response: httpx.Response, request_id: Any) -> dict[str, Any]:
        text = response.text or ""
        stripped = text.lstrip()
        if stripped.startswith(("{", "[")):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                if "result" in payload or "error" in payload:
                    return payload
            elif isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict) and item.get("id") == request_id:
                        return item
        for item in parse_sse_messages(text):
            if item.get("id") == request_id:
                return item
        raise McpError("MCP server returned no JSON-RPC response")

    def close(self) -> None:
        self._client.close()
