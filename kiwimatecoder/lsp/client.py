"""Minimal LSP client over a stdio subprocess.

One background reader thread decodes framed JSON-RPC and fans it out: responses
to per-request queues (matched by id), ``publishDiagnostics`` notifications to a
condition-guarded latest-wins map, and server-to-client requests to a benign
reply so a server waiting on ``workspace/configuration`` cannot stall. Every
wait is bounded, and teardown never raises.
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from kiwimatecoder import __version__
from kiwimatecoder.lsp.protocol import LspProtocolError, decode_messages, encode_message

CLIENT_INFO = {"name": "kiwimatecoder", "version": __version__}
SHUTDOWN_TIMEOUT = 2.0
CLOSE_TIMEOUT = 2.0

LANGUAGE_IDS = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescriptreact",
    ".js": "javascript",
    ".jsx": "javascriptreact",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".hh": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cxx": "cpp",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".swift": "swift",
}

SEVERITY_NAMES = {1: "error", 2: "warning", 3: "information", 4: "hint"}


class LspError(RuntimeError):
    """A language-server operation failed or is unavailable."""


class LspTimeout(LspError):
    """A language-server operation exceeded its time budget."""


def path_to_uri(path: str | Path) -> str:
    """Return the ``file://`` URI for ``path``."""
    return Path(path).expanduser().resolve().as_uri()


def uri_to_path(uri: str) -> str:
    """Convert a ``file://`` URI to a filesystem path (other URIs unchanged)."""
    try:
        parsed = urlparse(uri)
    except ValueError:
        return uri
    if parsed.scheme != "file":
        return uri
    path = url2pathname(unquote(parsed.path))
    if parsed.netloc and parsed.netloc not in ("", "localhost"):
        path = f"//{parsed.netloc}{path}"
    return path


def language_id_for(path: str | Path) -> str:
    """Guess an LSP ``languageId`` from a file suffix."""
    suffix = Path(path).suffix.lower()
    return LANGUAGE_IDS.get(suffix, suffix.lstrip(".") or "plaintext")


def _point(value: Any) -> tuple[int, int]:
    """Return a non-negative (line, character) pair from an LSP position."""
    if not isinstance(value, dict):
        return 0, 0
    try:
        line = int(value.get("line", 0))
        character = int(value.get("character", 0))
    except (TypeError, ValueError):
        return 0, 0
    return max(0, line), max(0, character)


def _range_start(value: Any) -> Any:
    if not isinstance(value, dict):
        return None
    return value.get("start")


def normalize_diagnostics(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize a ``publishDiagnostics`` payload into display-ready dicts.

    Lines and characters become 1-based; severity becomes a lowercase name.
    """
    raw = params.get("diagnostics")
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    for diagnostic in raw:
        if not isinstance(diagnostic, dict):
            continue
        line, character = _point(_range_start(diagnostic.get("range")))
        severity = diagnostic.get("severity")
        normalized.append(
            {
                "line": line + 1,
                "character": character + 1,
                "severity": (
                    SEVERITY_NAMES.get(severity, "error")
                    if isinstance(severity, int)
                    else "error"
                ),
                "message": str(diagnostic.get("message") or ""),
                "source": str(diagnostic.get("source") or ""),
            }
        )
    return normalized


def normalize_locations(result: Any) -> list[dict[str, Any]]:
    """Normalize definition/references results into ``{path, line, character}``.

    Accepts a single Location, a list of Locations, or LocationLinks. Lines and
    characters become 1-based.
    """
    if result is None:
        return []
    items = result if isinstance(result, list) else [result]
    locations: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        uri = item.get("uri") or item.get("targetUri")
        if not isinstance(uri, str) or not uri:
            continue
        start = _range_start(
            item.get("range")
            or item.get("targetSelectionRange")
            or item.get("targetRange")
        )
        line, character = _point(start)
        locations.append(
            {
                "path": uri_to_path(uri),
                "line": line + 1,
                "character": character + 1,
            }
        )
    return locations


class LspClient:
    """A live language server and its bounded request/diagnostics API."""

    def __init__(
        self,
        command: str,
        args: Sequence[str] | None = None,
        *,
        root: str | Path | None = None,
        initialize_timeout: float = 10.0,
        request_timeout: float = 10.0,
    ) -> None:
        self.command = command
        self.args = [str(arg) for arg in (args or [])]
        self.root = Path(root).expanduser().resolve() if root is not None else None
        self.initialize_timeout = float(initialize_timeout)
        self.request_timeout = float(request_timeout)

        self._next_id = 0
        self._responses: dict[Any, queue.Queue[dict[str, Any] | None]] = {}
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._doc_lock = threading.Lock()
        self._cond = threading.Condition()
        self._diagnostics: dict[str, dict[str, Any]] = {}
        self._dirty: set[str] = set()
        self._versions: dict[str, int] = {}
        self._texts: dict[str, str] = {}
        self._opened: set[str] = set()
        self._closed = False
        self._shutdown_done = False

        try:
            self._process: subprocess.Popen[bytes] = subprocess.Popen(
                [self.command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except OSError as exc:
            raise LspError(f"could not start {self.command!r}: {exc}") from exc
        self._reader = threading.Thread(
            target=self._read_loop,
            name=f"lsp-{Path(self.command).name}",
            daemon=True,
        )
        self._reader.start()

    @property
    def running(self) -> bool:
        """Whether the server process is still alive."""
        return self._process.poll() is None

    def initialize(self) -> dict[str, Any]:
        """Perform the LSP handshake; returns the server capabilities."""
        params: dict[str, Any] = {
            "processId": os.getpid(),
            "clientInfo": CLIENT_INFO,
            "capabilities": {
                "textDocument": {
                    "synchronization": {"dynamicRegistration": False},
                    "publishDiagnostics": {"relatedInformation": True},
                    "definition": {"linkSupport": True},
                    "references": {},
                },
                "workspace": {"workspaceFolders": True},
            },
        }
        if self.root is not None:
            root_uri = path_to_uri(self.root)
            params["rootUri"] = root_uri
            params["workspaceFolders"] = [
                {"uri": root_uri, "name": self.root.name}
            ]
        result = self._request("initialize", params, self.initialize_timeout)
        self._notify("initialized", {})
        return result if isinstance(result, dict) else {}

    def is_open(self, path: str | Path) -> bool:
        """Whether the document has been opened on this server."""
        with self._doc_lock:
            return path_to_uri(path) in self._opened

    def did_open(self, path: str | Path, text: str) -> None:
        """Notify the server a document was opened (Full sync, version 1)."""
        uri = path_to_uri(path)
        with self._doc_lock:
            self._opened.add(uri)
            self._versions[uri] = 1
            self._texts[uri] = text
        self._mark_dirty(uri)
        self._notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id_for(path),
                    "version": 1,
                    "text": text,
                }
            },
        )

    def did_change(self, path: str | Path, text: str) -> None:
        """Notify the server of new full document content."""
        uri = path_to_uri(path)
        with self._doc_lock:
            if uri not in self._opened:
                opened = False
                version = 1
            else:
                opened = True
                version = self._versions.get(uri, 1) + 1
                self._versions[uri] = version
                self._texts[uri] = text
        if not opened:
            self.did_open(path, text)
            return
        self._mark_dirty(uri)
        self._notify(
            "textDocument/didChange",
            {
                "textDocument": {"uri": uri, "version": version},
                "contentChanges": [{"text": text}],
            },
        )

    def sync(self, path: str | Path, text: str) -> None:
        """Open the document, or send a change only when its text differs."""
        uri = path_to_uri(path)
        with self._doc_lock:
            opened = uri in self._opened
            unchanged = opened and self._texts.get(uri) == text
        if not opened:
            self.did_open(path, text)
        elif not unchanged:
            self.did_change(path, text)

    def diagnostics(
        self, path: str | Path, timeout: float | None = None
    ) -> list[dict[str, Any]]:
        """Return the latest diagnostics for ``path``.

        Waits up to ``timeout`` for a publish that reflects the most recent
        open/change notification. Raises :class:`LspTimeout` when none arrives.
        """
        uri = path_to_uri(path)
        budget = self.request_timeout if timeout is None else float(timeout)
        deadline = time.monotonic() + max(0.0, budget)
        with self._cond:
            while True:
                stored = self._diagnostics.get(uri)
                if stored is not None and uri not in self._dirty:
                    return normalize_diagnostics(stored)
                if self._closed:
                    raise LspError("language server closed before diagnostics arrived")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if stored is not None:
                        return normalize_diagnostics(stored)
                    raise LspTimeout(
                        f"no diagnostics for {path} within {budget:g}s"
                    )
                self._cond.wait(remaining)

    def definition(
        self,
        path: str | Path,
        line: int,
        character: int,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Request ``textDocument/definition`` (0-based position)."""
        result = self._position_request(
            "textDocument/definition", path, line, character, None, timeout
        )
        return normalize_locations(result)

    def references(
        self,
        path: str | Path,
        line: int,
        character: int,
        include_declaration: bool = True,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Request ``textDocument/references`` (0-based position)."""
        context = {"includeDeclaration": bool(include_declaration)}
        result = self._position_request(
            "textDocument/references", path, line, character, context, timeout
        )
        return normalize_locations(result)

    def shutdown(self) -> None:
        """Best-effort graceful shutdown (shutdown/exit, then terminate)."""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        if not self._closed:
            try:
                self._request("shutdown", None, timeout=SHUTDOWN_TIMEOUT)
            except Exception:
                pass
            try:
                self._notify("exit", None)
            except Exception:
                pass
        self.close()

    def close(self) -> None:
        """Terminate the server process; never raises."""
        self._closed = True
        process = self._process
        try:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=CLOSE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        process.wait(timeout=CLOSE_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        pass
        except OSError:
            pass

    # -- internals ----------------------------------------------------------

    def _mark_dirty(self, uri: str) -> None:
        with self._cond:
            self._dirty.add(uri)
            self._cond.notify_all()

    def _read_loop(self) -> None:
        buffer = b""
        stream = self._process.stdout
        error: str | None = None
        if stream is not None:
            try:
                while True:
                    try:
                        chunk = stream.read(4096)
                    except (OSError, ValueError):
                        break
                    if not chunk:
                        break
                    buffer += chunk
                    try:
                        messages, buffer = decode_messages(buffer)
                    except LspProtocolError as exc:
                        error = f"protocol error: {exc}"
                        break
                    for message in messages:
                        self._dispatch(message)
            finally:
                self._fail_waiters(error)

    def _dispatch(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        if request_id is not None and ("result" in message or "error" in message):
            with self._lock:
                waiter = self._responses.get(request_id)
            if waiter is not None:
                waiter.put(message)
            return
        method = message.get("method")
        if method == "textDocument/publishDiagnostics":
            params = message.get("params")
            if isinstance(params, dict) and isinstance(params.get("uri"), str):
                uri = str(params["uri"])
                with self._cond:
                    self._diagnostics[uri] = params
                    self._dirty.discard(uri)
                    self._cond.notify_all()
            return
        if request_id is not None and method is not None:
            # Server -> client request. Answer benignly so servers that block
            # on configuration (clangd, pyright) can continue.
            result: Any = [] if method == "workspace/configuration" else None
            try:
                self._write(
                    {"jsonrpc": "2.0", "id": request_id, "result": result}
                )
            except LspError:
                pass

    def _fail_waiters(self, reason: str | None) -> None:
        with self._lock:
            self._closed = True
            waiters = list(self._responses.values())
            self._responses.clear()
        for waiter in waiters:
            waiter.put(None)
        with self._cond:
            self._cond.notify_all()

    def _write(self, message: dict[str, Any]) -> None:
        with self._write_lock:
            process = self._process
            if self._closed or process.poll() is not None:
                raise LspError("language server is not running")
            if process.stdin is None:
                raise LspError("language server has no stdin")
            try:
                process.stdin.write(encode_message(message))
                process.stdin.flush()
            except (OSError, ValueError) as exc:
                raise LspError(
                    f"could not write to language server: {exc}"
                ) from exc

    def _request(
        self, method: str, params: Any, timeout: float
    ) -> Any:
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            waiter: queue.Queue[dict[str, Any] | None] = queue.Queue()
            self._responses[request_id] = waiter
        try:
            self._write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                }
            )
            try:
                response = waiter.get(timeout=max(0.05, float(timeout)))
            except queue.Empty:
                raise LspTimeout(
                    f"{method} timed out after {timeout:g}s"
                ) from None
        finally:
            with self._lock:
                self._responses.pop(request_id, None)
        if response is None:
            raise LspError(f"{method}: language server closed the connection")
        error = response.get("error")
        if error:
            detail = (
                str(error.get("message") or error)
                if isinstance(error, dict)
                else str(error)
            )
            raise LspError(f"{method} failed: {detail}")
        return response.get("result")

    def _notify(self, method: str, params: Any) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _position_request(
        self,
        method: str,
        path: str | Path,
        line: int,
        character: int,
        context: dict[str, Any] | None,
        timeout: float | None,
    ) -> Any:
        params: dict[str, Any] = {
            "textDocument": {"uri": path_to_uri(path)},
            "position": {
                "line": max(0, int(line)),
                "character": max(0, int(character)),
            },
        }
        if context is not None:
            params["context"] = context
        budget = self.request_timeout if timeout is None else float(timeout)
        return self._request(method, params, budget)
