"""The ACP agent server: request dispatch, sessions, turns, and permissions.

This implements the agent side of a focused Agent Client Protocol subset over
newline-delimited JSON-RPC 2.0:

* client -> agent requests: ``initialize``, ``session/new``, ``session/prompt``
  and the ``session/cancel`` notification;
* agent -> client requests: ``session/request_permission`` (used to bridge the
  regular permission gate to the editor);
* agent -> client notifications: ``session/update`` for assistant text chunks
  and tool-call progress;
* optional file delegation through ``fs/read_text_file`` / ``fs/write_text_file``
  when the client advertises filesystem capabilities.

The transport is injected as two callables (``read_line``/``write_line``), so
tests drive the server with in-memory queues and the real process uses
:mod:`kiwimatecoder.acp.stdio`. Every wait is bounded: unanswered permission
requests are denied after ``acp.permission_timeout`` seconds (default 300).

Turns reuse :func:`kiwimatecoder.headless.run_agent_once`; the agent's block
and gate callbacks run inside the normal permission flow, so an editor in the
configured ``ask`` mode sees exactly the approvals the terminal would ask for.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from kiwimatecoder import __version__, config
from kiwimatecoder.acp.protocol import (
    ERROR_INTERNAL,
    ERROR_INVALID_PARAMS,
    ERROR_INVALID_REQUEST,
    ERROR_METHOD_NOT_FOUND,
    ERROR_PARSE,
    ERROR_SERVER,
    AcpError,
    AcpTimeoutError,
    IdAllocator,
    decode_lines,
    encode,
    error_response,
    notification_message,
    request_message,
    success_response,
)
from kiwimatecoder.headless import build_session, run_agent_once, shutdown_runtime
from kiwimatecoder.session import Session

_log = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
DEFAULT_PERMISSION_TIMEOUT = 300.0
SHUTDOWN_GRACE_SECONDS = 5.0

# Incoming transport hooks. ``read_line`` returns None (or "") at EOF; either
# callable may be sync or async.
ReadLine = Callable[[], Any]
WriteLine = Callable[[str], Any]

PERMISSION_OPTIONS: list[dict[str, Any]] = [
    {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
    {"optionId": "allow_always", "name": "Allow always", "kind": "allow_always"},
    {"optionId": "reject_once", "name": "Reject", "kind": "reject_once"},
]

# Tools whose result/args are read-like, edit-like, and so on. Anything not
# listed maps to ACP's generic "other".
TOOL_KINDS: dict[str, str] = {
    "read_file": "read",
    "list_dir": "read",
    "view_image": "read",
    "write_file": "edit",
    "edit_file": "edit",
    "search": "search",
    "run_bash": "execute",
    "shell": "execute",
    "shell_jobs": "execute",
    "web_fetch": "fetch",
    "web_search": "fetch",
    "http_get": "fetch",
    "http_request": "fetch",
    "task": "think",
}


def tool_kind(tool_name: str) -> str:
    """Map a KiwiMateCoder tool name to an ACP tool-call kind."""
    return TOOL_KINDS.get(tool_name, "other")


def _describe_tool(tool_name: str, args: dict[str, Any]) -> str:
    """A short human-readable title for a tool-call update."""
    for key in ("path", "command", "query", "url", "description", "action"):
        value = args.get(key)
        if value:
            text = str(value)
            if len(text) > 80:
                text = text[:77] + "..."
            return f"{tool_name} {text}"
    return tool_name


def prompt_text(prompt: Any) -> tuple[str, int]:
    """Concatenate a prompt's text parts; count any image parts skipped."""
    if not isinstance(prompt, list):
        return "", 0
    parts: list[str] = []
    images = 0
    for item in prompt:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "text":
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif kind == "image":
            images += 1
    return "\n".join(parts), images


@dataclass
class _SessionState:
    """One ``session/new`` conversation plus its in-flight tool calls."""

    session: Session
    tool_counter: int = 0
    in_progress: list[tuple[str, str]] = field(default_factory=list)
    pending_permission_id: str | None = None

    def allocate_call_id(self) -> str:
        self.tool_counter += 1
        return f"call-{self.tool_counter}"

    def begin_tool(self, tool_name: str, args: dict[str, Any]) -> tuple[str, str, str]:
        """Register a starting tool call and return ``(id, title, kind)``."""
        call_id = self.pending_permission_id or self.allocate_call_id()
        self.pending_permission_id = None
        self.in_progress.append((call_id, tool_name))
        return call_id, _describe_tool(tool_name, args), tool_kind(tool_name)

    def end_tool(self, tool_name: str) -> str | None:
        """Resolve the call id for a finished tool, preferring a name match."""
        for index, (call_id, name) in enumerate(self.in_progress):
            if name == tool_name:
                del self.in_progress[index]
                return call_id
        if self.in_progress:
            return self.in_progress.pop(0)[0]
        return None


@dataclass
class _TurnState:
    """Bookkeeping for one running ``session/prompt``."""

    session_id: str
    done_reasons: list[str] = field(default_factory=list)
    saw_text: bool = False
    cancelled: bool = False
    loop: asyncio.AbstractEventLoop | None = None
    task: asyncio.Task[Any] | None = None
    pending_updates: list[Any] = field(default_factory=list)


@dataclass
class _PendingRequest:
    """An agent -> client request awaiting its response."""

    future: asyncio.Future[Any]
    method: str
    session_id: str | None = None


def _consume_future_exception(future: Any) -> None:
    """Swallow an unobserved exception on a fire-and-forget future."""
    try:
        future.exception()
    except BaseException:
        pass


class AcpServer:
    """A transport-agnostic ACP agent.

    ``workspace`` is the default session root when ``session/new`` omits
    ``cwd``; ``permission_timeout`` overrides the configured
    ``acp.permission_timeout`` (seconds) and is mainly useful in tests.
    """

    def __init__(
        self,
        *,
        workspace: str | Path | None = None,
        permission_timeout: float | None = None,
    ) -> None:
        self.default_workspace = (
            Path(workspace).expanduser() if workspace is not None else None
        )
        if permission_timeout is None:
            permission_timeout = float(
                config.get_acp().get("permission_timeout", DEFAULT_PERMISSION_TIMEOUT)
            )
        self.permission_timeout = float(permission_timeout)

        self._sessions: dict[str, _SessionState] = {}
        self._turns: dict[str, _TurnState] = {}
        self._pending: dict[int, _PendingRequest] = {}
        self._inflight: set[asyncio.Task[Any]] = set()
        self._ids = IdAllocator()
        self._client_capabilities: dict[str, Any] = {}
        self._fs_read = False
        self._fs_write = False

        self._loop: asyncio.AbstractEventLoop | None = None
        self._read_line: ReadLine | None = None
        self._write_line: WriteLine | None = None
        self._write_lock: asyncio.Lock | None = None
        self._closed = False

    # -- lifecycle ----------------------------------------------------------

    async def serve(self, read_line: ReadLine, write_line: WriteLine) -> int:
        """Read and dispatch messages until EOF, then shut down cleanly.

        ``read_line`` returns the next line (with or without a trailing
        newline) or ``None``/``""`` at end of input; ``write_line`` accepts a
        terminated line. Both may be synchronous or asynchronous callables.
        Returns ``0`` after a clean EOF.
        """
        self._loop = asyncio.get_running_loop()
        self._read_line = read_line
        self._write_line = write_line
        self._write_lock = asyncio.Lock()
        self._closed = False
        try:
            while not self._closed:
                line = read_line()
                if inspect.isawaitable(line):
                    line = await line
                if line is None or line == "":
                    break
                await self._handle_line(str(line))
        finally:
            await self.shutdown()
        return 0

    async def shutdown(self) -> None:
        """Cancel active turns, release pending requests, and stop managers."""
        self._closed = True
        for turn in list(self._turns.values()):
            self._cancel_turn(turn)
        for request_id, pending in list(self._pending.items()):
            if not pending.future.done():
                if pending.method == "session/request_permission":
                    pending.future.set_result({"outcome": {"outcome": "cancelled"}})
                else:
                    pending.future.set_exception(
                        AcpError(ERROR_SERVER, "Server is shutting down")
                    )
            self._pending.pop(request_id, None)
        if self._inflight:
            done, still = await asyncio.wait(
                list(self._inflight), timeout=SHUTDOWN_GRACE_SECONDS
            )
            for task in still:
                task.cancel()
            if still:
                await asyncio.wait(still, timeout=SHUTDOWN_GRACE_SECONDS)
        try:
            shutdown_runtime()
        except Exception:  # pragma: no cover - defensive: cleanup must not raise
            _log.debug("ACP shutdown cleanup failed", exc_info=True)

    # -- incoming messages --------------------------------------------------

    async def _handle_line(self, line: str) -> None:
        """Parse one transport line and dispatch every message it contains."""
        text = line if line.endswith("\n") else line + "\n"
        messages, _remainder = decode_lines(text)
        if not messages:
            if line.strip():
                await self._send(
                    error_response(None, ERROR_PARSE, "Parse error: invalid JSON")
                )
            return
        for message in messages:
            await self._dispatch(message)

    async def _dispatch(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        if isinstance(method, str):
            if "id" in message:
                if method == "session/prompt":
                    task = asyncio.create_task(self._handle_prompt_request(message))
                    self._inflight.add(task)
                    task.add_done_callback(self._inflight.discard)
                    return
                await self._handle_request(message["id"], method, message.get("params"))
                return
            await self._handle_notification(method, message.get("params"))
            return
        if "id" in message and ("result" in message or "error" in message):
            self._resolve_response(message)
            return
        await self._send(
            error_response(
                message.get("id"), ERROR_INVALID_REQUEST, "Invalid JSON-RPC message"
            )
        )

    async def _handle_request(
        self, request_id: Any, method: str, params: Any
    ) -> None:
        payload = params if isinstance(params, dict) else {}
        try:
            result = await self._dispatch_request(method, payload)
        except AcpError as exc:
            await self._send(
                error_response(request_id, exc.code, exc.message, exc.data)
            )
            return
        except Exception as exc:  # pragma: no cover - defensive
            _log.debug("ACP request %s failed", method, exc_info=True)
            await self._send(
                error_response(request_id, ERROR_INTERNAL, f"Internal error: {exc}")
            )
            return
        await self._send(success_response(request_id, result))

    async def _dispatch_request(self, method: str, params: dict[str, Any]) -> Any:
        if method == "initialize":
            return await self._initialize(params)
        if method == "session/new":
            return await self._new_session(params)
        # ``session/prompt`` never reaches here (dispatched as a background
        # task), and ``session/request_permission`` is agent -> client only.
        raise AcpError(ERROR_METHOD_NOT_FOUND, f"Method not found: {method}")

    async def _handle_notification(self, method: str, params: Any) -> None:
        payload = params if isinstance(params, dict) else {}
        if method == "session/cancel":
            await self._cancel(payload)
        # Unknown notifications are ignored per JSON-RPC 2.0.

    async def _handle_prompt_request(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        params = message.get("params")
        try:
            result = await self._prompt(params if isinstance(params, dict) else {})
        except AcpError as exc:
            await self._send(
                error_response(request_id, exc.code, exc.message, exc.data)
            )
            return
        except Exception as exc:  # pragma: no cover - defensive
            _log.debug("ACP prompt failed", exc_info=True)
            await self._send(
                error_response(request_id, ERROR_INTERNAL, f"Internal error: {exc}")
            )
            return
        await self._send(success_response(request_id, result))

    # -- client -> agent handlers -------------------------------------------

    async def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        # The agent reports the protocol version it implements; a client on a
        # newer version negotiates down (or disconnects if incompatible).
        capabilities = params.get("clientCapabilities")
        self._client_capabilities = (
            capabilities if isinstance(capabilities, dict) else {}
        )
        fs = self._client_capabilities.get("fs")
        if isinstance(fs, dict):
            self._fs_read = bool(fs.get("readTextFile"))
            self._fs_write = bool(fs.get("writeTextFile"))
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "agentCapabilities": {
                "loadSession": False,
                "promptCapabilities": {"image": True},
            },
            "agentInfo": {"name": "kiwimatecoder", "version": __version__},
        }

    async def _new_session(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.default_workspace is not None:
            raw_cwd: Any = params.get("cwd") or self.default_workspace
        else:
            raw_cwd = params.get("cwd") or Path.cwd()
        root = Path(str(raw_cwd)).expanduser()
        if not root.is_dir():
            raise AcpError(
                ERROR_INVALID_PARAMS, f"Workspace is not a directory: {root}"
            )
        try:
            session = build_session(workspace=root)
        except (ValueError, KeyError) as exc:
            raise AcpError(ERROR_INVALID_PARAMS, str(exc)) from exc
        # ``mcpServers`` is accepted but not wired: MCP servers come from the
        # user's KiwiMateCoder config. The session uses the configured
        # permission mode (never an assumed auto-accept).
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = _SessionState(session=session)
        return {"sessionId": session_id}

    async def _prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = str(params.get("sessionId") or "")
        state = self._sessions.get(session_id)
        if state is None:
            raise AcpError(ERROR_INVALID_PARAMS, f"Unknown session '{session_id}'.")
        if session_id in self._turns:
            raise AcpError(
                ERROR_SERVER, "A turn is already running for this session."
            )

        text, images = prompt_text(params.get("prompt"))
        if not text.strip() and not images:
            raise AcpError(
                ERROR_INVALID_PARAMS, "prompt must contain at least one text part."
            )
        if images:
            note = f"[{images} image part(s) were ignored: this server has no image input]"
            text = f"{text}\n{note}".strip()

        turn = _TurnState(session_id=session_id)
        self._turns[session_id] = turn

        def on_event(name: str, payload: dict[str, Any]) -> None:
            self._translate_event(state, turn, name, payload)

        def confirm(summary: str, preview: str | None) -> bool:
            return self._request_permission_sync(turn, state, summary)

        try:
            outcome = await asyncio.to_thread(
                self._run_turn_thread, state, turn, text, confirm, on_event
            )
        finally:
            if self._turns.get(session_id) is turn:
                self._turns.pop(session_id, None)

        # The worker thread schedules notifications onto this loop; wait for
        # them so updates always precede the prompt's response.
        await self._flush_updates(turn)

        if turn.cancelled:
            return {"stopReason": "cancelled"}
        # Some paths (budget stop, provider error before text) never stream a
        # delta; surface their text as the one agent message chunk.
        if not turn.saw_text and outcome is not None and outcome.text:
            await self._emit_update(
                session_id,
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": outcome.text},
                },
            )
        return {"stopReason": self._stop_reason(turn, outcome)}

    async def _cancel(self, params: dict[str, Any]) -> None:
        session_id = str(params.get("sessionId") or "")
        turn = self._turns.get(session_id)
        if turn is not None:
            self._cancel_turn(turn)

    def _cancel_turn(self, turn: _TurnState) -> None:
        turn.cancelled = True
        # A thread blocked waiting for the editor's permission answer is woken
        # with a denial so cancellation is never stuck behind that wait.
        self._resolve_pending_for_session(
            turn.session_id, {"outcome": {"outcome": "cancelled"}}
        )
        state = self._sessions.get(turn.session_id)
        if state is not None and state.in_progress:
            for call_id, tool_name in state.in_progress:
                self._emit_update_threadsafe(
                    turn,
                    {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": call_id,
                        "status": "failed",
                        "content": [
                            {
                                "type": "content",
                                "content": {
                                    "type": "text",
                                    "text": f"{tool_name} cancelled",
                                },
                            }
                        ],
                    },
                )
            state.in_progress.clear()
        loop, task = turn.loop, turn.task
        if loop is not None and task is not None:
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass

    def _resolve_pending_for_session(
        self, session_id: str, result: dict[str, Any]
    ) -> None:
        for request_id, pending in list(self._pending.items()):
            if (
                pending.method == "session/request_permission"
                and pending.session_id == session_id
            ):
                if not pending.future.done():
                    pending.future.set_result(result)
                self._pending.pop(request_id, None)

    # -- turn execution (worker thread) -------------------------------------

    def _run_turn_thread(
        self,
        state: _SessionState,
        turn: _TurnState,
        text: str,
        confirm: Callable[[str, str | None], bool],
        on_event: Callable[[str, dict[str, Any]], None],
    ) -> Any:
        """Run one ``run_agent_once`` turn on a dedicated event loop.

        The agent's confirm callback is synchronous, so the turn runs on its
        own loop in a worker thread while this server's loop stays free to
        answer ``session/request_permission`` and ``session/cancel``.
        """
        loop = asyncio.new_event_loop()
        turn.loop = loop
        try:
            asyncio.set_event_loop(loop)
            task = loop.create_task(
                run_agent_once(
                    text,
                    session=state.session,
                    confirm=confirm,
                    on_event=on_event,
                    render_text=False,
                )
            )
            turn.task = task
            if turn.cancelled:
                task.cancel()
            try:
                return loop.run_until_complete(task)
            except asyncio.CancelledError:
                turn.cancelled = True
                return None
        finally:
            turn.loop = None
            turn.task = None
            loop.close()

    def _stop_reason(self, turn: _TurnState, outcome: Any) -> str:
        if turn.cancelled:
            return "cancelled"
        if "max_turns" in turn.done_reasons or "budget" in turn.done_reasons:
            return "max_tokens"
        if outcome is None or not getattr(outcome, "success", False):
            return "refusal"
        return "end_turn"

    # -- permission bridge --------------------------------------------------

    def _request_permission_sync(
        self, turn: _TurnState, state: _SessionState, summary: str
    ) -> bool:
        """Ask the editor to approve an action, denied on timeout/EOF."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return False
        tool_name = summary.split("(", 1)[0].strip()
        call_id = state.allocate_call_id()
        state.pending_permission_id = call_id
        params = {
            "sessionId": turn.session_id,
            "toolCall": {
                "toolCallId": call_id,
                "title": summary,
                "kind": tool_kind(tool_name),
                "status": "pending",
            },
            "options": PERMISSION_OPTIONS,
        }
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._request(
                    "session/request_permission",
                    params,
                    session_id=turn.session_id,
                    timeout=self.permission_timeout,
                ),
                loop,
            )
            result = future.result(timeout=self.permission_timeout + 5.0)
        except Exception:
            state.pending_permission_id = None
            return False

        selected = _selected_option(result)
        if selected is None:
            state.pending_permission_id = None
            return False
        if selected == "allow_always" and tool_name:
            try:
                config.persist_always_allowed_tool(tool_name)
            except OSError:  # pragma: no cover - unwritable config dir
                pass
            state.session.allow_always(tool_name)
        return True

    # -- agent -> client plumbing -------------------------------------------

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        session_id: str | None = None,
        timeout: float | None = None,
    ) -> Any:
        loop = self._loop
        assert loop is not None
        request_id = self._ids.next()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = _PendingRequest(
            future=future, method=method, session_id=session_id
        )
        try:
            await self._send(request_message(request_id, method, params))
            try:
                return await asyncio.wait_for(future, timeout)
            except asyncio.TimeoutError:
                raise AcpTimeoutError(
                    f"{method} timed out after {timeout:g}s"
                ) from None
        finally:
            self._pending.pop(request_id, None)

    def _resolve_response(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        if not isinstance(request_id, int):
            return
        pending = self._pending.get(request_id)
        if pending is None or pending.future.done():
            return
        error = message.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            pending.future.set_exception(
                AcpError(
                    code if isinstance(code, int) else ERROR_INTERNAL,
                    str(error.get("message") or "Client request failed"),
                    error.get("data"),
                )
            )
            return
        pending.future.set_result(message.get("result"))

    def _translate_event(
        self, state: _SessionState, turn: _TurnState, name: str, payload: dict[str, Any]
    ) -> None:
        """Forward one agent render event to the client as ``session/update``."""
        if name == "text_delta":
            text = str(payload.get("text") or "")
            if text:
                turn.saw_text = True
                self._emit_update_threadsafe(
                    turn,
                    {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": text},
                    },
                )
        elif name == "tool_start":
            tool_name = str(payload.get("tool") or "")
            args = payload.get("args")
            call_id, title, kind = state.begin_tool(
                tool_name, args if isinstance(args, dict) else {}
            )
            self._emit_update_threadsafe(
                turn,
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": call_id,
                    "title": title,
                    "kind": kind,
                    "status": "in_progress",
                },
            )
        elif name == "tool_end":
            tool_name = str(payload.get("tool") or "")
            finished_id = state.end_tool(tool_name)
            if finished_id is not None:
                ok = bool(payload.get("ok"))
                duration = payload.get("duration_ms")
                detail = f"{tool_name} {'completed' if ok else 'failed'}"
                if isinstance(duration, int):
                    detail += f" ({duration}ms)"
                self._emit_update_threadsafe(
                    turn,
                    {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": finished_id,
                        "status": "completed" if ok else "failed",
                        "content": [
                            {
                                "type": "content",
                                "content": {"type": "text", "text": detail},
                            }
                        ],
                    },
                )
        elif name == "done":
            turn.done_reasons.append(str(payload.get("reason") or ""))

    def _emit_update_threadsafe(
        self, turn: _TurnState, update: dict[str, Any]
    ) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._emit_update(turn.session_id, update), loop
            )
        except RuntimeError:
            return
        turn.pending_updates.append(future)
        future.add_done_callback(_consume_future_exception)

    async def _flush_updates(self, turn: _TurnState) -> None:
        """Await every ``session/update`` scheduled during the turn."""
        pending, turn.pending_updates = list(turn.pending_updates), []
        if not pending:
            return
        await asyncio.gather(
            *(asyncio.wrap_future(future) for future in pending),
            return_exceptions=True,
        )

    async def _emit_update(self, session_id: str, update: dict[str, Any]) -> None:
        await self._send(
            notification_message(
                "session/update", {"sessionId": session_id, "update": update}
            )
        )

    async def _send(self, message: dict[str, Any]) -> None:
        if self._write_line is None:
            return
        line = encode(message)
        lock = self._write_lock
        if lock is None:
            await self._write(line)
            return
        async with lock:
            await self._write(line)

    async def _write(self, line: str) -> None:
        if self._write_line is None:
            return
        result = self._write_line(line)
        if inspect.isawaitable(result):
            await result

    # -- optional client filesystem delegation ------------------------------

    async def read_text_file(self, session_id: str, path: str) -> str:
        """Read a file, delegating to the client when it advertises fs reads.

        Falls back to the local filesystem otherwise. This is an opt-in layer
        for hosts that call it; the built-in agent tools always operate on the
        local filesystem.
        """
        if self._fs_read and session_id in self._sessions:
            result = await self._request(
                "fs/read_text_file",
                {"sessionId": session_id, "path": path},
                session_id=session_id,
                timeout=self.permission_timeout,
            )
            content = result.get("content") if isinstance(result, dict) else None
            return str(content) if content is not None else ""
        return Path(path).read_text(encoding="utf-8")

    async def write_text_file(self, session_id: str, path: str, content: str) -> None:
        """Write a file, delegating to the client when it advertises fs writes."""
        if self._fs_write and session_id in self._sessions:
            await self._request(
                "fs/write_text_file",
                {"sessionId": session_id, "path": path, "content": content},
                session_id=session_id,
                timeout=self.permission_timeout,
            )
            return
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _selected_option(result: Any) -> str | None:
    """Return the selected option id when the client approved the action."""
    if not isinstance(result, dict):
        return None
    outcome = result.get("outcome")
    if not isinstance(outcome, dict):
        return None
    if outcome.get("outcome") != "selected":
        return None
    option_id = outcome.get("optionId")
    if option_id in {"allow_once", "allow_always"}:
        return str(option_id)
    return None
