"""The agentic tool-calling loop."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.markup import escape
from rich.text import Text

from kiwimatecoder import audit, events, hooks, images, tools, ui
from kiwimatecoder.client import (
    AssembledToolCall,
    Done,
    ProviderError,
    TextDelta,
    ToolCallAssembler,
    ToolCallDelta,
    UnifiedClient,
    Usage,
)
from kiwimatecoder.permissions import ConfirmFn, PermissionMode, gate
from kiwimatecoder.prompts import build_system_prompt
from kiwimatecoder.redaction import redact
from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult

POST_EDIT_DIAGNOSTICS_TIMEOUT = 3.0
MAX_EDIT_DIAGNOSTICS = 10
TASK_RESULT_MAX_CHARS = 8000

EventCallback = Callable[[str, dict[str, Any]], None]

_log = logging.getLogger(__name__)


class TaskConsole(Console):
    """Console wrapper that tags every child line with dim ``[task]``.

    Prints are delegated to the parent console so subagent output stays in the
    same stream (and under the same test harness); the tag is emitted once per
    line even when text is streamed with ``end=""``.
    """

    def __init__(self, parent: Console, label: str = "task") -> None:
        super().__init__(quiet=True)
        self._parent = parent
        self._tag = f"[dim]\\[{label}][/dim]"
        self._at_line_start = True

    def print(self, *objects: Any, **kwargs: Any) -> None:
        end = str(kwargs.get("end", "\n"))
        if objects and self._at_line_start:
            self._parent.print(self._tag, end=" ", markup=True, highlight=False)
            self._at_line_start = False
        self._parent.print(*objects, **kwargs)
        if end.endswith("\n"):
            self._at_line_start = True

    def status(self, status: Any, **kwargs: Any) -> Any:
        return self._parent.status(f"{self._tag} {status}", **kwargs)


@dataclass
class _PreparedCall:
    """A tool call that passed parsing, approval, hooks, and checkpointing."""

    call: AssembledToolCall
    tool: FunctionTool
    args: dict[str, Any]
    original_args: dict[str, Any]
    summary: str
    partial_hunks: tuple[int, ...] | None = None


class Agent:
    """Drives one conversational turn, including any tool calls it triggers."""

    session: Session
    console: Console
    confirm: ConfirmFn

    def __init__(
        self,
        session: Session,
        console: Console,
        confirm: ConfirmFn,
        bus: events.EventBus | None = None,
        ascii_mode: bool = False,
        output_mode: str = "normal",
        event_handler: EventCallback | None = None,
        max_turns: int | None = None,
        render_text: bool = True,
    ) -> None:
        self.session = session
        self.console = console
        self.confirm = confirm
        self.bus = bus if bus is not None else events.BUS
        self._budget_warned = False
        self.ascii_mode = bool(ascii_mode)
        self.output_mode = output_mode if output_mode in ui.OUTPUT_MODES else "normal"
        self.event_handler = event_handler
        self.max_turns = max_turns
        self.render_text = render_text

    def _emit(self, name: str, **payload: Any) -> None:
        """Forward one render event, never letting a bad subscriber break a run."""
        if self.event_handler is None:
            return
        try:
            self.event_handler(name, payload)
        except Exception:
            _log.debug("event handler failed for %s", name, exc_info=True)

    def _redacted_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return a JSON-safe copy of tool arguments with secrets redacted."""
        try:
            rendered = redact(
                json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
            )
            parsed = json.loads(rendered)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except (TypeError, ValueError):
            return {"value": redact(str(args))}

    def _glyph(self, name: str) -> str:
        """Active glyph escaped for direct use inside Rich markup."""
        return escape(ui.glyph(name, {"ui": {"ascii": self.ascii_mode}}))

    def _print_markup(self, markup: str) -> None:
        """Print styled markup as ``Text`` so logs keep the plain glyph text."""
        self.console.print(Text.from_markup(markup))

    def _client(self, provider_id: str | None = None) -> UnifiedClient:
        from kiwimatecoder.config import (
            get_key,
            get_prompt_cache,
            get_provider_config,
            get_sampling,
        )

        provider = get_provider_config(provider_id) if provider_id else self.session.provider
        key = get_key(provider.id)
        if not key and provider.needs_key:
            raise ProviderError(
                f"No API key for {provider.name}. Set one with "
                + f"`config set-key --provider {provider.id} <KEY>` or the "
                + f"{provider.key_env} environment variable."
            )
        return UnifiedClient(
            provider,
            key or "",
            sampling=get_sampling(),
            prompt_cache=get_prompt_cache(),
        )

    def _request_messages(self) -> list[dict[str, Any]]:
        self.session.trim_history(max_tokens=self.session.compact_at_tokens)
        return [build_system_prompt(self.session)] + self.session.messages

    def _drain_steering(self) -> bool:
        """Inject every queued steering message as a user message.

        Returns True when at least one message was queued and appended.
        """
        appended = False
        while self.session.steering:
            message = self.session.steering.popleft()
            self.session.messages.append({"role": "user", "content": message})
            appended = True
        return appended

    def _flush_pending_images(self) -> None:
        """Attach queued images as one user message and clear the queue.

        Called after the user message (for ``@path`` attachments) and after a
        tool batch that attached images, so the message always follows the
        content it belongs to and tool results keep their ordering.
        """
        if not self.session.pending_images:
            return
        pending = list(self.session.pending_images)
        self.session.pending_images = []
        self.session.messages.append(
            images.image_message(pending, note="Images attached for analysis:")
        )

    async def run_turn(self, user_input: str) -> None:
        """Process one user message, looping over tool calls until the model stops."""
        blocked, reason = self._budget_exceeded()
        if blocked:
            self.console.print(f"[red]{reason}[/red]")
            self._emit("done", reason="budget", error=reason)
            return

        from kiwimatecoder.config import get_model_routing
        from kiwimatecoder.routing import choose_turn_model

        routed_model = choose_turn_model(
            user_input, self.session, get_model_routing()
        )

        self._budget_warned = False
        self.session.messages.append({"role": "user", "content": user_input})
        self._flush_pending_images()

        edited = False
        verified = False
        first_pass = True
        tool_turns = 0
        while True:
            if first_pass:
                first_pass = False
            else:
                # Steering queued while the previous response streamed is
                # injected before the next model call.
                self._drain_steering()
            try:
                assistant_msg, tool_calls = await self._stream_once(
                    model_override=routed_model
                )
            except ProviderError as exc:
                self.console.print(f"\n[red]{exc}[/red]")
                self._emit("error", kind="provider", error=str(exc))
                self._emit("done", reason="provider_error")
                return

            self.session.messages.append(assistant_msg)

            if not tool_calls:
                if edited and not verified and self._should_verify():
                    verified = True
                    await self._run_verification()
                    continue
                if self._drain_steering():
                    # The model must answer the steered message.
                    continue
                self._warn_budget()
                self._emit("done", reason="stop")
                return

            if self.max_turns is not None and tool_turns >= self.max_turns:
                self._emit("done", reason="max_turns", limit=self.max_turns)
                return
            edited = await self._handle_tool_calls(tool_calls) or edited
            tool_turns += 1
            self._warn_budget()

    def _current_cost(self) -> float | None:
        from kiwimatecoder.pricing import estimate_cost

        return estimate_cost(
            self.session.prompt_tokens,
            self.session.completion_tokens,
            self.session.model,
            self.session.provider_id,
            is_local=self.session.provider.is_local,
        )

    def _budget_exceeded(self) -> tuple[bool, str]:
        """Whether a configured token/cost budget has already been spent."""
        from kiwimatecoder.config import get_budget

        budget = get_budget()
        max_tokens = budget.get("max_tokens")
        if max_tokens and self.session.total_tokens >= max_tokens:
            return True, (
                f"Budget reached: {self.session.total_tokens:,} tokens used "
                f"(limit {int(max_tokens):,}). Raise it with "
                "`/config budget tokens <n>` or clear it with `/config budget clear`."
            )
        max_cost = budget.get("max_cost_usd")
        if max_cost:
            cost = self._current_cost()
            if cost is not None and cost >= max_cost:
                return True, (
                    f"Budget reached: ~${cost:.4f} spent "
                    f"(limit ${max_cost:.4f}). Raise it with "
                    "`/config budget cost <usd>` or clear it with `/config budget clear`."
                )
        return False, ""

    def _warn_budget(self) -> None:
        """Print one warning when usage crosses 80% of a configured budget."""
        if self._budget_warned:
            return
        from kiwimatecoder.config import get_budget

        budget = get_budget()
        max_tokens = budget.get("max_tokens")
        if max_tokens and self.session.total_tokens >= 0.8 * max_tokens:
            self._budget_warned = True
            self.console.print(
                f"[yellow]Budget warning: {self.session.total_tokens:,} of "
                f"{int(max_tokens):,} tokens used.[/yellow]"
            )
            return
        max_cost = budget.get("max_cost_usd")
        if max_cost:
            cost = self._current_cost()
            if cost is not None and cost >= 0.8 * max_cost:
                self._budget_warned = True
                self.console.print(
                    f"[yellow]Budget warning: ~${cost:.4f} of "
                    f"${max_cost:.4f} spent.[/yellow]"
                )

    def _should_verify(self) -> bool:
        return bool(
            self.session.verify_command.strip()
            and self.session.mode is not PermissionMode.PLAN
        )

    async def _run_verification(self) -> None:
        """Run the configured verify command and feed its output back."""
        command = self.session.verify_command.strip()
        verify_tool = tools.get_tool("run_bash")
        if verify_tool is None:
            return
        self.console.print(f"\n[bold]Auto-verify:[/bold] [cyan]{command}[/cyan]")
        t0 = time.perf_counter()
        result = await asyncio.to_thread(
            verify_tool.execute, {"command": command}, self.session
        )
        duration_ms = int((time.perf_counter() - t0) * 1000)
        audit.record_tool_event(
            tool="run_bash",
            args={"command": command},
            decision="auto_verify",
            duration_ms=duration_ms,
            ok=result.ok,
        )
        status = "[green]passed[/green]" if result.ok else "[red]failed[/red]"
        self.console.print(f"Auto-verify {status} [dim]({duration_ms}ms)[/dim]")
        content = result.content
        if len(content) > 6000:
            content = content[:6000] + "\n... [truncated]"
        self.session.messages.append(
            {
                "role": "user",
                "content": (
                    f"[auto-verify] `{command}` "
                    f"{'passed' if result.ok else 'failed'}:\n"
                    f"```text\n{content}\n```\n"
                    + (
                        "Fix the failures above, then re-run the verification."
                        if not result.ok
                        else "Briefly summarize what changed, then stop."
                    )
                ),
            }
        )

    def _tool_schemas(self) -> list[dict[str, Any]]:
        """Tool schemas advertised for the next request.

        In plan mode only read-only tools are advertised (the gate blocks the
        rest as a second line of defense). Subagents never see ``task`` (no
        nested subagents) or ``ask_user`` (no interactive user), and ``task``
        is hidden for everyone while subagents are disabled. ``browser`` is
        hidden until browser automation is enabled in config.
        """
        from kiwimatecoder.config import get_browser, get_subagents

        schemas = tools.tool_schemas(
            read_only=self.session.mode is PermissionMode.PLAN
        )
        excluded: set[str] = set()
        if self.session.subagent:
            excluded.update({"task", "ask_user"})
        if not get_subagents()["enabled"]:
            excluded.add("task")
        if not get_browser()["enabled"]:
            excluded.add("browser")
        if excluded:
            schemas = [
                schema
                for schema in schemas
                if schema["function"]["name"] not in excluded
            ]
        return schemas

    async def _stream_once(
        self, model_override: str | None = None
    ) -> tuple[dict[str, Any], list[AssembledToolCall]]:
        """Stream one assistant response, rendering text and collecting tool calls.

        Tries each active provider in order (primary first); when a provider
        fails with a :class:`ProviderError`, the next active provider is tried
        with its own default model. Only when every active provider fails is the
        error surfaced.

        ``model_override`` (per-turn model routing) applies to the primary
        provider only; fallback providers keep their own model.
        """
        schemas = self._tool_schemas()

        errors: list[ProviderError] = []
        providers = self.session.active_providers
        for index, provider in enumerate(providers):
            try:
                client = self._client(provider.id)
            except ProviderError as exc:
                errors.append(exc)
                self._announce_failover(provider.name, exc, providers[index + 1 :])
                continue
            model = self.session.model_for(provider.id)
            if model_override and provider.id == self.session.provider_id:
                if model_override != model:
                    self.console.print(f"[dim]routed to {model_override}[/dim]")
                model = model_override
            try:
                return await self._stream_from(client, schemas, model)
            except ProviderError as exc:
                errors.append(exc)
                self._announce_failover(provider.name, exc, providers[index + 1 :])
                continue

        if errors:
            raise ProviderError(
                "All active providers failed: "
                + "; ".join(str(exc) for exc in errors)
            )
        raise ProviderError("No active providers are configured.")

    def _announce_failover(
        self,
        provider_name: str,
        exc: ProviderError,
        remaining: list[Any],
    ) -> None:
        if not remaining:
            return
        self.console.print(
            f"\n[yellow]{provider_name} failed ({exc}); "
            f"trying {remaining[0].name}.[/yellow]"
        )

    async def _stream_from(
        self,
        client: UnifiedClient,
        schemas: list[dict[str, Any]],
        model: str,
    ) -> tuple[dict[str, Any], list[AssembledToolCall]]:
        """Stream from one client, rendering text and collecting tool calls."""
        text_parts: list[str] = []
        assembler = ToolCallAssembler()
        printed_any = False
        status = self.console.status("[dim]Thinking…[/dim]")
        status.start()
        thinking = True

        try:
            async for event in client.stream_chat(
                self._request_messages(), schemas, model
            ):
                if isinstance(event, TextDelta):
                    if thinking:
                        status.stop()
                        thinking = False
                    if self.render_text:
                        self.console.print(
                            event.text, end="", markup=False, highlight=False
                        )
                        printed_any = True
                    self._emit("text_delta", text=event.text)
                    text_parts.append(event.text)
                elif isinstance(event, ToolCallDelta):
                    assembler.add(event)
                elif isinstance(event, Usage):
                    self.session.add_usage(event.prompt_tokens, event.completion_tokens)
                    self._emit(
                        "usage",
                        prompt_tokens=event.prompt_tokens,
                        completion_tokens=event.completion_tokens,
                    )
                elif isinstance(event, Done):
                    pass
        except BaseException:
            # Preserve the text the user already saw when a turn is interrupted
            # (Ctrl-C) or the stream fails partway through.
            partial = "".join(text_parts)
            if partial:
                partial_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": partial,
                }
                if (
                    not self.session.messages
                    or self.session.messages[-1] != partial_msg
                ):
                    self.session.messages.append(partial_msg)
            raise
        finally:
            if thinking:
                status.stop()

        if printed_any:
            self.console.print()

        calls = assembler.finalize()
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
        if calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in calls
            ]
        return assistant_msg, calls

    def _auto_confirm(self, summary: str, preview: str | None) -> bool:
        """Approval stand-in used in dry-run mode so nothing prompts."""
        return True

    def _format_call_summary(self, name: str, args: dict[str, Any]) -> str:
        """Produce a short human-readable string summarizing the tool call arguments."""
        if name in ("read_file", "write_file", "edit_file", "list_dir"):
            target = args.get("path", ".")
            return f"{name} [dim]{target}[/dim]"
        if name == "view_image":
            target = args.get("path", ".")
            return f"image [dim]{target}[/dim]"
        if name == "generate_image":
            prompt = str(args.get("prompt", "") or "")
            short = prompt if len(prompt) <= 50 else f"{prompt[:47]}..."
            return f"image-gen [dim]{short}[/dim]"
        if name == "search":
            pat = args.get("pattern", "")
            mode = args.get("mode", "grep")
            return f"search [dim]{pat}[/dim] ({mode})"
        if name == "run_bash":
            cmd = str(args.get("command", "") or "")
            cmd_short = cmd if len(cmd) <= 40 else f"{cmd[:37]}..."
            return f"bash [dim]`{cmd_short}`[/dim]"
        if name == "shell":
            cmd = str(args.get("command", "") or "")
            cmd_short = cmd if len(cmd) <= 40 else f"{cmd[:37]}..."
            return f"shell [dim]`{cmd_short}`[/dim]"
        if name == "shell_jobs":
            action = str(args.get("action", "") or "")
            detail = str(args.get("command") or args.get("id") or "")
            short = detail if len(detail) <= 40 else f"{detail[:37]}..."
            suffix = f" [dim]{short}[/dim]" if short else ""
            return f"shell_jobs [dim]{action}[/dim]{suffix}"
        if name == "update_todos":
            todos = args.get("todos")
            count = len(todos) if isinstance(todos, list) else 0
            return f"todos [dim]({count} item{'s' if count != 1 else ''})[/dim]"
        if name == "ask_user":
            question = str(args.get("question", "") or "")
            short = question if len(question) <= 50 else f"{question[:47]}..."
            return f"ask [dim]{short}[/dim]"
        if name == "load_skill":
            skill = str(args.get("name", "") or "")
            return f"skill [dim]{skill}[/dim]"
        if name == "web_fetch":
            url = str(args.get("url", "") or "")
            short = url if len(url) <= 60 else f"{url[:57]}..."
            return f"web_fetch [dim]{short}[/dim]"
        if name == "web_search":
            query = str(args.get("query", "") or "")
            short = query if len(query) <= 50 else f"{query[:47]}..."
            return f"web_search [dim]{short}[/dim]"
        if name in ("git", "git_write", "forge", "forge_write"):
            action = str(args.get("action", "") or "")
            return f"{name} [dim]{action}[/dim]"
        if name == "remember":
            return "memory [dim]remember[/dim]"
        if name == "recall":
            query = str(args.get("query", "") or "")
            short = query if len(query) <= 40 else f"{query[:37]}..."
            return f"memory [dim]recall {short}[/dim]" if short else "memory [dim]recall[/dim]"
        if name == "read_clipboard":
            return "clipboard [dim]read[/dim]"
        if name == "write_clipboard":
            text = str(args.get("text", "") or "")
            short = text if len(text) <= 40 else f"{text[:37]}..."
            return f"clipboard [dim]write {short}[/dim]"
        if name in ("http_get", "http_request"):
            url = str(args.get("url", "") or "")
            short = url if len(url) <= 60 else f"{url[:57]}..."
            method = str(args.get("method", "") or "GET").upper()
            label = "GET" if name == "http_get" else method
            return f"http [dim]{label} {short}[/dim]"
        if name in ("lsp_diagnostics", "lsp_definition", "lsp_references"):
            action = name[len("lsp_") :]
            path = str(args.get("path", "") or "")
            if action == "diagnostics" and not path:
                return "lsp [dim]diagnostics recent files[/dim]"
            return f"lsp [dim]{action} {path}[/dim]"
        if name == "task":
            description = str(args.get("description", "") or "")
            short = description if len(description) <= 50 else f"{description[:47]}..."
            return f"task [dim]{short}[/dim]"
        if name == "browser":
            action = str(args.get("action", "") or "")
            target = str(
                args.get("url")
                or args.get("selector")
                or args.get("script")
                or args.get("text")
                or ""
            )
            short = target if len(target) <= 60 else f"{target[:57]}..."
            detail = f"{action} {short}".strip()
            return f"browser [dim]{detail}[/dim]"
        return name

    # Only purely read-only tools are safe to run concurrently: they do not
    # mutate session state or the workspace.
    _PARALLEL_SAFE = frozenset({"read_file", "list_dir", "search"})

    def _can_run_parallel(self, calls: list[AssembledToolCall]) -> bool:
        if len(calls) < 2:
            return False
        for call in calls:
            tool = tools.get_tool(call.name)
            if tool is None or tool.needs_approval or call.name not in self._PARALLEL_SAFE:
                return False
        return True

    async def _handle_tool_calls(self, calls: list[AssembledToolCall]) -> bool:
        """Run a batch of tool calls; returns whether any file edit succeeded."""
        results: list[tuple[dict[str, Any], bool]]
        if self._can_run_parallel(calls):
            results = await asyncio.gather(
                *(asyncio.to_thread(self._run_tool_call, call) for call in calls)
            )
        else:
            results = []
            for call in calls:
                # ``task`` drives a nested async loop, so it cannot go through
                # the synchronous runner (or a worker thread).
                if call.name == "task":
                    results.append(await self._run_task_call(call))
                else:
                    results.append(self._run_tool_call(call))
        self.session.messages.extend(message for message, _edited in results)
        # Images attached by this batch go after its tool results (and only
        # once, even when several calls attached one).
        self._flush_pending_images()
        return any(edited for _message, edited in results)

    def _tool_message(self, tool_call_id: str, content: str) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    def _prepare_tool_call(
        self, call: AssembledToolCall
    ) -> tuple[_PreparedCall | None, tuple[dict[str, Any], bool] | None]:
        """Parse, gate, hook-check, and checkpoint one tool call.

        Returns ``(prepared, None)`` when the call may execute, or
        ``(None, early_result)`` with the tool message to return instead
        (unknown tool, bad JSON, denied, dry-run, or hook-blocked).
        """
        tool = tools.get_tool(call.name)
        if tool is None:
            return None, (
                self._tool_message(call.id, f"Error: unknown tool '{call.name}'"),
                False,
            )
        if call.name == "task" and self.session.subagent:
            return None, (
                self._tool_message(
                    call.id, "Error: subagents cannot spawn nested subagents."
                ),
                False,
            )

        try:
            args = call.parse_arguments()
        except json.JSONDecodeError as exc:
            return None, (
                self._tool_message(
                    call.id, f"Error: could not parse arguments as JSON: {exc}"
                ),
                False,
            )

        summary = self._format_call_summary(call.name, args)
        preview_text = tools.preview(call.name, args, self.session)
        confirm = (
            self._auto_confirm
            if self.session.dry_run and tool.needs_approval
            else self.confirm
        )
        decision = gate(tool, args, self.session, confirm, preview_text)
        if not decision.allowed:
            audit.record_tool_event(
                tool=call.name,
                args=args,
                decision="denied",
                reason=decision.reason,
            )
            self._print_markup(
                f"[yellow]{self._glyph('blocked')} {summary}: {decision.reason}[/yellow]"
            )
            return None, (self._tool_message(call.id, decision.reason), False)

        if self.session.dry_run and tool.needs_approval:
            audit.record_tool_event(tool=call.name, args=args, decision="dry_run")
            self.console.print(f"[yellow]dry-run[/yellow] {summary}")
            if preview_text:
                self.console.print(preview_text, markup=False, highlight=False)
            return None, (
                self._tool_message(
                    call.id,
                    f"DRY RUN: {summary} was not executed; no files or state changed.",
                ),
                False,
            )

        original_args = args
        partial_hunks: tuple[int, ...] | None = None
        if decision.selected_hunks is not None:
            selected_args = (
                tools.select_hunks(
                    call.name, args, self.session, decision.selected_hunks
                )
                if call.name in ("write_file", "edit_file")
                else None
            )
            if selected_args is None:
                reason = (
                    "Partial hunk selection could not be applied; no changes "
                    "were made."
                )
                audit.record_tool_event(
                    tool=call.name,
                    args=original_args,
                    decision="denied",
                    reason=reason,
                    hunks=decision.selected_hunks,
                )
                self._print_markup(
                    f"[yellow]{self._glyph('blocked')} {summary}: {reason}[/yellow]"
                )
                return None, (self._tool_message(call.id, reason), False)
            args = selected_args
            partial_hunks = decision.selected_hunks

        pre_results = self._run_pre_tool_hooks(call.name, args)
        blocked = next((item for item in pre_results if item.blocked), None)
        if blocked is not None:
            reason = self._hook_block_reason(blocked)
            audit.record_tool_event(
                tool=call.name,
                args=original_args,
                decision="hook_blocked",
                reason=reason,
            )
            self._print_markup(
                f"[red]{self._glyph('blocked')} {summary}: {reason}[/red]"
            )
            return None, (
                self._tool_message(call.id, self._hook_block_message(blocked)),
                False,
            )

        if call.name in ("write_file", "edit_file"):
            path = str(args.get("path") or "").strip()
            if path:
                self.session.checkpoint([path], f"{call.name} {path}")

        return (
            _PreparedCall(
                call=call,
                tool=tool,
                args=args,
                original_args=original_args,
                summary=summary,
                partial_hunks=partial_hunks,
            ),
            None,
        )

    def _run_tool_call(
        self, call: AssembledToolCall
    ) -> tuple[dict[str, Any], bool]:
        """Execute one tool call; returns (tool message, edited_file)."""
        prepared, early = self._prepare_tool_call(call)
        if early is not None:
            return early
        assert prepared is not None
        self._emit(
            "tool_start",
            tool=prepared.call.name,
            args=self._redacted_args(prepared.args),
        )
        if self.output_mode == "verbose":
            self.console.print(self._verbose_args(prepared.args))

        t0 = time.perf_counter()
        try:
            with self.console.status(f"{prepared.summary}…"):
                result = prepared.tool.execute(prepared.args, self.session)
        except Exception as exc:
            result = ToolResult.error(f"Tool crashed: {exc!r}")
        return self._finish_tool_call(prepared, result, t0)

    async def _run_task_call(
        self, call: AssembledToolCall
    ) -> tuple[dict[str, Any], bool]:
        """Execute a ``task`` call, awaiting the nested subagent loop."""
        prepared, early = self._prepare_tool_call(call)
        if early is not None:
            return early
        assert prepared is not None
        self._emit(
            "tool_start",
            tool=prepared.call.name,
            args=self._redacted_args(prepared.args),
        )
        if self.output_mode == "verbose":
            self.console.print(self._verbose_args(prepared.args))

        t0 = time.perf_counter()
        try:
            with self.console.status(f"{prepared.summary}…"):
                result = await self._run_task(prepared.args)
        except Exception as exc:
            result = ToolResult.error(f"Tool crashed: {exc!r}")
        return self._finish_tool_call(prepared, result, t0)

    def _finish_tool_call(
        self,
        prepared: _PreparedCall,
        result: ToolResult,
        started_at: float,
    ) -> tuple[dict[str, Any], bool]:
        """Record, render, and package the outcome of a prepared call."""
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        decision_value = (
            "allowed_partial" if prepared.partial_hunks is not None else "allowed"
        )
        audit.record_tool_event(
            tool=prepared.call.name,
            args=prepared.original_args,
            decision=decision_value,
            duration_ms=duration_ms,
            ok=result.ok,
            hunks=prepared.partial_hunks,
        )
        self._run_post_tool_hooks(
            prepared.call.name,
            prepared.original_args,
            result.ok,
            duration_ms,
            decision_value,
        )
        self._emit(
            "tool_end",
            tool=prepared.call.name,
            ok=result.ok,
            duration_ms=duration_ms,
        )

        if result.ok:
            if self.output_mode != "compact":
                self._print_markup(
                    f"[bold green]{self._glyph('check')}[/bold green] "
                    f"{prepared.summary} [dim]({duration_ms}ms)[/dim]"
                )
        else:
            self._print_markup(
                f"[bold red]{self._glyph('cross')}[/bold red] "
                f"{prepared.summary} [red](failed)[/red] [dim]({duration_ms}ms)[/dim]"
            )
        if self.output_mode == "verbose":
            self.console.print(f"[dim]result: {len(result.content):,} chars[/dim]")
        edited = prepared.call.name in ("write_file", "edit_file") and result.ok
        content = result.content
        if result.ok and prepared.partial_hunks is not None:
            content += (
                f" (applied hunks: {', '.join(str(h) for h in prepared.partial_hunks)})"
            )
        if edited:
            content += self._post_edit_diagnostics(prepared.args)
        return self._tool_message(prepared.call.id, content), edited

    def _build_child_session(self, model: str) -> Session:
        """Create the isolated session a subagent runs in.

        The child inherits the workspace, provider roster/model, permission
        mode, rules, and trust settings, but starts with empty conversation
        history, context, todos, checkpoints, and steering.
        """
        parent = self.session
        return Session(
            provider_id=parent.provider_id,
            model=model,
            mode=parent.mode,
            workspace_root=parent.workspace_root,
            always_allowed=set(parent.always_allowed),
            active_provider_ids=list(parent.active_provider_ids),
            models=dict(parent.models),
            output_style=parent.output_style,
            custom_system_prompt=parent.custom_system_prompt,
            command_rules={
                kind: list(patterns)
                for kind, patterns in parent.command_rules.items()
            },
            dry_run=parent.dry_run,
            trusted_workspace=parent.trusted_workspace,
            compact_at_tokens=parent.compact_at_tokens,
            context_window=parent.context_window,
            subagent=True,
        )

    async def _run_task(self, args: dict[str, Any]) -> ToolResult:
        """Run one subagent to completion and return its final report."""
        from kiwimatecoder.config import get_budget, get_subagents

        settings = get_subagents()
        if not settings["enabled"]:
            return ToolResult.error(
                "Subagents are disabled. Enable them with "
                "`/config subagents enable on`."
            )
        prompt = str(args.get("prompt") or "").strip()
        if not prompt:
            return ToolResult.error("'prompt' is required")

        token_cap: int | None = None
        max_tokens = get_budget().get("max_tokens")
        if max_tokens:
            remaining = int(max_tokens) - self.session.total_tokens
            if remaining <= 0:
                return ToolResult.error(
                    "The token budget is already exhausted; raise it with "
                    "`/config budget tokens <n>` before delegating."
                )
            token_cap = remaining

        model = (
            str(args.get("model") or "").strip()
            or str(settings["model"])
            or self.session.model
        )
        max_steps = int(settings["max_steps"])
        child_session = self._build_child_session(model)
        child = Agent(
            child_session,
            TaskConsole(self.console),
            self.confirm,
            bus=self.bus,
            ascii_mode=self.ascii_mode,
            output_mode=self.output_mode,
        )

        child_session.messages.append({"role": "user", "content": prompt})
        steps = 0
        last_text = ""
        last_state = ""
        stop_note = ""
        try:
            while True:
                try:
                    assistant_msg, tool_calls = await child._stream_once()
                except ProviderError as exc:
                    return ToolResult.error(f"Subagent failed: {exc}")
                child_session.messages.append(assistant_msg)
                if assistant_msg.get("content"):
                    last_text = str(assistant_msg["content"])
                if not tool_calls:
                    break
                await child._handle_tool_calls(tool_calls)
                steps += 1
                last_message = child_session.messages[-1]
                if last_message.get("role") == "tool":
                    last_state = str(last_message.get("content") or "")
                if steps >= max_steps:
                    stop_note = f"step limit ({max_steps}) reached"
                    break
                if token_cap is not None and child_session.total_tokens >= token_cap:
                    stop_note = (
                        f"inherited token budget ({token_cap:,} tokens) reached"
                    )
                    break
        finally:
            # Subagent usage counts against the parent's budget and its edits
            # stay undoable from the parent conversation.
            self.session.add_usage(
                child_session.prompt_tokens, child_session.completion_tokens
            )
            for path in child_session.touched_files:
                self.session.record_touched(path)
            if child_session.checkpoints:
                self.session.checkpoints.extend(child_session.checkpoints)
                if self.session.checkpoint_store is None:
                    self.session.checkpoint_store = child_session.checkpoint_store

        report = last_text.strip() or last_state.strip() or "(no final answer)"
        if stop_note:
            report += f"\n\n[task stopped: {stop_note}; partial report]"
        content = (
            f"[task result]\n{report}\n\n"
            f"[task: {steps} step(s), {child_session.total_tokens:,} token(s)]"
        )
        if len(content) > TASK_RESULT_MAX_CHARS:
            content = content[:TASK_RESULT_MAX_CHARS] + "\n... [truncated]"
        return ToolResult(content=content)

    def _post_edit_diagnostics(self, args: dict[str, Any]) -> str:
        """Return ``LSP:`` lines for an edited file, or an empty string.

        Best-effort and bounded: disabled config, a missing server, a timeout,
        or any other failure appends nothing and never delays past the
        configured timeout (capped further for post-edit latency).
        """
        from kiwimatecoder import config, lsp
        from kiwimatecoder.tools.lsp import format_diagnostic

        path = str(args.get("path") or "").strip()
        if not path:
            return ""
        try:
            settings = config.get_lsp()
            if not (settings["enabled"] and settings["diagnostics_after_edits"]):
                return ""
            target = Path(path)
            if not target.is_absolute():
                target = self.session.workspace_root / target
            manager = lsp.get_manager()
            if manager is None:
                manager = lsp.ensure_manager(root=self.session.workspace_root)
            timeout = min(float(settings["timeout"]), POST_EDIT_DIAGNOSTICS_TIMEOUT)
            diagnostics = manager.diagnostics_for(str(target), timeout=timeout)
        except Exception:
            return ""
        if not diagnostics:
            return ""
        shown = diagnostics[:MAX_EDIT_DIAGNOSTICS]
        lines = [f"LSP: {format_diagnostic(path, item)}" for item in shown]
        hidden = len(diagnostics) - len(shown)
        if hidden > 0:
            lines.append(f"LSP: … {hidden} more diagnostic(s)")
        return "\n" + "\n".join(lines)

    def _verbose_args(self, args: dict[str, Any]) -> str:
        """Render the verbose-mode argument block: redacted and capped."""
        try:
            rendered = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            rendered = str(args)
        rendered = redact(rendered)
        if len(rendered) > 500:
            rendered = rendered[:497] + "..."
        return f"[dim]args: {escape(rendered)}[/dim]"

    def _run_pre_tool_hooks(
        self, name: str, args: dict[str, Any]
    ) -> list[hooks.HookResult]:
        """Emit PRE_TOOL and run configured pre-tool shell hooks."""
        self.bus.emit(events.PRE_TOOL, tool=name, args=args)
        return hooks.run_hooks(
            events.PRE_TOOL,
            session=self.session,
            console=self.console,
            tool_name=name,
            tool_args=args,
        )

    def _run_post_tool_hooks(
        self,
        name: str,
        args: dict[str, Any],
        ok: bool,
        duration_ms: int,
        decision: str,
    ) -> None:
        """Emit POST_TOOL and run configured post-tool shell hooks."""
        self.bus.emit(
            events.POST_TOOL,
            tool=name,
            args=args,
            ok=ok,
            duration_ms=duration_ms,
            decision=decision,
        )
        hooks.run_hooks(
            events.POST_TOOL,
            session=self.session,
            console=self.console,
            tool_name=name,
            tool_args=args,
            ok=ok,
            duration_ms=duration_ms,
        )

    def _hook_block_reason(self, result: hooks.HookResult) -> str:
        detail = "timed out" if result.timed_out else f"exit code {result.exit_code}"
        return f"blocked by pre_tool hook ({detail}): {redact(result.command)}"

    def _hook_block_message(self, result: hooks.HookResult) -> str:
        message = (
            f"Error: {self._hook_block_reason(result)}. The action was not "
            "executed. Fix the problem the hook reported, then try again."
        )
        output = redact(result.output.strip())
        if output:
            if len(output) > 2000:
                output = output[:2000] + "\n... [truncated]"
            message += f"\n{output}"
        return message
