"""The agentic tool-calling loop."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from rich.console import Console

from kiwimatecoder import audit, events, hooks, tools
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
from kiwimatecoder.tools.base import ToolResult


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
    ) -> None:
        self.session = session
        self.console = console
        self.confirm = confirm
        self.bus = bus if bus is not None else events.BUS
        self._budget_warned = False

    def _client(self, provider_id: str | None = None) -> UnifiedClient:
        from kiwimatecoder.config import get_key, get_provider_config, get_sampling

        provider = get_provider_config(provider_id) if provider_id else self.session.provider
        key = get_key(provider.id)
        if not key and provider.needs_key:
            raise ProviderError(
                f"No API key for {provider.name}. Set one with "
                + f"`config set-key --provider {provider.id} <KEY>` or the "
                + f"{provider.key_env} environment variable."
            )
        return UnifiedClient(provider, key or "", sampling=get_sampling())

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

    async def run_turn(self, user_input: str) -> None:
        """Process one user message, looping over tool calls until the model stops."""
        blocked, reason = self._budget_exceeded()
        if blocked:
            self.console.print(f"[red]{reason}[/red]")
            return

        self._budget_warned = False
        self.session.messages.append({"role": "user", "content": user_input})

        edited = False
        verified = False
        first_pass = True
        while True:
            if first_pass:
                first_pass = False
            else:
                # Steering queued while the previous response streamed is
                # injected before the next model call.
                self._drain_steering()
            try:
                assistant_msg, tool_calls = await self._stream_once()
            except ProviderError as exc:
                self.console.print(f"\n[red]{exc}[/red]")
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
                return

            edited = await self._handle_tool_calls(tool_calls) or edited
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

    async def _stream_once(self) -> tuple[dict[str, Any], list[AssembledToolCall]]:
        """Stream one assistant response, rendering text and collecting tool calls.

        Tries each active provider in order (primary first); when a provider
        fails with a :class:`ProviderError`, the next active provider is tried
        with its own default model. Only when every active provider fails is the
        error surfaced.
        """
        read_only = self.session.mode is PermissionMode.PLAN
        schemas = tools.tool_schemas(read_only=read_only)

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
                    self.console.print(
                        event.text, end="", markup=False, highlight=False
                    )
                    text_parts.append(event.text)
                    printed_any = True
                elif isinstance(event, ToolCallDelta):
                    assembler.add(event)
                elif isinstance(event, Usage):
                    self.session.add_usage(event.prompt_tokens, event.completion_tokens)
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
        if name == "search":
            pat = args.get("pattern", "")
            mode = args.get("mode", "grep")
            return f"search [dim]{pat}[/dim] ({mode})"
        if name == "run_bash":
            cmd = str(args.get("command", "") or "")
            cmd_short = cmd if len(cmd) <= 40 else f"{cmd[:37]}..."
            return f"bash [dim]`{cmd_short}`[/dim]"
        if name == "update_todos":
            todos = args.get("todos")
            count = len(todos) if isinstance(todos, list) else 0
            return f"todos [dim]({count} item{'s' if count != 1 else ''})[/dim]"
        if name == "ask_user":
            question = str(args.get("question", "") or "")
            short = question if len(question) <= 50 else f"{question[:47]}..."
            return f"ask [dim]{short}[/dim]"
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
        if self._can_run_parallel(calls):
            results = await asyncio.gather(
                *(asyncio.to_thread(self._run_tool_call, call) for call in calls)
            )
        else:
            results = [self._run_tool_call(call) for call in calls]
        self.session.messages.extend(message for message, _edited in results)
        return any(edited for _message, edited in results)

    def _tool_message(self, tool_call_id: str, content: str) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    def _run_tool_call(
        self, call: AssembledToolCall
    ) -> tuple[dict[str, Any], bool]:
        """Execute one tool call; returns (tool message, edited_file)."""
        tool = tools.get_tool(call.name)
        if tool is None:
            return (
                self._tool_message(call.id, f"Error: unknown tool '{call.name}'"),
                False,
            )

        try:
            args = call.parse_arguments()
        except json.JSONDecodeError as exc:
            return (
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
            self.console.print(f"[yellow]⊘ {summary}: {decision.reason}[/yellow]")
            return self._tool_message(call.id, decision.reason), False

        if self.session.dry_run and tool.needs_approval:
            audit.record_tool_event(tool=call.name, args=args, decision="dry_run")
            self.console.print(f"[yellow]dry-run[/yellow] {summary}")
            if preview_text:
                self.console.print(preview_text, markup=False, highlight=False)
            return (
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
                self.console.print(f"[yellow]⊘ {summary}: {reason}[/yellow]")
                return self._tool_message(call.id, reason), False
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
            self.console.print(f"[red]⊘ {summary}: {reason}[/red]")
            return self._tool_message(call.id, self._hook_block_message(blocked)), False

        if call.name in ("write_file", "edit_file"):
            path = str(args.get("path") or "").strip()
            if path:
                self.session.checkpoint([path], f"{call.name} {path}")

        t0 = time.perf_counter()
        try:
            with self.console.status(f"{summary}…"):
                result = tool.execute(args, self.session)
        except Exception as exc:
            result = ToolResult.error(f"Tool crashed: {exc!r}")
        duration_ms = int((time.perf_counter() - t0) * 1000)
        decision_value = (
            "allowed_partial" if partial_hunks is not None else "allowed"
        )
        audit.record_tool_event(
            tool=call.name,
            args=original_args,
            decision=decision_value,
            duration_ms=duration_ms,
            ok=result.ok,
            hunks=partial_hunks,
        )
        self._run_post_tool_hooks(
            call.name, original_args, result.ok, duration_ms, decision_value
        )

        if result.ok:
            self.console.print(f"[bold green]✓[/bold green] {summary} [dim]({duration_ms}ms)[/dim]")
        else:
            self.console.print(f"[bold red]✗[/bold red] {summary} [red](failed)[/red] [dim]({duration_ms}ms)[/dim]")
        edited = call.name in ("write_file", "edit_file") and result.ok
        content = result.content
        if result.ok and partial_hunks is not None:
            content += f" (applied hunks: {', '.join(str(h) for h in partial_hunks)})"
        return self._tool_message(call.id, content), edited

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
