"""The agentic tool-calling loop."""

from __future__ import annotations

import json
import time
from typing import Any

from rich.console import Console

from kiwimatecoder import tools
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
from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import ToolResult


class Agent:
    """Drives one conversational turn, including any tool calls it triggers."""

    session: Session
    console: Console
    confirm: ConfirmFn

    def __init__(self, session: Session, console: Console, confirm: ConfirmFn) -> None:
        self.session = session
        self.console = console
        self.confirm = confirm

    def _client(self, provider_id: str | None = None) -> UnifiedClient:
        from kiwimatecoder.config import get_key, get_provider_config

        provider = get_provider_config(provider_id) if provider_id else self.session.provider
        key = get_key(provider.id)
        if not key and provider.needs_key:
            raise ProviderError(
                f"No API key for {provider.name}. Set one with "
                + f"`config set-key --provider {provider.id} <KEY>` or the "
                + f"{provider.key_env} environment variable."
            )
        return UnifiedClient(provider, key or "")

    def _request_messages(self) -> list[dict[str, Any]]:
        self.session.trim_history()
        return [build_system_prompt(self.session)] + self.session.messages

    async def run_turn(self, user_input: str) -> None:
        """Process one user message, looping over tool calls until the model stops."""
        self.session.messages.append({"role": "user", "content": user_input})

        while True:
            try:
                assistant_msg, tool_calls = await self._stream_once()
            except ProviderError as exc:
                self.console.print(f"\n[red]{exc}[/red]")
                return

            self.session.messages.append(assistant_msg)

            if not tool_calls:
                return

            for call in tool_calls:
                self._handle_tool_call(call)

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

        async for event in client.stream_chat(self._request_messages(), schemas, model):
            if isinstance(event, TextDelta):
                self.console.print(event.text, end="", markup=False, highlight=False)
                text_parts.append(event.text)
                printed_any = True
            elif isinstance(event, ToolCallDelta):
                assembler.add(event)
            elif isinstance(event, Usage):
                self.session.add_usage(event.prompt_tokens, event.completion_tokens)
            elif isinstance(event, Done):
                pass

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
        return name

    def _handle_tool_call(self, call: AssembledToolCall) -> None:
        """Execute one tool call (with the permission gate) and append the result."""
        tool = tools.get_tool(call.name)
        if tool is None:
            self._append_result(call.id, f"Error: unknown tool '{call.name}'")
            return

        try:
            args = call.parse_arguments()
        except json.JSONDecodeError as exc:
            self._append_result(
                call.id, f"Error: could not parse arguments as JSON: {exc}"
            )
            return

        summary = self._format_call_summary(call.name, args)
        preview_text = tools.preview(call.name, args, self.session)
        decision = gate(tool, args, self.session, self.confirm, preview_text)
        if not decision.allowed:
            self.console.print(f"[yellow]⊘ {summary}: {decision.reason}[/yellow]")
            self._append_result(call.id, decision.reason)
            return

        t0 = time.perf_counter()
        try:
            result = tool.execute(args, self.session)
        except Exception as exc:
            result = ToolResult.error(f"Tool crashed: {exc!r}")
        duration_ms = int((time.perf_counter() - t0) * 1000)

        if result.ok:
            self.console.print(f"[bold green]✓[/bold green] {summary} [dim]({duration_ms}ms)[/dim]")
        else:
            self.console.print(f"[bold red]✗[/bold red] {summary} [red](failed)[/red] [dim]({duration_ms}ms)[/dim]")
        self._append_result(call.id, result.content)

    def _append_result(self, tool_call_id: str, content: str) -> None:
        self.session.messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        )
