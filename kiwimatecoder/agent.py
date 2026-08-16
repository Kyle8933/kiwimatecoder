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

MAX_TOOL_ROUNDS = 25


class Agent:
    """Drives one conversational turn, including any tool calls it triggers."""

    session: Session
    console: Console
    confirm: ConfirmFn

    def __init__(self, session: Session, console: Console, confirm: ConfirmFn) -> None:
        self.session = session
        self.console = console
        self.confirm = confirm

    def _client(self) -> UnifiedClient:
        from kiwimatecoder.config import get_key

        provider = self.session.provider
        key = get_key(provider.id)
        if not key and not provider.is_local:
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

        for round_num in range(MAX_TOOL_ROUNDS):
            try:
                assistant_msg, tool_calls = await self._stream_once()
            except ProviderError as exc:
                self.console.print(f"\n[red]{exc}[/red]")
                return

            self.session.messages.append(assistant_msg)

            if not tool_calls:
                return

            # On the final allowed round, don't execute the model's new tool
            # calls — instead record synthetic results so the conversation
            # history stays valid (every tool_call id has a matching result)
            # and the model can wrap up on the next turn.
            if round_num == MAX_TOOL_ROUNDS - 1:
                for call in tool_calls:
                    self._append_result(
                        call.id, "Reached the step limit for this turn."
                    )
                self.console.print(
                    f"\n[yellow]Reached the {MAX_TOOL_ROUNDS}-step limit "
                    + "for this turn.[/yellow]"
                )
                return

            for call in tool_calls:
                self._handle_tool_call(call)

    async def _stream_once(self) -> tuple[dict[str, Any], list[AssembledToolCall]]:
        """Stream one assistant response, rendering text and collecting tool calls."""
        client = self._client()
        read_only = self.session.mode is PermissionMode.PLAN
        schemas = tools.tool_schemas(read_only=read_only)

        text_parts: list[str] = []
        assembler = ToolCallAssembler()
        printed_any = False

        async for event in client.stream_chat(
            self._request_messages(), schemas, self.session.model
        ):
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
