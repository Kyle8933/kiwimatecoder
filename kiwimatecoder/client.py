"""Unified OpenAI and Anthropic streaming client with tool-calling support.

A single :class:`UnifiedClient` drives every provider in the registry. It supports
both OpenAI-compatible ``/chat/completions`` SSE streaming and native Anthropic
``/messages`` SSE streaming.

Streamed responses are surfaced as :class:`StreamEvent` objects. Tool calls
arrive as fragments indexed by position; :class:`ToolCallAssembler` reassembles
them into complete calls. The assembler is a pure, network-free object so it can
be unit-tested directly.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from kiwimatecoder.providers import ProviderConfig


# ---------------------------------------------------------------------------
# Stream events
# ---------------------------------------------------------------------------


@dataclass
class TextDelta:
    """A chunk of assistant text content."""

    text: str


@dataclass
class ToolCallDelta:
    """A fragment of a tool call. Fragments share an ``index`` per call."""

    index: int
    id: str | None = None
    name: str | None = None
    args_fragment: str = ""


@dataclass
class Usage:
    """Token usage reported by the provider (when available)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Done:
    """Marks the end of a streamed completion."""

    finish_reason: str | None = None


StreamEvent = TextDelta | ToolCallDelta | Usage | Done


# ---------------------------------------------------------------------------
# Tool-call assembly
# ---------------------------------------------------------------------------


@dataclass
class AssembledToolCall:
    """A fully reassembled tool call ready for dispatch."""

    id: str
    name: str
    arguments: str  # raw JSON string as emitted by the model

    def parse_arguments(self) -> dict:
        """Parse ``arguments`` as JSON, returning ``{}`` for an empty string."""
        if not self.arguments.strip():
            return {}
        return json.loads(self.arguments)


class ToolCallAssembler:
    """Reassembles indexed tool-call fragments from a streamed completion."""

    def __init__(self) -> None:
        self._calls: dict[int, dict] = {}
        self._order: list[int] = []

    def add(self, delta: ToolCallDelta) -> None:
        slot = self._calls.get(delta.index)
        if slot is None:
            slot = {"id": None, "name": None, "arguments": ""}
            self._calls[delta.index] = slot
            self._order.append(delta.index)
        if delta.id is not None:
            slot["id"] = delta.id
        if delta.name is not None:
            slot["name"] = delta.name
        if delta.args_fragment:
            slot["arguments"] += delta.args_fragment

    def finalize(self) -> list[AssembledToolCall]:
        """Return the assembled calls in the order they first appeared."""
        result: list[AssembledToolCall] = []
        for index in self._order:
            slot = self._calls[index]
            if not slot["name"]:
                continue
            result.append(
                AssembledToolCall(
                    id=slot["id"] or f"call_{index}",
                    name=slot["name"],
                    arguments=slot["arguments"],
                )
            )
        return result

    def __bool__(self) -> bool:
        return bool(self._calls)


def parse_sse_chunk(data: str) -> list[StreamEvent]:
    """Convert one ``data:`` SSE payload into stream events for OpenAI endpoints."""
    if data.strip() == "[DONE]":
        return [Done()]
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        return []

    events: list[StreamEvent] = []

    if usage := chunk.get("usage"):
        events.append(
            Usage(
                prompt_tokens=usage.get("prompt_tokens", 0) or 0,
                completion_tokens=usage.get("completion_tokens", 0) or 0,
            )
        )

    for choice in chunk.get("choices", []):
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if content:
            events.append(TextDelta(text=content))
        for tc in delta.get("tool_calls", []) or []:
            fn = tc.get("function") or {}
            events.append(
                ToolCallDelta(
                    index=tc.get("index", 0),
                    id=tc.get("id"),
                    name=fn.get("name"),
                    args_fragment=fn.get("arguments") or "",
                )
            )
        if choice.get("finish_reason"):
            events.append(Done(finish_reason=choice["finish_reason"]))

    return events


def format_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert OpenAI-format message list to Anthropic (system_prompt, messages)."""
    system_parts: list[str] = []
    converted: list[dict] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            if content:
                system_parts.append(str(content))
            continue

        if role == "user":
            converted.append(
                {"role": "user", "content": str(content) if content else ""}
            )
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                blocks: list[dict] = []
                if content:
                    blocks.append({"type": "text", "text": str(content)})
                for idx, tc in enumerate(tool_calls):
                    fn = tc.get("function", {})
                    args_str = fn.get("arguments", "{}")
                    try:
                        parsed_args = json.loads(args_str) if args_str else {}
                    except Exception:
                        parsed_args = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id") or f"call_{idx}",
                            "name": fn.get("name", ""),
                            "input": parsed_args,
                        }
                    )
                converted.append({"role": "assistant", "content": blocks})
            else:
                converted.append(
                    {
                        "role": "assistant",
                        "content": str(content) if content is not None else "",
                    }
                )
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id") or ""
            tool_content = str(content) if content is not None else ""
            tool_block = {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": tool_content,
            }
            if (
                converted
                and converted[-1]["role"] == "user"
                and isinstance(converted[-1]["content"], list)
            ):
                converted[-1]["content"].append(tool_block)
            else:
                converted.append({"role": "user", "content": [tool_block]})

    system_prompt = "\n\n".join(system_parts)
    return system_prompt, converted


def format_anthropic_tools(tools: list[dict] | None) -> list[dict] | None:
    """Convert OpenAI tool schemas to Anthropic tools format."""
    if not tools:
        return None
    anthropic_tools: list[dict] = []
    for t in tools:
        fn = t.get("function", {})
        anthropic_tools.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
        )
    return anthropic_tools


def parse_anthropic_sse_event(
    event_type: str | None, data: str
) -> list[StreamEvent]:
    """Convert an Anthropic SSE event payload into stream events."""
    if not data.strip():
        return []
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        return []

    if not isinstance(chunk, dict):
        return []

    c_type = chunk.get("type") or event_type

    if c_type == "error":
        err = chunk.get("error") or {}
        msg = err.get("message") if isinstance(err, dict) else str(err)
        raise ProviderError(f"Anthropic stream error: {msg}")

    events: list[StreamEvent] = []

    if c_type == "message_start":
        msg = chunk.get("message") or {}
        usage = msg.get("usage") or {}
        in_tok = usage.get("input_tokens", 0) or 0
        out_tok = usage.get("output_tokens", 0) or 0
        if in_tok or out_tok:
            events.append(Usage(prompt_tokens=in_tok, completion_tokens=out_tok))

    elif c_type == "content_block_start":
        idx = chunk.get("index", 0)
        block = chunk.get("content_block") or {}
        if block.get("type") == "tool_use":
            events.append(
                ToolCallDelta(
                    index=idx,
                    id=block.get("id"),
                    name=block.get("name"),
                    args_fragment="",
                )
            )
        elif block.get("type") == "text" and block.get("text"):
            events.append(TextDelta(text=block["text"]))

    elif c_type == "content_block_delta":
        idx = chunk.get("index", 0)
        delta = chunk.get("delta") or {}
        d_type = delta.get("type")
        if d_type == "text_delta":
            text = delta.get("text")
            if text:
                events.append(TextDelta(text=text))
        elif d_type == "input_json_delta":
            frag = delta.get("partial_json") or ""
            events.append(ToolCallDelta(index=idx, args_fragment=frag))

    elif c_type == "message_delta":
        delta = chunk.get("delta") or {}
        usage = chunk.get("usage") or {}
        out_tok = usage.get("output_tokens", 0) or 0
        if out_tok:
            events.append(Usage(prompt_tokens=0, completion_tokens=out_tok))
        stop_reason = delta.get("stop_reason")
        if stop_reason:
            events.append(Done(finish_reason=stop_reason))

    elif c_type == "message_stop":
        events.append(Done(finish_reason=None))

    return events


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


class ProviderError(RuntimeError):
    """Raised when a provider returns a non-200 response."""


class UnifiedClient:
    """Streams chat completions against OpenAI-compatible or Anthropic providers."""

    def __init__(
        self, provider: ProviderConfig, api_key: str, timeout: float = 120.0
    ):
        self.provider = provider
        self.api_key = api_key
        self.timeout = timeout

    @property
    def is_anthropic(self) -> bool:
        return self.provider.compat == "anthropic"

    @property
    def _url(self) -> str:
        base = self.provider.base_url.rstrip("/")
        if self.is_anthropic:
            return f"{base}/messages"
        return f"{base}/chat/completions"

    def _headers(self) -> dict[str, str]:
        if self.is_anthropic:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
        else:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        headers.update(self.provider.extra_headers)
        return headers

    def _payload(
        self, messages: list[dict], tools: list[dict] | None, model: str
    ) -> dict:
        if self.is_anthropic:
            system_prompt, anthropic_msgs = format_anthropic_messages(messages)
            anthropic_tools = format_anthropic_tools(tools)
            payload: dict = {
                "model": model,
                "messages": anthropic_msgs,
                "max_tokens": 8192,
                "stream": True,
            }
            if system_prompt:
                payload["system"] = system_prompt
            if anthropic_tools:
                payload["tools"] = anthropic_tools
            return payload

        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    async def stream_chat(
        self, messages: list[dict], tools: list[dict] | None, model: str
    ) -> AsyncIterator[StreamEvent]:
        """Yield :class:`StreamEvent` objects for one completion, with retry on transient errors."""
        payload = self._payload(messages, tools, model)
        headers = self._headers()
        url = self._url

        max_attempts = 3
        backoff_delays = [0.5, 1.0, 2.0]

        for attempt in range(max_attempts):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream(
                        "POST", url, json=payload, headers=headers
                    ) as response:
                        if (
                            response.status_code in (429, 500, 502, 503, 504)
                            and attempt < max_attempts - 1
                        ):
                            await asyncio.sleep(backoff_delays[attempt])
                            continue

                        if response.status_code != 200:
                            body = (await response.aread()).decode(
                                "utf-8", "replace"
                            )
                            raise ProviderError(
                                f"{self.provider.name} returned HTTP {response.status_code}: "
                                f"{body[:500]}"
                            )

                        current_event_type: str | None = None
                        async for line in response.aiter_lines():
                            if self.is_anthropic and line.startswith("event: "):
                                current_event_type = line[7:].strip()
                                continue
                            if not line.startswith("data: "):
                                continue
                            data_part = line[6:]

                            if not self.is_anthropic:
                                try:
                                    chunk = json.loads(data_part)
                                    if (
                                        isinstance(chunk, dict)
                                        and "error" in chunk
                                    ):
                                        err = chunk["error"]
                                        msg = (
                                            err.get("message")
                                            if isinstance(err, dict)
                                            else str(err)
                                        )
                                        raise ProviderError(
                                            f"{self.provider.name} stream error: {str(msg)[:500]}"
                                        )
                                except (
                                    json.JSONDecodeError,
                                    TypeError,
                                    AttributeError,
                                    KeyError,
                                ):
                                    pass

                            events = (
                                parse_anthropic_sse_event(
                                    current_event_type, data_part
                                )
                                if self.is_anthropic
                                else parse_sse_chunk(data_part)
                            )
                            for event in events:
                                yield event
                                if (
                                    isinstance(event, Done)
                                    and event.finish_reason is None
                                ):
                                    return
                        return
            except (
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.ConnectError,
                httpx.RemoteProtocolError,
            ) as exc:
                if attempt < max_attempts - 1:
                    await asyncio.sleep(backoff_delays[attempt])
                    continue
                raise ProviderError(
                    f"{self.provider.name} connection error: {exc}"
                ) from exc
