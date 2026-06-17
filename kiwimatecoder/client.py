"""Unified OpenAI-compatible streaming client with tool-calling support.

A single :class:`UnifiedClient` drives every provider in the registry because
they all expose the OpenAI ``/chat/completions`` SSE streaming protocol.

Streamed responses are surfaced as :class:`StreamEvent` objects. Tool calls
arrive as fragments indexed by position; :class:`ToolCallAssembler` reassembles
them into complete calls. The assembler is a pure, network-free object so it can
be unit-tested directly.
"""

from __future__ import annotations

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
    """Convert one ``data:`` SSE payload into stream events.

    ``data`` is the raw text after the ``data: `` prefix. ``[DONE]`` yields a
    terminal :class:`Done`. Malformed JSON yields no events (callers skip it).
    """
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


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


class ProviderError(RuntimeError):
    """Raised when a provider returns a non-200 response."""


class UnifiedClient:
    """Streams chat completions against an OpenAI-compatible provider."""

    def __init__(self, provider: ProviderConfig, api_key: str, timeout: float = 120.0):
        self.provider = provider
        self.api_key = api_key
        self.timeout = timeout

    @property
    def _url(self) -> str:
        return f"{self.provider.base_url.rstrip('/')}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.provider.extra_headers)
        return headers

    def _payload(
        self, messages: list[dict], tools: list[dict] | None, model: str
    ) -> dict:
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        # Omit the tools key entirely when there are none — some providers
        # reject an empty array.
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    async def stream_chat(
        self, messages: list[dict], tools: list[dict] | None, model: str
    ) -> AsyncIterator[StreamEvent]:
        """Yield :class:`StreamEvent` objects for one completion."""
        payload = self._payload(messages, tools, model)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", self._url, json=payload, headers=self._headers()
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise ProviderError(
                        f"{self.provider.name} returned HTTP {response.status_code}: "
                        f"{body[:500]}"
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    for event in parse_sse_chunk(line[6:]):
                        yield event
                        if isinstance(event, Done) and event.finish_reason is None:
                            return
