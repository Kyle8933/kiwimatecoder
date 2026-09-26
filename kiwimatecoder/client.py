"""Unified OpenAI and Anthropic streaming client with tool-calling support.

A single :class:`UnifiedClient` drives every provider in the registry. It supports
both OpenAI-compatible ``/chat/completions`` SSE streaming and native Anthropic
``/messages`` SSE streaming. Chat-only providers (``supports_tools=False``) are
sent no tool schemas, and :func:`flatten_tool_messages` rewrites any tool
traffic in their history as plain text.

Streamed responses are surfaced as :class:`StreamEvent` objects. Tool calls
arrive as fragments indexed by position; :class:`ToolCallAssembler` reassembles
them into complete calls. The assembler is a pure, network-free object so it can
be unit-tested directly.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator
from urllib.parse import urlparse

import httpx

from kiwimatecoder import network, telemetry
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

    def parse_arguments(self) -> dict[str, Any]:
        """Parse ``arguments`` as JSON, returning ``{}`` for an empty string."""
        if not self.arguments.strip():
            return {}
        result: dict[str, Any] = json.loads(self.arguments)
        return result


class ToolCallAssembler:
    """Reassembles indexed tool-call fragments from a streamed completion."""

    def __init__(self) -> None:
        self._calls: dict[int, dict[str, Any]] = {}
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
            slot["arguments"] = str(slot.get("arguments") or "") + delta.args_fragment

    def finalize(self) -> list[AssembledToolCall]:
        """Return the assembled calls in the order they first appeared."""
        result: list[AssembledToolCall] = []
        for index in self._order:
            slot = self._calls[index]
            if not slot["name"]:
                continue
            result.append(
                AssembledToolCall(
                    id=str(slot["id"] or f"call_{index}"),
                    name=str(slot["name"]),
                    arguments=str(slot["arguments"]),
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


def _anthropic_image_block(url: str) -> dict[str, Any] | None:
    """Convert one OpenAI image URL into an Anthropic image block, or None."""
    if url.startswith("data:"):
        header, separator, payload = url[5:].partition(";base64,")
        if not separator or not payload:
            return None
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": header or "image/png",
                "data": payload,
            },
        }
    if url.startswith(("http://", "https://")):
        return {"type": "image", "source": {"type": "url", "url": url}}
    return None


def _anthropic_content_blocks(content: list[Any]) -> list[dict[str, Any]]:
    """Convert OpenAI content parts (text/image_url) into Anthropic blocks.

    Unrecognized parts are skipped so an unexpected part cannot break a whole
    request.
    """
    blocks: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if text:
                blocks.append({"type": "text", "text": str(text)})
        elif part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            block = _anthropic_image_block(str(image_url or ""))
            if block is not None:
                blocks.append(block)
    return blocks


def format_anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Convert OpenAI-format message list to Anthropic (system_prompt, messages)."""
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            if content:
                system_parts.append(str(content))
            continue

        if role == "user":
            if isinstance(content, list):
                converted.append(
                    {
                        "role": "user",
                        "content": _anthropic_content_blocks(content) or "",
                    }
                )
            else:
                converted.append(
                    {"role": "user", "content": str(content) if content else ""}
                )
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                blocks: list[dict[str, Any]] = []
                if isinstance(content, list):
                    blocks.extend(_anthropic_content_blocks(content))
                elif content:
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
            elif isinstance(content, list):
                converted.append(
                    {
                        "role": "assistant",
                        "content": _anthropic_content_blocks(content) or "",
                    }
                )
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


# Flattened tool results and call arguments longer than this are cut so one huge
# file read (or write) cannot crowd the rest of the history out of a chat-only
# endpoint's context.
_FLATTENED_TEXT_LIMIT = 8000


def _clip(text: str) -> str:
    """Cut ``text`` to the flattened-history limit, marking the cut."""
    if len(text) <= _FLATTENED_TEXT_LIMIT:
        return text
    return text[:_FLATTENED_TEXT_LIMIT] + "\n…[truncated]"


def _content_text(content: Any) -> str:
    """Return the text of a string or OpenAI content-parts value."""
    if content is None:
        return ""
    if isinstance(content, list):
        texts = [
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n\n".join(text for text in texts if text)
    return str(content)


def _content_parts(content: Any) -> list[Any]:
    """Return ``content`` as a new list of OpenAI content parts."""
    if isinstance(content, list):
        return list(content)
    text = str(content or "")
    return [{"type": "text", "text": text}] if text else []


def _merge_content(first: Any, second: Any) -> Any:
    """Join two message contents, producing parts if either side has parts."""
    if isinstance(first, list) or isinstance(second, list):
        return _content_parts(first) + _content_parts(second)
    if not first:
        return second
    if not second:
        return first
    return f"{first}\n\n{second}"


def flatten_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrite an OpenAI-format transcript for a chat-only endpoint.

    Tool calls become assistant text, tool results become user text, developer
    messages become system messages, consecutive same-role turns are merged,
    and the transcript is made to end on a user turn. Only ``role`` and
    ``content`` survive. The input is never mutated.
    """
    flattened: list[dict[str, Any]] = []
    tool_names: dict[str, str] = {}

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role in ("system", "developer"):
            flattened.append({"role": "system", "content": _content_text(content)})
            continue

        if role == "assistant":
            pieces: list[str] = []
            text = _content_text(content)
            if text.strip():
                pieces.append(text)
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = str(fn.get("name") or "tool")
                if tc.get("id"):
                    tool_names[str(tc["id"])] = name
                args = fn.get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False, default=str) if args else ""
                pieces.append(f"[called tool {name} with {_clip(args.strip()) or '{}'}]")
            if not pieces:
                continue
            entry: dict[str, Any] = {"role": "assistant", "content": "\n\n".join(pieces)}
        elif role == "tool":
            name = tool_names.get(str(msg.get("tool_call_id") or ""), "tool")
            result = _clip(_content_text(content))
            entry = {"role": "user", "content": f"[result of {name}]\n{result}"}
        else:
            entry = {
                "role": "user",
                "content": list(content) if isinstance(content, list) else content or "",
            }

        previous = flattened[-1] if flattened else None
        if previous is not None and previous["role"] == entry["role"]:
            # ``previous`` was built here, never taken from the input.
            previous["content"] = _merge_content(previous["content"], entry["content"])
        else:
            flattened.append(entry)

    last_turn = next((m for m in reversed(flattened) if m["role"] != "system"), None)
    if last_turn is not None and last_turn["role"] == "assistant":
        # E.g. a partial reply left behind by a failed stream.
        flattened.append({"role": "user", "content": "Continue."})
    return flattened


def _image_url(part: Any) -> str | None:
    """Return an ``image_url`` part's URL, or None for any other part."""
    if not isinstance(part, dict) or part.get("type") != "image_url":
        return None
    image = part.get("image_url")
    url = image.get("url") if isinstance(image, dict) else image
    return str(url or "")


def limit_images(
    messages: list[dict[str, Any]],
    provider: ProviderConfig,
) -> list[dict[str, Any]]:
    """Drop images a provider would reject, noting each drop as text.

    Applies ``provider.max_image_url_chars`` (oversized images) and then
    ``provider.max_images_per_message`` (keeping the newest images) to every
    message whose content is a parts list; a limit of 0 means none. A request
    the endpoint rejects would otherwise fail on every later turn, since the
    images stay in the history. The input is never mutated.
    """
    max_images = provider.max_images_per_message
    max_chars = provider.max_image_url_chars
    if not max_images and not max_chars:
        return messages
    limited: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            limited.append(msg)
            continue
        oversized = 0
        kept: list[Any] = []
        for part in content:
            url = _image_url(part)
            if url is not None and max_chars and len(url) > max_chars:
                oversized += 1
                continue
            kept.append(part)
        image_rows = [index for index, part in enumerate(kept) if _image_url(part) is not None]
        excess = image_rows[: max(0, len(image_rows) - max_images)] if max_images else []
        if not oversized and not excess:
            limited.append(msg)
            continue
        dropped = set(excess)
        parts = [part for index, part in enumerate(kept) if index not in dropped]
        notes: list[str] = []
        if oversized:
            notes.append(f"{oversized} image(s) too large for {provider.name}")
        if excess:
            notes.append(
                f"{len(excess)} earlier image(s): {provider.name} accepts at most "
                f"{max_images} per message"
            )
        parts.insert(0, {"type": "text", "text": f"[omitted {'; '.join(notes)}]"})
        limited.append({**msg, "content": parts})
    return limited


def format_anthropic_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Convert OpenAI tool schemas to Anthropic tools format."""
    if not tools:
        return None
    anthropic_tools: list[dict[str, Any]] = []
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
        self,
        provider: ProviderConfig,
        api_key: str,
        timeout: float = 120.0,
        sampling: dict[str, Any] | None = None,
        prompt_cache: bool = False,
    ):
        self.provider = provider
        self.api_key = api_key
        self.timeout = timeout
        self.sampling = sampling or {}
        self.prompt_cache = bool(prompt_cache)

    @property
    def is_anthropic(self) -> bool:
        return self.provider.compat == "anthropic"

    @property
    def _url(self) -> str:
        base = self.provider.base_url.rstrip("/")
        if self.is_anthropic:
            return self.provider.versioned_url(f"{base}/messages")
        return self.provider.versioned_url(f"{base}/chat/completions")

    def _headers(self) -> dict[str, str]:
        # Auth headers are only sent when there is a key — local servers
        # (Ollama, LM Studio) accept keyless requests.
        headers = {"Content-Type": "application/json"}
        if self.is_anthropic:
            headers["anthropic-version"] = "2023-06-01"
            if self.api_key:
                headers["x-api-key"] = self.api_key
        elif self.api_key:
            # Cloud providers usually want Authorization: Bearer <key>; Azure
            # OpenAI wants `api-key: <key>` with no prefix. The provider config
            # carries both so custom providers can follow either scheme.
            headers[self.provider.key_header] = (
                f"{self.provider.key_prefix}{self.api_key}"
            )
        headers.update(self.provider.extra_headers)
        return headers

    def _payload(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, model: str
    ) -> dict[str, Any]:
        if self.is_anthropic:
            system_prompt, anthropic_msgs = format_anthropic_messages(messages)
            anthropic_tools = format_anthropic_tools(tools)
            payload: dict[str, Any] = {
                "model": model,
                "messages": anthropic_msgs,
                "max_tokens": int(self.sampling.get("max_tokens") or 8192),
                "stream": True,
            }
            if self.sampling.get("temperature") is not None:
                payload["temperature"] = self.sampling["temperature"]
            if self.sampling.get("top_p") is not None:
                payload["top_p"] = self.sampling["top_p"]
            if system_prompt:
                if self.prompt_cache:
                    # Anthropic prompt caching: mark the stable system prompt
                    # and the end of the tool definitions as cache breakpoints.
                    payload["system"] = [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                else:
                    payload["system"] = system_prompt
            if anthropic_tools:
                if self.prompt_cache:
                    anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
                payload["tools"] = anthropic_tools
            return payload

        # OpenAI-compatible providers cache automatically; no payload change. The
        # ``prompt_cache`` toggle only affects native Anthropic requests.

        # Chat-only endpoints (KiwiMate) reject tool traffic: flatten it, send no tools.
        chat_only = not self.provider.supports_tools
        if chat_only:
            messages = flatten_tool_messages(messages)
        openai_payload: dict[str, Any] = {
            "model": model,
            "messages": limit_images(messages, self.provider),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        for key in ("temperature", "top_p", "max_tokens", "reasoning_effort"):
            value = self.sampling.get(key)
            if value is not None:
                openai_payload[key] = value
        if tools and not chat_only:
            openai_payload["tools"] = tools
            # Ollama rejects tool_choice; "auto" is the default behavior anyway.
            if not self.provider.is_local:
                openai_payload["tool_choice"] = "auto"
        return openai_payload

    async def stream_chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, model: str
    ) -> AsyncIterator[StreamEvent]:
        """Yield :class:`StreamEvent` objects for one completion, with retry on transient errors."""
        # There are no default models, so an unset model fails here (as a
        # failover-compatible ProviderError) rather than as an HTTP 400.
        if not model or not model.strip():
            raise ProviderError(
                f"No model chosen for {self.provider.name}; "
                "choose one with /model or pass --model."
            )
        if not self.provider.is_local and network.offline_enabled():
            raise ProviderError(
                network.offline_message(f"{self.provider.name} chat completions")
            )
        payload = self._payload(messages, tools, model)
        headers = self._headers()
        url = self._url
        host = urlparse(url).hostname or ""
        connect_options = network.current_options()

        max_attempts = 3
        backoff_delays = [0.5, 1.0, 2.0]

        for attempt in range(max_attempts):
            started = time.perf_counter()
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout, **connect_options
                ) as client:
                    async with client.stream(
                        "POST", url, json=payload, headers=headers
                    ) as response:
                        if (
                            response.status_code in (429, 500, 502, 503, 504)
                            and attempt < max_attempts - 1
                        ):
                            telemetry.log_event(
                                "provider_retry",
                                level="error",
                                provider=self.provider.id,
                                host=host,
                                status=response.status_code,
                                attempt=attempt + 1,
                            )
                            await asyncio.sleep(backoff_delays[attempt])
                            continue

                        telemetry.log_event(
                            "http_request",
                            level="debug",
                            method="POST",
                            host=host,
                            status=response.status_code,
                            duration_ms=int((time.perf_counter() - started) * 1000),
                            provider=self.provider.id,
                        )

                        if response.status_code != 200:
                            body = (await response.aread()).decode(
                                "utf-8", "replace"
                            )
                            telemetry.log_event(
                                "provider_error",
                                level="error",
                                provider=self.provider.id,
                                model=model,
                                status=response.status_code,
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
                    telemetry.log_event(
                        "provider_retry",
                        level="error",
                        provider=self.provider.id,
                        host=host,
                        error=exc.__class__.__name__,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(backoff_delays[attempt])
                    continue
                telemetry.log_event(
                    "provider_error",
                    level="error",
                    provider=self.provider.id,
                    host=host,
                    error=exc.__class__.__name__,
                )
                raise ProviderError(
                    f"{self.provider.name} connection error: {exc}"
                ) from exc
