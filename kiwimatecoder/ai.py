"""One-shot streaming helper for the ``ask`` command.

This is the simple, non-agentic path: send a single prompt and stream the
answer to the console. It runs on top of :class:`UnifiedClient` so it
benefits from the full provider registry (OpenRouter unless overridden). There
is no default model: the caller passes one, or the provider's chosen model is
used.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console

from kiwimatecoder.client import ProviderError, TextDelta, UnifiedClient
from kiwimatecoder.config import get_prompt_cache, get_sampling, no_model_message, resolve_model
from kiwimatecoder.providers import ProviderConfig, default_provider

console = Console()

SYSTEM_PROMPT = (
    "You are KiwiMateCoder, an expert coding assistant. Give clear, concise, and "
    + "accurate coding help. Prefer showing code over lengthy explanations."
)


async def stream_response(
    prompt: str,
    api_key: str,
    model: str | None = None,
    provider: ProviderConfig | None = None,
) -> None:
    """Stream a single answer to the console."""
    provider = provider or default_provider()
    model = model or resolve_model(provider)
    if not model:
        console.print(f"[red]{no_model_message(provider)}[/red]")
        return
    client = UnifiedClient(
        provider,
        api_key,
        sampling=get_sampling(),
        prompt_cache=get_prompt_cache(),
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    status = console.status("[dim]Thinking…[/dim]")
    status.start()
    thinking = True
    try:
        try:
            async for event in client.stream_chat(messages, tools=None, model=model):
                if isinstance(event, TextDelta):
                    if thinking:
                        status.stop()
                        thinking = False
                    console.print(event.text, end="", markup=False, highlight=False)
        finally:
            if thinking:
                status.stop()
    except ProviderError as exc:
        console.print(f"\n[red]{exc}[/red]")
