"""One-shot streaming helper for the ``ask`` command.

This is the simple, non-agentic path: send a single prompt and stream the
answer to the console. It now runs on top of :class:`UnifiedClient` so it
benefits from the full provider registry while keeping the original behavior
(OpenRouter + the project's default model unless overridden).
"""

from __future__ import annotations

from rich.console import Console

from kiwimatecoder.client import ProviderError, TextDelta, UnifiedClient
from kiwimatecoder.providers import ProviderConfig, default_provider

console = Console()

SYSTEM_PROMPT = (
    "You are KiwiMateCoder, an expert coding assistant. Give clear, concise, and "
    "accurate coding help. Prefer showing code over lengthy explanations."
)


async def stream_response(
    prompt: str,
    api_key: str,
    model: str | None = None,
    provider: ProviderConfig | None = None,
) -> None:
    """Stream a single answer to the console."""
    provider = provider or default_provider()
    model = model or provider.default_model
    client = UnifiedClient(provider, api_key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        async for event in client.stream_chat(messages, tools=None, model=model):
            if isinstance(event, TextDelta):
                console.print(event.text, end="")
    except ProviderError as exc:
        console.print(f"\n[red]{exc}[/red]")
