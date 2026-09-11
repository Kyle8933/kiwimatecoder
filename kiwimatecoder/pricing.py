"""Per-model token pricing and a dependency-free token estimator.

Prices are USD per million tokens and are best-effort estimates curated
offline; providers change them over time. The estimator is heuristic (no
tokenizer dependency), refined enough for context-budget gauges and cost
approximations. Actual billed usage from the provider always takes precedence
for cost totals.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ModelPrice:
    """USD per million input/output tokens."""

    input_per_mtok: float
    output_per_mtok: float


FREE = ModelPrice(0.0, 0.0)

# Longest matching key wins, so provider-prefixed ids (OpenRouter style) and
# bare ids both resolve. Keep keys lowercase.
PRICES: dict[str, ModelPrice] = {
    "gpt-5.6-sol": ModelPrice(1.25, 10.00),
    "gpt-5.6": ModelPrice(1.25, 10.00),
    "gpt-5.5": ModelPrice(1.25, 10.00),
    "gpt-5": ModelPrice(1.25, 10.00),
    "claude-sonnet-5": ModelPrice(3.00, 15.00),
    "claude-opus-4-8": ModelPrice(15.00, 75.00),
    "claude-haiku-4-5": ModelPrice(0.80, 4.00),
    "claude": ModelPrice(3.00, 15.00),
    "gemini-3.5-flash": ModelPrice(0.30, 2.50),
    "gemini-3.5-pro": ModelPrice(1.25, 10.00),
    "gemini": ModelPrice(1.25, 10.00),
    "grok-4.5": ModelPrice(3.00, 15.00),
    "grok-build-0.1": ModelPrice(1.00, 5.00),
    "grok": ModelPrice(3.00, 15.00),
    "mistral-medium-3.5": ModelPrice(0.40, 2.00),
    "devstral-2512": ModelPrice(0.40, 2.00),
    "mistral": ModelPrice(0.40, 2.00),
    "deepseek-v4-pro": ModelPrice(0.55, 2.19),
    "deepseek-chat": ModelPrice(0.27, 1.10),
    "deepseek-reasoner": ModelPrice(0.55, 2.19),
    "deepseek": ModelPrice(0.27, 1.10),
    "qwen3.7-max": ModelPrice(1.20, 6.00),
    "qwen-plus": ModelPrice(0.40, 1.20),
    "qwen-turbo": ModelPrice(0.05, 0.20),
    "qwen": ModelPrice(1.20, 6.00),
    "kimi-k2.7-code": ModelPrice(0.60, 2.50),
    "kimi-latest": ModelPrice(0.60, 2.50),
    "kimi": ModelPrice(0.60, 2.50),
    "glm-5.2": ModelPrice(0.60, 2.20),
    "glm": ModelPrice(0.60, 2.20),
    "llama": FREE,
    "qwen2.5-coder": FREE,
    "qwen3": FREE,
    "gemma": FREE,
}


def normalize_model(model: str) -> str:
    """Lowercase and strip a provider prefix and ``:tag`` suffix."""
    name = (model or "").strip().lower()
    if "/" in name:
        name = name.rsplit("/", 1)[1]
    return name.split(":", 1)[0]


def find_price(
    model: str,
    provider_id: str | None = None,
    *,
    is_local: bool = False,
) -> ModelPrice | None:
    """Return the price for ``model``, or None when it is unknown.

    Local providers and OpenRouter ``:free`` variants cost nothing. Matching is
    exact first, then longest substring (so ``openai/gpt-5.6-sol`` resolves via
    its bare id).
    """
    raw = (model or "").strip().lower()
    if is_local or raw.endswith(":free"):
        return FREE

    name = normalize_model(raw)
    if not name:
        return None
    for candidate in (name, raw):
        if candidate in PRICES:
            return PRICES[candidate]
    best_key = ""
    for key in PRICES:
        if key in name and len(key) > len(best_key):
            best_key = key
    if best_key:
        return PRICES[best_key]
    return None


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    provider_id: str | None = None,
    *,
    is_local: bool = False,
) -> float | None:
    """Estimate USD cost for the given usage, or None without a price entry."""
    price = find_price(model, provider_id, is_local=is_local)
    if price is None:
        return None
    return (
        prompt_tokens * price.input_per_mtok
        + completion_tokens * price.output_per_mtok
    ) / 1_000_000


_WHITESPACE_RE = re.compile(r"\S+")


def estimate_text_tokens(text: str) -> int:
    """Heuristically estimate token count for a piece of text.

    Uses the common ~4 characters/token rule and the ~4 tokens per 3 words
    rule, taking the larger. Never returns a negative count.
    """
    if not text:
        return 0
    chars_estimate = (len(text) + 3) // 4
    words = len(_WHITESPACE_RE.findall(text))
    word_estimate = (words * 4 + 2) // 3
    return max(chars_estimate, word_estimate, 1)


def estimate_messages_tokens(messages: Iterable[dict[str, Any]]) -> int:
    """Estimate tokens for a chat transcript, including per-message overhead."""
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            text = " ".join(json.dumps(block, default=str) for block in content)
        else:
            text = str(content or "")
        total += estimate_text_tokens(text) + 4
        if message.get("tool_calls"):
            total += estimate_text_tokens(json.dumps(message["tool_calls"], default=str))
    return total
