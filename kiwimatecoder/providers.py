"""Built-in registry of model providers.

Most providers expose an OpenAI-compatible ``/chat/completions`` API so a single
:class:`~kiwimatecoder.client.UnifiedClient` can drive them. The ``compat`` field
and the ``anthropic`` entry are reserved for future native code paths (e.g.
Anthropic's native Messages API). Callers must not assume every registered id
yields a fully compatible endpoint today.

Model ids drift fast — the defaults and ``models`` catalogs below were verified
in July 2026. They are only starting points: the user can override the model for
any provider at runtime with ``/model`` (typing any id works, listed or not) or
persist a choice via ``config set-model``, and can reshape the offered list with
``/config models allow|deny``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderConfig:
    """Static configuration for a single model provider."""

    id: str
    name: str
    base_url: str  # includes /v1, never a trailing /chat/completions
    default_model: str
    key_env: str
    compat: str = "openai"  # "openai" | "anthropic" (reserved; native paths not yet implemented)
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Curated catalog offered by /model; not exhaustive, and any id can still
    # be set by name. The default model is always offered even if absent here.
    models: tuple[str, ...] = ()


class UnknownProviderError(KeyError):
    """KeyError subclass for unknown provider IDs.

    Subclassing preserves all existing ``except KeyError`` sites (in main,
    commands, config, and tests). Overrides __str__ so f"{exc}" and the red
    error prints produce clean messages without Python's extra repr quotes.
    """

    def __str__(self) -> str:
        return self.args[0] if self.args else super().__str__()


REGISTRY: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        id="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        default_model="gpt-5.6-sol",
        key_env="OPENAI_API_KEY",
        models=("gpt-5.6-sol", "gpt-5.5"),
    ),
    "anthropic": ProviderConfig(
        id="anthropic",
        name="Anthropic",
        base_url="https://api.anthropic.com/v1",
        default_model="claude-sonnet-5",
        key_env="ANTHROPIC_API_KEY",
        compat="anthropic",
        models=("claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5"),
    ),
    "google": ProviderConfig(
        id="google",
        name="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_model="gemini-3.5-flash",
        key_env="GEMINI_API_KEY",
        models=("gemini-3.5-flash", "gemini-3.5-pro"),
    ),
    "xai": ProviderConfig(
        id="xai",
        name="xAI Grok",
        base_url="https://api.x.ai/v1",
        default_model="grok-4.5",
        key_env="XAI_API_KEY",
        models=("grok-4.5", "grok-build-0.1"),
    ),
    "mistral": ProviderConfig(
        id="mistral",
        name="Mistral",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-medium-3.5",
        key_env="MISTRAL_API_KEY",
        models=("mistral-medium-3.5", "devstral-2512"),
    ),
    "deepseek": ProviderConfig(
        id="deepseek",
        name="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-v4-pro",
        key_env="DEEPSEEK_API_KEY",
        models=("deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"),
    ),
    "qwen": ProviderConfig(
        id="qwen",
        name="Qwen (Alibaba DashScope)",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        default_model="qwen3.7-max",
        key_env="DASHSCOPE_API_KEY",
        models=("qwen3.7-max", "qwen-plus", "qwen-turbo"),
    ),
    "moonshot": ProviderConfig(
        id="moonshot",
        name="Moonshot (Kimi)",
        base_url="https://api.moonshot.ai/v1",
        default_model="kimi-k2.7-code",
        key_env="MOONSHOT_API_KEY",
        models=("kimi-k2.7-code", "kimi-latest"),
    ),
    "openrouter": ProviderConfig(
        id="openrouter",
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="anthropic/claude-sonnet-5",
        key_env="OPENROUTER_API_KEY",
        extra_headers={
            "HTTP-Referer": "https://kiwimatecoder.com",
            "X-Title": "KiwiMateCoder",
        },
        models=(
            "anthropic/claude-sonnet-5",
            "anthropic/claude-opus-4-8",
            "openai/gpt-5.6-sol",
            "google/gemini-3.5-flash",
            "x-ai/grok-4.5",
            "deepseek/deepseek-v4-pro",
            "qwen/qwen3.7-max",
            "moonshotai/kimi-k2.7-code",
            "mistralai/devstral-2512",
            "z-ai/glm-5.2",
        ),
    ),
}

DEFAULT_PROVIDER_ID = "openrouter"


def get_provider(provider_id: str) -> ProviderConfig:
    """Return the provider config for ``provider_id`` or raise ``UnknownProviderError`` (a ``KeyError`` subclass)."""
    try:
        return REGISTRY[provider_id]
    except KeyError:
        raise UnknownProviderError(
            f"Unknown provider '{provider_id}'. "
            f"Known providers: {', '.join(sorted(REGISTRY))}"
        ) from None


def list_providers() -> list[ProviderConfig]:
    """Return all registered providers in a stable order."""
    return list(REGISTRY.values())


def default_provider() -> ProviderConfig:
    """Return the default provider (OpenRouter, preserving legacy behavior)."""
    return REGISTRY[DEFAULT_PROVIDER_ID]
