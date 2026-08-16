"""Built-in registry of model providers.

Most providers expose an OpenAI-compatible ``/chat/completions`` API so a single
:class:`~kiwimatecoder.client.UnifiedClient` can drive them. The ``compat`` field
and the ``anthropic`` entry are reserved for future native code paths (e.g.
Anthropic's native Messages API). Callers must not assume every registered id
yields a fully compatible endpoint today.

Model ids drift fast — the defaults and ``models`` catalogs below were verified
in July 2026. They are only the offline starting point: once a provider is in
use, :mod:`kiwimatecoder.catalog` fetches its live ``/models`` listing so newly
released ids are offered and retired ones disappear (see
``config.get_model_catalog``). The user can still override the model for any
provider at runtime with ``/model`` (typing any id works, listed or not) or
persist a choice via ``config set-model``, and can reshape the offered list with
``/config models allow|deny``.

Local providers (Ollama, LM Studio) need no API key and serve whatever models
are loaded, so they ship no static ``default_model`` — the session model is
resolved live from the running server (see ``config.resolve_default_model``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse

# Hosts that serve the local machine and therefore need no API key (LM Studio,
# Ollama, llama.cpp, ...). ``*.local`` hosts are treated the same way.
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"}


@dataclass(frozen=True)
class ProviderConfig:
    """Static configuration for a single model provider."""

    id: str
    name: str
    base_url: str  # includes /v1, never a trailing /chat/completions
    default_model: str  # may be "" for local providers (resolved live)
    key_env: str
    compat: str = "openai"  # "openai" | "anthropic" (reserved; native paths not yet implemented)
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Curated catalog offered by /model; not exhaustive, and any id can still
    # be set by name. The default model is always offered even if absent here.
    models: tuple[str, ...] = ()

    @property
    def is_local(self) -> bool:
        """Whether the provider serves the local machine (no API key needed)."""
        host = (urlparse(self.base_url).hostname or "").lower()
        return host in _LOCAL_HOSTS or host.endswith(".local")


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
    "ollama": ProviderConfig(
        id="ollama",
        name="Ollama (local)",
        base_url="http://localhost:11434/v1",
        # No static default: the model is resolved live from the running
        # server. The curated tuple is only the offline fallback for /model.
        default_model="",
        key_env="OLLAMA_API_KEY",  # optional; only if the server enforces auth
        models=("llama3.1:8b", "qwen3:8b", "deepseek-r1:8b"),
    ),
    "lmstudio": ProviderConfig(
        id="lmstudio",
        name="LM Studio (local)",
        base_url="http://localhost:1234/v1",
        default_model="",
        key_env="LMSTUDIO_API_KEY",  # optional; only if the server enforces auth
        models=("qwen2.5-coder-7b-instruct", "llama-3.1-8b-instruct"),
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
            + f"Known providers: {', '.join(sorted(REGISTRY))}"
        ) from None


def list_providers() -> list[ProviderConfig]:
    """Return all registered providers in a stable order."""
    return list(REGISTRY.values())


def default_provider() -> ProviderConfig:
    """Return the default provider (OpenRouter, preserving legacy behavior)."""
    return REGISTRY[DEFAULT_PROVIDER_ID]
