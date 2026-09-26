"""Built-in registry of model providers.

Most providers expose an OpenAI-compatible ``/chat/completions`` API so a single
:class:`~kiwimatecoder.client.UnifiedClient` can drive them. Providers whose
``compat`` is ``"anthropic"`` (Anthropic itself, or a custom provider configured
that way) are driven through the native Messages API with SSE streaming and
tool-use support instead.

There are no default models. The user chooses a model when they add a
provider (``kiwimatecoder setup``, ``/provider`` in the REPL, or
``config provider add``), and that choice is persisted per provider (see
``config.get_provider_model``). The ``models`` tuples below were verified in
July 2026 and are only suggestions for the pickers when the live listing is
unavailable: once a provider has a key, :mod:`kiwimatecoder.catalog` fetches
its live ``/models`` listing so newly released ids are offered and retired
ones disappear (see ``config.get_model_catalog``). Any id can still be set by
name with ``/model`` or ``config model set``, and the offered list can be
reshaped with ``/config models allow|deny``.

Local providers (Ollama, LM Studio, Unsloth) serve whatever models are loaded,
so when no model has been chosen the session model is read live from the
running server (see ``config.resolve_model``). Ollama and LM Studio need no API
key; Unsloth enforces auth even locally (``requires_key=True``), so its
``sk-unsloth-…`` key must be configured before the server can be used.

Experimental providers (``experimental=True``, currently KiwiMate) are listed
first and drawn in their own "Experimental" box by the provider pickers.
Providers with ``supports_tools=False`` are chat-only: the client sends them no
tool schemas and flattens any tool traffic in the history into plain text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import quote, urlparse

# Hosts that serve the local machine (LM Studio, Ollama, llama.cpp, Unsloth,
# ...). ``*.local`` hosts are treated the same way. Local does not imply
# keyless: see ``ProviderConfig.requires_key``.
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"}


@dataclass(frozen=True)
class ProviderConfig:
    """Static configuration for a single model provider."""

    id: str
    name: str
    base_url: str  # includes /v1, never a trailing /chat/completions
    key_env: str
    # Set for local servers that enforce auth anyway (Unsloth's sk-unsloth-…
    # key). Keyless locals (Ollama, LM Studio) and custom providers leave it off.
    requires_key: bool = False
    compat: str = "openai"  # "openai" | "anthropic" (native Messages API)
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Suggested models offered by the pickers until a live listing is
    # available; not exhaustive, and any id can still be set by name. None of
    # them is a default: the user always chooses.
    models: tuple[str, ...] = ()
    # Auth header used for OpenAI-compatible requests. Cloud providers want
    # ``Authorization: Bearer <key>``; Azure OpenAI wants ``api-key: <key>``.
    # Both fields are ignored for native Anthropic providers (x-api-key).
    key_header: str = "Authorization"
    key_prefix: str = "Bearer "
    # Azure-style API versioning: appended as ``?api-version=<value>`` to chat,
    # catalog, and embedding URLs when non-empty.
    api_version: str = ""
    # Shown in its own "Experimental" box at the top of the provider pickers.
    experimental: bool = False
    # False for chat-only endpoints: no tool schemas are sent and tool calls /
    # results already in the history are flattened into plain text.
    supports_tools: bool = True
    # Short note shown next to the provider in pickers and tables.
    description: str = ""
    # Per-message image limits enforced by the endpoint (0 = no limit). The
    # client drops what would be rejected and says so in a text part.
    max_images_per_message: int = 0
    max_image_url_chars: int = 0

    @property
    def is_local(self) -> bool:
        """Whether the provider serves the local machine."""
        host = (urlparse(self.base_url).hostname or "").lower()
        return host in _LOCAL_HOSTS or host.endswith(".local")

    @property
    def needs_key(self) -> bool:
        """Whether requests fail without an API key (cloud or auth-enforcing local server)."""
        return self.requires_key or not self.is_local

    def versioned_url(self, url: str) -> str:
        """Append ``api_version`` as an ``api-version`` query when set.

        Azure OpenAI versions its endpoint with a date query parameter, so the
        client, catalog, and embeddings all build URLs through here. Providers
        without an ``api_version`` get the URL back untouched.
        """
        if not self.api_version:
            return url
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}api-version={quote(self.api_version, safe='')}"


class UnknownProviderError(KeyError):
    """KeyError subclass for unknown provider IDs.

    Subclassing preserves all existing ``except KeyError`` sites (in main,
    commands, config, and tests). Overrides __str__ so f"{exc}" and the red
    error prints produce clean messages without Python's extra repr quotes.
    """

    def __str__(self) -> str:
        return self.args[0] if self.args else super().__str__()


REGISTRY: dict[str, ProviderConfig] = {
    # Experimental: KiwiMate's own OpenAI-compatible API (kiwimate.net, keys
    # start with sk-km-; see kiwimate.net/developers/docs). The endpoint is
    # chat-only for now — it ignores tool schemas, rejects tool-role
    # messages, drops system prompts, and caps replies at 1,024 tokens — so
    # it runs with supports_tools=False and cannot edit files or run commands.
    "kiwimate": ProviderConfig(
        id="kiwimate",
        name="KiwiMate",
        base_url="https://gznrhppouxwpfihlfgpb.supabase.co/functions/v1/api-chat-completions",
        key_env="KIWIMATE_API_KEY",
        models=(
            "kiwimate-mini-1-0",
            "kiwimate-small-1-0",
            "kiwimate-medium-1-0",
            "kiwimate-large-1-0",
        ),
        experimental=True,
        supports_tools=False,
        description="kiwimate.net · chat only, no tool use yet",
        # data: URLs only, at most 4 per message, each under 3,000,000 chars.
        max_images_per_message=4,
        max_image_url_chars=3_000_000,
    ),
    "openai": ProviderConfig(
        id="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        key_env="OPENAI_API_KEY",
        models=("gpt-5.6-sol", "gpt-5.5"),
    ),
    "anthropic": ProviderConfig(
        id="anthropic",
        name="Anthropic",
        base_url="https://api.anthropic.com/v1",
        key_env="ANTHROPIC_API_KEY",
        compat="anthropic",
        models=("claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5"),
    ),
    "google": ProviderConfig(
        id="google",
        name="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        key_env="GEMINI_API_KEY",
        models=("gemini-3.5-flash", "gemini-3.5-pro"),
    ),
    "xai": ProviderConfig(
        id="xai",
        name="xAI Grok",
        base_url="https://api.x.ai/v1",
        key_env="XAI_API_KEY",
        models=("grok-4.5", "grok-build-0.1"),
    ),
    "mistral": ProviderConfig(
        id="mistral",
        name="Mistral",
        base_url="https://api.mistral.ai/v1",
        key_env="MISTRAL_API_KEY",
        models=("mistral-medium-3.5", "devstral-2512"),
    ),
    "deepseek": ProviderConfig(
        id="deepseek",
        name="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        key_env="DEEPSEEK_API_KEY",
        models=("deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"),
    ),
    "qwen": ProviderConfig(
        id="qwen",
        name="Qwen (Alibaba DashScope)",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        key_env="DASHSCOPE_API_KEY",
        models=("qwen3.7-max", "qwen-plus", "qwen-turbo"),
    ),
    "moonshot": ProviderConfig(
        id="moonshot",
        name="Moonshot (Kimi)",
        base_url="https://api.moonshot.ai/v1",
        key_env="MOONSHOT_API_KEY",
        models=("kimi-k2.7-code", "kimi-latest"),
    ),
    "openrouter": ProviderConfig(
        id="openrouter",
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
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
    "azure": ProviderConfig(
        id="azure",
        name="Azure OpenAI",
        # Placeholder only: replace <resource> with your Azure resource name.
        # Because built-in entries are not editable, the supported path is a
        # custom provider pointed at your resource, e.g.
        # ``config provider add my-azure "My Azure" \
        #   https://my-resource.openai.azure.com/openai/v1 my-deployment \
        #   --key-env AZURE_OPENAI_API_KEY --key-header api-key \
        #   --key-prefix "" --api-version 2024-10-21`` (the fourth argument is
        # the model — your deployment name — to use with it).
        base_url="https://<resource>.openai.azure.com/openai/v1",
        key_env="AZURE_OPENAI_API_KEY",
        key_header="api-key",
        key_prefix="",
        api_version="2024-10-21",
        models=("gpt-5.6-sol", "gpt-5.5"),
    ),
    "bedrock": ProviderConfig(
        id="bedrock",
        name="AWS Bedrock",
        # Placeholder only: replace <region> with your AWS region. Bedrock's
        # OpenAI-compatible runtime accepts a bearer token; SigV4/IAM signing is
        # out of scope (use a signing proxy or custom provider if you need it).
        base_url="https://bedrock-runtime.<region>.amazonaws.com/openai/v1",
        key_env="AWS_BEARER_TOKEN_BEDROCK",
        models=("openai.gpt-5.6-sol", "mistral.devstral-2512"),
    ),
    "groq": ProviderConfig(
        id="groq",
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        key_env="GROQ_API_KEY",
        models=("llama-4.1-70b-versatile", "qwen3.7-32b"),
    ),
    "together": ProviderConfig(
        id="together",
        name="Together AI",
        base_url="https://api.together.xyz/v1",
        key_env="TOGETHER_API_KEY",
        models=(
            "meta-llama/Llama-4.1-70B-Instruct-Turbo",
            "Qwen/Qwen3.7-72B-Instruct-Turbo",
        ),
    ),
    "fireworks": ProviderConfig(
        id="fireworks",
        name="Fireworks AI",
        base_url="https://api.fireworks.ai/inference/v1",
        key_env="FIREWORKS_API_KEY",
        models=(
            "accounts/fireworks/models/llama-v4-70b-instruct",
            "accounts/fireworks/models/qwen3.7-32b-instruct",
        ),
    ),
    "cerebras": ProviderConfig(
        id="cerebras",
        name="Cerebras",
        base_url="https://api.cerebras.ai/v1",
        key_env="CEREBRAS_API_KEY",
        models=("llama-4.1-70b", "qwen-3.7-32b"),
    ),
    "deepinfra": ProviderConfig(
        id="deepinfra",
        name="DeepInfra",
        base_url="https://api.deepinfra.com/v1/openai",
        key_env="DEEPINFRA_API_KEY",
        models=(
            "meta-llama/Llama-4.1-70B-Instruct",
            "Qwen/Qwen3.7-72B-Instruct",
        ),
    ),
    "ollama": ProviderConfig(
        id="ollama",
        name="Ollama (local)",
        base_url="http://localhost:11434/v1",
        # With no chosen model, the session uses whatever the running server
        # lists first. The suggested tuple only fills the offline picker.
        key_env="OLLAMA_API_KEY",  # optional; only if the server enforces auth
        models=("llama3.1:8b", "qwen3:8b", "deepseek-r1:8b"),
    ),
    "lmstudio": ProviderConfig(
        id="lmstudio",
        name="LM Studio (local)",
        base_url="http://localhost:1234/v1",
        key_env="LMSTUDIO_API_KEY",  # optional; only if the server enforces auth
        models=("qwen2.5-coder-7b-instruct", "llama-3.1-8b-instruct"),
    ),
    "unsloth": ProviderConfig(
        id="unsloth",
        name="Unsloth (local)",
        base_url="http://localhost:8888/v1",
        # With no chosen model, the session uses whatever GGUF is loaded in
        # Unsloth Studio. The suggested tuple only fills the offline picker.
        # Required: Unsloth's local server enforces auth (Settings → API,
        # the key starts with sk-unsloth-).
        key_env="UNSLOTH_API_KEY",
        requires_key=True,
        models=("unsloth/Qwen3.6-27B-GGUF", "unsloth/gemma-4-26B-A4B-it-GGUF"),
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
