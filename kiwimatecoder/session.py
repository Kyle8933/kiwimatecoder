"""Mutable runtime state for an interactive session."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.providers import ProviderConfig, get_provider


@dataclass
class Session:
    """All mutable state for one REPL session."""

    provider_id: str
    model: str
    mode: PermissionMode = PermissionMode.ASK
    workspace_root: Path = field(default_factory=Path.cwd)
    messages: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    touched_files: list[str] = field(default_factory=list)
    always_allowed: set[str] = field(default_factory=set)

    @property
    def provider(self) -> ProviderConfig:
        return get_provider(self.provider_id)

    def set_provider(self, provider_id: str, model: str | None = None) -> None:
        """Switch provider; reset to the provider default model unless given."""
        provider = get_provider(provider_id)
        self.provider_id = provider_id
        self.model = model or provider.default_model
        # Tool/command approvals don't carry across providers.
        self.always_allowed.clear()

    def record_touched(self, path: str) -> None:
        if path not in self.touched_files:
            self.touched_files.append(path)

    def is_always_allowed(self, tool_name: str) -> bool:
        return tool_name in self.always_allowed

    def allow_always(self, tool_name: str) -> None:
        self.always_allowed.add(tool_name)

    def add_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def reset_history(self) -> None:
        """Clear conversation history (the system prompt is rebuilt per turn)."""
        self.messages = []
