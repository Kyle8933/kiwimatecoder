"""Mutable runtime state for an interactive session."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.config import get_provider_config
from kiwimatecoder.providers import ProviderConfig


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
    context_files: list[str] = field(default_factory=list)
    always_allowed: set[str] = field(default_factory=set)

    @property
    def provider(self) -> ProviderConfig:
        return get_provider_config(self.provider_id)

    def set_provider(self, provider_id: str, model: str | None = None) -> None:
        """Switch provider; reset to the provider default model unless given."""
        provider = get_provider_config(provider_id)
        self.provider_id = provider_id
        self.model = model or provider.default_model
        # Tool/command approvals don't carry across providers.
        self.always_allowed.clear()

    def record_touched(self, path: str) -> None:
        if path not in self.touched_files:
            self.touched_files.append(path)

    def add_context_file(self, path: str) -> bool:
        """Track a workspace-relative file as pinned context.

        Returns True when the file was newly added and False when it was already
        present. The caller is responsible for resolving and validating paths.
        """
        if path in self.context_files:
            return False
        self.context_files.append(path)
        return True

    def remove_context_file(self, path: str) -> bool:
        """Remove a pinned context file, returning whether anything changed."""
        try:
            self.context_files.remove(path)
        except ValueError:
            return False
        return True

    def clear_context_files(self) -> int:
        """Remove all pinned context files and return the number removed."""
        count = len(self.context_files)
        self.context_files = []
        return count

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
