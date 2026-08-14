"""Mutable runtime state for an interactive session."""

from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kiwimatecoder.config import ensure_config_dir, get_provider_config
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.providers import ProviderConfig


@dataclass
class Session:
    """All mutable state for one REPL session."""

    provider_id: str
    model: str
    mode: PermissionMode = PermissionMode.ASK
    workspace_root: Path = field(default_factory=Path.cwd)
    messages: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    touched_files: list[str] = field(default_factory=list)
    context_files: list[str] = field(default_factory=list)
    always_allowed: set[str] = field(default_factory=set)

    @property
    def provider(self) -> ProviderConfig:
        return get_provider_config(self.provider_id)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_history_tokens(self) -> int:
        return sum(
            len(str(m.get("content") or "")) // 4 + 10 for m in self.messages
        )

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

    def trim_history(self, max_tokens: int = 64_000) -> int:
        """Prune older conversation turns if token budget is exceeded.

        Preserves the initial user message and full turn boundaries so tool calls
        and tool results are never decoupled.
        """
        if len(self.messages) <= 4:
            return 0

        current_tokens = self.estimated_history_tokens
        if current_tokens <= max_tokens:
            return 0

        turns: list[list[dict[str, Any]]] = []
        current_turn: list[dict[str, Any]] = []
        for msg in self.messages:
            if msg.get("role") == "user" and not (
                isinstance(msg.get("content"), list)
                and any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in msg["content"]
                )
            ):
                if current_turn:
                    turns.append(current_turn)
                current_turn = [msg]
            else:
                current_turn.append(msg)
        if current_turn:
            turns.append(current_turn)

        if len(turns) <= 2:
            return 0

        original_count = len(self.messages)
        # Drop oldest intermediate turns starting from index 1
        while len(turns) > 2 and current_tokens > max_tokens:
            dropped_turn = turns.pop(1)
            dropped_tokens = sum(
                len(str(m.get("content") or "")) // 4 + 10 for m in dropped_turn
            )
            current_tokens -= dropped_tokens

        new_messages: list[dict[str, Any]] = []
        for turn in turns:
            new_messages.extend(turn)
        self.messages = new_messages
        return original_count - len(new_messages)

    def to_dict(self) -> dict[str, Any]:
        """Serialize session state for persistence."""
        return {
            "provider_id": self.provider_id,
            "model": self.model,
            "mode": self.mode.value,
            "workspace_root": str(self.workspace_root),
            "messages": self.messages,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "touched_files": self.touched_files,
            "context_files": self.context_files,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        """Deserialize session state."""
        return cls(
            provider_id=str(data.get("provider_id", "openrouter")),
            model=str(data.get("model", "")),
            mode=PermissionMode.from_str(str(data.get("mode", "ask"))),
            workspace_root=Path(str(data.get("workspace_root", "."))),
            messages=list(data.get("messages", [])),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            touched_files=list(data.get("touched_files", [])),
            context_files=list(data.get("context_files", [])),
        )


def _sessions_dir() -> Path:
    s_dir = ensure_config_dir() / "sessions"
    s_dir.mkdir(mode=0o700, exist_ok=True)
    return s_dir


def save_session(session: Session, name: str | None = None) -> Path:
    """Save session state to ~/.kiwimatecoder/sessions/<name>.json."""
    if not name:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"session_{ts}"
    cleaned = re.sub(r"[^\w\-.]", "_", name.strip())
    if not cleaned.endswith(".json"):
        cleaned += ".json"
    dest = _sessions_dir() / cleaned
    data = session.to_dict()
    data["saved_at"] = datetime.datetime.now().isoformat()
    dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return dest


def load_session(name_or_path: str, workspace_root: Path | None = None) -> Session:
    """Load session state from name or path."""
    target = Path(name_or_path)
    if not target.is_file():
        cleaned = name_or_path if name_or_path.endswith(".json") else f"{name_or_path}.json"
        target = _sessions_dir() / cleaned
    if not target.is_file():
        raise FileNotFoundError(f"Session '{name_or_path}' not found.")

    data: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    sess = Session.from_dict(data)
    if workspace_root is not None:
        sess.workspace_root = workspace_root
    return sess


def list_saved_sessions() -> list[dict[str, Any]]:
    """List all saved sessions sorted newest first."""
    s_dir = _sessions_dir()
    results: list[dict[str, Any]] = []
    for p in s_dir.glob("*.json"):
        try:
            data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
            results.append(
                {
                    "name": p.stem,
                    "file": p.name,
                    "provider": data.get("provider_id", ""),
                    "model": data.get("model", ""),
                    "saved_at": data.get("saved_at", ""),
                    "messages": len(data.get("messages", [])),
                    "tokens": int(data.get("prompt_tokens", 0))
                    + int(data.get("completion_tokens", 0)),
                }
            )
        except Exception:
            continue
    results.sort(key=lambda x: str(x.get("saved_at", "")), reverse=True)
    return results
