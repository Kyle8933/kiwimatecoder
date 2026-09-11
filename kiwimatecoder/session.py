"""Mutable runtime state for an interactive session."""

from __future__ import annotations

import datetime
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kiwimatecoder.checkpoints import Checkpoint, CheckpointStore
from kiwimatecoder.config import (
    ensure_config_dir,
    get_provider_config,
    resolve_default_model,
)
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.pricing import estimate_messages_tokens
from kiwimatecoder.providers import ProviderConfig

SESSION_FORMAT_VERSION = 2


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
    active_provider_ids: list[str] = field(default_factory=list)
    # Per-provider model overrides; empty value means "use the provider default".
    models: dict[str, str] = field(default_factory=dict)
    # Output style name and a user-supplied system-prompt addition.
    output_style: str = "default"
    custom_system_prompt: str | None = None
    # Command allow/deny regexes (see permissions.gate).
    command_rules: dict[str, list[str]] = field(default_factory=dict)
    # Show actions without running them when True.
    dry_run: bool = False
    # Whether reads outside the workspace are permitted.
    trusted_workspace: bool = False
    # Agent-maintained task list.
    todos: list[dict[str, Any]] = field(default_factory=list)
    # File snapshots for /undo (in-memory; cleared when the process exits).
    checkpoints: list[Checkpoint] = field(default_factory=list)
    checkpoint_store: CheckpointStore | None = field(default=None, repr=False)
    # Injected by the REPL so the ask_user tool can prompt interactively.
    ask_user: Callable[[str, list[str]], str] | None = field(default=None, repr=False)

    @property
    def format_version(self) -> int:
        return SESSION_FORMAT_VERSION

    @property
    def provider(self) -> ProviderConfig:
        return get_provider_config(self.provider_id)

    @property
    def active_providers(self) -> list[ProviderConfig]:
        """The active provider roster, primary first.

        Falls back to the session's own provider when the roster is empty so a
        resumed pre-feature session cannot pick up a different vendor from
        today's config. Always resolves to at least one provider.
        """
        ids = self.active_provider_ids or [self.provider_id]
        providers: list[ProviderConfig] = []
        seen: set[str] = set()
        for pid in ids:
            if pid in seen:
                continue
            seen.add(pid)
            try:
                providers.append(get_provider_config(pid))
            except KeyError:
                continue
        return providers or [self.provider]

    def model_for(self, provider_id: str) -> str:
        """Return the model to use for ``provider_id``.

        The primary provider uses ``session.model``; fallback providers use any
        per-provider override, otherwise their default model (resolved live for
        local servers).
        """
        if provider_id == self.provider_id:
            return self.model
        override = self.models.get(provider_id)
        if override:
            return override
        provider = get_provider_config(provider_id)
        return resolve_default_model(provider)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_history_tokens(self) -> int:
        return estimate_messages_tokens(self.messages)

    def set_provider(self, provider_id: str, model: str | None = None) -> None:
        """Switch provider; reset to the provider default model unless given.

        Local providers have no static default — their model is resolved from
        the running server's catalog (see ``config.resolve_default_model``).
        """
        provider = get_provider_config(provider_id)
        self.provider_id = provider_id
        self.model = model or resolve_default_model(provider)
        # Tool approvals are persisted user preferences, so they survive both
        # provider switches and restarts.

    def set_active_providers(self, provider_ids: list[str]) -> None:
        """Set the active-provider roster; the first id becomes the primary.

        ``provider_ids`` must be a non-empty list of known providers. The
        primary's model is resolved (or kept when unchanged), and fallback
        providers fall back to their default models.
        """
        if not provider_ids:
            raise ValueError("At least one active provider is required.")
        cleaned: list[str] = []
        seen: set[str] = set()
        for pid in provider_ids:
            if pid in seen:
                continue
            get_provider_config(pid)  # raises UnknownProviderError (a KeyError)
            seen.add(pid)
            cleaned.append(pid)
        if not cleaned:
            raise ValueError("At least one active provider is required.")
        previous_primary = self.provider_id
        self.active_provider_ids = cleaned
        if cleaned[0] != previous_primary:
            self.set_provider(cleaned[0])

    def record_touched(self, path: str) -> None:
        if path not in self.touched_files:
            self.touched_files.append(path)

    def checkpoint(self, paths: list[str], label: str) -> Checkpoint | None:
        """Snapshot ``paths`` before a mutation so it can be undone."""
        if self.checkpoint_store is None:
            self.checkpoint_store = CheckpointStore()
        try:
            captured = self.checkpoint_store.capture(self.workspace_root, paths, label)
        except OSError:
            return None
        self.checkpoints.append(captured)
        return captured

    def undo_checkpoints(self, count: int = 1) -> list[Checkpoint]:
        """Restore the most recent ``count`` checkpoints and drop them.

        Snapshots are restored newest-first so the oldest selected checkpoint
        wins, which rewinds the workspace to just before that action.
        """
        if not self.checkpoints:
            return []
        count = max(1, min(count, len(self.checkpoints)))
        selected = self.checkpoints[-count:]
        if self.checkpoint_store is not None:
            for item in reversed(selected):
                self.checkpoint_store.restore(self.workspace_root, item)
        self.checkpoints = self.checkpoints[: len(self.checkpoints) - count]
        return selected

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
            dropped_tokens = estimate_messages_tokens(dropped_turn)
            current_tokens -= dropped_tokens

        new_messages: list[dict[str, Any]] = []
        for turn in turns:
            new_messages.extend(turn)
        self.messages = new_messages
        return original_count - len(new_messages)

    def to_dict(self) -> dict[str, Any]:
        """Serialize session state for persistence."""
        return {
            "format_version": SESSION_FORMAT_VERSION,
            "provider_id": self.provider_id,
            "model": self.model,
            "mode": self.mode.value,
            "workspace_root": str(self.workspace_root),
            "messages": self.messages,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "touched_files": self.touched_files,
            "context_files": self.context_files,
            "always_allowed": sorted(self.always_allowed),
            "active_provider_ids": self.active_provider_ids,
            "models": self.models,
            "output_style": self.output_style,
            "custom_system_prompt": self.custom_system_prompt,
            "command_rules": self.command_rules,
            "dry_run": self.dry_run,
            "trusted_workspace": self.trusted_workspace,
            "todos": self.todos,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        """Deserialize session state."""
        provider_id = str(data.get("provider_id", "openrouter"))
        raw_active = data.get("active_provider_ids") or []
        active_provider_ids = [
            str(pid) for pid in raw_active if str(pid).strip()
        ]
        return cls(
            provider_id=provider_id,
            model=str(data.get("model", "")),
            mode=PermissionMode.from_str(str(data.get("mode", "ask"))),
            workspace_root=Path(str(data.get("workspace_root", "."))),
            messages=list(data.get("messages", [])),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            touched_files=list(data.get("touched_files", [])),
            context_files=list(data.get("context_files", [])),
            always_allowed={
                str(name) for name in (data.get("always_allowed") or []) if str(name)
            },
            active_provider_ids=active_provider_ids or [provider_id],
            models={
                str(k): str(v) for k, v in (data.get("models") or {}).items()
            },
            output_style=str(data.get("output_style") or "default"),
            custom_system_prompt=(
                str(data["custom_system_prompt"])
                if data.get("custom_system_prompt")
                else None
            ),
            command_rules={
                str(kind): [str(pattern) for pattern in (patterns or []) if str(pattern)]
                for kind, patterns in (data.get("command_rules") or {}).items()
            },
            dry_run=bool(data.get("dry_run", False)),
            trusted_workspace=bool(data.get("trusted_workspace", False)),
            todos=[
                dict(todo)
                for todo in (data.get("todos") or [])
                if isinstance(todo, dict)
            ],
        )


def _sessions_dir() -> Path:
    s_dir = ensure_config_dir() / "sessions"
    s_dir.mkdir(mode=0o700, exist_ok=True)
    return s_dir


# Name used by the implicit end-of-session autosave. ``--continue`` and
# ``/load last`` resolve to it (or to the newest saved session when it is gone).
AUTOSAVE_NAME = "last"


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
    """Load session state from name or path.

    ``"last"``/``"latest"`` resolve to the end-of-session autosave, falling
    back to the newest saved session when that file is missing.
    """
    target = Path(name_or_path)
    if not target.is_file():
        cleaned = name_or_path if name_or_path.endswith(".json") else f"{name_or_path}.json"
        target = _sessions_dir() / cleaned
        if (
            not target.is_file()
            and name_or_path.lower() in {AUTOSAVE_NAME, "latest"}
            and (latest := latest_saved_session())
        ):
            target = _sessions_dir() / latest
    if not target.is_file():
        raise FileNotFoundError(f"Session '{name_or_path}' not found.")

    data: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    sess = Session.from_dict(data)
    if workspace_root is not None:
        sess.workspace_root = workspace_root
    return sess


def save_autosave(session: Session) -> Path | None:
    """Save the session as the implicit ``last`` autosave.

    Empty sessions are not written so ``--continue`` keeps pointing at the most
    recent session that actually had a conversation. Returns the written path,
    or None when nothing was saved.
    """
    if not session.messages:
        return None
    return save_session(session, AUTOSAVE_NAME)


def latest_saved_session() -> str | None:
    """Return the file name of the most recently saved session, if any."""
    saved = list_saved_sessions()
    if not saved:
        return None
    return str(saved[0]["file"])


def fork_session(session: Session, name: str | None = None) -> Path:
    """Persist an independent copy of ``session`` under a new name."""
    if not name:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"fork_{stamp}"
    return save_session(session, name)


def export_session_markdown(session: Session, max_tool_chars: int = 2000) -> str:
    """Render the conversation as a Markdown transcript."""
    lines = [
        "# KiwiMateCoder session",
        "",
        f"- Provider: `{session.provider_id}`",
        f"- Model: `{session.model}`",
        f"- Mode: `{session.mode.value}`",
        f"- Messages: {len(session.messages)}",
        f"- Tokens: {session.total_tokens:,}",
        "",
    ]
    for message in session.messages:
        role = str(message.get("role") or "?")
        content = message.get("content")
        if isinstance(content, list):
            text = "\n".join(str(block) for block in content)
        else:
            text = str(content or "")
        if role == "tool":
            clipped = text[:max_tool_chars]
            if len(text) > max_tool_chars:
                clipped += f"\n\n[truncated {len(text) - max_tool_chars} chars]"
            lines.extend(["## Tool result", "", "```text", clipped, "```", ""])
            continue
        lines.extend([f"## {role.capitalize()}", "", text, ""])
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            lines.extend(
                [
                    f"- Tool call `{function.get('name', '?')}`:",
                    "",
                    "```json",
                    str(function.get("arguments") or "{}"),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


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
