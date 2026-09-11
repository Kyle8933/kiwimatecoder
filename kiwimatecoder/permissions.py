"""Permission modes and the approval gate for tool actions.

Three modes, toggleable at runtime:

* ``ASK`` (default) — reads run freely; writes and commands show a preview and
  require y/n approval.
* ``AUTO`` — everything runs without prompting.
* ``PLAN`` — read-only; writes and commands are denied and the model is told so.

The confirm prompt is injected (``confirm`` callable) so tests run without a TTY.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol


class PermissionMode(str, Enum):
    ASK = "ask"
    AUTO = "auto-accept"
    PLAN = "plan"

    @classmethod
    def from_str(cls, value: str) -> "PermissionMode":
        value = value.strip().lower()
        aliases = {
            "ask": cls.ASK,
            "auto": cls.AUTO,
            "auto-accept": cls.AUTO,
            "accept": cls.AUTO,
            "plan": cls.PLAN,
            "read-only": cls.PLAN,
            "readonly": cls.PLAN,
        }
        if value not in aliases:
            raise ValueError(
                f"Unknown mode '{value}'. Choose: ask, auto-accept, plan."
            )
        return aliases[value]


@dataclass(frozen=True)
class ApprovalResult:
    """Outcome of a confirm prompt.

    ``selected_hunks`` is ``None`` for whole-action approval, or a tuple of
    1-based hunk indices when the user approved only part of a diff.
    """

    allowed: bool
    selected_hunks: tuple[int, ...] | None = None


@dataclass
class Decision:
    allowed: bool
    reason: str = ""
    selected_hunks: tuple[int, ...] | None = None


class SessionLike(Protocol):
    mode: PermissionMode
    command_rules: dict[str, list[str]]

    def is_always_allowed(self, tool_name: str) -> bool: ...


class ToolLike(Protocol):
    name: str

    @property
    def needs_approval(self) -> bool: ...


# A confirm callable receives (action_summary, preview_text) and returns True to
# allow the whole action, or an :class:`ApprovalResult` to allow a selection.
ConfirmFn = Callable[[str, str | None], bool | ApprovalResult]


def _command_rule_decision(
    tool_name: str, args: dict[str, Any], session: SessionLike
) -> Decision | None:
    """Apply run_bash allow/deny regexes.

    Deny patterns are a hard block (even in auto-accept mode). Allow patterns
    auto-approve a command that would otherwise prompt, but never override plan
    mode because the plan check runs before allow is consulted.
    """
    if tool_name != "run_bash":
        return None
    command = str(args.get("command") or "")
    if not command:
        return None
    rules = getattr(session, "command_rules", None)
    if not isinstance(rules, dict):
        return None

    patterns = rules.get("deny")
    for pattern in patterns if isinstance(patterns, list) else []:
        try:
            if re.search(str(pattern), command):
                return Decision(
                    allowed=False,
                    reason=f"Blocked by command deny rule '{pattern}'.",
                )
        except re.error:
            continue

    patterns = rules.get("allow")
    for pattern in patterns if isinstance(patterns, list) else []:
        try:
            if re.search(str(pattern), command):
                return Decision(allowed=True)
        except re.error:
            continue
    return None


def gate(
    tool: ToolLike,
    args: dict[str, Any],
    session: SessionLike,
    confirm: ConfirmFn,
    preview_text: str | None = None,
) -> Decision:
    """Decide whether a tool call may run under the session's current mode."""
    if not tool.needs_approval:
        return Decision(allowed=True)

    rule_decision = _command_rule_decision(tool.name, args, session)
    if rule_decision is not None and not rule_decision.allowed:
        return rule_decision

    if session.mode is PermissionMode.PLAN:
        return Decision(
            allowed=False,
            reason=(
                f"Blocked: {tool.name} cannot run in plan (read-only) mode. "
                "Describe the change instead, or ask the user to switch modes."
            ),
        )

    if session.mode is PermissionMode.AUTO:
        return Decision(allowed=True)

    if rule_decision is not None:
        return rule_decision

    if session.is_always_allowed(tool.name):
        return Decision(allowed=True)

    summary = f"{tool.name}({_summarize_args(args)})"
    outcome = confirm(summary, preview_text)
    if isinstance(outcome, ApprovalResult):
        if not outcome.allowed:
            return Decision(allowed=False, reason="Denied by user.")
        return Decision(allowed=True, selected_hunks=outcome.selected_hunks)
    if outcome:
        return Decision(allowed=True)
    return Decision(allowed=False, reason="Denied by user.")


def _summarize_args(args: dict[str, Any]) -> str:
    parts = []
    for key in ("path", "command"):
        if key in args:
            parts.append(f"{key}={args[key]!r}")
    return ", ".join(parts)
