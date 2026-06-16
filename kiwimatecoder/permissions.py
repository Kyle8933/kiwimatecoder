"""Permission modes and the approval gate for tool actions.

Three modes, toggleable at runtime:

* ``ASK`` (default) — reads run freely; writes and commands show a preview and
  require y/n approval.
* ``AUTO`` — everything runs without prompting.
* ``PLAN`` — read-only; writes and commands are denied and the model is told so.

The confirm prompt is injected (``confirm`` callable) so tests run without a TTY.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from kiwimatecoder.session import Session
    from kiwimatecoder.tools.base import FunctionTool


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


@dataclass
class Decision:
    allowed: bool
    reason: str = ""


# A confirm callable receives (action_summary, preview_text) and returns True to allow.
ConfirmFn = Callable[[str, str | None], bool]


def gate(
    tool: "FunctionTool",
    args: dict,
    session: "Session",
    confirm: ConfirmFn,
    preview_text: str | None = None,
) -> Decision:
    """Decide whether a tool call may run under the session's current mode."""
    if not tool.needs_approval:
        return Decision(allowed=True)

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

    if session.is_always_allowed(tool.name):
        return Decision(allowed=True)

    summary = f"{tool.name}({_summarize_args(args)})"
    approved = confirm(summary, preview_text)
    if approved:
        return Decision(allowed=True)
    return Decision(allowed=False, reason="Denied by user.")


def _summarize_args(args: dict) -> str:
    parts = []
    for key in ("path", "command"):
        if key in args:
            parts.append(f"{key}={args[key]!r}")
    return ", ".join(parts)
