"""Headless one-shot agent runs shared by the CLI and the SDK.

The CLI's ``-p/--print`` mode and :mod:`kiwimatecoder.sdk` both drive the
regular :class:`~kiwimatecoder.agent.Agent` through this module so session
construction, event collection, and cost accounting live in one place.

Rendering stays with the caller: pass ``on_event`` to receive
``(name, payload)`` notifications for every emitted event and ``console`` for
the agent's own tool/progress output. Without a console the run is silent.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rich.console import Console

from kiwimatecoder import config, ui
from kiwimatecoder.agent import Agent
from kiwimatecoder.permissions import ConfirmFn, PermissionMode
from kiwimatecoder.pricing import estimate_cost
from kiwimatecoder.session import Session

EventCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class RunOutcome:
    """Everything one headless run produces for its caller."""

    text: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float | None = None
    provider: str = ""
    model: str = ""
    mode: str = ""
    tools_used: list[str] = field(default_factory=list)
    messages: int = 0
    success: bool = True
    error: str | None = None


def _resolve_workspace(workspace: str | Path | None) -> Path:
    root = Path(workspace).expanduser() if workspace is not None else Path.cwd()
    if not root.is_dir():
        raise ValueError(f"Workspace is not a directory: {root}")
    return root.resolve()


def build_session(
    *,
    workspace: str | Path | None = None,
    provider: str | None = None,
    model: str | None = None,
    mode: PermissionMode | str | None = None,
) -> Session:
    """Build a fresh session exactly like the interactive launch path.

    ``provider``/``model``/``mode`` override the configured defaults for this
    run only; nothing is persisted. Raises ``ValueError`` for an unknown
    workspace or mode and ``KeyError`` for an unknown provider.
    """
    cfg = config.load_config()
    provider_id = provider or config.get_selected_provider_id(cfg)
    provider_cfg = config.get_provider_config(provider_id, cfg)
    resolved_model = (
        model
        or str(cfg.get("selected_model") or "")
        or config.resolve_default_model(provider_cfg)
    )
    if mode is None:
        try:
            resolved_mode = PermissionMode.from_str(str(cfg.get("default_mode", "ask")))
        except ValueError:
            resolved_mode = PermissionMode.ASK
    else:
        resolved_mode = PermissionMode.from_str(mode) if isinstance(mode, str) else mode
    return Session(
        provider_id=provider_id,
        model=resolved_model,
        mode=resolved_mode,
        workspace_root=_resolve_workspace(workspace),
        active_provider_ids=config.get_active_provider_ids(cfg),
        always_allowed=set(config.get_always_allowed_tools(cfg)),
        output_style=config.get_output_style(cfg),
        custom_system_prompt=config.get_system_prompt(cfg),
        command_rules=config.get_command_rules(cfg),
        trusted_workspace=config.get_trusted_workspace(cfg),
        verify_command=config.get_verify_command(cfg),
        compact_at_tokens=config.get_compact_at_tokens(cfg),
        context_window=config.get_context_window(cfg),
    )


def _apply_overrides(
    session: Session,
    *,
    workspace: str | Path | None,
    provider: str | None,
    model: str | None,
    mode: PermissionMode | str | None,
) -> None:
    """Apply per-run overrides to an existing session."""
    if mode is not None:
        session.mode = PermissionMode.from_str(mode) if isinstance(mode, str) else mode
    if provider and provider != session.provider_id:
        session.set_provider(provider, model)
    elif model:
        session.model = model
    if workspace is not None:
        session.workspace_root = _resolve_workspace(workspace)


def _deny(summary: str, preview: str | None) -> bool:
    """Default approval behavior for runs with no interactive user."""
    return False


def _final_text(session: Session, deltas: list[str]) -> str:
    """The last assistant message, falling back to the streamed text."""
    for message in reversed(session.messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return "".join(deltas)


def shutdown_runtime() -> None:
    """Stop anything a headless run lazily started (MCP, LSP, browser)."""
    from kiwimatecoder import browser, lsp, mcp

    manager = mcp.get_manager()
    if manager is not None:
        mcp.set_manager(None)
        manager.shutdown()
    lsp_manager = lsp.get_manager()
    if lsp_manager is not None:
        lsp.set_manager(None)
        lsp_manager.shutdown()
    browser.reset_driver()


async def run_agent_once(
    prompt: str,
    *,
    workspace: str | Path | None = None,
    provider: str | None = None,
    model: str | None = None,
    mode: PermissionMode | str | None = None,
    confirm: ConfirmFn | None = None,
    on_event: EventCallback | None = None,
    max_turns: int | None = 30,
    console: Console | None = None,
    session: Session | None = None,
    render_text: bool = True,
    output_mode: str | None = None,
) -> RunOutcome:
    """Drive one agent turn and return its outcome.

    Approvals are denied when ``confirm`` is omitted. Text deltas are written
    to ``console`` unless ``render_text`` is False; every emitted event is
    forwarded to ``on_event``. Lazily started managers are shut down before
    returning.
    """
    if session is None:
        session = build_session(
            workspace=workspace, provider=provider, model=model, mode=mode
        )
    else:
        _apply_overrides(
            session, workspace=workspace, provider=provider, model=model, mode=mode
        )

    if console is None:
        console = Console(file=io.StringIO(), no_color=True, highlight=False)

    text_parts: list[str] = []
    tools_used: list[str] = []
    errors: list[str] = []
    done_reasons: list[str] = []

    def collect(name: str, payload: dict[str, Any]) -> None:
        if name == "text_delta":
            text_parts.append(str(payload.get("text") or ""))
        elif name == "tool_start":
            tool = str(payload.get("tool") or "")
            if tool and tool not in tools_used:
                tools_used.append(tool)
        elif name == "error":
            errors.append(str(payload.get("error") or "provider error"))
        elif name == "done":
            reason = str(payload.get("reason") or "")
            done_reasons.append(reason)
            if reason == "budget" and payload.get("error"):
                errors.append(str(payload["error"]))
        if on_event is not None:
            on_event(name, payload)

    ui_config = ui.ui_config()
    agent = Agent(
        session,
        console,
        confirm if confirm is not None else _deny,
        ascii_mode=ui_config["ascii"],
        output_mode=output_mode or ui_config["output_mode"],
        event_handler=collect,
        max_turns=max_turns,
        render_text=render_text,
        spinner=ui_config["spinner"],
    )

    try:
        await agent.run_turn(prompt)
    finally:
        shutdown_runtime()

    if max_turns is not None and "max_turns" in done_reasons:
        errors.append(
            f"Reached the maximum of {max_turns} tool-loop turn(s) without a "
            "final answer."
        )
    error = errors[0] if errors else None
    return RunOutcome(
        text=_final_text(session, text_parts),
        usage={
            "prompt_tokens": session.prompt_tokens,
            "completion_tokens": session.completion_tokens,
        },
        cost_usd=estimate_cost(
            session.prompt_tokens,
            session.completion_tokens,
            session.model,
            session.provider_id,
            is_local=session.provider.is_local,
        ),
        provider=session.provider_id,
        model=session.model,
        mode=session.mode.value,
        tools_used=tools_used,
        messages=len(session.messages),
        success=error is None,
        error=error,
    )
