"""Embedded SDK for programmatic, headless agent runs.

The SDK wraps the same implementation the CLI's ``-p/--print`` mode uses, so a
host application gets identical session setup, tool behavior, permissions, and
cost accounting without spawning a subprocess.

Example::

    from kiwimatecoder.sdk import run_agent_sync

    result = run_agent_sync("Summarize README.md", workspace=".", mode="plan")
    if result.success:
        print(result.text)

``run_agent`` is the async entry point for hosts that already own an event
loop. Both entry points return a :class:`RunResult`. Approvals default to
deny; pass ``confirm`` to grant them (for example
``confirm=lambda summary, preview: True``) or run with
``mode="auto-accept"``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rich.console import Console

from kiwimatecoder.headless import run_agent_once
from kiwimatecoder.permissions import ConfirmFn, PermissionMode
from kiwimatecoder.session import Session

EventCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class RunResult:
    """The outcome of one programmatic agent run.

    ``usage`` totals the billed prompt/completion tokens across every model
    call in the turn. ``cost_usd`` is ``None`` when the model has no price
    entry (see :mod:`kiwimatecoder.pricing`). ``error`` is set only when
    ``success`` is False.
    """

    text: str
    usage: dict[str, int]
    cost_usd: float | None
    provider: str
    model: str
    mode: str
    tools_used: list[str]
    messages: int
    success: bool
    error: str | None = None


async def run_agent(
    prompt: str,
    *,
    workspace: str | Path | None = None,
    provider: str | None = None,
    model: str | None = None,
    mode: PermissionMode | str = PermissionMode.ASK,
    on_event: EventCallback | None = None,
    max_turns: int = 30,
    console: Console | None = None,
    session: Session | None = None,
    confirm: ConfirmFn | None = None,
) -> RunResult:
    """Run one agentic turn and return its :class:`RunResult`.

    ``workspace`` defaults to the current directory. ``provider``, ``model``,
    and ``mode`` override the configured defaults for this run only. Text is
    written to ``console`` when one is supplied, otherwise the run is silent.
    ``on_event`` receives ``(name, payload)`` for every render event. Without
    ``confirm`` (and outside auto-accept/plan modes) approvals are denied.
    """
    outcome = await run_agent_once(
        prompt,
        workspace=workspace,
        provider=provider,
        model=model,
        mode=mode,
        confirm=confirm,
        on_event=on_event,
        max_turns=max_turns,
        console=console,
        session=session,
        render_text=True,
    )
    return RunResult(
        text=outcome.text,
        usage=outcome.usage,
        cost_usd=outcome.cost_usd,
        provider=outcome.provider,
        model=outcome.model,
        mode=outcome.mode,
        tools_used=outcome.tools_used,
        messages=outcome.messages,
        success=outcome.success,
        error=outcome.error,
    )


def run_agent_sync(
    prompt: str,
    *,
    workspace: str | Path | None = None,
    provider: str | None = None,
    model: str | None = None,
    mode: PermissionMode | str = PermissionMode.ASK,
    on_event: EventCallback | None = None,
    max_turns: int = 30,
    console: Console | None = None,
    session: Session | None = None,
    confirm: ConfirmFn | None = None,
) -> RunResult:
    """Blocking wrapper around :func:`run_agent`.

    Raises :class:`RuntimeError` when called from inside a running event loop;
    ``await run_agent(...)`` instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            run_agent(
                prompt,
                workspace=workspace,
                provider=provider,
                model=model,
                mode=mode,
                on_event=on_event,
                max_turns=max_turns,
                console=console,
                session=session,
                confirm=confirm,
            )
        )
    raise RuntimeError(
        "run_agent_sync() cannot run inside a running event loop; "
        "await run_agent(...) instead."
    )
