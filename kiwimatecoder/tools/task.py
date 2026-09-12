"""task tool: delegate a focused investigation to a subagent.

The tool is declared with ``runs=True`` so spawning a subagent is
approval-gated in ASK mode: the subagent can write files and run commands, and
its own actions still go through the normal permission gate.

Execution itself is special-cased in :class:`kiwimatecoder.agent.Agent` (the
subagent runs an async loop), so the sync callable below only exists for
dispatch paths without an agent driving them.
"""

from __future__ import annotations

from typing import Any

from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult


def _run_task(args: dict[str, Any], session: Session) -> ToolResult:
    return ToolResult.error(
        "The task tool must run inside the agent loop; use it from a normal "
        "conversation turn."
    )


task_tool = FunctionTool(
    name="task",
    description=(
        "Delegate a focused, self-contained investigation to a subagent that "
        "runs in its own context with the same workspace and tools. It cannot "
        "ask the user questions and it cannot delegate further, so put every "
        "needed decision in the prompt. It returns a final report and its "
        "intermediate tool output stays out of this conversation. Its file "
        "writes and commands still require the same approvals as yours."
    ),
    parameters={
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Short (3-5 word) description of the task.",
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Complete, self-contained instructions for the subagent, "
                    "including what to report back."
                ),
            },
            "model": {
                "type": "string",
                "description": "Optional model override for this subagent.",
            },
        },
        "required": ["description", "prompt"],
    },
    func=_run_task,
    writes=False,
    runs=True,
)
