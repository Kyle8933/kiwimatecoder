"""update_goal tool: report progress on, or the outcome of, the session goal.

Only offered to the model while a ``/goal`` is active (see ``Agent._tool_schemas``).
"""

from __future__ import annotations

from typing import Any

from kiwimatecoder import goal as goal_module
from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult

REPORT_STATUSES = ("in_progress", "achieved", "blocked")


def _update_goal(args: dict[str, Any], session: Session) -> ToolResult:
    goal = session.goal
    if goal is None or goal.status != goal_module.ACTIVE:
        return ToolResult.error("No goal is active, so there is nothing to update.")

    status = str(args.get("status") or "").strip().lower()
    summary = str(args.get("summary") or "").strip()
    if status not in REPORT_STATUSES:
        return ToolResult.error(
            f"Unknown status '{status}'. Choose: {', '.join(REPORT_STATUSES)}."
        )
    if not summary:
        return ToolResult.error("'summary' is required: say what was done or what is missing.")

    if status == "in_progress":
        goal.progress = summary
        return ToolResult(content="Progress recorded. Keep working until the goal is achieved.")
    if status == "achieved":
        goal_module.close(goal, goal_module.ACHIEVED, summary)
        return ToolResult(
            content="Goal marked achieved. Stop here and give the user a short final summary."
        )
    goal_module.close(goal, goal_module.BLOCKED, summary)
    return ToolResult(content="Goal marked blocked. Stop and tell the user what you need from them.")


update_goal_tool = FunctionTool(
    name="update_goal",
    description=(
        "Report on the goal the user set with /goal. Call it with status 'in_progress' "
        "and a one-line summary after meaningful progress. Call it with 'achieved' only "
        "once you have verified the objective is fully met, and say how in the summary. "
        "Call it with 'blocked' only when you need something only the user can provide, "
        "and say exactly what in the summary."
    ),
    parameters={
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": list(REPORT_STATUSES),
                "description": "Where the goal stands.",
            },
            "summary": {
                "type": "string",
                "description": "One or two sentences: progress made, how the result was "
                "verified, or what is needed from the user.",
            },
        },
        "required": ["status", "summary"],
    },
    func=_update_goal,
)
