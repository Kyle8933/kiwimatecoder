"""ask_user tool: ask the user a clarifying question mid-turn."""

from __future__ import annotations

from typing import Any

from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult


def _ask_user(args: dict[str, Any], session: Session) -> ToolResult:
    question = str(args.get("question") or "").strip()
    if not question:
        return ToolResult.error("'question' is required")

    raw_options = args.get("options")
    options = (
        [str(option).strip() for option in raw_options if str(option).strip()]
        if isinstance(raw_options, list)
        else []
    )

    callback = getattr(session, "ask_user", None)
    if callback is None:
        return ToolResult.error(
            "No interactive user is available to answer; make a reasonable "
            "assumption and state it."
        )
    try:
        answer = callback(question, options)
    except (EOFError, KeyboardInterrupt):
        return ToolResult.error("The user did not answer.")
    answer = str(answer or "").strip()
    if not answer:
        return ToolResult.error("The user gave an empty answer.")
    return ToolResult(content=f"User answer: {answer}")


ask_user_tool = FunctionTool(
    name="ask_user",
    description=(
        "Ask the user a clarifying question when a decision cannot be inferred "
        "from the workspace. Provide 2-4 concrete options when possible. Use "
        "sparingly: prefer reading the relevant files first."
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask.",
            },
            "options": {
                "type": "array",
                "description": "Optional suggested answers.",
                "items": {"type": "string"},
            },
        },
        "required": ["question"],
    },
    func=_ask_user,
)
