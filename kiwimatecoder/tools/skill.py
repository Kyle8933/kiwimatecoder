"""load_skill tool: fetch an Agent Skill's full instructions on demand."""

from __future__ import annotations

from typing import Any

from kiwimatecoder.session import Session
from kiwimatecoder.skills import discover_skills, load_skill as _load_skill
from kiwimatecoder.tools.base import FunctionTool, ToolResult


def _load_skill_tool(args: dict[str, Any], session: Session) -> ToolResult:
    name = str(args.get("name") or "").strip()
    if not name:
        return ToolResult.error("'name' is required")
    content = _load_skill(name, session.workspace_root)
    if content is None:
        available = [skill.name for skill in discover_skills(session.workspace_root)]
        hint = ", ".join(available) if available else "none"
        return ToolResult.error(
            f"Unknown skill '{name}'. Available skills: {hint}."
        )
    return ToolResult(content=content)


load_skill_tool = FunctionTool(
    name="load_skill",
    description=(
        "Load the full instructions for an available Agent Skill by name. "
        "Use this when the current task matches a skill listed in the system "
        "prompt; the return value is the skill's complete markdown guide."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the skill to load, exactly as listed.",
            }
        },
        "required": ["name"],
    },
    func=_load_skill_tool,
    writes=False,
    runs=False,
)
