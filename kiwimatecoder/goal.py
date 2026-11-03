"""Session goals: an objective the agent keeps working on across rounds.

``/goal <objective>`` sets one and the REPL runs agent rounds back to back until
the model reports it achieved or blocked, the round cap is reached, or a round
ends abnormally (budget, provider error, tool-call limit, Ctrl-C). This module
holds the state and builds the prompts; the REPL drives the rounds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

ACTIVE = "active"
PAUSED = "paused"
ACHIEVED = "achieved"
BLOCKED = "blocked"
STATUSES = (ACTIVE, PAUSED, ACHIEVED, BLOCKED)

DEFAULT_MAX_ROUNDS = 10
MAX_ROUNDS_LIMIT = 100
MAX_OBJECTIVE_CHARS = 4000

# Why a round ended, from the agent's "done" reason. Anything but a normal stop
# pauses the goal, and the reason is shown so the user knows what happened.
_STOP_REASONS = {
    "interrupted": "you interrupted it",
    "budget": "the budget is spent",
    "provider_error": "the provider returned an error",
    "max_turns": "the tool-call limit was reached",
}

_USAGE = (
    "Usage: /goal <objective> · /goal --rounds N <objective> · "
    "/goal pause | resume | clear · /goal"
)
_ROUNDS_FLAG = re.compile(r"^--rounds(?=\s|$)\s*(\S*)\s*(.*)$", re.DOTALL)


@dataclass
class Goal:
    """One standing objective and how far the agent has got with it."""

    objective: str
    max_rounds: int = DEFAULT_MAX_ROUNDS
    status: str = ACTIVE
    rounds: int = 0  # rounds run since the goal was set or last resumed
    progress: str = ""  # latest in_progress note from update_goal
    outcome: str = ""  # summary given with achieved/blocked, or why it paused
    due: bool = True  # runtime only: another round should run now

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "max_rounds": self.max_rounds,
            "status": self.status,
            "rounds": self.rounds,
            "progress": self.progress,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Goal | None:
        """Restore a saved goal. A goal that was running is restored as paused,
        so the user decides when to continue it (``/goal resume``)."""
        if not isinstance(data, dict):
            return None
        raw_objective = data.get("objective")
        objective = raw_objective.strip() if isinstance(raw_objective, str) else ""
        if not objective:
            return None
        status = str(data.get("status") or PAUSED)
        outcome = str(data.get("outcome") or "")
        if status not in STATUSES or status == ACTIVE:
            status = PAUSED
            outcome = "restored from a saved session"
        return cls(
            objective=objective,
            max_rounds=_clamp_rounds(data.get("max_rounds")),
            status=status,
            rounds=max(0, _as_int(data.get("rounds"))),
            progress=str(data.get("progress") or ""),
            outcome=outcome,
            due=False,
        )


@dataclass(frozen=True)
class GoalCommand:
    """A parsed ``/goal`` argument."""

    verb: str  # "show", "set", "pause", "resume" or "clear"
    objective: str = ""
    max_rounds: int = DEFAULT_MAX_ROUNDS


def parse_goal_command(arg: str) -> GoalCommand:
    """Parse the text after ``/goal``. Raises ``ValueError`` with a message."""
    text = arg.strip()
    if not text:
        return GoalCommand("show")

    # "--" makes the rest an objective even when it reads like a verb.
    if text == "--" or text.startswith("-- "):
        objective = text[2:].strip()
        if not objective:
            raise ValueError(_USAGE)
        return GoalCommand("set", _checked_objective(objective))

    max_rounds = DEFAULT_MAX_ROUNDS
    flag = _ROUNDS_FLAG.match(text)
    if flag:
        value, text = flag.group(1), flag.group(2).strip()
        # isascii: str.isdigit() accepts forms such as "²" that int() rejects.
        if not (value.isascii() and value.isdigit()) or not (
            1 <= int(value) <= MAX_ROUNDS_LIMIT
        ):
            raise ValueError(f"--rounds takes a number from 1 to {MAX_ROUNDS_LIMIT}.")
        max_rounds = int(value)
        if not text:
            raise ValueError("--rounds needs an objective, e.g. /goal --rounds 20 fix the tests.")
        return GoalCommand("set", _checked_objective(text), max_rounds)

    verb = text.lower()
    if verb in ("pause", "resume", "clear"):
        return GoalCommand(verb)
    return GoalCommand("set", _checked_objective(text), max_rounds)


def _checked_objective(text: str) -> str:
    if len(text) > MAX_OBJECTIVE_CHARS:
        raise ValueError(f"A goal can be at most {MAX_OBJECTIVE_CHARS} characters.")
    return text


def new_goal(objective: str, max_rounds: int = DEFAULT_MAX_ROUNDS) -> Goal:
    return Goal(objective=objective.strip(), max_rounds=_clamp_rounds(max_rounds))


def start_round(goal: Goal) -> str:
    """Begin the next round and return the user message that starts it."""
    goal.rounds += 1
    goal.due = False
    if goal.rounds == 1:
        return opening_prompt(goal)
    return continuation_prompt(goal)


def finish_round(goal: Goal, stop_reason: str | None) -> None:
    """Decide, after a round, whether another round is due.

    The goal keeps going only when the model stopped on its own and the round
    budget is not spent. A status set by ``update_goal`` during the round
    (achieved or blocked) is left alone.
    """
    goal.due = False
    if goal.status != ACTIVE:
        return
    if stop_reason != "stop":
        pause(goal, _STOP_REASONS.get(stop_reason or "", "the round ended early"))
        return
    if goal.rounds >= goal.max_rounds:
        pause(goal, f"reached the {goal.max_rounds}-round limit")
        return
    goal.due = True


def pause(goal: Goal, reason: str) -> None:
    goal.status = PAUSED
    goal.outcome = reason
    goal.due = False


def resume(goal: Goal) -> None:
    """Make the goal active again with a fresh round allowance."""
    goal.status = ACTIVE
    goal.rounds = 0
    goal.outcome = ""
    goal.due = True


def close(goal: Goal, status: str, summary: str) -> None:
    """Record the outcome reported through ``update_goal``."""
    goal.status = status
    goal.outcome = summary
    goal.due = False


def opening_prompt(goal: Goal) -> str:
    return (
        "You are working toward a goal the user set. Work through it step by step, "
        "using tools as needed, until the objective is fully met. Do not stop to ask "
        "for confirmation unless you are genuinely blocked.\n\n"
        f"Goal: {goal.objective}\n\n"
        "Report through the update_goal tool:\n"
        '- status "in_progress" with a one-line summary after meaningful progress;\n'
        '- status "achieved" only once you have checked the result, with a summary of '
        "how you verified it;\n"
        '- status "blocked" only if you need something only the user can provide, with '
        "a summary of exactly what you need."
    )


def continuation_prompt(goal: Goal) -> str:
    note = (
        f"Last progress note: {goal.progress}"
        if goal.progress
        else "No progress note has been recorded yet."
    )
    return (
        f"Keep working toward the goal (round {goal.rounds} of {goal.max_rounds}).\n"
        f"Goal: {goal.objective}\n"
        f"{note}\n\n"
        "Continue from where you left off. When the objective is fully met and "
        'verified, call update_goal with status "achieved"; if you are blocked on '
        'something only the user can give, call it with status "blocked".'
    )


def _clamp_rounds(value: Any) -> int:
    return min(MAX_ROUNDS_LIMIT, max(1, _as_int(value, DEFAULT_MAX_ROUNDS)))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
