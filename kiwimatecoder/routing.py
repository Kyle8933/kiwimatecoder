"""Per-turn model routing.

A deliberately small, pure heuristic: short, plain user messages can be sent
to a cheaper "simple" model, while anything that looks like real engineering
work stays on the session's model. Nothing here touches the network or global
state, so the decision is trivial to unit-test.
"""

from __future__ import annotations

import re
from typing import Any

#: A fenced code block means the user is showing/handling code.
_CODE_FENCE = "```"

#: A token like ``main.py`` or ``tests/test_x.py`` looks like a file path.
_FILE_PATH_RE = re.compile(r"\S+\.\w{1,6}")

#: A slash command such as ``/help`` or ``/config set``.
_SLASH_COMMAND_RE = re.compile(r"(?:^|\s)/[A-Za-z]")


def _contains_exclude_keyword(text: str, keywords: Any) -> bool:
    if not isinstance(keywords, list):
        return False
    lowered = text.lower()
    for keyword in keywords:
        word = str(keyword).strip().lower()
        if word and re.search(rf"\b{re.escape(word)}\b", lowered):
            return True
    return False


def choose_turn_model(
    user_input: str, session: Any, routing: dict[str, Any]
) -> str | None:
    """Return the simple model to use for this turn, or None to stay put.

    The override applies only when routing is enabled, a simple model is
    configured, the message is short, and it shows no sign of real work: no
    code fence, no file path, no slash command, and no exclude keyword
    (whole-word, case-insensitive).
    """
    if not routing.get("enabled"):
        return None
    simple_model = str(routing.get("simple_model") or "").strip()
    if not simple_model:
        return None
    text = user_input or ""
    try:
        max_chars = int(routing.get("simple_max_chars") or 0)
    except (TypeError, ValueError):
        return None
    if max_chars <= 0 or len(text) > max_chars:
        return None
    if _CODE_FENCE in text:
        return None
    if _FILE_PATH_RE.search(text):
        return None
    if _SLASH_COMMAND_RE.search(text):
        return None
    if _contains_exclude_keyword(text, routing.get("exclude_keywords")):
        return None
    return simple_model
