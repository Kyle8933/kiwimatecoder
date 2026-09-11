"""UI preferences: color control, themes, glyphs, and output modes.

The persistent values live under the ``ui`` key in the config file and are
normalized by :func:`kiwimatecoder.config.get_ui`. This module keeps the pure
decision logic (color precedence, glyph selection, theme accents) so it can be
unit-tested without a terminal.
"""

from __future__ import annotations

import os
from typing import Any

from rich.console import Console

UI_DEFAULTS: dict[str, Any] = {
    "color": "auto",
    "output_mode": "normal",
    "ascii": False,
    "theme": "default",
}

COLOR_MODES = ("auto", "always", "never")
OUTPUT_MODES = ("normal", "compact", "verbose")
THEMES = ("default", "ocean", "magenta", "mono")

UNICODE_GLYPHS: dict[str, str] = {
    "check": "✓",
    "cross": "✗",
    "blocked": "⊘",
    "folder": "📁",
    "bullet": "•",
}

ASCII_GLYPHS: dict[str, str] = {
    "check": "[ok]",
    "cross": "[fail]",
    "blocked": "[blocked]",
    "folder": "",
    "bullet": "-",
}

_THEME_ACCENTS: dict[str, str] = {
    "default": "green",
    "ocean": "cyan",
    "magenta": "magenta",
    "mono": "white",
}


def _ui_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize ``cfg`` (a config dict or nothing) into a UI settings dict."""
    from kiwimatecoder.config import get_ui

    if not isinstance(cfg, dict):
        cfg = None
    return get_ui(cfg)


def resolve_color(cfg: dict[str, Any] | None = None) -> bool:
    """Decide whether colored output is enabled, in this precedence order:

    1. An explicit ``ui.color`` of ``"always"`` enables color and ``"never"``
       disables it, overriding the environment.
    2. ``NO_COLOR`` set to a non-empty value disables color (https://no-color.org).
    3. ``FORCE_COLOR`` set to a non-empty value enables color.
    4. Otherwise True. Rich still auto-detects a non-TTY and strips styling,
       so a pipe or log file stays plain even when this returns True.
    """
    color = _ui_config(cfg)["color"]
    if color == "always":
        return True
    if color == "never":
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return True


def make_console(**kwargs: Any) -> Console:
    """Build a :class:`Console` honoring the color config and NO_COLOR.

    Extra keyword arguments (``file``, ``width``, ``force_terminal``, ...) are
    forwarded to Rich; an explicit ``no_color`` override wins.
    """
    kwargs.setdefault("no_color", not resolve_color())
    return Console(**kwargs)


def glyph(name: str, cfg: dict[str, Any] | None = None) -> str:
    """Return the active glyph for ``name`` (ASCII map when ascii mode is on).

    Unknown names fall back to the name itself so callers get a predictable
    label instead of a crash.
    """
    active = ASCII_GLYPHS if _ui_config(cfg)["ascii"] else UNICODE_GLYPHS
    return active.get(name, name)


def glyphs(cfg: dict[str, Any] | None = None) -> dict[str, str]:
    """Return the whole active glyph map (a copy, safe to mutate)."""
    return dict(ASCII_GLYPHS if _ui_config(cfg)["ascii"] else UNICODE_GLYPHS)


def theme_accent(cfg: dict[str, Any] | None = None) -> str:
    """Map the configured theme to a Rich color name.

    ``cfg`` may be a config dict or a session-like object; non-dicts fall back
    to the stored config.
    """
    return _THEME_ACCENTS.get(_ui_config(cfg)["theme"], "green")
