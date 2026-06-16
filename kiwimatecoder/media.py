"""Media generation extension point (deferred).

This module is a placeholder for future image/video generation support. The
intended design mirrors the model provider registry: a small registry of media
providers (each with a base URL, default model, and key env var), invoked
through a dedicated tool or a ``/image`` slash command.

It is intentionally not wired into the tool registry yet.
"""

from __future__ import annotations


def generate_media(*args, **kwargs):  # pragma: no cover - placeholder
    raise NotImplementedError(
        "Media generation is not implemented yet. This is a planned feature; "
        "see kiwimatecoder/media.py for the intended design."
    )
