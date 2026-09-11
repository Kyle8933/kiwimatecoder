"""A tiny synchronous event bus for lifecycle notifications.

Dependency-free by design: subscribers receive an :class:`Event` and may return
any value. :meth:`EventBus.emit` calls them in registration order and collects
their results. A raising subscriber never breaks the caller -- the exception is
captured in the results list and dispatch continues to the next subscriber.

Module-level helpers proxy to a default :data:`BUS`, which the agent and REPL
use unless a fresh bus is injected (tests do this for isolation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

SESSION_START = "session_start"
SESSION_END = "session_end"
PRE_TOOL = "pre_tool"
POST_TOOL = "post_tool"

EVENT_NAMES = (SESSION_START, SESSION_END, PRE_TOOL, POST_TOOL)

Subscriber = Callable[["Event"], Any]
Unsubscribe = Callable[[], None]


@dataclass
class Event:
    """One named lifecycle notification with a free-form payload."""

    name: str
    payload: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """Synchronous publish/subscribe over named events."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Subscriber]] = {}

    def subscribe(self, name: str, callback: Subscriber) -> Unsubscribe:
        """Register ``callback`` for ``name``; returns an unsubscribe callable."""
        self._subscribers.setdefault(name, []).append(callback)

        def unsubscribe() -> None:
            self.unsubscribe(name, callback)

        return unsubscribe

    def unsubscribe(self, name: str, callback: Subscriber) -> bool:
        """Remove one subscription; returns whether it was registered."""
        callbacks = self._subscribers.get(name)
        if not callbacks:
            return False
        for index, registered in enumerate(callbacks):
            if registered is callback:
                del callbacks[index]
                if not callbacks:
                    self._subscribers.pop(name, None)
                return True
        return False

    def emit(self, name: str, **payload: Any) -> list[Any]:
        """Call every subscriber for ``name``; collect results and exceptions."""
        event = Event(name=name, payload=payload)
        results: list[Any] = []
        for callback in list(self._subscribers.get(name, [])):
            try:
                results.append(callback(event))
            except Exception as exc:  # a bad subscriber must not break emit
                results.append(exc)
        return results

    def clear(self) -> None:
        """Drop every subscription."""
        self._subscribers.clear()


BUS = EventBus()


def subscribe(name: str, callback: Subscriber) -> Unsubscribe:
    """Subscribe to ``name`` on the default bus."""
    return BUS.subscribe(name, callback)


def unsubscribe(name: str, callback: Subscriber) -> bool:
    """Unsubscribe from ``name`` on the default bus."""
    return BUS.unsubscribe(name, callback)


def emit(name: str, **payload: Any) -> list[Any]:
    """Emit ``name`` on the default bus."""
    return BUS.emit(name, **payload)
