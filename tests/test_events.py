from __future__ import annotations

from kiwimatecoder import events
from kiwimatecoder.events import (
    POST_TOOL,
    PRE_TOOL,
    SESSION_END,
    SESSION_START,
    Event,
    EventBus,
)


def test_event_names_are_stable():
    assert SESSION_START == "session_start"
    assert SESSION_END == "session_end"
    assert PRE_TOOL == "pre_tool"
    assert POST_TOOL == "post_tool"
    assert events.EVENT_NAMES == (
        "session_start",
        "session_end",
        "pre_tool",
        "post_tool",
    )


def test_event_carries_name_and_payload():
    event = Event(name=PRE_TOOL, payload={"tool": "read_file"})

    assert event.name == "pre_tool"
    assert event.payload == {"tool": "read_file"}
    assert Event("bare").payload == {}


def test_emit_calls_subscribers_in_registration_order():
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe("ping", lambda event: seen.append("first") or 1)
    bus.subscribe("ping", lambda event: seen.append(event.payload["who"]) or 2)

    results = bus.emit("ping", who="second")

    assert seen == ["first", "second"]
    assert results == [1, 2]


def test_emit_without_subscribers_returns_empty():
    assert EventBus().emit("nobody") == []


def test_subscribe_returns_unsubscribe_callable():
    bus = EventBus()
    calls: list[str] = []
    unsubscribe = bus.subscribe("ping", lambda event: calls.append("hit"))

    bus.emit("ping")
    unsubscribe()
    bus.emit("ping")
    unsubscribe()  # idempotent

    assert calls == ["hit"]


def test_unsubscribe_method_reports_whether_removed():
    bus = EventBus()
    callback = lambda event: None  # noqa: E731

    bus.subscribe("ping", callback)
    assert bus.unsubscribe("ping", callback) is True
    assert bus.unsubscribe("ping", callback) is False
    assert bus.unsubscribe("other", callback) is False


def test_raising_subscriber_is_isolated_and_recorded():
    bus = EventBus()
    calls: list[str] = []

    def boom(event: Event) -> None:
        raise ValueError("bad subscriber")

    bus.subscribe("ping", boom)
    bus.subscribe("ping", lambda event: calls.append("after"))

    results = bus.emit("ping")

    assert len(results) == 2
    assert isinstance(results[0], ValueError)
    assert str(results[0]) == "bad subscriber"
    assert calls == ["after"]


def test_clear_removes_every_subscription():
    bus = EventBus()
    calls: list[str] = []
    bus.subscribe("ping", lambda event: calls.append("hit"))
    bus.subscribe("pong", lambda event: calls.append("hit"))

    bus.clear()
    bus.emit("ping")
    bus.emit("pong")

    assert calls == []


def test_module_helpers_use_default_bus():
    calls: list[Event] = []

    def record(event: Event) -> None:
        calls.append(event)

    unsubscribe = events.subscribe("custom", record)
    try:
        results = events.emit("custom", value=42)
        assert calls == [Event(name="custom", payload={"value": 42})]
        assert results == [None]
    finally:
        unsubscribe()

    events.emit("custom")
    assert len(calls) == 1