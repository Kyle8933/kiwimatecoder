from __future__ import annotations

import json

import pytest

from kiwimatecoder import audit, config


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    return tmp_path


def _entries() -> list[dict]:
    text = audit.audit_log_path().read_text(encoding="utf-8")
    return [json.loads(line) for line in text.strip().splitlines()]


def test_record_tool_event_redacts_secrets():
    audit.record_tool_event(
        tool="run_bash",
        args={"command": "export API_KEY=abcdefghijklmnop123456"},
        decision="allowed",
        duration_ms=5,
        ok=True,
    )

    entry = _entries()[0]

    assert entry["tool"] == "run_bash"
    assert entry["decision"] == "allowed"
    assert entry["duration_ms"] == 5
    assert entry["ok"] is True
    assert "abcdefghijklmnop123456" not in entry["args"]
    assert "[REDACTED]" in entry["args"]


def test_record_denied_event_has_reason():
    audit.record_tool_event(
        tool="run_bash",
        args={"command": "rm -rf /"},
        decision="denied",
        reason="Blocked by command deny rule.",
    )

    entry = _entries()[0]

    assert entry["decision"] == "denied"
    assert entry["reason"] == "Blocked by command deny rule."


def test_audit_log_appends_multiple_entries():
    audit.record_tool_event(tool="read_file", args={"path": "a"}, decision="allowed")
    audit.record_tool_event(tool="read_file", args={"path": "b"}, decision="allowed")

    assert [entry["tool"] for entry in _entries()] == ["read_file", "read_file"]


def test_audit_never_raises_on_unwritable_path(monkeypatch, tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")

    monkeypatch.setattr(audit, "audit_log_path", lambda: blocked / "audit.log")

    audit.record_tool_event(tool="read_file", args={}, decision="allowed")
