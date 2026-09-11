"""Append-only audit trail of tool actions.

Every tool decision (allowed, denied, or dry-run) is written as one JSON line
to ``~/.kiwimatecoder/audit.log`` with secrets redacted. Logging is best-effort
so a read-only home directory never breaks a session.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any

from kiwimatecoder.config import ensure_config_dir
from kiwimatecoder.redaction import redact

AUDIT_LOG_NAME = "audit.log"


def audit_log_path() -> Path:
    """Return the audit log path (resolved lazily so tests can patch CONFIG_DIR)."""
    return ensure_config_dir() / AUDIT_LOG_NAME


def record_tool_event(
    *,
    tool: str,
    args: dict[str, Any],
    decision: str,
    reason: str = "",
    duration_ms: int | None = None,
    ok: bool | None = None,
) -> None:
    """Append one redacted JSON record; never raises."""
    try:
        encoded_args = redact(json.dumps(args, default=str, ensure_ascii=False))
    except (TypeError, ValueError):
        encoded_args = redact(str(args))
    entry: dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "tool": tool,
        "args": encoded_args,
        "decision": decision,
    }
    if reason:
        entry["reason"] = redact(reason)
    if duration_ms is not None:
        entry["duration_ms"] = duration_ms
    if ok is not None:
        entry["ok"] = ok
    try:
        path = audit_log_path()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except OSError:
        pass
