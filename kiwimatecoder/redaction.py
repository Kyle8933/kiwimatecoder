"""Secret detection and redaction for logs and audit records.

Patterns cover common provider keys, cloud credentials, bearer tokens, and
``key = value`` assignments. Redaction is intentionally conservative: it is
applied to audit records, never to tool output that the model needs verbatim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PLACEHOLDER = "[REDACTED]"

_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?key|secret|password|passwd|token)"
    r"(\s*[:=]\s*)['\"]?([A-Za-z0-9._\-/+]{12,})"
)

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{8,}")),
    ("openrouter_key", re.compile(r"\bsk-or-[A-Za-z0-9_\-]{8,}")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}")),
    (
        "github_token",
        re.compile(
            r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"
            r"|\bgithub_pat_[A-Za-z0-9_]{20,}"
        ),
    ),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("bearer_token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{20,}")),
]


@dataclass(frozen=True)
class Finding:
    """One detected secret, with a short non-reversible preview."""

    kind: str
    preview: str


def find_secrets(text: str) -> list[Finding]:
    """Return the secrets detected in ``text``."""
    findings: list[Finding] = []
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0)
            findings.append(Finding(kind=kind, preview=value[:4] + "…"))
    for match in _ASSIGNMENT.finditer(text):
        value = match.group(3)
        findings.append(Finding(kind="assignment", preview=value[:4] + "…"))
    return findings


def redact(text: str, placeholder: str = PLACEHOLDER) -> str:
    """Replace detected secrets in ``text`` with ``placeholder``."""
    result = text
    for _kind, pattern in _PATTERNS:
        result = pattern.sub(placeholder, result)
    result = _ASSIGNMENT.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{placeholder}", result
    )
    return result
