from __future__ import annotations

from kiwimatecoder.redaction import find_secrets, redact


def test_redacts_openai_style_key():
    text = "export OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz123456"

    cleaned = redact(text)

    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in cleaned
    assert "[REDACTED]" in cleaned


def test_redacts_provider_prefixes():
    assert "sk-ant-" not in redact("key sk-ant-api03-abcdefghijklmnop")
    assert "sk-or-" not in redact("key sk-or-v1-abcdefghijklmnop")
    assert "ghp_" not in redact("token ghp_abcdefghijklmnopqrstuvwxyz012345")


def test_redacts_bearer_token():
    assert "abcdefghijklmnopqrstuvwx" not in redact(
        "Authorization: Bearer abcdefghijklmnopqrstuvwx"
    )


def test_redacts_assignment_preserving_name():
    cleaned = redact('API_KEY = "supersecretvalue123"')

    assert "API_KEY" in cleaned
    assert "supersecretvalue123" not in cleaned


def test_find_secrets_reports_kinds():
    findings = find_secrets("password=abcdefghijklmnop")

    assert findings
    assert findings[0].kind == "assignment"
    assert "abcdefghijklmnop" not in findings[0].preview


def test_find_secrets_empty_for_plain_code():
    assert find_secrets("def add(a, b):\n    return a + b") == []


def test_redact_leaves_plain_text_alone():
    text = "def add(a, b):\n    return a + b"

    assert redact(text) == text
