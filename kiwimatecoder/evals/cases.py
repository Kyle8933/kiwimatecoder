"""Load and validate declarative eval cases from JSON files.

A case is a small unit of expected agent behavior: a prompt, an optional
starting workspace, and deterministic expectations about the run's text,
tools, and resulting files. See ``evals/README.md`` for the schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES_DIR = PACKAGE_ROOT / "evals" / "cases"

_TOP_LEVEL_KEYS = frozenset({"name", "description", "prompt", "files", "expect"})
_EXPECT_KEYS = frozenset(
    {
        "text_contains",
        "text_not_contains",
        "tools_called",
        "tools_not_called",
        "files",
        "success",
    }
)
_LIST_FIELDS = (
    "text_contains",
    "text_not_contains",
    "tools_called",
    "tools_not_called",
)


class CaseError(ValueError):
    """Raised when an eval case file is missing, unreadable, or malformed."""


def _fail(source: Path | None, message: str) -> CaseError:
    where = f"{source}: " if source is not None else ""
    return CaseError(f"{where}{message}")


def _non_empty_str(value: Any, *, key: str, source: Path | None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(source, f"'{key}' must be a non-empty string")
    return value


def _string_list(value: Any, *, key: str, source: Path | None) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _fail(source, f"'{key}' must be a list of strings")
    for item in value:
        if not isinstance(item, str) or not item:
            raise _fail(source, f"'{key}' entries must be non-empty strings")
    return tuple(value)


def _safe_relpath(value: Any, *, key: str, source: Path | None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(source, f"'{key}' paths must be non-empty strings")
    if "\\" in value:
        raise _fail(source, f"'{key}' path {value!r} must use forward slashes")
    path = Path(value)
    if path.is_absolute() or value.startswith("~"):
        raise _fail(source, f"'{key}' path {value!r} must be relative to the workspace")
    if ".." in path.parts:
        raise _fail(source, f"'{key}' path {value!r} must not contain '..'")
    return value


def _string_map(value: Any, *, key: str, source: Path | None) -> dict[str, str]:
    if not isinstance(value, dict):
        raise _fail(source, f"'{key}' must be an object mapping paths to contents")
    result: dict[str, str] = {}
    for raw_path, content in value.items():
        relpath = _safe_relpath(raw_path, key=key, source=source)
        if not isinstance(content, str):
            raise _fail(source, f"'{key}' value for {relpath!r} must be a string")
        result[relpath] = content
    return result


def _expect_files(value: Any, *, key: str, source: Path | None) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        raise _fail(
            source, f"expect.'{key}' must be an object mapping paths to string lists"
        )
    result: dict[str, tuple[str, ...]] = {}
    for raw_path, needles in value.items():
        relpath = _safe_relpath(raw_path, key=f"expect.{key}", source=source)
        result[relpath] = _string_list(needles, key=f"expect.{key}.{relpath}", source=source)
    return result


@dataclass
class Expectation:
    """Deterministic checks applied to one :class:`~kiwimatecoder.sdk.RunResult`."""

    text_contains: tuple[str, ...] = ()
    text_not_contains: tuple[str, ...] = ()
    tools_called: tuple[str, ...] = ()
    tools_not_called: tuple[str, ...] = ()
    files: dict[str, tuple[str, ...]] = field(default_factory=dict)
    success: bool | None = None


@dataclass
class EvalCase:
    """One prompt plus the workspace files and expectations for its run."""

    name: str
    prompt: str
    expect: Expectation = field(default_factory=Expectation)
    description: str = ""
    files: dict[str, str] = field(default_factory=dict)
    source: Path | None = None


def parse_case(payload: Any, *, source: Path | None = None) -> EvalCase:
    """Validate an already-decoded case payload and return an :class:`EvalCase`."""
    if not isinstance(payload, dict):
        raise _fail(source, "case must be a JSON object")

    unknown = sorted(set(payload) - _TOP_LEVEL_KEYS)
    if unknown:
        raise _fail(source, f"unknown key(s): {', '.join(unknown)}")

    name = _non_empty_str(payload.get("name"), key="name", source=source)
    prompt = _non_empty_str(payload.get("prompt"), key="prompt", source=source)

    description = payload.get("description", "")
    if not isinstance(description, str):
        raise _fail(source, "'description' must be a string")

    files: dict[str, str] = {}
    if "files" in payload:
        files = _string_map(payload["files"], key="files", source=source)

    if "expect" not in payload:
        raise _fail(source, "'expect' is required")
    expect_payload = payload["expect"]
    if not isinstance(expect_payload, dict):
        raise _fail(source, "'expect' must be an object")
    unknown_expect = sorted(set(expect_payload) - _EXPECT_KEYS)
    if unknown_expect:
        raise _fail(source, f"unknown expect key(s): {', '.join(unknown_expect)}")

    success = expect_payload.get("success")
    if success is not None and not isinstance(success, bool):
        raise _fail(source, "'expect.success' must be a boolean")

    expect = Expectation(
        text_contains=_string_list(
            expect_payload.get("text_contains", []), key="expect.text_contains", source=source
        ),
        text_not_contains=_string_list(
            expect_payload.get("text_not_contains", []),
            key="expect.text_not_contains",
            source=source,
        ),
        tools_called=_string_list(
            expect_payload.get("tools_called", []), key="expect.tools_called", source=source
        ),
        tools_not_called=_string_list(
            expect_payload.get("tools_not_called", []),
            key="expect.tools_not_called",
            source=source,
        ),
        files=_expect_files(expect_payload.get("files", {}), key="files", source=source),
        success=success,
    )
    return EvalCase(
        name=name,
        prompt=prompt,
        expect=expect,
        description=description,
        files=files,
        source=source,
    )


def load_case(path: str | Path) -> EvalCase:
    """Read and validate one ``*.json`` case file."""
    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise _fail(source, f"could not read case: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _fail(source, f"invalid JSON: {exc}") from exc
    return parse_case(payload, source=source)


def resolve_cases_dir(directory: str | Path | None = None) -> Path:
    """Return the cases directory, falling back to ``./evals/cases`` when installed."""
    if directory is not None:
        return Path(directory).expanduser()
    if DEFAULT_CASES_DIR.is_dir():
        return DEFAULT_CASES_DIR
    return Path.cwd() / "evals" / "cases"


def discover_cases(directory: str | Path | None = None) -> list[EvalCase]:
    """Load every ``*.json`` case under ``directory``, sorted by case name.

    Raises :class:`CaseError` for a missing directory, a malformed case file,
    or duplicate case names.
    """
    root = resolve_cases_dir(directory)
    if not root.is_dir():
        raise CaseError(f"Eval cases directory not found: {root}")
    cases: list[EvalCase] = []
    seen: dict[str, Path] = {}
    for path in sorted(root.glob("*.json")):
        case = load_case(path)
        if case.name in seen:
            raise CaseError(
                f"Duplicate eval case name {case.name!r} in {path} and {seen[case.name]}"
            )
        seen[case.name] = path
        cases.append(case)
    cases.sort(key=lambda case: case.name.lower())
    return cases
