"""Run eval cases against the SDK and evaluate their expectations.

``run_case`` materializes a case into a fresh temp workspace, drives
:func:`kiwimatecoder.sdk.run_agent_sync`, and turns every expectation into a
human-readable failure reason. ``run_suite`` runs a directory of cases and can
write a JSON report. Neither raises for a failing case: only invalid input
(missing directory, malformed case, no matching cases) raises ``CaseError``.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kiwimatecoder import sdk
from kiwimatecoder.evals.cases import (
    CaseError,
    EvalCase,
    discover_cases,
    resolve_cases_dir,
)
from kiwimatecoder.tools.paths import atomic_write_text

DEFAULT_TIMEOUT = 300.0
DEFAULT_MAX_TURNS = 20


class EvalTimeout(TimeoutError):
    """Raised internally when a case exceeds its ``timeout``."""


@dataclass
class EvalResult:
    """The outcome of one eval case: a pass/fail plus human-readable reasons."""

    name: str
    success: bool
    reasons: list[str] = field(default_factory=list)
    text: str = ""
    tools_used: list[str] = field(default_factory=list)
    error: str | None = None
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "success": self.success,
            "reasons": list(self.reasons),
            "duration_s": round(self.duration_s, 3),
            "error": self.error,
            "tools_used": list(self.tools_used),
            "text": self.text,
        }


@dataclass
class EvalReport:
    """The results of running a suite of eval cases."""

    results: list[EvalResult] = field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    cases_dir: str = ""

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.success)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def success(self) -> bool:
        return self.failed == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "provider": self.provider,
            "model": self.model,
            "cases_dir": self.cases_dir,
            "results": [result.to_dict() for result in self.results],
        }


def materialize_case(case: EvalCase, workspace: str | Path) -> dict[str, Path]:
    """Write the case's starting files into ``workspace`` and return their paths."""
    root = Path(workspace)
    written: dict[str, Path] = {}
    for relpath in sorted(case.files):
        target = root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target, case.files[relpath])
        written[relpath] = target
    return written


def _describe_tools(tools: list[str]) -> str:
    return ", ".join(tools) if tools else "none"


def evaluate_case(case: EvalCase, result: sdk.RunResult, workspace: str | Path) -> list[str]:
    """Return human-readable failures for a completed run (empty means passed)."""
    expect = case.expect
    problems: list[str] = []

    for needle in expect.text_contains:
        if needle not in result.text:
            problems.append(f"text_contains: {needle!r} not found in the reply")
    for needle in expect.text_not_contains:
        if needle in result.text:
            problems.append(f"text_not_contains: {needle!r} found in the reply")

    used = list(result.tools_used)
    for tool in expect.tools_called:
        if tool not in used:
            problems.append(
                f"tools_called: {tool!r} was not used (used: {_describe_tools(used)})"
            )
    for tool in expect.tools_not_called:
        if tool in used:
            problems.append(f"tools_not_called: {tool!r} was used")

    root = Path(workspace)
    for relpath in sorted(expect.files):
        target = root / relpath
        if not target.is_file():
            problems.append(f"files: {relpath!r} was not created")
            continue
        content = target.read_text(encoding="utf-8", errors="replace")
        for needle in expect.files[relpath]:
            if needle not in content:
                problems.append(f"files: {relpath!r} does not contain {needle!r}")

    if expect.success is not None:
        if result.success != expect.success:
            problems.append(f"success: expected {expect.success}, got {result.success}")
    elif result.error:
        problems.append(f"run failed: {result.error}")
    return problems


def _run_with_timeout(
    prompt: str,
    *,
    workspace: Path,
    provider: str | None,
    model: str | None,
    mode: str,
    timeout: float,
    max_turns: int,
) -> sdk.RunResult:
    """Run the SDK on a worker thread so ``timeout`` can bound the wait.

    The worker is a daemon thread, so a timed-out run never blocks process
    exit; its result is discarded.
    """
    box: list[sdk.RunResult] = []
    errors: list[BaseException] = []

    def target() -> None:
        try:
            box.append(
                sdk.run_agent_sync(
                    prompt,
                    workspace=workspace,
                    provider=provider,
                    model=model,
                    mode=mode,
                    max_turns=max_turns,
                )
            )
        except BaseException as exc:  # re-raised on the caller's thread
            errors.append(exc)

    worker = threading.Thread(target=target, name="kiwimatecoder-eval", daemon=True)
    worker.start()
    worker.join(timeout=timeout)
    if worker.is_alive():
        raise EvalTimeout(f"timed out after {timeout:g}s")
    if errors:
        raise errors[0]
    if not box:
        raise RuntimeError("eval run produced no result")
    return box[0]


def run_case(
    case: EvalCase,
    *,
    provider: str | None = None,
    model: str | None = None,
    mode: str = "auto-accept",
    timeout: float = DEFAULT_TIMEOUT,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> EvalResult:
    """Materialize ``case``, run the SDK, and evaluate every expectation.

    A run error, provider exception, or timeout becomes a failed result with a
    reason; this function does not raise for a failing case.
    """
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="kiwimatecoder-eval-") as tmp:
        workspace = Path(tmp)
        try:
            materialize_case(case, workspace)
        except OSError as exc:
            return EvalResult(
                name=case.name,
                success=False,
                reasons=[f"could not materialize case files: {exc}"],
                duration_s=time.monotonic() - started,
            )
        try:
            result = _run_with_timeout(
                case.prompt,
                workspace=workspace,
                provider=provider,
                model=model,
                mode=mode,
                timeout=timeout,
                max_turns=max_turns,
            )
        except EvalTimeout as exc:
            return EvalResult(
                name=case.name,
                success=False,
                reasons=[str(exc)],
                error="timeout",
                duration_s=time.monotonic() - started,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            return EvalResult(
                name=case.name,
                success=False,
                reasons=[f"run failed: {message}"],
                error=message,
                duration_s=time.monotonic() - started,
            )
        reasons = evaluate_case(case, result, workspace)
        return EvalResult(
            name=case.name,
            success=not reasons,
            reasons=reasons,
            text=result.text,
            tools_used=list(result.tools_used),
            error=result.error,
            duration_s=time.monotonic() - started,
        )


def run_suite(
    cases_dir: str | Path | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
    mode: str = "auto-accept",
    filter: str | None = None,
    report_path: str | Path | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> EvalReport:
    """Run every matching case and optionally write a JSON report.

    Raises :class:`CaseError` for invalid input: a missing directory, no
    matching cases, or malformed case files. A case whose expectations fail is
    reported, never raised.
    """
    root = resolve_cases_dir(cases_dir)
    cases = discover_cases(root)
    if filter:
        needle = filter.lower()
        cases = [case for case in cases if needle in case.name.lower()]
    if not cases:
        if filter:
            raise CaseError(f"No eval cases matched {filter!r} in {root}")
        raise CaseError(f"No eval cases found in {root}")

    report = EvalReport(provider=provider, model=model, cases_dir=str(root))
    for case in cases:
        report.results.append(
            run_case(
                case,
                provider=provider,
                model=model,
                mode=mode,
                timeout=timeout,
                max_turns=max_turns,
            )
        )

    if report_path is not None:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target, json.dumps(report.to_dict(), indent=2) + "\n")
    return report
