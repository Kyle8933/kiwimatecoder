"""Declarative eval harness for prompt/tool regressions."""

from kiwimatecoder.evals.cases import (
    DEFAULT_CASES_DIR,
    CaseError,
    EvalCase,
    Expectation,
    discover_cases,
    load_case,
    parse_case,
    resolve_cases_dir,
)
from kiwimatecoder.evals.runner import (
    EvalReport,
    EvalResult,
    EvalTimeout,
    evaluate_case,
    materialize_case,
    run_case,
    run_suite,
)

__all__ = [
    "DEFAULT_CASES_DIR",
    "CaseError",
    "EvalCase",
    "EvalReport",
    "EvalResult",
    "EvalTimeout",
    "Expectation",
    "discover_cases",
    "evaluate_case",
    "load_case",
    "materialize_case",
    "parse_case",
    "resolve_cases_dir",
    "run_case",
    "run_suite",
]
