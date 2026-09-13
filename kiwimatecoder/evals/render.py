"""Rich table and JSON rendering for eval reports."""

from __future__ import annotations

import json

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from kiwimatecoder.evals.runner import EvalReport


def report_to_json(report: EvalReport) -> str:
    """Serialize a report as pretty-printed JSON."""
    return json.dumps(report.to_dict(), indent=2)


def build_report_table(report: EvalReport) -> Table:
    """Build the per-case result table shown by ``kiwimatecoder eval run``."""
    table = Table(title="Eval results")
    table.add_column("Case", style="cyan", no_wrap=True)
    table.add_column("Result", no_wrap=True)
    table.add_column("Details", overflow="fold")
    table.add_column("Time", justify="right", no_wrap=True)
    for result in report.results:
        label = "PASS" if result.success else "FAIL"
        style = "green" if result.success else "red"
        details = "; ".join(result.reasons)
        table.add_row(
            escape(result.name),
            f"[{style}]{label}[/{style}]",
            escape(details),
            f"{result.duration_s:.1f}s",
        )
    return table


def print_report(report: EvalReport, console: Console | None = None) -> None:
    """Print the result table plus a one-line summary."""
    console = console or Console()
    console.print(build_report_table(report))
    style = "green" if report.success else "red"
    console.print(
        f"[{style}]{report.passed}/{report.total} eval case(s) passed[/{style}]"
    )
