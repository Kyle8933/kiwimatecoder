"""Environment, config, and provider diagnostics for ``/doctor``."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from kiwimatecoder import __version__, audit, catalog, config
from kiwimatecoder.providers import ProviderConfig
from kiwimatecoder.session import Session

OK = "ok"
WARN = "warn"
FAIL = "fail"


@dataclass(frozen=True)
class Check:
    """One diagnostic result."""

    name: str
    status: str
    detail: str


def _git_branch(root: Path) -> str | None:
    head = root / ".git" / "HEAD"
    if not head.is_file():
        return None
    try:
        ref = head.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if ref.startswith("ref: refs/heads/"):
        return ref.split("/")[-1]
    return ref[:7] or None


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _config_checks() -> list[Check]:
    checks: list[Check] = []
    try:
        config_dir = config.ensure_config_dir()
    except OSError as exc:
        return [Check("Config dir", FAIL, f"cannot create {config.CONFIG_DIR}: {exc}")]

    writable = os.access(config_dir, os.W_OK)
    checks.append(
        Check(
            "Config dir",
            OK if writable else FAIL,
            f"{config_dir}" + ("" if writable else " (not writable)"),
        )
    )

    if config.CONFIG_FILE.is_file():
        checks.append(Check("Global config", OK, str(config.CONFIG_FILE)))
    else:
        checks.append(Check("Global config", WARN, "not created yet (defaults in use)"))

    project_path = config.project_config_path()
    if project_path is not None:
        checks.append(Check("Project config", OK, str(project_path)))
    else:
        candidate = Path.cwd() / config.PROJECT_CONFIG_NAME
        checks.append(Check("Project config", OK, f"none (optional; {candidate} not found)"))

    from kiwimatecoder import team as team_module

    team_settings = config.get_team()
    if not team_settings["policy_path"]:
        checks.append(Check("Team policy", OK, "none (optional)"))
    else:
        issues = team_module.policy_issues()
        errors = [issue for issue in issues if issue["level"] == "error"]
        detail = (
            f"{team_settings['policy_path']} "
            f"({'enforced' if team_settings['enforce'] else 'advisory'})"
        )
        if errors:
            status = FAIL
            detail += f" — {errors[0]['message']}"
        elif issues:
            status = WARN
            detail += f" — {issues[0]['message']}"
        else:
            status = OK
        checks.append(Check("Team policy", status, detail))

    sessions = config.ensure_config_dir() / "sessions"
    count = len(list(sessions.glob("*.json"))) if sessions.is_dir() else 0
    checks.append(Check("Saved sessions", OK, f"{count} in {sessions}"))

    audit_path = audit.audit_log_path()
    if audit_path.exists():
        try:
            size = audit_path.stat().st_size
        except OSError:
            size = 0
        checks.append(Check("Audit log", OK, f"{audit_path} ({size} bytes)"))
    else:
        checks.append(Check("Audit log", OK, f"{audit_path} (not created yet)"))

    from kiwimatecoder import telemetry as telemetry_module

    telemetry_settings = config.get_telemetry()
    crash_count = len(telemetry_module.crash_report_paths())
    checks.append(
        Check(
            "Telemetry",
            OK,
            f"level {telemetry_settings['level']} "
            f"({'on' if telemetry_settings['enabled'] else 'off'}) — "
            f"log {telemetry_module.current_log_path()}; "
            f"{crash_count} crash report(s)",
        )
    )
    return checks


def _provider_checks(session: Session) -> list[Check]:
    checks: list[Check] = []
    provider: ProviderConfig = session.provider
    checks.append(
        Check("Provider", OK, f"{provider.id} ({provider.name}) — {provider.base_url}")
    )
    key_source = config.key_source(provider.id)
    if provider.needs_key and key_source["origin"] == "missing":
        checks.append(
            Check(
                "API key",
                WARN,
                f"missing — export {provider.key_env} or run "
                f"`kiwimatecoder setup --provider {provider.id}`",
            )
        )
    else:
        checks.append(Check("API key", OK, config.describe_key(provider.id)))

    if provider.is_local:
        running = catalog.probe(provider, api_key=config.get_key(provider.id))
        checks.append(
            Check(
                "Local server",
                OK if running else WARN,
                f"{provider.base_url} " + ("is responding" if running else "did not answer"),
            )
        )

    try:
        model_catalog = config.get_model_catalog(provider.id)
        detail = f"{len(model_catalog.models)} models ({model_catalog.source})"
        if model_catalog.error:
            detail += f" — {model_catalog.error}"
        checks.append(
            Check(
                "Model catalog",
                WARN if model_catalog.error else OK,
                detail,
            )
        )
        if session.model and model_catalog.models:
            if session.model in model_catalog.models:
                checks.append(Check("Current model", OK, session.model))
            else:
                checks.append(
                    Check(
                        "Current model",
                        WARN,
                        f"{session.model} is not in the current catalog (still usable by name)",
                    )
                )
    except Exception as exc:  # pragma: no cover - defensive
        checks.append(Check("Model catalog", FAIL, str(exc)))
    return checks


def _workspace_checks(session: Session) -> list[Check]:
    root = session.workspace_root
    checks: list[Check] = []
    if not root.exists():
        return [Check("Workspace", FAIL, f"{root} does not exist")]
    writable = os.access(root, os.W_OK)
    branch = _git_branch(root)
    detail = str(root) + (f" (git:{branch})" if branch else "")
    checks.append(Check("Workspace", OK if writable else WARN, detail))
    checks.append(
        Check(
            "Permission mode",
            OK,
            f"{session.mode.value}; always-allowed: "
            + (", ".join(sorted(session.always_allowed)) or "none"),
        )
    )
    style = session.output_style
    custom = "custom prompt set" if session.custom_system_prompt else "no custom prompt"
    checks.append(Check("Output style", OK, f"{style} ({custom})"))
    sampling = config.get_sampling()
    checks.append(
        Check(
            "Sampling",
            OK,
            ", ".join(f"{key}={value}" for key, value in sampling.items())
            or "provider defaults",
        )
    )
    return checks


def run_checks(session: Session) -> list[Check]:
    """Collect all diagnostic checks, never raising."""
    checks: list[Check] = [
        Check(
            "Version",
            OK,
            f"kiwimatecoder {__version__} (Python {platform.python_version()}, "
            f"{sys.platform})",
        ),
        Check(
            "Dependencies",
            OK,
            "httpx " + _package_version("httpx")
            + ", rich " + _package_version("rich")
            + ", prompt_toolkit " + _package_version("prompt_toolkit")
            + ", typer " + _package_version("typer"),
        ),
    ]
    checks.extend(_config_checks())
    try:
        checks.extend(_provider_checks(session))
    except Exception as exc:  # pragma: no cover - defensive
        checks.append(Check("Provider", FAIL, str(exc)))
    checks.extend(_workspace_checks(session))
    return checks


_STATUS_STYLE = {OK: "green", WARN: "yellow", FAIL: "red"}


def render(checks: list[Check], console: Console) -> None:
    """Print checks as a table and a one-line summary."""
    table = Table(title="KiwiMateCoder diagnostics", show_header=True)
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Status")
    table.add_column("Details")
    for check in checks:
        style = _STATUS_STYLE.get(check.status, "white")
        table.add_row(check.name, f"[{style}]{check.status}[/{style}]", check.detail)
    console.print(table)
    failures = sum(1 for check in checks if check.status == FAIL)
    warnings = sum(1 for check in checks if check.status == WARN)
    if failures:
        console.print(f"[red]{failures} check(s) failed.[/red]")
    elif warnings:
        console.print(f"[yellow]{warnings} warning(s); nothing critical.[/yellow]")
    else:
        console.print("[green]All checks passed.[/green]")
