from __future__ import annotations

import io

import pytest
from rich.console import Console

from kiwimatecoder import config, diagnostics
from kiwimatecoder.session import Session


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    return tmp_path


def test_run_checks_reports_core_checks(tmp_path):
    session = Session(provider_id="openrouter", model="test-model", workspace_root=tmp_path)

    checks = diagnostics.run_checks(session)
    names = {check.name for check in checks}

    assert {"Version", "Dependencies", "Config dir", "Provider", "Workspace"} <= names
    assert all(check.status in {diagnostics.OK, diagnostics.WARN, diagnostics.FAIL} for check in checks)
    provider = next(check for check in checks if check.name == "Provider")
    assert "openrouter" in provider.detail


def test_run_checks_flags_missing_workspace(tmp_path):
    session = Session(
        provider_id="openrouter",
        model="test-model",
        workspace_root=tmp_path / "does-not-exist",
    )

    checks = diagnostics.run_checks(session)
    workspace = next(check for check in checks if check.name == "Workspace")

    assert workspace.status == diagnostics.FAIL


def test_run_checks_never_raises_for_unknown_provider(tmp_path):
    session = Session(
        provider_id="openrouter",
        model="test-model",
        workspace_root=tmp_path,
    )
    session.provider_id = "not-a-provider"

    checks = diagnostics.run_checks(session)

    assert checks
    provider = next(check for check in checks if check.name == "Provider")
    assert provider.status == diagnostics.FAIL


def test_render_prints_table_and_summary(tmp_path):
    session = Session(provider_id="openrouter", model="test-model", workspace_root=tmp_path)
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, width=120)

    diagnostics.render(diagnostics.run_checks(session), console)

    text = output.getvalue().lower()
    assert "diagnostics" in text
    assert "check(s)" in text or "all checks passed" in text or "warning" in text
