from __future__ import annotations

import io

import pytest
from rich.console import Console

from kiwimatecoder import config, diagnostics
from kiwimatecoder.providers import REGISTRY
from kiwimatecoder.session import Session


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for provider in REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)
    return tmp_path


def _check(checks, name):
    return next(check for check in checks if check.name == name)


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


def test_team_policy_check_reports_missing_and_valid(tmp_path):
    session = Session(provider_id="openrouter", model="test-model", workspace_root=tmp_path)
    checks = diagnostics.run_checks(session)
    policy = next(check for check in checks if check.name == "Team policy")
    assert policy.status == diagnostics.OK
    assert "none" in policy.detail

    missing = tmp_path / "gone.json"
    cfg = config.load_config()
    cfg["team"] = {"policy_path": str(missing), "enforce": True}
    config.save_config(cfg)

    checks = diagnostics.run_checks(session)
    policy = next(check for check in checks if check.name == "Team policy")
    assert policy.status == diagnostics.FAIL
    assert "not found" in policy.detail

    valid = tmp_path / "policy.json"
    valid.write_text('{"default_mode": "plan"}')
    cfg = config.load_config()
    cfg["team"] = {"policy_path": str(valid), "enforce": True}
    config.save_config(cfg)

    checks = diagnostics.run_checks(session)
    policy = next(check for check in checks if check.name == "Team policy")
    assert policy.status == diagnostics.OK
    assert "enforced" in policy.detail


def test_current_model_check_warns_when_none_chosen(tmp_path):
    session = Session(provider_id="openrouter", model="", workspace_root=tmp_path)

    checks = diagnostics.run_checks(session)

    current = _check(checks, "Current model")
    assert current.status == diagnostics.WARN
    assert current.detail.startswith("none chosen")
    assert "/model" in current.detail
    assert "config model set <model> --provider openrouter" in current.detail


def test_current_model_check_ok_for_a_listed_model(tmp_path):
    model = REGISTRY["openrouter"].models[0]
    session = Session(provider_id="openrouter", model=model, workspace_root=tmp_path)

    current = _check(diagnostics.run_checks(session), "Current model")

    assert current.status == diagnostics.OK
    assert current.detail == model


def test_current_model_check_warns_for_an_unlisted_model(tmp_path):
    session = Session(
        provider_id="openrouter", model="vendor/unlisted", workspace_root=tmp_path
    )

    current = _check(diagnostics.run_checks(session), "Current model")

    assert current.status == diagnostics.WARN
    assert "not in the current catalog" in current.detail
