"""Shared team policy overlays."""

from __future__ import annotations

import json

import pytest

from kiwimatecoder import config, team


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    team.clear_policy_cache()
    yield tmp_path
    team.clear_policy_cache()


def _write(path, data) -> str:
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Config CRUD and validation
# ---------------------------------------------------------------------------


def test_team_defaults():
    cfg = config.load_config()

    assert config.get_team(cfg) == {"policy_path": "", "enforce": False}
    assert cfg["team"] == {"policy_path": "", "enforce": False}
    assert team.load_policy(cfg=cfg) == {}


def test_set_team_requires_existing_policy_file(isolate_config):
    missing = isolate_config / "nope.json"

    with pytest.raises(ValueError):
        config.set_team(policy_path=str(missing))

    policy_file = _write(isolate_config / "policy.json", {"default_mode": "plan"})
    settings = config.set_team(policy_path=policy_file, enforce=True)

    assert settings["policy_path"] == policy_file
    assert settings["enforce"] is True
    assert config.get_team() == {"policy_path": policy_file, "enforce": True}


def test_set_team_rejects_bad_enforce_and_clears_path(isolate_config):
    with pytest.raises(ValueError):
        config.set_team(enforce="yes")

    policy_file = _write(isolate_config / "policy.json", {"default_mode": "plan"})
    config.set_team(policy_path=policy_file)
    assert config.set_team(policy_path="")["policy_path"] == ""


def test_validate_config_reports_team_types(isolate_config):
    cfg = config.load_config()
    cfg["team"] = {"policy_path": 42, "enforce": "sometimes"}
    config.save_config(cfg)

    issues = config.validate_config()
    keys = {issue["key"] for issue in issues}

    assert "team.policy_path" in keys
    assert "team.enforce" in keys


def test_validate_config_reports_missing_policy(isolate_config):
    cfg = config.load_config()
    cfg["team"] = {"policy_path": str(isolate_config / "gone.json"), "enforce": True}
    config.save_config(cfg)

    issues = config.validate_config()
    errors = [issue for issue in issues if issue["key"] == "team.policy_path"]

    assert errors
    assert errors[0]["level"] == "error"
    assert "not found" in errors[0]["message"]


def test_validate_config_reports_corrupt_policy(isolate_config):
    policy_file = isolate_config / "policy.json"
    policy_file.write_text("{not json", encoding="utf-8")
    assert config.set_team(policy_path=str(policy_file))["policy_path"]
    team.clear_policy_cache()

    issues = config.validate_config()
    errors = [issue for issue in issues if issue["key"] == "team.policy_path"]

    assert errors
    assert errors[0]["level"] == "error"


# ---------------------------------------------------------------------------
# load_policy
# ---------------------------------------------------------------------------


def test_policy_keys_are_the_documented_set():
    keys = team.policy_keys()

    assert "default_mode" in keys
    assert "command_rules" in keys
    assert "mcp_servers" in keys
    assert "network" in keys
    assert "team" not in keys
    assert "keys" not in keys


def test_load_policy_filters_unknown_and_secret_keys(isolate_config):
    policy_file = isolate_config / "policy.json"
    raw = {
        "version": 1,
        "keys": {"openai": "stolen"},
        "team": {"enforce": False},
        "bogus": True,
        "default_mode": "plan",
        "sampling": {"temperature": 0.2},
    }
    _write(policy_file, raw)

    policy = team.load_policy(policy_file)

    assert policy == {"default_mode": "plan", "sampling": {"temperature": 0.2}}
    assert config.load_config()["keys"] == {}


def test_load_policy_missing_and_corrupt(isolate_config):
    with pytest.raises(team.PolicyError):
        team.load_policy(isolate_config / "missing.json")

    bad = isolate_config / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(team.PolicyError):
        team.load_policy(bad)

    scalar = isolate_config / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    with pytest.raises(team.PolicyError):
        team.load_policy(scalar)


def test_policy_issues_unknown_key_warning(isolate_config):
    policy_file = _write(
        isolate_config / "policy.json", {"default_mode": "plan", "nope": 1}
    )
    config.set_team(policy_path=policy_file)

    issues = team.policy_issues()

    assert [issue["level"] for issue in issues] == ["warning"]
    assert "nope" in issues[0]["message"]


# ---------------------------------------------------------------------------
# Overlay precedence and enforcement
# ---------------------------------------------------------------------------


def _project(isolate_config, values) -> "object":
    project = isolate_config / "project"
    project.mkdir(exist_ok=True)
    (project / config.PROJECT_CONFIG_NAME).write_text(json.dumps(values))
    return project


def test_policy_overlay_precedence(isolate_config):
    config.set_default_mode("ask")
    config.set_sampling({"temperature": 0.9})
    project = _project(
        isolate_config,
        {"default_mode": "plan", "sampling": {"temperature": 0.3}},
    )
    policy_file = _write(
        isolate_config / "policy.json",
        {"default_mode": "auto-accept", "sampling": {"temperature": 0.1}},
    )
    config.set_team(policy_path=policy_file)

    cfg = config.load_config(project_root=project)

    assert cfg["default_mode"] == "auto-accept"
    assert config.get_sampling(cfg) == {"temperature": 0.1}


def test_advisory_policy_keeps_untouched_project_keys(isolate_config):
    policy_file = _write(isolate_config / "policy.json", {"default_mode": "plan"})
    config.set_team(policy_path=policy_file)
    project = _project(
        isolate_config,
        {"default_mode": "ask", "sampling": {"temperature": 0.3}},
    )

    cfg = config.load_config(project_root=project)

    assert cfg["default_mode"] == "plan"
    assert config.get_sampling(cfg) == {"temperature": 0.3}


def test_enforce_blocks_project_overrides(isolate_config):
    policy_file = _write(
        isolate_config / "policy.json",
        {
            "default_mode": "plan",
            "command_rules": {"deny": [r"rm -rf"]},
        },
    )
    config.set_team(policy_path=policy_file, enforce=True)
    project = _project(
        isolate_config,
        {
            "default_mode": "auto-accept",
            "command_rules": {"deny": [r"curl"]},
            "sampling": {"temperature": 0.5},
        },
    )

    cfg = config.load_config(project_root=project)

    assert cfg["default_mode"] == "plan"
    assert config.get_command_rules(cfg)["deny"] == [r"rm -rf"]
    # sampling is not defined by the policy, so the project value is rolled
    # back to global (unset), not merged in.
    assert config.get_sampling(cfg) == {}


def test_project_cannot_disable_team_policy(isolate_config):
    policy_file = _write(isolate_config / "policy.json", {"default_mode": "plan"})
    config.set_team(policy_path=policy_file, enforce=True)
    project = _project(
        isolate_config,
        {"default_mode": "ask", "team": {"policy_path": "", "enforce": False}},
    )

    cfg = config.load_config(project_root=project)

    assert config.get_team(cfg) == {"policy_path": policy_file, "enforce": True}
    assert cfg["default_mode"] == "plan"


def test_policy_selected_provider_seeds_roster(isolate_config):
    policy_file = _write(
        isolate_config / "policy.json", {"selected_provider": "openai"}
    )
    config.set_team(policy_path=policy_file)

    cfg = config.load_config()

    assert config.get_active_provider_ids(cfg) == ["openai"]


def test_enforce_rolls_back_project_seeded_roster(isolate_config):
    config.set_selected_provider("openrouter")
    policy_file = _write(isolate_config / "policy.json", {"default_mode": "plan"})
    config.set_team(policy_path=policy_file, enforce=True)
    project = _project(isolate_config, {"selected_provider": "openai"})

    cfg = config.load_config(project_root=project)

    assert config.get_active_provider_ids(cfg) == ["openrouter"]


def test_corrupt_policy_is_tolerated_with_issue(isolate_config):
    config.set_default_mode("ask")
    policy_file = isolate_config / "policy.json"
    policy_file.write_text("{not json", encoding="utf-8")
    config.set_team(policy_path=str(policy_file))
    team.clear_policy_cache()

    cfg = config.load_config()

    assert cfg["default_mode"] == "ask"
    issues = team.policy_issues(cfg)
    assert any(issue["level"] == "error" for issue in issues)


def test_missing_policy_is_tolerated_with_issue(isolate_config):
    cfg = config.load_config()
    cfg["team"] = {"policy_path": str(isolate_config / "gone.json"), "enforce": True}
    config.save_config(cfg)

    loaded = config.load_config()

    assert loaded["default_mode"] == "ask"
    assert any(
        issue["level"] == "error" for issue in team.policy_issues(loaded)
    )


def test_policy_file_change_is_picked_up(isolate_config):
    config.set_default_mode("ask")
    policy_file = isolate_config / "policy.json"
    _write(policy_file, {"default_mode": "plan"})
    config.set_team(policy_path=str(policy_file))
    assert config.load_config()["default_mode"] == "plan"

    _write(policy_file, {"default_mode": "auto-accept"})
    team.clear_policy_cache()
    assert config.load_config()["default_mode"] == "auto-accept"
