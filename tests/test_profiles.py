from __future__ import annotations

import pytest

from kiwimatecoder import config


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    return tmp_path


# ---------------------------------------------------------------------------
# profile storage
# ---------------------------------------------------------------------------


def test_profiles_default_empty():
    cfg = config.load_config()

    assert cfg["profiles"] == {}
    assert config.get_profiles(cfg) == {}
    assert config.get_profile("nope", cfg) is None


def test_save_profile_captures_current_settings():
    config.set_selected_provider("openai")
    config.set_selected_model("gpt-5.6-sol")
    config.set_default_mode("plan")
    config.set_sampling({"temperature": 0.2})
    config.set_output_style("concise")
    config.set_system_prompt("Be terse.")
    config.set_verify_command("pytest -q")
    config.set_trusted_workspace(True)
    config.persist_always_allowed_tool("run_bash")
    config.add_command_rule("deny", r"rm -rf")

    profile = config.save_profile("work")

    assert profile["provider"] == "openai"
    assert profile["model"] == "gpt-5.6-sol"
    assert profile["mode"] == "plan"
    assert profile["sampling"] == {"temperature": 0.2}
    assert profile["output_style"] == "concise"
    assert profile["system_prompt"] == "Be terse."
    assert profile["verify_command"] == "pytest -q"
    assert profile["trusted_workspace"] is True
    assert profile["always_allowed"] == ["run_bash"]
    assert profile["command_rules"]["deny"] == [r"rm -rf"]


def test_save_profile_with_explicit_values_validates():
    with pytest.raises(ValueError):
        config.save_profile("bad", {"nope": 1})
    with pytest.raises(ValueError):
        config.save_profile("bad", {"mode": "nonsense"})
    with pytest.raises(ValueError):
        config.save_profile("bad", {"sampling": {"temperature": 99}})
    with pytest.raises(ValueError):
        config.save_profile("bad", {"sampling": {"bogus": 1}})
    with pytest.raises(ValueError):
        config.save_profile("bad", {"budget": {"max_cost_usd": -1}})
    with pytest.raises(ValueError):
        config.save_profile("bad", {"output_style": "fancy"})
    with pytest.raises(ValueError):
        config.save_profile("bad", {"command_rules": {"deny": ["("]}})
    with pytest.raises(ValueError):
        config.save_profile("bad", {"provider": "does-not-exist"})

    assert config.get_profiles() == {}


def test_save_profile_rejects_empty_name():
    with pytest.raises(ValueError):
        config.save_profile("   ", {"mode": "plan"})


def test_apply_profile_writes_global_config_and_persists():
    config.set_selected_provider("openai")
    config.set_selected_model("gpt-5.6-sol")
    config.set_default_mode("plan")
    config.save_profile("work")

    # Drift everything away, then apply the snapshot.
    config.set_selected_provider("deepseek")
    config.set_selected_model("other")
    config.set_default_mode("ask")

    applied = config.apply_profile("work")

    assert applied["provider"] == "openai"
    assert config.get_selected_provider_id() == "openai"
    assert config.load_config()["selected_model"] == "gpt-5.6-sol"
    assert config.get_default_mode() == "plan"


def test_apply_profile_unknown_raises():
    with pytest.raises(ValueError):
        config.apply_profile("missing")


def test_apply_profile_replaces_always_allowed_section():
    config.persist_always_allowed_tool("write_file")
    config.save_profile("work", {"always_allowed": ["run_bash"]})

    config.apply_profile("work")

    assert config.get_always_allowed_tools() == ["run_bash"]


def test_remove_and_rename_profile():
    config.save_profile("one", {"mode": "plan"})

    assert config.rename_profile("one", "two") is True
    assert config.get_profile("one") is None
    assert config.get_profile("two") == {"mode": "plan"}
    assert config.rename_profile("missing", "x") is False

    assert config.remove_profile("two") is True
    assert config.remove_profile("two") is False


def test_profile_roundtrip_through_reload():
    config.save_profile("solo", {"model": "m", "mode": "plan"})

    fresh = config.load_config()

    assert config.get_profile("solo", fresh) == {"model": "m", "mode": "plan"}


def test_profile_with_provider_becomes_primary_on_apply():
    config.set_active_providers(["openrouter", "openai"])
    config.save_profile("openai-mode", {"provider": "openai"})

    config.apply_profile("openai-mode")

    assert config.get_active_provider_ids() == ["openai"]


def test_get_profiles_drops_malformed_entries():
    config.save_profile("good", {"mode": "plan"})
    cfg = config.load_config()
    cfg["profiles"]["broken"] = {"mode": "nope"}
    cfg["profiles"]["not-an-object"] = "nope"

    profiles = config.get_profiles(cfg)

    assert set(profiles) == {"good"}


# ---------------------------------------------------------------------------
# schema validation
# ---------------------------------------------------------------------------


def test_validate_default_config_has_no_issues():
    assert config.validate_config() == []


def test_validate_flags_unknown_top_level_key_as_warning():
    cfg = config.load_config()
    cfg["future_key"] = 1

    issues = config.validate_config(cfg)

    assert issues == [
        {
            "level": "warning",
            "key": "future_key",
            "message": "Unknown top-level key 'future_key' is ignored.",
        }
    ]


def test_validate_flags_representative_bad_values():
    cfg = config.load_config()
    cfg["default_mode"] = "bogus"
    cfg["output_style"] = "fancy"
    cfg["trusted_workspace"] = "yes"
    cfg["sampling"] = {"temperature": 9, "unknown": 1}
    cfg["budget"] = {"max_tokens": -2}
    cfg["hooks"] = {"bogus_event": ["x"], "pre_tool": "not-a-list"}
    cfg["command_rules"] = {"deny": ["("]}
    cfg["profiles"] = {"broken": {"mode": "nope"}}
    cfg["mcp_servers"] = {"bad name": {"command": "x", "url": "http://x"}}
    cfg["plugins"] = {"disabled": "nope"}
    cfg["providers"] = {"custom": {"name": "Custom"}}
    cfg["keys"] = {"openai": 123}

    issues = config.validate_config(cfg)
    keys = {issue["key"] for issue in issues}

    assert any(issue["level"] == "error" for issue in issues)
    assert {
        "keys.openai",
        "providers.custom",
        "default_mode",
        "output_style",
        "trusted_workspace",
        "sampling.temperature",
        "sampling.unknown",
        "budget",
        "hooks.bogus_event",
        "hooks.pre_tool",
        "command_rules.deny[0]",
        "profiles.broken",
        "mcp_servers.bad name",
        "plugins.disabled",
    } <= keys


def test_validate_flags_unknown_active_provider_and_model_filter():
    cfg = config.load_config()
    cfg["active_providers"] = ["nope"]
    cfg["model_filters"] = {"openai": {"mode": "allow", "models": []}}

    issues = config.validate_config(cfg)
    keys = {issue["key"] for issue in issues}

    assert "active_providers[0]" in keys
    assert "model_filters.openai.models" in keys


def test_validate_passes_a_fully_populated_config():
    config.set_selected_provider("openai")
    config.set_selected_model("gpt-5.6-sol")
    config.set_default_mode("plan")
    config.set_sampling({"temperature": 0.2, "reasoning_effort": "medium"})
    config.set_output_style("concise")
    config.set_system_prompt("Be terse.")
    config.set_verify_command("pytest -q")
    config.set_trusted_workspace(True)
    config.set_budget(max_tokens=1000, max_cost_usd=2.5)
    config.add_command_rule("deny", r"rm -rf")
    config.add_hook("pre_tool", "echo hi")
    config.persist_always_allowed_tool("run_bash")
    config.save_profile("work")
    config.set_mcp_server("files", {"command": "npx", "args": ["-y", "server"]})

    assert config.validate_config() == []
