from __future__ import annotations

import json

import pytest

from kiwimatecoder import config


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    return tmp_path


def test_load_config_sets_version_and_new_defaults():
    cfg = config.load_config()

    assert cfg["version"] == config.CONFIG_VERSION
    assert cfg["tool_permissions"] == {}
    assert cfg["sampling"] == {}
    assert cfg["output_style"] == "default"
    assert cfg["system_prompt"] is None


def test_save_config_stamps_version():
    config.save_config({"keys": {}})

    stored = json.loads(config.CONFIG_FILE.read_text())
    assert stored["version"] == config.CONFIG_VERSION


def test_project_config_overrides_global(tmp_path):
    config.set_default_mode("ask")
    project = tmp_path / "project"
    project.mkdir()
    (project / config.PROJECT_CONFIG_NAME).write_text(
        json.dumps(
            {
                "default_mode": "plan",
                "selected_provider": "openai",
                "sampling": {"temperature": 0.3},
            }
        )
    )

    cfg = config.load_config(project_root=project)

    assert cfg["default_mode"] == "plan"
    assert config.get_sampling(cfg) == {"temperature": 0.3}


def test_project_config_cannot_carry_keys(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / config.PROJECT_CONFIG_NAME).write_text(
        json.dumps({"keys": {"openai": "stolen"}})
    )

    assert config.load_config(project_root=project)["keys"] == {}


def test_project_config_env_override(tmp_path, monkeypatch):
    project_file = tmp_path / "elsewhere.json"
    project_file.write_text(json.dumps({"default_mode": "auto-accept"}))
    monkeypatch.setenv(config.PROJECT_CONFIG_ENV, str(project_file))

    assert config.load_config()["default_mode"] == "auto-accept"


def test_project_selected_provider_seeds_roster(tmp_path):
    config.set_active_providers(["openrouter"])
    project = tmp_path / "project"
    project.mkdir()
    (project / config.PROJECT_CONFIG_NAME).write_text(
        json.dumps({"selected_provider": "openai"})
    )

    cfg = config.load_config(project_root=project)

    assert config.get_active_provider_ids(cfg) == ["openai"]


def test_project_config_corrupt_is_ignored(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / config.PROJECT_CONFIG_NAME).write_text("{not json")

    assert config.load_config(project_root=project)["default_mode"] == "ask"


def test_always_allowed_tool_persistence():
    assert config.get_always_allowed_tools() == []

    config.persist_always_allowed_tool("run_bash")
    config.persist_always_allowed_tool("run_bash")

    assert config.get_always_allowed_tools() == ["run_bash"]
    assert config.remove_always_allowed_tool("run_bash") is True
    assert config.remove_always_allowed_tool("run_bash") is False
    assert config.get_always_allowed_tools() == []


def test_clear_always_allowed_tools():
    config.persist_always_allowed_tool("run_bash")
    config.persist_always_allowed_tool("write_file")

    assert config.clear_always_allowed_tools() == 2
    assert config.get_always_allowed_tools() == []


def test_persist_always_allowed_tool_survives_reload():
    config.persist_always_allowed_tool("run_bash")

    fresh = config.load_config()

    assert config.get_always_allowed_tools(fresh) == ["run_bash"]


def test_command_rules_roundtrip_and_validation():
    rules = config.add_command_rule("deny", r"rm -rf")

    assert rules["deny"] == [r"rm -rf"]
    assert config.get_command_rules()["deny"] == [r"rm -rf"]

    with pytest.raises(ValueError):
        config.add_command_rule("deny", "[")
    with pytest.raises(ValueError):
        config.add_command_rule("nope", "x")

    assert config.remove_command_rule("deny", r"rm -rf") is True
    assert config.remove_command_rule("deny", r"rm -rf") is False

    config.add_command_rule("allow", "^pytest")
    assert config.get_command_rules()["allow"] == ["^pytest"]

    config.clear_command_rules()
    assert config.get_command_rules() == {"allow": [], "deny": []}


def test_set_command_rules_replaces_kinds():
    config.add_command_rule("deny", "one")

    config.set_command_rules(deny=["two"], allow=["three"])

    assert config.get_command_rules() == {"allow": ["three"], "deny": ["two"]}


def test_sampling_roundtrip_and_validation():
    effective = config.set_sampling({"temperature": "0.2", "max_tokens": "4096"})

    assert effective == {"temperature": 0.2, "max_tokens": 4096}
    assert config.get_sampling() == effective

    with pytest.raises(ValueError):
        config.set_sampling({"temperature": 5})
    with pytest.raises(ValueError):
        config.set_sampling({"nope": 1})
    with pytest.raises(ValueError):
        config.set_sampling({"reasoning_effort": "extreme"})

    config.set_sampling({"temperature": None})
    assert config.get_sampling() == {"max_tokens": 4096}

    config.reset_sampling()
    assert config.get_sampling() == {}


def test_output_style_validation():
    assert config.set_output_style("concise") == "concise"
    assert config.get_output_style() == "concise"

    with pytest.raises(ValueError):
        config.set_output_style("fancy")

    cfg = config.load_config()
    cfg["output_style"] = "garbage"
    config.save_config(cfg)
    assert config.get_output_style() == "default"


def test_system_prompt_roundtrip():
    config.set_system_prompt("Always write tests.")

    assert config.get_system_prompt() == "Always write tests."

    config.set_system_prompt("   ")
    assert config.get_system_prompt() is None


def test_trusted_workspace_roundtrip():
    assert config.get_trusted_workspace() is False
    assert config.set_trusted_workspace(True) is True
    assert config.get_trusted_workspace() is True
    assert config.set_trusted_workspace(False) is False


def test_verify_command_roundtrip():
    assert config.get_verify_command() == ""

    assert config.set_verify_command("pytest -q") == "pytest -q"
    assert config.get_verify_command() == "pytest -q"

    assert config.set_verify_command("") == ""
    assert config.get_verify_command() == ""


def test_budget_roundtrip_and_validation():
    assert config.get_budget() == {}

    assert config.set_budget(max_tokens=1000) == {"max_tokens": 1000}
    assert config.set_budget(max_cost_usd=1.5) == {
        "max_tokens": 1000,
        "max_cost_usd": 1.5,
    }

    with pytest.raises(ValueError):
        config.set_budget(max_tokens=0)
    with pytest.raises(ValueError):
        config.set_budget(max_cost_usd=-1)

    assert config.set_budget(max_tokens=None) == {"max_cost_usd": 1.5}
    config.clear_budget()
    assert config.get_budget() == {}


def test_compact_and_context_settings():
    assert config.get_compact_at_tokens() == 64000
    assert config.set_compact_at_tokens(20000) == 20000
    with pytest.raises(ValueError):
        config.set_compact_at_tokens(10)

    assert config.get_context_window() == 128000
    assert config.set_context_window(200000) == 200000
    with pytest.raises(ValueError):
        config.set_context_window(10)


def test_hooks_roundtrip_and_validation():
    assert config.get_hooks() == {event: [] for event in config.HOOK_EVENTS}

    hooks = config.add_hook("pre_tool", "echo first")
    assert hooks["pre_tool"] == ["echo first"]
    config.add_hook("pre_tool", "echo second")
    config.add_hook("pre_tool", "echo first")  # duplicate is ignored
    config.add_hook("post_tool", "echo done")

    assert config.get_hooks()["pre_tool"] == ["echo first", "echo second"]
    assert config.get_hooks()["post_tool"] == ["echo done"]
    assert config.get_hooks()["session_start"] == []

    with pytest.raises(ValueError):
        config.add_hook("nope", "echo x")
    with pytest.raises(ValueError):
        config.add_hook("pre_tool", "   ")
    with pytest.raises(ValueError):
        config.remove_hook("nope", 0)

    assert config.remove_hook("pre_tool", 0) is True
    assert config.remove_hook("pre_tool", 5) is False
    assert config.remove_hook("pre_tool", -1) is False
    assert config.get_hooks()["pre_tool"] == ["echo second"]

    config.clear_hooks()
    assert config.get_hooks() == {event: [] for event in config.HOOK_EVENTS}
    assert config.load_config()["hooks"] == {}


def test_hooks_persist_across_reload():
    config.add_hook("session_start", "echo hi")

    fresh = config.load_config()

    assert config.get_hooks(fresh)["session_start"] == ["echo hi"]
    assert fresh["hooks"]["session_start"] == ["echo hi"]


def test_hooks_tolerate_corrupt_section():
    cfg = config.load_config()
    cfg["hooks"] = "not a dict"
    config.save_config(cfg)

    assert config.get_hooks() == {event: [] for event in config.HOOK_EVENTS}


def test_project_config_can_add_hooks(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / config.PROJECT_CONFIG_NAME).write_text(
        json.dumps({"hooks": {"session_end": ["echo bye"]}})
    )

    cfg = config.load_config(project_root=project)

    assert config.get_hooks(cfg)["session_end"] == ["echo bye"]


def test_plugins_config_defaults_and_roundtrip():
    assert config.load_config()["plugins"] == {}
    assert config.get_plugins_config() == {"allow_project": False, "disabled": []}

    assert config.set_plugins_config(allow_project=True) == {
        "allow_project": True,
        "disabled": [],
    }
    assert config.set_plugins_config(disabled=["one", "one", " two "]) == {
        "allow_project": True,
        "disabled": ["one", "two"],
    }

    cfg = config.load_config()
    assert config.get_plugins_config(cfg) == {
        "allow_project": True,
        "disabled": ["one", "two"],
    }


def test_plugins_config_validation_and_corruption():
    with pytest.raises(ValueError):
        config.set_plugins_config(disabled=[""])
    with pytest.raises(ValueError):
        config.set_plugins_config(disabled="not-a-list")  # type: ignore[arg-type]

    cfg = config.load_config()
    cfg["plugins"] = "garbage"
    config.save_config(cfg)

    assert config.get_plugins_config() == {"allow_project": False, "disabled": []}
