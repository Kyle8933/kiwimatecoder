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
    assert cfg["ui"] == {}


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


def test_mcp_servers_crud_and_normalization():
    assert config.load_config()["mcp_servers"] == {}
    assert config.get_mcp_servers() == {}

    servers = config.set_mcp_server(
        "files", {"command": "npx", "args": ["-y", "srv"], "env": {"TOKEN": "x"}}
    )
    assert servers["files"] == {
        "command": "npx",
        "args": ["-y", "srv"],
        "env": {"TOKEN": "x"},
        "disabled": False,
    }

    config.set_mcp_server(
        "remote", {"url": "https://host/mcp", "headers": {"Authorization": "Bearer t"}}
    )
    assert config.get_mcp_servers()["remote"] == {
        "url": "https://host/mcp",
        "headers": {"Authorization": "Bearer t"},
        "disabled": False,
    }

    fresh = config.load_config()
    assert set(config.get_mcp_servers(fresh)) == {"files", "remote"}

    config.set_mcp_server("off", {"command": "srv", "disabled": True})
    assert config.get_mcp_servers()["off"]["disabled"] is True
    assert config.remove_mcp_server("off") is True
    assert config.remove_mcp_server("off") is False

    replaced = config.set_mcp_servers({"only": {"command": "srv"}})
    assert replaced == {
        "only": {"command": "srv", "args": [], "env": {}, "disabled": False}
    }
    assert config.get_mcp_servers() == replaced


def test_mcp_servers_validation_rejects_bad_names_and_specs():
    with pytest.raises(ValueError):
        config.set_mcp_server("Bad Name", {"command": "x"})
    with pytest.raises(ValueError):
        config.set_mcp_server("UPPER", {"command": "x"})
    with pytest.raises(ValueError):
        config.set_mcp_server("-leading", {"command": "x"})
    with pytest.raises(ValueError):
        config.set_mcp_server("both", {"command": "x", "url": "https://y"})
    with pytest.raises(ValueError):
        config.set_mcp_server("neither", {})
    with pytest.raises(ValueError):
        config.set_mcp_server("badargs", {"command": "x", "args": "nope"})
    with pytest.raises(ValueError):
        config.set_mcp_server("badenv", {"command": "x", "env": ["nope"]})
    with pytest.raises(ValueError):
        config.set_mcp_server("badurl", {"url": "ftp://host/mcp"})
    with pytest.raises(ValueError):
        config.set_mcp_servers("nope")  # type: ignore[arg-type]

    assert config.get_mcp_servers() == {}


def test_mcp_servers_tolerate_corrupt_section():
    cfg = config.load_config()
    cfg["mcp_servers"] = "garbage"
    config.save_config(cfg)

    assert config.get_mcp_servers() == {}


def test_prompt_cache_defaults_and_toggle_roundtrip():
    cfg = config.load_config()

    assert cfg["prompt_cache"] is False
    assert config.get_prompt_cache() is False

    assert config.set_prompt_cache(True) is True
    assert config.get_prompt_cache() is True
    assert config.load_config()["prompt_cache"] is True

    assert config.set_prompt_cache(False) is False
    assert config.get_prompt_cache() is False


def test_validate_flags_non_bool_prompt_cache():
    cfg = config.load_config()
    cfg["prompt_cache"] = "yes"

    issues = config.validate_config(cfg)

    assert any(
        issue["level"] == "error" and issue["key"] == "prompt_cache"
        for issue in issues
    )
    assert config.validate_config() == []


def test_validate_flags_bad_ui_values():
    cfg = config.load_config()
    cfg["ui"] = {
        "color": "rainbow",
        "output_mode": "loud",
        "ascii": "yes",
        "theme": "neon",
    }

    issues = config.validate_config(cfg)

    error_keys = {issue["key"] for issue in issues if issue["level"] == "error"}
    assert {
        "ui.color",
        "ui.output_mode",
        "ui.ascii",
        "ui.theme",
    } <= error_keys


def test_validate_flags_non_object_ui_section():
    cfg = config.load_config()
    cfg["ui"] = ["nope"]

    issues = config.validate_config(cfg)

    assert any(
        issue["level"] == "error" and issue["key"] == "ui" for issue in issues
    )


# ---------------------------------------------------------------------------
# Custom provider auth fields (Azure-style)
# ---------------------------------------------------------------------------


def test_custom_provider_roundtrip_persists_auth_fields():
    provider = config.add_provider(
        "my-azure",
        "My Azure",
        "https://my-resource.openai.azure.com/openai/v1",
        "my-deployment",
        "AZURE_OPENAI_API_KEY",
        key_header="api-key",
        key_prefix="",
        api_version="2024-10-21",
    )

    assert provider.key_header == "api-key"
    assert provider.key_prefix == ""
    assert provider.api_version == "2024-10-21"

    # A fresh load reads the same values back from disk.
    fresh = config.get_provider_config("my-azure", config.load_config())
    assert fresh.key_header == "api-key"
    assert fresh.key_prefix == ""
    assert fresh.api_version == "2024-10-21"

    stored = json.loads(config.CONFIG_FILE.read_text())
    assert stored["providers"]["my-azure"]["key_header"] == "api-key"


def test_update_provider_changes_and_clears_auth_fields():
    config.add_provider(
        "my-azure",
        "My Azure",
        "https://r.openai.azure.com/openai/v1",
        "deploy",
        "AZURE_OPENAI_API_KEY",
    )

    updated = config.update_provider(
        "my-azure",
        key_header="X-API-Key",
        key_prefix="Token ",
        api_version="2025-01-01",
    )
    assert updated.key_header == "X-API-Key"
    assert updated.key_prefix == "Token "
    assert updated.api_version == "2025-01-01"

    cleared = config.update_provider(
        "my-azure", key_prefix="", api_version=""
    )
    assert cleared.key_prefix == ""
    assert cleared.api_version == ""

    with pytest.raises(ValueError):
        config.update_provider("my-azure", key_header="   ")


def test_legacy_custom_provider_without_auth_fields_gets_defaults():
    cfg = config.load_config()
    cfg["providers"]["old"] = {
        "name": "Old",
        "base_url": "https://old.example.com/v1",
        "default_model": "old-model",
        "key_env": "OLD_API_KEY",
        "compat": "openai",
    }
    config.save_config(cfg)

    provider = config.get_provider_config("old")

    assert provider.key_header == "Authorization"
    assert provider.key_prefix == "Bearer "
    assert provider.api_version == ""
