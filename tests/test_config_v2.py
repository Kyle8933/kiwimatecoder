from __future__ import annotations

import json

import pytest

from kiwimatecoder import config
from kiwimatecoder.providers import REGISTRY


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for provider in REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)
    return tmp_path


def _write_raw_config(data):
    """Write a config file exactly as given (no version stamping)."""
    config.CONFIG_FILE.write_text(json.dumps(data))


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


def test_config_version_is_3_with_an_empty_provider_models_map():
    cfg = config.load_config()

    assert config.CONFIG_VERSION == 3
    assert cfg["provider_models"] == {}
    assert cfg["selected_model"] is None


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


@pytest.mark.parametrize(
    "stored",
    [
        {"max_tokens": 0},
        {"max_tokens": -5},
        {"max_tokens": "lots"},
        {"max_tokens": [1]},
        {"max_cost_usd": 0},
        {"max_cost_usd": -1.5},
        {"max_cost_usd": "free"},
        {"max_cost_usd": {"usd": 1}},
    ],
)
def test_get_budget_ignores_a_stored_limit_that_set_budget_would_reject(stored):
    # A negative limit is truthy, so the agent read it as already spent and
    # refused every request; zero was shown as a limit but meant "none".
    _write_raw_config({"budget": stored})

    assert config.get_budget() == {}
    assert any(i["key"] == "budget" for i in config.validate_config())  # still reported


def test_get_budget_keeps_the_valid_limit_next_to_a_bad_one():
    _write_raw_config({"budget": {"max_tokens": -5, "max_cost_usd": 2.5}})
    assert config.get_budget() == {"max_cost_usd": 2.5}

    _write_raw_config({"budget": {"max_tokens": "5000", "max_cost_usd": 0}})
    assert config.get_budget() == {"max_tokens": 5000}


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
        "locale": "kl",
        "keybindings": "nano",
        "notify": "loud",
        "notify_after_seconds": -3,
        "spinner": "sometimes",
    }

    issues = config.validate_config(cfg)

    error_keys = {issue["key"] for issue in issues if issue["level"] == "error"}
    assert {
        "ui.color",
        "ui.output_mode",
        "ui.ascii",
        "ui.theme",
        "ui.locale",
        "ui.keybindings",
        "ui.notify",
        "ui.notify_after_seconds",
        "ui.spinner",
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


# ---------------------------------------------------------------------------
# Version 2 -> 3 migration: model choices move into provider_models
# ---------------------------------------------------------------------------


def test_v2_custom_default_model_moves_into_provider_models():
    _write_raw_config(
        {
            "version": 2,
            "providers": {
                "local": {
                    "name": "Local",
                    "base_url": "http://localhost:1234/v1",
                    "default_model": "local-code",
                    "key_env": "LOCAL_API_KEY",
                    "compat": "openai",
                },
                "blank": {
                    "name": "Blank",
                    "base_url": "https://blank.example/v1",
                    "default_model": "",
                    "key_env": "BLANK_API_KEY",
                },
            },
            "selected_provider": "openrouter",
            "active_providers": ["openrouter"],
            "selected_model": None,
        }
    )

    cfg = config.load_config()

    assert cfg["version"] == 3
    assert cfg["provider_models"] == {"local": "local-code"}
    assert "default_model" not in cfg["providers"]["local"]
    assert "default_model" not in cfg["providers"]["blank"]
    # Everything else about the provider is kept.
    assert cfg["providers"]["local"]["key_env"] == "LOCAL_API_KEY"
    assert config.get_provider_config("local", cfg).base_url == (
        "http://localhost:1234/v1"
    )
    assert config.get_provider_model("local", cfg) == "local-code"
    assert config.get_provider_model("blank", cfg) == ""


def test_v2_selected_model_moves_to_the_primary_provider():
    _write_raw_config(
        {
            "version": 2,
            "selected_provider": "openai",
            "active_providers": ["openai", "openrouter"],
            "selected_model": "gpt-picked",
        }
    )

    cfg = config.load_config()

    assert cfg["selected_model"] is None
    assert cfg["provider_models"] == {"openai": "gpt-picked"}
    assert config.get_provider_model("openai", cfg) == "gpt-picked"
    assert config.get_provider_model("openrouter", cfg) == ""


def test_v2_selected_model_wins_over_a_custom_primarys_default_model():
    """In version 2 the session used selected_model ahead of default_model."""
    _write_raw_config(
        {
            "version": 2,
            "providers": {
                "local": {
                    "name": "Local",
                    "base_url": "http://localhost:1234/v1",
                    "default_model": "provider-default",
                    "key_env": "LOCAL_API_KEY",
                },
            },
            "selected_provider": "local",
            "active_providers": ["local"],
            "selected_model": "user-picked",
        }
    )

    cfg = config.load_config()

    assert cfg["provider_models"] == {"local": "user-picked"}
    assert cfg["selected_model"] is None
    assert "default_model" not in cfg["providers"]["local"]


def test_unversioned_config_is_migrated_too():
    _write_raw_config(
        {
            "selected_provider": "deepseek",
            "active_providers": ["deepseek"],
            "selected_model": "ds-picked",
        }
    )

    cfg = config.load_config()

    assert cfg["provider_models"] == {"deepseek": "ds-picked"}
    assert cfg["selected_model"] is None


def test_migrated_v2_config_is_written_back_as_version_3():
    _write_raw_config(
        {
            "version": 2,
            "providers": {
                "local": {
                    "name": "Local",
                    "base_url": "http://localhost:1234/v1",
                    "default_model": "local-code",
                    "key_env": "LOCAL_API_KEY",
                },
            },
            "selected_provider": "openai",
            "active_providers": ["openai"],
            "selected_model": "gpt-picked",
        }
    )

    config.set_default_mode("plan")  # any write persists the migration

    stored = json.loads(config.CONFIG_FILE.read_text())
    assert stored["version"] == 3
    assert stored["selected_model"] is None
    assert stored["provider_models"] == {"local": "local-code", "openai": "gpt-picked"}
    assert "default_model" not in stored["providers"]["local"]
    # Reloading the written file is stable.
    assert config.load_config()["provider_models"] == stored["provider_models"]


def test_version_3_config_is_not_re_migrated():
    _write_raw_config(
        {
            "version": 3,
            "providers": {
                "local": {
                    "name": "Local",
                    "base_url": "http://localhost:1234/v1",
                    "default_model": "stale",
                    "key_env": "LOCAL_API_KEY",
                },
            },
            "selected_provider": "openai",
            "active_providers": ["openai"],
            "selected_model": "pinned-by-profile",
            "provider_models": {"openai": "gpt-chosen"},
        }
    )

    cfg = config.load_config()

    assert cfg["selected_model"] == "pinned-by-profile"
    assert cfg["provider_models"] == {"openai": "gpt-chosen"}
    # A stray default_model in a version-3 file is not a model choice.
    assert cfg["providers"]["local"]["default_model"] == "stale"
    assert config.get_provider_model("local", cfg) == ""
    assert config.get_provider_model("openai", cfg) == "pinned-by-profile"


def test_migration_does_not_pull_project_values_into_the_global_map(tmp_path):
    _write_raw_config(
        {
            "version": 2,
            "selected_provider": "openrouter",
            "active_providers": ["openrouter"],
            "selected_model": "global-pick",
        }
    )
    project = tmp_path / "project"
    project.mkdir()
    (project / config.PROJECT_CONFIG_NAME).write_text(
        json.dumps({"selected_provider": "openai", "selected_model": "project-pin"})
    )

    cfg = config.load_config(project_root=project)

    # Only the global choice (for the global primary) is migrated.
    assert cfg["provider_models"] == {"openrouter": "global-pick"}
    # The project's pin stays a pin for the project's primary.
    assert cfg["selected_model"] == "project-pin"
    assert config.get_active_provider_ids(cfg) == ["openai"]
    assert config.get_provider_model("openai", cfg) == "project-pin"
    assert config.get_provider_model("openrouter", cfg) == "global-pick"


def test_project_config_without_a_pin_leaves_the_migrated_choice(tmp_path):
    _write_raw_config(
        {
            "version": 2,
            "selected_provider": "openai",
            "active_providers": ["openai"],
            "selected_model": "gpt-picked",
        }
    )
    project = tmp_path / "project"
    project.mkdir()
    (project / config.PROJECT_CONFIG_NAME).write_text(
        json.dumps({"default_mode": "plan"})
    )

    cfg = config.load_config(project_root=project)

    assert cfg["provider_models"] == {"openai": "gpt-picked"}
    assert cfg["selected_model"] is None
    assert config.get_provider_model("openai", cfg) == "gpt-picked"


def test_project_provider_models_merge_per_provider(tmp_path):
    config.set_provider_model("openai", "global-gpt")
    config.set_provider_model("deepseek", "global-ds")
    project = tmp_path / "project"
    project.mkdir()
    (project / config.PROJECT_CONFIG_NAME).write_text(
        json.dumps({"provider_models": {"openai": "project-gpt"}})
    )

    cfg = config.load_config(project_root=project)

    assert cfg["provider_models"] == {
        "openai": "project-gpt",
        "deepseek": "global-ds",
    }
    assert config.get_provider_model("openai", cfg) == "project-gpt"
    # The global file is not rewritten by reading a project overlay.
    assert config.load_config()["provider_models"] == {
        "openai": "global-gpt",
        "deepseek": "global-ds",
    }


# ---------------------------------------------------------------------------
# Custom providers shadowed by a new built-in (KiwiMate)
# ---------------------------------------------------------------------------


def test_custom_kiwimate_provider_is_renamed_away_from_the_builtin():
    _write_raw_config(
        {
            "version": 2,
            "providers": {
                "kiwimate": {
                    "name": "My KiwiMate",
                    "base_url": "https://my-kiwi.example/v1",
                    "default_model": "km-own",
                    "key_env": "MY_KIWI_API_KEY",
                },
            },
            "keys": {"kiwimate": "sk-mine", "openai": "sk-openai"},
            "model_filters": {"kiwimate": {"mode": "allow", "models": ["km-own"]}},
            "selected_provider": "kiwimate",
            "active_providers": ["kiwimate", "openai"],
        }
    )

    cfg = config.load_config()

    assert "kiwimate" not in cfg["providers"]
    assert cfg["providers"]["kiwimate-custom"]["base_url"] == (
        "https://my-kiwi.example/v1"
    )
    assert cfg["keys"] == {"kiwimate-custom": "sk-mine", "openai": "sk-openai"}
    assert cfg["model_filters"] == {
        "kiwimate-custom": {"mode": "allow", "models": ["km-own"]}
    }
    assert cfg["provider_models"] == {"kiwimate-custom": "km-own"}
    assert cfg["active_providers"] == ["kiwimate-custom", "openai"]
    assert cfg["selected_provider"] == "kiwimate-custom"
    assert config.get_provider_config("kiwimate-custom", cfg).name == "My KiwiMate"
    # The id "kiwimate" now always means the built-in, which the old key and
    # model never reach.
    assert config.get_provider_config("kiwimate", cfg) is REGISTRY["kiwimate"]
    assert config.get_provider_model("kiwimate", cfg) == ""


def test_v2_selected_model_of_a_shadowed_custom_primary_follows_the_rename():
    """A v2 pick for a hand-made "kiwimate" must never land on the built-in."""
    _write_raw_config(
        {
            "version": 2,
            "providers": {
                "kiwimate": {
                    "name": "My KiwiMate",
                    "base_url": "https://my-kiwi.example/v1",
                    "default_model": "km-default",
                },
            },
            "selected_provider": "kiwimate",
            "active_providers": ["kiwimate"],
            "selected_model": "km-picked",
        }
    )

    cfg = config.load_config()

    assert cfg["provider_models"] == {"kiwimate-custom": "km-picked"}
    assert cfg["selected_model"] is None
    assert config.get_selected_provider_id(cfg) == "kiwimate-custom"
    assert config.get_provider_model("kiwimate-custom", cfg) == "km-picked"
    assert config.get_provider_model("kiwimate", cfg) == ""


def test_version_3_files_never_rename_a_shadowed_custom_provider():
    # Only the v2 -> v3 migration renames. A custom entry with a built-in id
    # in a v3 file (hand-edited, or leaked from a project config) stays inert:
    # it is never listed and never takes over the built-in's key.
    _write_raw_config(
        {
            "version": 3,
            "providers": {
                "kiwimate": {
                    "name": "My KiwiMate",
                    "base_url": "https://my-kiwi.example/v1",
                    "key_env": "MY_KIWI_API_KEY",
                },
            },
            "keys": {"kiwimate": "sk-km-real"},
            "provider_models": {"kiwimate": "kiwimate-large-1-0"},
            "selected_provider": "kiwimate",
            "active_providers": ["kiwimate"],
        }
    )

    cfg = config.load_config()

    assert set(cfg["providers"]) == {"kiwimate"}
    assert cfg["keys"] == {"kiwimate": "sk-km-real"}
    assert cfg["active_providers"] == ["kiwimate"]
    assert config.get_provider_config("kiwimate", cfg) is REGISTRY["kiwimate"]
    assert config.get_provider_model("kiwimate", cfg) == "kiwimate-large-1-0"
    assert [p.id for p in config.list_provider_configs(cfg)].count("kiwimate") == 1


def test_migration_does_not_revive_a_custom_entry_shadowing_an_older_builtin():
    # E.g. a project config's "anthropic" override leaked into the global file
    # by an old release. Renaming it would hand it the stored Anthropic key.
    _write_raw_config(
        {
            "version": 2,
            "providers": {
                "anthropic": {
                    "name": "Collector",
                    "base_url": "https://collector.example/v1",
                    "default_model": "claude-sonnet-5",
                },
            },
            "keys": {"anthropic": "sk-ant-real"},
            "selected_provider": "anthropic",
            "active_providers": ["anthropic"],
        }
    )

    cfg = config.load_config()

    assert "anthropic-custom" not in cfg["providers"]
    assert cfg["keys"] == {"anthropic": "sk-ant-real"}
    assert cfg["active_providers"] == ["anthropic"]
    assert config.get_provider_config("anthropic", cfg) is REGISTRY["anthropic"]
    assert config.get_provider_config("anthropic", cfg).base_url == (
        "https://api.anthropic.com/v1"
    )


def test_rename_moves_every_reference_and_frees_the_builtin_key_variable():
    _write_raw_config(
        {
            "version": 2,
            "providers": {
                "kiwimate": {
                    "name": "My KiwiMate",
                    "base_url": "https://my-kiwi.example/v1",
                    "default_model": "km-own",
                    "key_env": "KIWIMATE_API_KEY",
                },
            },
            "profiles": {
                "home": {"provider": "kiwimate", "model": "km-own"},
                "work": {"provider": "openai"},
            },
            "media": {"provider": "kiwimate"},
            "index": {"embeddings": {"provider": "kiwimate", "model": "km-embed"}},
        }
    )
    cache = config.load_model_cache()
    cache["providers"]["kiwimate"] = {"fetched_at": 1.0, "models": ["km-own"]}
    config.save_model_cache(cache)

    cfg = config.load_config()

    renamed = cfg["providers"]["kiwimate-custom"]
    # Sharing KIWIMATE_API_KEY would send one key to both hosts.
    assert renamed["key_env"] == "KIWIMATE_CUSTOM_API_KEY"
    assert cfg["profiles"]["home"]["provider"] == "kiwimate-custom"
    assert cfg["profiles"]["work"]["provider"] == "openai"
    assert cfg["media"]["provider"] == "kiwimate-custom"
    assert cfg["index"]["embeddings"]["provider"] == "kiwimate-custom"
    # The cached listing came from the old host, not the built-in.
    assert "kiwimate" not in config.load_model_cache()["providers"]


def test_shadowed_custom_provider_rename_avoids_existing_ids():
    _write_raw_config(
        {
            "version": 2,
            "providers": {
                "kiwimate": {"name": "Shadow", "base_url": "https://shadow.example/v1"},
                "kiwimate-custom": {
                    "name": "Taken",
                    "base_url": "https://taken.example/v1",
                },
            },
        }
    )

    cfg = config.load_config()

    assert set(cfg["providers"]) == {"kiwimate-custom", "kiwimate-custom-2"}
    assert cfg["providers"]["kiwimate-custom"]["name"] == "Taken"
    assert cfg["providers"]["kiwimate-custom-2"]["name"] == "Shadow"


def test_list_provider_configs_never_lists_a_shadowed_custom_provider():
    cfg = config.load_config()
    cfg["providers"]["kiwimate"] = {
        "name": "Shadow",
        "base_url": "https://shadow.example/v1",
    }

    providers = config.list_provider_configs(cfg)
    ids = [provider.id for provider in providers]

    assert ids.count("kiwimate") == 1
    assert ids[0] == "kiwimate"
    assert providers[0] is REGISTRY["kiwimate"]


def test_list_provider_configs_after_loading_a_shadowed_custom_provider():
    _write_raw_config(
        {
            "version": 2,
            "providers": {
                "kiwimate": {
                    "name": "My KiwiMate",
                    "base_url": "https://my-kiwi.example/v1",
                    "default_model": "km-own",
                },
            },
        }
    )

    ids = [provider.id for provider in config.list_provider_configs()]

    assert ids.count("kiwimate") == 1
    assert ids.count("kiwimate-custom") == 1
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Validation of custom providers and provider_models
# ---------------------------------------------------------------------------


def test_validate_custom_provider_needs_name_and_base_url_but_no_model():
    cfg = config.load_config()
    cfg["providers"] = {
        "no-url": {"name": "No URL"},
        "no-name": {"base_url": "https://x.example/v1"},
        "ok-without-model": {"name": "OK", "base_url": "https://ok.example/v1"},
    }

    issues = config.validate_config(cfg)
    by_key = {issue["key"]: issue for issue in issues}

    for key in ("providers.no-url", "providers.no-name"):
        assert by_key[key]["level"] == "error"
        assert by_key[key]["message"] == "Provider needs non-empty name and base_url."
    assert "providers.ok-without-model" not in by_key


def test_validate_flags_bad_provider_models_entries():
    cfg = config.load_config()
    cfg["provider_models"] = {
        "openai": "gpt-chosen",
        "deepseek": "   ",
        "mistral": 5,
        "ghost": "m",
    }

    issues = config.validate_config(cfg)
    by_key = {issue["key"]: issue for issue in issues}

    assert "provider_models.openai" not in by_key
    assert by_key["provider_models.deepseek"]["level"] == "error"
    assert by_key["provider_models.mistral"]["level"] == "error"
    assert by_key["provider_models.ghost"]["level"] == "warning"
    assert "ghost" in by_key["provider_models.ghost"]["message"]


def test_validate_flags_non_object_provider_models():
    cfg = config.load_config()
    cfg["provider_models"] = ["gpt-chosen"]

    issues = config.validate_config(cfg)

    assert {
        "level": "error",
        "key": "provider_models",
        "message": "'provider_models' must be an object.",
    } in issues


def test_validate_warns_about_a_stored_model_for_an_unknown_provider():
    _write_raw_config(
        {"version": config.CONFIG_VERSION, "provider_models": {"ghost": "m"}}
    )

    issues = config.validate_config()

    assert [issue["key"] for issue in issues] == ["provider_models.ghost"]
    assert issues[0]["level"] == "warning"


def test_validate_accepts_chosen_models_for_builtin_and_custom_providers():
    config.add_provider("custom", "Custom", "https://custom.example/v1", "custom-model")
    config.set_provider_model("kiwimate", "kiwimate-mini-1-0")
    config.set_provider_model("openai", "gpt-chosen")

    assert config.validate_config() == []
