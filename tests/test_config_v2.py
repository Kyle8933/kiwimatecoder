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
