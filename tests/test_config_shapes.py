"""A config section of the wrong type must not stop the program.

``keys``, ``providers`` and ``model_filters`` are maps that the rest of the
module indexes directly. ``load_config`` used to copy them with ``dict(...)``, so
a list, a string or a number there (a hand-edited file, a project
``.kiwimatecoder.json``) raised ``ValueError`` or ``TypeError`` and every command
failed, not just the ones that use the section. An entry of the wrong type
(``"keys": {"openrouter": 5}``) crashed ``describe_key``, ``get_model_filter`` and
``update_provider``.

Such a value now counts as empty, the rest of the config still loads, and
``validate_config`` still reports it.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from kiwimatecoder import config, main, team
from kiwimatecoder.commands import dispatch
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.providers import REGISTRY, UnknownProviderError
from kiwimatecoder.session import Session
from tests.test_cli_markup import (  # noqa: F401  (fixtures and helpers)
    _install,
    _is_argument,
    _is_wipe,
    _string_params,
    cli,
    leaves,
    root,
)
from tests.test_config_markup import SHOW_COMMANDS, StrictConsole

SECTIONS = ("keys", "providers", "model_filters")
NOT_MAPS = {
    "list": ["x"],
    "str": "x",
    "int": 5,
    "true": True,
    "pairs": [["a", "b"]],  # dict() used to turn this into {"a": "b"}
}
# Right container, wrong entries.
BAD_ENTRIES = {
    "str": {"openrouter": "x", "a": "x"},
    "int": {"openrouter": 5, "a": 5},
    "list": {"openrouter": ["x"], "a": ["x"]},
    "none": {"openrouter": None, "a": None},
}


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for provider in REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)
    team.clear_policy_cache()
    yield
    team.clear_policy_cache()


def _store(**sections) -> None:
    config.CONFIG_FILE.write_text(
        json.dumps({"version": config.CONFIG_VERSION, **sections}), encoding="utf-8"
    )


@pytest.mark.parametrize("shape", NOT_MAPS)
@pytest.mark.parametrize("section", SECTIONS)
def test_a_section_that_is_not_a_map_counts_as_empty(section, shape):
    _store(**{section: NOT_MAPS[shape], "output_style": "concise"})

    cfg = config.load_config()

    assert cfg[section] == {}
    assert cfg["output_style"] == "concise"  # the rest of the file is still read


@pytest.mark.parametrize("shape", NOT_MAPS)
@pytest.mark.parametrize("section", SECTIONS)
def test_the_config_api_works_over_a_section_that_is_not_a_map(section, shape):
    _store(**{section: NOT_MAPS[shape]})

    assert config.get_key("openrouter") is None
    assert config.describe_key("openrouter") == "missing"
    assert config.get_model_filter("openrouter") == {"mode": "all", "models": []}
    assert [p.id for p in config.list_provider_configs()] == [p.id for p in REGISTRY.values()]
    with pytest.raises(UnknownProviderError):
        config.get_provider_config("nope")


@pytest.mark.parametrize("shape", NOT_MAPS)
@pytest.mark.parametrize("section", SECTIONS)
def test_writers_replace_a_section_that_is_not_a_map(section, shape):
    # set_key, add_provider and set_model_filter used to raise.
    def store_again():
        _store(**{section: NOT_MAPS[shape]})

    store_again()
    config.set_key("openrouter", "sk-test")
    assert config.get_key("openrouter") == "sk-test"

    store_again()
    config.add_provider("mine", "Mine", "https://api.example.com/v1", "m")
    assert config.get_provider_config("mine").name == "Mine"

    store_again()
    config.set_model_filter("openrouter", "allow", ["m1"])
    assert config.get_model_filter("openrouter") == {"mode": "allow", "models": ["m1"]}

    store_again()
    assert config.remove_key("openrouter") is False
    assert isinstance(json.loads(config.CONFIG_FILE.read_text())[section], dict)


@pytest.mark.parametrize("shape", NOT_MAPS)
@pytest.mark.parametrize("section", SECTIONS)
def test_validate_config_still_reports_a_section_that_is_not_a_map(section, shape):
    _store(**{section: NOT_MAPS[shape]})

    issues = config.validate_config()

    assert any(i["key"] == section and i["level"] == "error" for i in issues), issues


def test_validate_config_reports_nothing_for_a_well_formed_file():
    _store(keys={"openrouter": "sk"}, providers={}, model_filters={})

    assert config.validate_config() == []


@pytest.mark.parametrize("section", ("providers", "model_filters"))
def test_a_project_config_cannot_replace_a_map_with_a_list(section, tmp_path, monkeypatch):
    config.add_provider("mine", "Mine", "https://api.example.com/v1", "m")
    config.set_model_filter("mine", "allow", ["m1"])
    project = tmp_path / "project.json"
    project.write_text(json.dumps({section: ["x"]}), encoding="utf-8")
    monkeypatch.setenv(config.PROJECT_CONFIG_ENV, str(project))

    cfg = config.load_config()

    assert cfg[section]  # the global value is kept
    assert config.get_provider_config("mine").name == "Mine"
    assert any(i["key"] == section for i in config.validate_config())


@pytest.mark.parametrize("section", SECTIONS)
def test_a_project_config_that_is_not_a_map_does_not_break_loading(section, tmp_path, monkeypatch):
    project = tmp_path / "project.json"
    project.write_text(json.dumps({section: "x"}), encoding="utf-8")
    monkeypatch.setenv(config.PROJECT_CONFIG_ENV, str(project))

    assert config.load_config()[section] == {}


def test_a_team_policy_that_is_not_a_map_does_not_break_loading(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"model_filters": ["x"]}), encoding="utf-8")
    _store(team={"policy_path": str(policy), "enforce": True})

    assert config.load_config()["model_filters"] == {}
    assert config.get_model_filter("openrouter") == {"mode": "all", "models": []}


# --- entries of the wrong type ------------------------------------------------


@pytest.mark.parametrize("stored", [5, ["x"], None, {"k": "v"}, True], ids=repr)
def test_a_stored_key_that_is_not_a_string_counts_as_missing(stored):
    _store(keys={"openrouter": stored})

    assert config.get_key("openrouter") is None
    assert config.key_source("openrouter")["origin"] == "missing"
    assert config.describe_key("openrouter") == "missing"
    assert any(i["key"] == "keys.openrouter" for i in config.validate_config())


@pytest.mark.parametrize("shape", BAD_ENTRIES)
def test_a_model_filter_that_is_not_a_map_means_no_filter(shape):
    _store(model_filters=BAD_ENTRIES[shape])

    assert config.get_model_filter("openrouter") == {"mode": "all", "models": []}


@pytest.mark.parametrize("models", ["abc", 5, {"a": 1}, None])
def test_a_model_filter_whose_models_are_not_a_list_lists_none(models):
    _store(model_filters={"openrouter": {"mode": "allow", "models": models}})

    assert config.get_model_filter("openrouter") == {"mode": "allow", "models": []}


@pytest.mark.parametrize("shape", BAD_ENTRIES)
def test_a_provider_entry_that_is_not_a_map_is_unknown_but_can_be_removed(shape):
    _store(providers=BAD_ENTRIES[shape])

    assert "a" not in [p.id for p in config.list_provider_configs()]
    with pytest.raises(UnknownProviderError):
        config.get_provider_config("a")
    with pytest.raises(ValueError, match="Unknown custom provider"):
        config.update_provider("a", name="Z")  # used to raise TypeError / ValueError

    config.remove_provider("a")  # the bad entry can be cleaned up

    assert "a" not in json.loads(config.CONFIG_FILE.read_text())["providers"]


# --- commands -------------------------------------------------------------------

CLI_SHAPES = {
    **{f"not-a-map-{name}": value for name, value in NOT_MAPS.items() if name != "pairs"},
    **{f"entries-{name}": value for name, value in BAD_ENTRIES.items()},
}


@pytest.mark.parametrize("shape", CLI_SHAPES)
@pytest.mark.parametrize("section", SECTIONS)
def test_the_commands_that_show_state_survive_a_section_of_the_wrong_type(
    section, shape, cli, monkeypatch, leaves, tmp_path  # noqa: F811 (imported fixtures)
):
    _install(monkeypatch, "zzz")
    runner = CliRunner()
    crashes = []
    _store(**{section: CLI_SHAPES[shape]})

    for path, command in leaves:
        required = [p for p in command.params if _is_argument(p) and p.required]
        if _is_wipe(path) or (_string_params(command) and required):
            continue
        result = runner.invoke(main.app, list(path), env={"HOME": str(cli)})
        if result.exception is not None and not isinstance(result.exception, SystemExit):
            crashes.append(
                f"{' '.join(path)} -> {type(result.exception).__name__}: {result.exception}"
            )
    session = Session(
        provider_id="openrouter", model="m", mode=PermissionMode.ASK, workspace_root=tmp_path
    )
    console = StrictConsole("zzz")
    for command in (*SHOW_COMMANDS, "/provider", "/model", "/mode", "/cost"):
        try:
            dispatch(command.replace("{V}", "x"), session, console)
        except Exception as exc:
            crashes.append(f"{command} -> {type(exc).__name__}: {exc}")

    assert not crashes, "\n".join(crashes)
