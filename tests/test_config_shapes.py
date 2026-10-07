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

``budget``, ``sampling`` and ``profiles`` had the same flaw in their writers
(``set_budget``, ``set_sampling``, ``save_profile``, ``remove_profile``,
``rename_profile`` copied the stored value with ``dict(...)``), and
``get_sampling`` accepted values ``set_sampling`` would refuse. They are covered
at the end of this file.
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


# --- budget, sampling and profiles ---------------------------------------------

WRITTEN_SECTIONS = ("budget", "sampling", "profiles")


@pytest.mark.parametrize("shape", NOT_MAPS)
@pytest.mark.parametrize("section", WRITTEN_SECTIONS)
def test_a_section_that_is_not_a_map_is_read_as_empty(section, shape):
    _store(**{section: NOT_MAPS[shape], "output_style": "concise"})

    assert config.load_config()["output_style"] == "concise"
    assert config.get_budget() == {}
    assert config.get_sampling() == {}
    assert config.get_profiles() == {}
    assert any(i["key"] == section for i in config.validate_config())


@pytest.mark.parametrize("shape", NOT_MAPS)
def test_set_budget_replaces_a_budget_that_is_not_a_map(shape):
    _store(budget=NOT_MAPS[shape])

    assert config.set_budget(max_tokens=100) == {"max_tokens": 100}

    _store(budget=NOT_MAPS[shape])
    assert config.set_budget(max_cost_usd=1.5) == {"max_cost_usd": 1.5}
    assert config.load_config()["budget"] == {"max_cost_usd": 1.5}

    _store(budget=NOT_MAPS[shape])
    config.clear_budget()
    assert config.load_config()["budget"] == {}


@pytest.mark.parametrize("shape", NOT_MAPS)
def test_set_sampling_replaces_a_sampling_value_that_is_not_a_map(shape):
    _store(sampling=NOT_MAPS[shape])

    assert config.set_sampling({"temperature": 0.5}) == {"temperature": 0.5}

    _store(sampling=NOT_MAPS[shape])
    config.reset_sampling()
    assert config.load_config()["sampling"] == {}


@pytest.mark.parametrize("shape", NOT_MAPS)
def test_profiles_can_be_saved_removed_and_renamed_over_a_value_that_is_not_a_map(shape):
    _store(profiles=NOT_MAPS[shape])
    config.save_profile("work", {"model": "m"})
    assert list(config.get_profiles()) == ["work"]

    _store(profiles=NOT_MAPS[shape])
    assert config.remove_profile("work") is False
    assert config.rename_profile("work", "home") is False

    _store(profiles=NOT_MAPS[shape])
    config.save_profile("snapshot")  # captures the current settings
    assert "snapshot" in config.get_profiles()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("reasoning_effort", "[/x]"),
        ("reasoning_effort", ["low"]),
        ("reasoning_effort", "extreme"),
        ("temperature", 5),
        ("temperature", -1),
        ("temperature", "hot"),
        ("top_p", 1.5),
        ("max_tokens", 0),
        ("max_tokens", -5),
        ("max_tokens", [1]),
        ("max_tokens", "many"),
    ],
    ids=repr,
)
def test_a_sampling_value_set_sampling_would_refuse_is_ignored(key, value):
    other = {"temperature": 0.9} if key == "top_p" else {"top_p": 0.9}
    _store(sampling={key: value, **other})

    assert config.get_sampling() == other
    assert any(i["key"] == f"sampling.{key}" for i in config.validate_config())
    # ... and a profile can still be captured from the settings
    config.save_profile("snapshot")
    assert config.get_profile("snapshot")["sampling"] == other


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("reasoning_effort", " High ", "high"),
        ("temperature", "0.7", 0.7),
        ("temperature", 2, 2.0),
        ("top_p", 0, 0.0),
        ("max_tokens", 100.0, 100),
    ],
)
def test_usable_sampling_values_are_kept(key, value, expected):
    _store(sampling={key: value})

    assert config.get_sampling() == {key: expected}


BAD_PROFILE_ENTRIES = {
    "str": {"p": "x"},
    "int": {"p": 5},
    "list": {"p": ["x"]},
    "none": {"p": None},
    "inner-lists": {"p": {"budget": ["x"], "sampling": ["x"], "command_rules": "x"}},
    "inner-dicts": {"p": {"budget": "x", "sampling": "x", "always_allowed": {"a": 1}}},
}


@pytest.mark.parametrize("shape", BAD_PROFILE_ENTRIES)
def test_a_profile_that_does_not_validate_is_left_out_but_can_be_removed(shape):
    _store(profiles={**BAD_PROFILE_ENTRIES[shape], "ok": {"model": "m"}})

    # `None` is an empty profile, which is valid; the others do not validate
    assert sorted(config.get_profiles()) == (["ok", "p"] if shape == "none" else ["ok"])
    if shape != "none":
        with pytest.raises(ValueError):  # a clean error, not a TypeError
            config.apply_profile("p")

    assert config.remove_profile("p") is True
    assert list(config.load_config()["profiles"]) == ["ok"]


@pytest.mark.parametrize("shape", ["str", "int", "list", "inner-lists", "inner-dicts"])
def test_renaming_a_profile_that_does_not_validate_says_so(shape):
    _store(profiles=BAD_PROFILE_ENTRIES[shape])

    with pytest.raises(ValueError):
        config.rename_profile("p", "q")


# The commands that write these sections, run over a value that is not a map.
WRITING_COMMANDS = [
    (["config", "budget", "tokens", "100"], "budget", {"max_tokens": 100}),
    (["config", "budget", "cost", "1.5"], "budget", {"max_cost_usd": 1.5}),
    (["config", "budget", "clear"], "budget", {}),
    (["config", "sampling", "set", "temperature=0.5"], "sampling", {"temperature": 0.5}),
    (["config", "sampling", "reset"], "sampling", {}),
]


@pytest.mark.parametrize("shape", NOT_MAPS)
@pytest.mark.parametrize(("argv", "section", "expected"), WRITING_COMMANDS, ids=" ".join)
def test_the_commands_that_write_a_section_replace_a_value_that_is_not_a_map(
    argv, section, expected, shape, cli, monkeypatch  # noqa: F811 (imported fixture)
):
    _install(monkeypatch, "zzz")
    _store(**{section: NOT_MAPS[shape]})

    result = CliRunner().invoke(main.app, argv, env={"HOME": str(cli)})

    assert result.exception is None, result.exception
    assert result.exit_code == 0, result.output
    assert config.load_config()[section] == expected


@pytest.mark.parametrize("shape", NOT_MAPS)
def test_the_profile_commands_replace_a_value_that_is_not_a_map(shape, cli, monkeypatch):  # noqa: F811
    _install(monkeypatch, "zzz")
    runner = CliRunner()
    env = {"HOME": str(cli)}
    _store(profiles=NOT_MAPS[shape])

    save = runner.invoke(main.app, ["config", "profile", "save", "work"], env=env)
    rename = runner.invoke(main.app, ["config", "profile", "rename", "work", "home"], env=env)
    remove = runner.invoke(main.app, ["config", "profile", "remove", "home"], env=env)

    for result in (save, rename, remove):
        assert result.exception is None, result.exception
        assert result.exit_code == 0, result.output
    assert config.load_config()["profiles"] == {}


@pytest.mark.parametrize("section", WRITTEN_SECTIONS)
def test_every_command_that_shows_state_survives_these_sections_too(
    section, cli, monkeypatch, leaves, tmp_path  # noqa: F811 (imported fixtures)
):
    _install(monkeypatch, "zzz")
    runner = CliRunner()
    crashes = []
    for shape, value in {**NOT_MAPS, "entries": {"p": "x", "max_tokens": ["x"]}}.items():
        _store(**{section: value})
        for path, command in leaves:
            required = [p for p in command.params if _is_argument(p) and p.required]
            if _is_wipe(path) or (_string_params(command) and required):
                continue
            result = runner.invoke(main.app, list(path), env={"HOME": str(cli)})
            if result.exception is not None and not isinstance(result.exception, SystemExit):
                crashes.append(f"{shape}: {' '.join(path)} -> {type(result.exception).__name__}")
        session = Session(
            provider_id="openrouter", model="m", mode=PermissionMode.ASK, workspace_root=tmp_path
        )
        console = StrictConsole("zzz")
        for command in (*SHOW_COMMANDS, "/cost"):
            try:
                dispatch(command.replace("{V}", "x"), session, console)
            except Exception as exc:
                crashes.append(f"{shape}: {command} -> {type(exc).__name__}: {exc}")

    assert not crashes, "\n".join(crashes)
