"""The shell CLI echoes typed values literally, never as Rich markup.

Same bug class as the REPL's /config handlers (see test_config_markup.py): an
unescaped ``tests/a.py::test_b[/tmp/x]`` raised ``MarkupError`` after the value
had been saved, and ``[bold]x[/bold]`` lost its brackets. The CLI had about 75
such prints, so every command that takes a string is driven here with each
hostile value, then every show/list command runs so saved values are displayed.

The detector is ``StrictConsole`` from test_config_markup: it sits at
``Console.render_str``, which every print string and table cell goes through.
"""

from __future__ import annotations

import collections
import json
import os
import sys

import pytest
import typer
import typer.testing
from rich.text import Text
from typer.testing import CliRunner

from kiwimatecoder import config, i18n, main, ui
from kiwimatecoder.providers import REGISTRY
from tests.test_config_markup import HOSTILE, StrictConsole, hand_edit_config, seed_every_section

# Anything that launches the agent, needs the network, or spawns processes.
DENY_PREFIX = (
    "acp", "ask", "setup", "update", "doctor", "eval run", "jobs run",
    "jobs schedule", "jobs tick", "sync", "config models refresh", "share",
    "config index clear",
)
# A sensible value for each parameter name, so a command reaches the line that
# echoes the one parameter under test instead of failing on the others first.
DEFAULT_BY_NAME = {
    "state": "on", "mode": "on", "provider": "openrouter", "value": "1",
    "seconds": "5", "kind": "allow", "size": "1024x1024", "level": "error",
    "style": "concise", "path": "/tmp/x", "values": "temperature=0.5",
    "models": "m1", "pattern": "^x$", "name": "x", "key": "sk-x",
    "tool": "run_bash", "base_url": "http://localhost:1/v1",
}
# Run first, so values that need existing state can persist (an embedding model
# is only shown once an embedding provider is set, and `x` is not a provider).
SETUP = [["config", "index", "embed-provider", "openrouter"]]
OVERRIDE = {"config budget": {"action": "cost"}, "config verify": {"action": "set"}}


def _walk(command, path):
    if hasattr(command, "commands"):  # a group (typer ships its own click)
        for name, sub in sorted(command.commands.items()):
            yield from _walk(sub, [*path, name])
    else:
        yield path, command


def _is_argument(param) -> bool:
    return type(param).__name__ == "TyperArgument"


def _string_params(command):
    return [
        p for p in command.params
        if type(p.type).__name__ in ("StringParamType", "Path") and p.name != "help"
    ]


def _argv(path, command, target, value, serial=0):
    overrides = OVERRIDE.get(" ".join(path), {})
    argv = list(path)
    for param in command.params:
        if param.name == "help":
            continue
        default = overrides.get(param.name, DEFAULT_BY_NAME.get(param.name, "x"))
        if param.name == "provider_id":  # unique, so each run adds a provider
            default = f"p{serial}"
        text = value if param is target else default
        if _is_argument(param):
            argv.append(text)
        elif param is target:
            argv += [f"--{param.name.replace('_', '-')}", text]
    return argv


@pytest.fixture(scope="module")
def root():
    return typer.main.get_command(main.app)


@pytest.fixture(scope="module")
def leaves(root):
    return [
        (path, command)
        for path, command in _walk(root, [])
        if path and not any(" ".join(path).startswith(d) for d in DENY_PREFIX)
    ]


@pytest.fixture
def cli(tmp_path, monkeypatch, root):
    """Isolated config and cwd, a cached command tree, and a console we control."""
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for provider in REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)
    monkeypatch.chdir(tmp_path)
    # Typer would rebuild all ~150 commands on every invoke; build them once.
    monkeypatch.setattr(typer.testing, "_get_command", lambda app: root)
    i18n.set_locale(i18n.DEFAULT_LOCALE)
    yield tmp_path
    i18n.set_locale(i18n.DEFAULT_LOCALE)


def _install(monkeypatch, hostile) -> StrictConsole:
    strict = StrictConsole(hostile)
    monkeypatch.setattr(main, "console", strict)
    # The app callback rebuilds main.console from this factory on every command.
    monkeypatch.setattr(ui, "make_console", lambda **kwargs: strict)
    return strict


def test_the_harness_reaches_the_cli_console(cli, monkeypatch):
    # Control: the app callback rebuilds the console on every command, so a hook
    # that did not take would capture nothing and "no problems" would mean nothing.
    strict = _install(monkeypatch, "[bold]x[/bold]")

    result = CliRunner().invoke(
        main.app,
        ["config", "verify", "set", "[bold]x[/bold]"],
        env={"HOME": str(cli)},
    )

    assert result.exit_code == 0
    assert "[bold]x[/bold]" in strict.file.getvalue()  # printed, brackets and all
    assert result.output == ""  # nothing went to a console we are not watching


def test_the_sweep_covers_the_commands_that_took_typed_text(leaves):
    driven = {" ".join(path) for path, _ in leaves}

    assert len(driven) > 100
    assert {
        "config verify", "config budget", "config sandbox add-path",
        "config media model", "config profile save", "config commands allow",
        "config provider add", "config key set", "config network proxy",
        "config mcp add", "config model set", "jobs show",
    } <= driven


# Leaves that wipe what the others show: run last, or `config prompt show` would
# only ever see an empty prompt.
WIPES = {"clear", "reset", "remove", "clear-paths", "unset", "disable", "forget"}


def _is_wipe(path) -> bool:
    return path[-1] in WIPES


@pytest.mark.parametrize("hostile", HOSTILE)
def test_cli_echoes_typed_values_literally(hostile, cli, monkeypatch, leaves):
    strict = _install(monkeypatch, hostile)
    runner = CliRunner()
    crashes: collections.Counter[str] = collections.Counter()
    problems: dict[str, str] = {}

    def run(argv):
        strict.problems.clear()
        result = runner.invoke(main.app, argv, env={"HOME": str(cli)})
        label = " ".join("<H>" if part == hostile else part for part in argv)
        if result.exception is not None and not isinstance(result.exception, SystemExit):
            crashes[f"{label} -> {type(result.exception).__name__}"] += 1
        for problem in strict.problems:
            problems.setdefault(label, problem.replace(hostile, "<H>")[:140])

    def show(*, wipes: bool) -> None:
        for path, command in leaves:
            required = [p for p in command.params if _is_argument(p) and p.required]
            if _is_wipe(path) == wipes and (not _string_params(command) or not required):
                run(list(path))

    for argv in SETUP:
        run(argv)
    serial = 0
    for path, command in leaves:
        for target in _string_params(command):
            serial += 1
            run(_argv(path, command, target, hostile, serial))
    # Then everything that displays saved state, wiping commands last ...
    show(wipes=False)
    # ... and again after the file is edited by hand, which no typed command
    # checks: whatever the getters let through is displayed as is.
    seed_every_section()
    hand_edit_config(hostile)
    show(wipes=False)
    show(wipes=True)

    assert not crashes, "commands crashed:\n" + "\n".join(sorted(crashes))
    assert not problems, "echoed without escaping:\n" + "\n".join(
        f"  {label}: {text}" for label, text in sorted(problems.items())
    )


# Commands the sweep above skips (they normally launch the agent or use the
# network) but that fail fast on a bad value, before doing either: an unknown
# provider is rejected before any request, and `share` only reads files.
EARLY_FAILURES = [
    ["config", "commands", "allow", "(?P<{H}>a)"],  # re.error quotes the group name
    ["config", "commands", "deny", "(?P<{H}>a)"],
    ["config", "profile", "use", "{H}"],  # before it exists: "Unknown profile"
    ["config", "profile", "rename", "{H}", "newname"],
    ["config", "profile", "show", "{H}"],
    ["ask", "hello", "--provider", "{H}"],
    ["setup", "--provider", "{H}"],
    ["eval", "run", "--provider", "{H}"],
    ["share", "create", "{H}"],
    ["--profile", "{H}", "version"],
    ["--resume", "{H}", "version"],
]


@pytest.mark.parametrize("hostile", HOSTILE)
def test_cli_early_failures_echo_typed_values_literally(hostile, cli, monkeypatch):
    strict = _install(monkeypatch, hostile)
    runner = CliRunner()
    bad: list[str] = []

    for template in EARLY_FAILURES:
        argv = [part.replace("{H}", hostile) for part in template]
        strict.problems.clear()
        result = runner.invoke(main.app, argv, env={"HOME": str(cli)}, input="")
        label = " ".join(template)
        if result.exception is not None and not isinstance(result.exception, SystemExit):
            bad.append(f"{label}: crashed with {type(result.exception).__name__}")
        bad += [f"{label}: {p.replace(hostile, '<H>')[:100]}" for p in strict.problems]

    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("hostile", HOSTILE)
def test_the_quick_start_panel_shows_the_provider_and_key_variable_literally(
    hostile, cli, monkeypatch
):
    # Printed on every bare launch when the chosen provider has no key.
    strict = _install(monkeypatch, hostile)
    config.add_provider("hp", hostile, "https://api.example.com/v1", "m")  # a cloud API needs a key
    config.set_selected_provider("hp")
    stored = json.loads(config.CONFIG_FILE.read_text(encoding="utf-8"))
    stored["providers"]["hp"]["key_env"] = hostile  # as if edited by hand
    config.CONFIG_FILE.write_text(json.dumps(stored), encoding="utf-8")

    result = CliRunner().invoke(main.app, [], env={"HOME": str(cli)})

    output = strict.file.getvalue()
    assert result.exit_code == 2, result.output  # no terminal to start the REPL in
    assert not strict.problems, strict.problems
    assert f"No API key set for {hostile}." in output
    assert f"export {hostile}." in output


@pytest.mark.parametrize("hostile", HOSTILE)
def test_the_saved_system_prompt_is_shown_as_typed(hostile, cli, monkeypatch):
    strict = _install(monkeypatch, hostile)
    runner = CliRunner()

    runner.invoke(main.app, ["config", "prompt", "set", hostile], env={"HOME": str(cli)})
    strict.file.truncate(0)
    strict.file.seek(0)
    result = runner.invoke(main.app, ["config", "prompt", "show"], env={"HOME": str(cli)})

    assert result.exit_code == 0
    assert not strict.problems, strict.problems
    assert strict.file.getvalue().strip() == hostile


@pytest.mark.parametrize("hostile", HOSTILE)
def test_the_prompt_to_run_setup_keeps_its_y_n_hint(hostile, cli, monkeypatch):
    # "[y/N]" was read as a style tag, so the question showed as "Run setup now? : ".
    strict = _install(monkeypatch, hostile)
    answers = []
    monkeypatch.setattr(
        strict, "input", lambda prompt="", **_kw: answers.append(prompt) or "n"
    )

    assert main._prompt_yes_no("Run setup now?") is False

    rendered = Text.from_markup(answers[0]).plain
    assert rendered == "Run setup now? [y/N]: "


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="a program is only found on Windows with an extension such as .exe",
)
@pytest.mark.parametrize("hostile", HOSTILE)
def test_lsp_show_lists_servers_and_overrides_literally(hostile, cli, monkeypatch):
    # A server whose command can be run is listed as available, with its command.
    strict = _install(monkeypatch, hostile)
    program = cli / hostile  # found on PATH, or relative to the cwd for a value with "/"
    program.parent.mkdir(parents=True, exist_ok=True)
    program.write_text("#!/bin/sh\n")
    program.chmod(0o755)
    monkeypatch.setenv("PATH", f"{cli}{os.pathsep}{os.environ['PATH']}")
    stored = config.load_config()
    stored["lsp"] = {
        "enabled": True,
        "servers": {hostile: {"command": hostile, "extensions": [".zz"]}},
    }
    config.save_config(stored)

    result = CliRunner().invoke(main.app, ["config", "lsp", "show"], env={"HOME": str(cli)})

    output = strict.file.getvalue()
    assert result.exit_code == 0
    assert not strict.problems, strict.problems
    assert f"Available servers: {hostile} ({hostile})" in output
    assert f"Overrides: {hostile}" in output
