"""Typed /config values must be echoed literally, never parsed as Rich markup.

The /config handlers print values the user typed (a verify command, a path, a
regex, a model id, a name...). Rich treats ``[...]`` in a string as markup, so
an unescaped ``tests/a.py::test_b[/tmp/x]`` raised ``MarkupError`` after the
value had already been saved, and ``pytest -k "[slow]"`` was echoed without
its ``[slow]``.

The detector below sits at ``Console.render_str``: every ``console.print``
string and every table cell is rendered through it. Any string that contains a
hostile value must parse and keep that value verbatim. Each hostile value is
fed through every value-taking typed command, then every show/list command is
run so values that were saved get displayed too.

What this does and does not prove:

* It sees everything rendered through ``render_str`` (``console.print``
  strings and table cells), plus the label of every ``console.status`` spinner
  and each of its updates, which Rich parses elsewhere.
* A few ``[red]{escape(str(exc))}[/red]`` sites guard setters whose messages
  are fixed text (browser/shell/web/vision timeouts, enable flags...). Escaping
  there is defensive and no input here can fail it; the sites whose messages do
  quote the input (a bad regex group name, an unknown profile or provider, a
  non-numeric budget) are exercised and were checked by removing each escape.
"""

import io
import json
import re
import shlex
import sys

import pytest
from rich.console import Console
from rich.errors import MarkupError
from rich.text import Text

from kiwimatecoder import catalog, config, i18n
from kiwimatecoder.commands import SelectionPrompt, dispatch
from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.providers import REGISTRY
from kiwimatecoder.session import Session

HOSTILE = [
    "[/x]",  # a closing tag with nothing open: raises MarkupError
    "[bold]x[/bold]",  # valid markup: the tags vanish from the echo
    "a[/b]c",
    "[slow]",  # an unknown tag: silently swallowed
    "C:\\[x]",  # a backslash before a bracket: the backslash is eaten
]


def path_safe(name: str) -> str:
    """``name`` as something that can be a file or directory name on this platform.

    Windows rejects ``< > : " | ? *`` in a name and reads a backslash as a
    separator, so a value such as ``C:\\[x]`` cannot be created there; elsewhere
    the value is used as it is.
    """
    return re.sub(r'[<>:"|?*\\]', "_", name) if sys.platform == "win32" else name


class StrictConsole(Console):
    """A console that records every string that does not survive rendering."""

    def __init__(self, hostile: str) -> None:
        super().__init__(file=io.StringIO(), force_terminal=False, width=200)
        self.hostile = hostile
        self.problems: list[str] = []

    def _check(self, text) -> None:
        if isinstance(text, str) and self.hostile in text:
            try:
                plain = Text.from_markup(text).plain
            except MarkupError as exc:
                self.problems.append(f"MarkupError ({exc}) in {text!r}")
            else:
                # every occurrence must survive, not just one of several
                if plain.count(self.hostile) < text.count(self.hostile):
                    self.problems.append(f"{self.hostile!r} lost: {text!r} -> {plain!r}")

    def render_str(self, text, *args, **kwargs):  # type: ignore[override]
        markup = kwargs.get("markup")
        if self._markup if markup is None else markup:
            self._check(text)
        return super().render_str(text, *args, **kwargs)

    def status(self, status, **kwargs):  # type: ignore[override]
        # A spinner parses its label as markup too, at creation and on every update,
        # but not through render_str.
        self._check(status)
        handle = super().status(status, **kwargs)
        update = handle.update

        def checked(new_status=None, **options):
            self._check(new_status)
            return update(new_status, **options)

        handle.update = checked  # type: ignore[method-assign,assignment]
        return handle


# Every value-taking typed command. ``{V}`` is the shell-quoted hostile value.
# Several are expected to be rejected (the value is invalid for that setting);
# the rejection message is echoed too, which is what makes them worth running.
SET_COMMANDS = [
    # providers and keys
    "/config provider add hp1 {V} http://localhost:1/v1 m1",
    "/config provider add hp2 Name http://localhost:1/v1 {V}",
    "/config provider add hp3 Name http://localhost:1/v1 m KEYENV {V}",
    "/config provider add hp4 Name http://localhost:1/v1 m {V}=1",
    "/config provider add {V} Name http://localhost:1/v1 m",
    "/config provider add hp5 Name http://localhost:1/v1 m {V}",
    "/config provider edit hp1 {V}",
    "/config provider edit hp1 {V}=x",
    "/config provider edit hp1 model={V}",
    "/config provider edit hp1 key_prefix={V}",
    "/config provider edit {V} name=x",
    "/config provider remove {V}",
    "/config provider use {V}",
    "/config key set {V} k",
    "/config key remove {V}",
    "/config key edit {V}",
    "/model {V}",
    "/provider {V}",
    "/mode {V}",
    # models
    "/config model set {V}",
    "/config model set m {V}",
    "/config model reset {V}",
    "/config models allow {V}",
    "/config models deny {V}",
    "/config mode set {V}",
    # approvals and rules
    "/config permissions remove {V}",
    "/config commands allow {V}",
    "/config commands deny {V}",
    "/config commands allow (?P<{V}>a)",  # re.error: bad character in group name '...'
    "/config commands deny (?P<{V}>a)",
    "/config commands remove allow {V}",
    "/config commands remove {V} x",
    "/config trust {V}",
    # verify, budget, subagents, sandbox
    "/config verify set {V}",
    "/config budget tokens {V}",
    "/config budget cost {V}",
    "/config subagents max-steps {V}",
    "/config subagents model {V}",
    "/config subagents enable {V}",
    "/config sandbox enable {V}",
    "/config sandbox network {V}",
    "/config sandbox add-path {V}",
    "/config sandbox remove-path {V}",
    # tools and execution
    "/config browser enable {V}",
    "/config browser headless {V}",
    "/config browser timeout {V}",
    "/config shell persistent {V}",
    "/config shell timeout {V}",
    "/config shell max-jobs {V}",
    "/config remote enable {V}",
    "/config remote host {V}",
    "/config remote user {V}",
    "/config remote port {V}",
    "/config remote identity {V}",
    "/config remote workspace {V}",
    "/config remote devcontainer {V}",
    "/config acp timeout {V}",
    "/config sampling set temperature={V}",
    "/config sampling set reasoning_effort={V}",
    "/config sampling set {V}",
    "/config sampling set {V}=1",
    # network and limits
    "/config web max-chars {V}",
    "/config web timeout {V}",
    "/config web allow-local {V}",
    "/config network proxy {V}",
    "/config network ca {V}",
    "/config network offline {V}",
    "/config vision max-bytes {V}",
    "/config vision max-images {V}",
    # media and telemetry
    "/config media enable {V}",
    "/config media model {V}",
    "/config media provider {V}",
    "/config media size {V}",
    "/config media video-model {V}",
    "/config media duration {V}",
    "/config media video-size {V}",
    "/config media output-dir {V}",
    "/config media {V}",
    "/config telemetry enable {V}",
    "/config telemetry level {V}",
    "/config telemetry log-file {V}",
    "/config telemetry max-bytes {V}",
    "/config telemetry {V}",
    # appearance
    "/config ui color {V}",
    "/config ui output {V}",
    "/config ui ascii {V}",
    "/config ui theme {V}",
    "/config ui locale {V}",
    "/config ui keybindings {V}",
    "/config ui notify {V}",
    "/config ui notify-after {V}",
    "/config ui spinner {V}",
    "/config ui {V}",
    "/config style set {V}",
    "/config prompt set {V}",
    "/config cache {V}",
    # profiles, team, unknown section
    "/config profile use {V}",  # before it is saved: "Unknown profile '...'"
    "/config profile save {V}",
    "/config profile show {V}",
    "/config profile use {V}",
    "/config profile remove {V}",
    "/config team set-policy {V}",
    "/config team enforce {V}",
    "/config {V}",
]

# Everything that displays saved state (so saved hostile values get shown).
SHOW_COMMANDS = [
    "/config",
    "/config providers",
    "/config key list",
    "/config model",
    "/config models",
    "/config mode",
    "/config permissions",
    "/config commands list",
    "/config trust",
    "/config verify",
    "/config budget",
    "/config subagents",
    "/config browser",
    "/config shell",
    "/config sandbox",
    "/config remote",
    "/config acp",
    "/config sampling",
    "/config web",
    "/config network",
    "/config vision",
    "/config media",
    "/config telemetry",
    "/config ui",
    "/config style",
    "/config prompt",
    "/config profile list",
    "/config profile show {V}",
    "/config team",
    "/config cache",
    "/config help",
]


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    for provider in REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    i18n.set_locale(i18n.DEFAULT_LOCALE)
    yield
    i18n.set_locale(i18n.DEFAULT_LOCALE)


def _run(command: str, session: Session, console: StrictConsole, crashes: list[str]) -> None:
    try:
        dispatch(command, session, console)
    except Exception as exc:  # a handler crashing is exactly what we look for
        crashes.append(f"{command} -> {type(exc).__name__}: {exc}")


@pytest.mark.parametrize("hostile", HOSTILE)
def test_typed_values_are_echoed_literally(hostile, tmp_path):
    session = Session(
        provider_id="openrouter",
        model="test-model",
        mode=PermissionMode.ASK,
        workspace_root=tmp_path,
    )
    console = StrictConsole(hostile)
    crashes: list[str] = []
    quoted = shlex.quote(hostile)

    # State that only exists from outside the typed commands.
    config.persist_always_allowed_tool(hostile)
    session.always_allowed.add(hostile)

    for command in SET_COMMANDS:
        _run(command.format(V=quoted), session, console, crashes)
    for command in SHOW_COMMANDS:
        _run(command.format(V=quoted), session, console, crashes)

    assert not crashes, "handlers crashed:\n" + "\n".join(crashes)
    assert not console.problems, "echoed without escaping:\n" + "\n".join(
        dict.fromkeys(console.problems)
    )


# Sections whose getters return the full set of settings, defaults included.
SECTIONS = (
    "acp", "browser", "budget", "index", "lsp", "media", "memory", "model_routing",
    "network", "plugins", "remote", "sampling", "sandbox", "shell", "subagents",
    "sync", "team", "telemetry", "ui", "vision", "web",
)


def seed_every_section() -> None:
    """Store a value for every setting, so that hand-editing has something to edit."""
    cfg = config.load_config()
    for section in SECTIONS:
        for name in (f"get_{section}", f"get_{section}_config"):
            getter = getattr(config, name, None)
            if callable(getter):
                effective = getter()
                if isinstance(effective, dict):
                    cfg[section] = json.loads(json.dumps(effective, default=list))
                break
    cfg["sampling"] = {
        "temperature": 0.5, "top_p": 0.9, "max_tokens": 100, "reasoning_effort": "low",
    }
    cfg["lsp"] = {
        "enabled": True,
        "servers": {"srv": {"command": "c", "args": ["a"], "extensions": [".x"]}},
    }
    cfg["mcp_servers"] = {"srv": {"command": "c", "args": ["a"], "env": {"K": "V"}}}
    cfg["hooks"] = {"pre_tool": ["a"], "post_tool": ["b"], "session_start": ["c"]}
    cfg["command_rules"] = {"allow": ["a"], "deny": ["b"]}
    cfg["tool_permissions"] = {"always_allow": ["run_bash"]}
    cfg["provider_models"] = {"openrouter": "m"}
    cfg["profiles"] = {"p": {"output_style": "concise", "default_mode": "ask", "model": "m"}}
    cfg.update(
        default_mode="ask", output_style="concise", system_prompt="be brief",
        verify_command="pytest", selected_provider="openrouter", selected_model="m",
    )
    config.save_config(cfg)


def hand_edit_config(hostile: str) -> None:
    """Rewrite the saved config as editing the file by hand could.

    Every string, in every section, becomes ``hostile``, and so does every name
    in a map of named entries (providers, profiles, language servers...). The
    typed commands check what they store; a hand-edited file or a project
    ``.kiwimatecoder.json`` is only checked by the getters that read it, so
    whatever a getter lets through is shown as is.
    """
    stored = json.loads(config.CONFIG_FILE.read_text(encoding="utf-8"))

    def rewrite(node):
        if isinstance(node, dict):
            named = bool(node) and all(isinstance(value, dict) for value in node.values())
            return {(hostile if named else key): rewrite(value) for key, value in node.items()}
        if isinstance(node, list):
            return [rewrite(item) for item in node]
        return hostile if isinstance(node, str) else node

    edited = {
        key: value if key == "version" else rewrite(value) for key, value in stored.items()
    }
    config.CONFIG_FILE.write_text(json.dumps(edited), encoding="utf-8")


@pytest.mark.parametrize("hostile", HOSTILE)
def test_a_hand_edited_config_is_shown_literally(hostile, tmp_path):
    session = Session(
        provider_id="openrouter",
        model="test-model",
        mode=PermissionMode.ASK,
        workspace_root=tmp_path,
    )
    console = StrictConsole(hostile)
    crashes: list[str] = []
    quoted = shlex.quote(hostile)
    for command in SET_COMMANDS:
        _run(command.format(V=quoted), session, console, crashes)
    crashes.clear()  # typed values are the other test's business
    console.problems.clear()

    seed_every_section()
    hand_edit_config(hostile)
    for command in SHOW_COMMANDS:
        _run(command.format(V=quoted), session, console, crashes)

    assert not crashes, "handlers crashed:\n" + "\n".join(crashes)
    assert not console.problems, "echoed without escaping:\n" + "\n".join(
        dict.fromkeys(console.problems)
    )


def test_path_safe_only_changes_what_windows_cannot_name(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert path_safe("C:\\[x]") == "C:\\[x]"

    monkeypatch.setattr(sys, "platform", "win32")
    assert path_safe("C:\\[x]") == "C__[x]"
    assert path_safe("a[/b]c") == "a[/b]c"  # "/" is a separator there too
    assert path_safe("[bold]x[/bold]") == "[bold]x[/bold]"


def test_the_detector_sees_an_unescaped_value():
    # Guard against the test above passing because the detector is blind.
    console = StrictConsole("[/x]")

    with pytest.raises(MarkupError):
        console.print("[green]saved:[/green] [/x]")
    assert console.problems

    console = StrictConsole("[slow]")
    console.print("[green]saved:[/green] [slow]")
    assert console.problems

    console = StrictConsole("[/x]")
    from rich.markup import escape

    console.print(f"[green]saved:[/green] {escape('[/x]')}")
    assert console.problems == []
    assert "[/x]" in console.file.getvalue()

    # one value escaped and another not: the second one is lost, and that counts
    console = StrictConsole("[slow]")
    console.print(f"{escape('[slow]')} then [slow]")
    assert console.problems
    console = StrictConsole("[slow]")
    console.print(f"{escape('[slow]')} then {escape('[slow]')}")
    assert console.problems == []


class Walk:
    """Open one menu section, take its n-th entry, then keep taking first entries.

    Confirmations answer "no" so saved state survives for later walks, and the
    walk stops after a few prompts so looping sections terminate.
    """

    def __init__(self, section: str, index: int, limit: int = 8) -> None:
        self.section, self.index, self.limit, self.calls = section, index, limit, 0

    def __call__(self, prompt: SelectionPrompt) -> str | None:
        self.calls += 1
        if self.calls > self.limit:
            return None
        if self.calls == 1:
            return self.section
        values = [option.value for option in prompt.options]
        if prompt.title == "Are you sure?":
            return "no"
        return values[min(self.index if self.calls == 2 else 0, len(values) - 1)]


def _menu_sections(session: Session) -> list[str]:
    seen: list[SelectionPrompt] = []

    def probe(prompt: SelectionPrompt) -> None:
        seen.append(prompt)

    dispatch("/config", session, StrictConsole("unused"), selector=probe)
    return [option.value for option in seen[0].options]


@pytest.mark.parametrize("hostile", HOSTILE)
def test_menu_walks_echo_hostile_values_literally(hostile, tmp_path, monkeypatch):
    # A provider that lists a hostile model id, so the model pickers, catalog
    # reports and tables all carry one.
    def fake_fetch(provider, api_key=None, **kwargs):
        return [catalog.RemoteModel(hostile, 2.0), catalog.RemoteModel("ok-model", 1.0)]

    monkeypatch.setattr(config.catalog, "fetch_models", fake_fetch)
    session = Session(
        provider_id="openrouter",
        model=hostile,
        mode=PermissionMode.ASK,
        workspace_root=tmp_path,
    )
    console = StrictConsole(hostile)
    crashes: list[str] = []
    quoted = shlex.quote(hostile)
    config.persist_always_allowed_tool(hostile)
    session.always_allowed.add(hostile)
    for command in SET_COMMANDS:
        _run(command.format(V=quoted), session, console, crashes)

    typed = lambda _message: hostile  # noqa: E731 - every typed prompt gets the value
    for section in _menu_sections(session):
        for index in range(10):
            try:
                dispatch(
                    "/config",
                    session,
                    console,
                    selector=Walk(section, index),
                    prompt_input=typed,
                )
            except Exception as exc:
                crashes.append(f"menu {section}[{index}] -> {type(exc).__name__}: {exc}")

    assert not crashes, "menu crashed:\n" + "\n".join(dict.fromkeys(crashes))
    assert not console.problems, "echoed without escaping:\n" + "\n".join(
        dict.fromkeys(console.problems)
    )


def _hostile_provider_session(tmp_path, hostile):
    config.add_provider("hp", hostile, "http://localhost:1/v1", "m1")
    config.set_selected_provider("hp")
    return Session(
        provider_id="hp", model="m1", mode=PermissionMode.ASK, workspace_root=tmp_path
    )


@pytest.mark.parametrize("hostile", HOSTILE)
def test_catalog_refresh_with_a_hostile_provider_name_and_model_ids(
    hostile, tmp_path, monkeypatch
):
    # `console.status(...)` parses its text as markup outside render_str, so the
    # spinner label needs its own check: it used to raise MarkupError here.
    def fake_fetch(provider, api_key=None, **kwargs):
        return [catalog.RemoteModel(hostile, 2.0), catalog.RemoteModel("ok-model", 1.0)]

    monkeypatch.setattr(config.catalog, "fetch_models", fake_fetch)
    session = _hostile_provider_session(tmp_path, hostile)
    console = StrictConsole(hostile)
    crashes: list[str] = []

    for command in (
        "/config models refresh",
        "/config models",
        "/model search ok",
        f"/model search {shlex.quote(hostile)}",  # the spinner label embeds the query
    ):
        _run(command, session, console, crashes)

    assert not crashes, "\n".join(crashes)
    assert not console.problems, "\n".join(dict.fromkeys(console.problems))
    assert hostile in console.file.getvalue()


@pytest.mark.parametrize("hostile", HOSTILE)
def test_empty_picker_message_keeps_a_hostile_provider_id(hostile, tmp_path, monkeypatch):
    # The message embeds the provider id; it is printed by /model, /provider and
    # the /config model editor.
    from kiwimatecoder import commands

    hostile = hostile.lower()  # provider ids are stored lower-case
    config.add_provider(hostile, "Name", "http://localhost:1/v1", "m1")
    session = Session(
        provider_id=hostile,
        model="m1",
        mode=PermissionMode.ASK,
        workspace_root=tmp_path,
    )

    def no_models(*_args, **_kwargs):
        return SelectionPrompt(
            title="t",
            text="t",
            options=(),
            empty_message=f"No models are visible for {hostile}. Use /model <name>.",
        )

    monkeypatch.setattr(commands, "model_selection_prompt", no_models)
    console = StrictConsole(hostile)
    crashes: list[str] = []
    answers = iter(["model", "choose"])

    try:
        dispatch("/config", session, console, selector=lambda _p: next(answers, None))
        dispatch("/model", session, console, selector=lambda _p: None)
    except Exception as exc:
        crashes.append(f"{type(exc).__name__}: {exc}")

    assert not crashes, "\n".join(crashes)
    assert not console.problems, "\n".join(console.problems)
    assert console.file.getvalue().count(hostile) >= 2
