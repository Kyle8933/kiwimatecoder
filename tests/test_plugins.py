from __future__ import annotations

import io
from collections.abc import Iterator

import pytest
from rich.console import Console

from kiwimatecoder import commands, config, plugins, tools
from kiwimatecoder.commands import CommandResult
from kiwimatecoder.events import EventBus


def _console():
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def _output(console) -> str:
    return console.file.getvalue()


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    config_dir = tmp_path / "cfg"
    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "CONFIG_FILE", config_dir / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", config_dir / "config")
    return config_dir


@pytest.fixture(autouse=True)
def restore_registries() -> Iterator[None]:
    """Snapshot/restore tools and commands so plugin tests stay isolated."""
    original_commands = dict(commands._COMMANDS)
    original_descriptions = dict(commands._COMMAND_DESCRIPTIONS)
    name_sources = {
        name: source for source, names in tools.TOOLS.sources().items() for name in names
    }
    original_tools = list(tools.TOOLS.all())
    yield
    plugins.unload_plugins()
    commands._COMMANDS.clear()
    commands._COMMANDS.update(original_commands)
    commands._COMMAND_DESCRIPTIONS.clear()
    commands._COMMAND_DESCRIPTIONS.update(original_descriptions)
    for name in list(tools.TOOLS):
        tools.TOOLS.unregister(name, force=True)
    for tool in original_tools:
        tools.TOOLS.register(tool, source=name_sources.get(tool.name, "builtin"))


def _write_plugin(directory, name: str, source: str):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.py"
    path.write_text(source)
    return path


GREETER_SOURCE = """
from kiwimatecoder.tools.base import FunctionTool, ToolResult


def _shout(args, session):
    return ToolResult(content=f"hello {args.get('who', 'world')}")


def _shout_command(arg, session, console):
    console.print(f"shouted {arg}")
    return "continue"


def register(api):
    api.register_tool(
        FunctionTool(
            name="shout",
            description="Shout a greeting.",
            parameters={
                "type": "object",
                "properties": {"who": {"type": "string"}},
            },
            func=_shout,
        )
    )
    api.register_command("shout", _shout_command, "Shout at someone.")
    api.subscribe("custom_event", lambda event: event.payload.get("value"))
"""

BREAKING_SOURCE = """
from kiwimatecoder.tools.base import FunctionTool, ToolResult


def register(api):
    api.register_tool(
        FunctionTool(
            name="bad_tool",
            description="partial registration",
            parameters={"type": "object", "properties": {}},
            func=lambda args, session: ToolResult(content="bad"),
        )
    )
    raise RuntimeError("boom")
"""

DUPLICATE_SOURCE = """
from kiwimatecoder.tools.base import FunctionTool, ToolResult


def register(api):
    api.register_tool(
        FunctionTool(
            name="read_file",
            description="shadow the builtin",
            parameters={"type": "object", "properties": {}},
            func=lambda args, session: ToolResult(content="shadow"),
        )
    )
"""

DUPLICATE_COMMAND_SOURCE = """
def _handler(arg, session, console):
    return "continue"


def register(api):
    api.register_command("help", _handler)
"""

KEYWORD_SOURCE = """
from kiwimatecoder.tools.base import ToolResult


def _echo(args, session):
    return ToolResult(content=str(args.get("text", "")))


def register(api):
    api.register_tool(
        name="echo",
        description="Echo text.",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        func=_echo,
    )
    api.log("loaded")
"""


def test_user_plugin_registers_tool_command_and_subscriber(session, isolate_config):
    _write_plugin(isolate_config / "plugins", "greeter", GREETER_SOURCE)
    console = _console()
    bus = EventBus()

    result = plugins.load_plugins(session.workspace_root, console=console, bus=bus)

    assert result.loaded == ["greeter"]
    assert result.failed == []
    assert result.skipped == []

    tool_result = tools.dispatch("shout", {"who": "kiwi"}, session)
    assert tool_result.ok
    assert tool_result.content == "hello kiwi"

    assert commands.dispatch("/shout there", session, console) == CommandResult.CONTINUE
    assert "shouted there" in _output(console)

    assert bus.emit("custom_event", value=42) == [42]


def test_plugin_keyword_tool_and_log(session, isolate_config):
    _write_plugin(isolate_config / "plugins", "kw", KEYWORD_SOURCE)
    console = _console()

    result = plugins.load_plugins(session.workspace_root, console=console, bus=EventBus())

    assert result.loaded == ["kw"]
    assert tools.dispatch("echo", {"text": "hi"}, session).content == "hi"
    assert "plugin kw: loaded" in _output(console)


def test_project_plugins_require_opt_in(session):
    _write_plugin(
        session.workspace_root / ".kiwimatecoder" / "plugins", "proj", GREETER_SOURCE
    )

    skipped = plugins.load_plugins(
        session.workspace_root, console=_console(), bus=EventBus()
    )

    assert skipped.skipped == ["proj"]
    assert skipped.loaded == []
    assert tools.get_tool("shout") is None

    config.set_plugins_config(allow_project=True)
    loaded = plugins.load_plugins(
        session.workspace_root, console=_console(), bus=EventBus()
    )

    assert loaded.loaded == ["proj"]
    assert tools.get_tool("shout") is not None


def test_disabled_plugin_is_skipped(session, isolate_config):
    _write_plugin(isolate_config / "plugins", "greeter", GREETER_SOURCE)
    config.set_plugins_config(disabled=["greeter"])

    result = plugins.load_plugins(
        session.workspace_root, console=_console(), bus=EventBus()
    )

    assert result.skipped == ["greeter"]
    assert result.loaded == []
    assert tools.get_tool("shout") is None


def test_failing_plugin_is_recorded_and_cleaned_up(session, isolate_config):
    _write_plugin(isolate_config / "plugins", "bad_plugin", BREAKING_SOURCE)
    _write_plugin(isolate_config / "plugins", "good_plugin", GREETER_SOURCE)

    result = plugins.load_plugins(
        session.workspace_root, console=_console(), bus=EventBus()
    )

    assert result.loaded == ["good_plugin"]
    assert len(result.failed) == 1
    name, reason = result.failed[0]
    assert name == "bad_plugin"
    assert "boom" in reason
    assert tools.get_tool("bad_tool") is None
    assert tools.get_tool("shout") is not None


def test_duplicate_tool_fails_and_builtin_survives(session, isolate_config):
    _write_plugin(isolate_config / "plugins", "dupe", DUPLICATE_SOURCE)

    result = plugins.load_plugins(
        session.workspace_root, console=_console(), bus=EventBus()
    )

    assert result.loaded == []
    assert len(result.failed) == 1
    assert result.failed[0][0] == "dupe"
    assert "already registered" in result.failed[0][1]
    builtin = tools.get_tool("read_file")
    assert builtin is not None
    assert builtin.description.startswith("Read the contents")


def test_duplicate_command_is_recorded_as_failed(session, isolate_config):
    _write_plugin(isolate_config / "plugins", "dupe_cmd", DUPLICATE_COMMAND_SOURCE)
    console = _console()

    result = plugins.load_plugins(session.workspace_root, console=console, bus=EventBus())

    assert result.loaded == []
    assert "already registered" in result.failed[0][1]
    assert commands.dispatch("/help", session, console) == CommandResult.CONTINUE
    assert "Leave the session" in _output(console)


def test_import_error_is_recorded(session, isolate_config):
    _write_plugin(isolate_config / "plugins", "broken", "import nope_xyz\n")

    result = plugins.load_plugins(
        session.workspace_root, console=_console(), bus=EventBus()
    )

    assert result.loaded == []
    assert result.failed[0][0] == "broken"
    assert "import failed" in result.failed[0][1]


def test_missing_register_is_recorded(session, isolate_config):
    _write_plugin(isolate_config / "plugins", "empty", "VALUE = 1\n")

    result = plugins.load_plugins(
        session.workspace_root, console=_console(), bus=EventBus()
    )

    assert result.loaded == []
    assert "no register" in result.failed[0][1]


def test_underscore_files_are_not_loaded(session, isolate_config):
    _write_plugin(isolate_config / "plugins", "_helper", "raise RuntimeError('nope')\n")

    result = plugins.load_plugins(
        session.workspace_root, console=_console(), bus=EventBus()
    )

    assert result.loaded == []
    assert result.failed == []


def test_unload_plugins_removes_registrations(session, isolate_config):
    _write_plugin(isolate_config / "plugins", "greeter", GREETER_SOURCE)
    bus = EventBus()
    plugins.load_plugins(session.workspace_root, console=_console(), bus=bus)

    plugins.unload_plugins()

    assert tools.get_tool("shout") is None
    assert not commands.has_command("shout")
    assert bus.emit("custom_event", value=1) == []
