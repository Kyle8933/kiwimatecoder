"""External plugins: user and project Python files extending the CLI.

A plugin is a ``*.py`` file with a ``register(api)`` function. It can add
tools, slash commands, and event subscribers without touching core:

    from kiwimatecoder.tools.base import FunctionTool, ToolResult

    def register(api):
        api.register_tool(FunctionTool(...))
        api.register_command("hello", hello_handler, "Say hello.")
        api.subscribe("post_tool", on_tool)

Plugins load from ``~/.kiwimatecoder/plugins/*.py`` always. Project plugins in
``<workspace>/.kiwimatecoder/plugins/*.py`` are disabled by default — cloning
a repository must never execute code just because the CLI started there — and
are enabled with ``"plugins": {"allow_project": true}``. A failed plugin is
recorded and skipped so it can never prevent startup.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from rich.console import Console

from kiwimatecoder import commands, config, events, tools
from kiwimatecoder.events import EventBus, Subscriber
from kiwimatecoder.session import Session
from kiwimatecoder.tools.base import FunctionTool, ToolResult
from kiwimatecoder.tools.registry import PLUGIN_SOURCE

PLUGINS_DIR_NAME = "plugins"
PROJECT_PLUGINS_DIR = Path(".kiwimatecoder") / "plugins"

ToolFactory = Callable[[], FunctionTool]


@dataclass
class PluginLoadResult:
    """Outcome of one :func:`load_plugins` pass."""

    loaded: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


class PluginAPI:
    """The extension surface handed to each plugin's ``register(api)``.

    Registrations are tracked so a partially loaded plugin can be cleaned up
    and so :func:`unload_plugins` can reverse everything on reload.
    """

    def __init__(
        self,
        name: str,
        console: Console | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.name = name
        self._console = console if console is not None else Console()
        self._bus = bus if bus is not None else events.BUS
        self._tools: list[str] = []
        self._commands: list[str] = []
        self._unsubscribes: list[Callable[[], None]] = []

    def register_tool(
        self,
        tool: FunctionTool | ToolFactory | None = None,
        *,
        name: str | None = None,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        func: Callable[[dict[str, Any], Any], ToolResult] | None = None,
        writes: bool = False,
        runs: bool = False,
    ) -> FunctionTool:
        """Register a tool (an instance, a zero-arg factory, or keyword args)."""
        resolved: Any = tool
        if resolved is not None and not isinstance(resolved, FunctionTool):
            if not callable(resolved):
                raise ValueError(
                    "register_tool expects a FunctionTool, a factory, or keyword args."
                )
            resolved = resolved()
        if resolved is None:
            if not name or parameters is None or func is None:
                raise ValueError(
                    "register_tool needs a FunctionTool or name, parameters, and func."
                )
            resolved = FunctionTool(
                name=name,
                description=description,
                parameters=parameters,
                func=func,
                writes=writes,
                runs=runs,
            )
        if not isinstance(resolved, FunctionTool):
            raise ValueError("register_tool expects a FunctionTool instance.")
        registered = tools.register_tool(resolved, source=PLUGIN_SOURCE)
        self._tools.append(registered.name)
        return registered

    def register_command(
        self,
        name: str,
        handler: Callable[[str, Session, Console], str],
        description: str = "",
    ) -> str:
        """Register a slash command with the given handler."""
        key = commands.register_command(name, handler, description)
        self._commands.append(key)
        return key

    def subscribe(self, event: str, callback: Subscriber) -> Callable[[], None]:
        """Subscribe to an event on the loader's bus for the whole session."""
        unsubscribe = self._bus.subscribe(event, callback)
        self._unsubscribes.append(unsubscribe)
        return unsubscribe

    def log(self, message: str) -> None:
        """Print a dim status line through the loader's console."""
        self._console.print(f"[dim]plugin {self.name}: {message}[/dim]")

    def cleanup(self) -> None:
        """Undo every registration this plugin made."""
        for name in self._tools:
            tools.unregister_tool(name)
        for name in self._commands:
            commands.unregister_command(name)
        for unsubscribe in self._unsubscribes:
            unsubscribe()
        self._tools.clear()
        self._commands.clear()
        self._unsubscribes.clear()


_ACTIVE: dict[str, PluginAPI] = {}


def unload_plugins() -> None:
    """Unregister every plugin loaded by :func:`load_plugins`."""
    for api in list(_ACTIVE.values()):
        api.cleanup()
    _ACTIVE.clear()


def _plugin_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [
        path
        for path in sorted(directory.glob("*.py"))
        if path.is_file() and not path.name.startswith("_")
    ]


def _load_module(name: str, path: Path) -> ModuleType:
    module_name = f"kiwimatecoder_plugin_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load plugin module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_one(
    name: str,
    path: Path,
    console: Console | None,
    bus: EventBus | None,
    result: PluginLoadResult,
) -> None:
    try:
        module = _load_module(name, path)
    except (Exception, SystemExit) as exc:
        result.failed.append((name, f"import failed: {exc}"))
        return

    register = getattr(module, "register", None)
    if not callable(register):
        result.failed.append((name, "no register(api) function found"))
        return

    api = PluginAPI(name, console=console, bus=bus)
    try:
        register(api)
    except (Exception, SystemExit) as exc:
        api.cleanup()
        result.failed.append((name, f"register failed: {exc}"))
        return
    _ACTIVE[str(path)] = api
    result.loaded.append(name)


def load_plugins(
    workspace_root: Path,
    console: Console | None = None,
    bus: EventBus | None = None,
) -> PluginLoadResult:
    """Load user plugins and, when opted in, project plugins. Never raises.

    Reloading first unloads everything registered by a previous call so tools
    and commands do not pile up across sessions.
    """
    unload_plugins()
    result = PluginLoadResult()
    settings = config.get_plugins_config()

    try:
        user_dir = config.ensure_config_dir() / PLUGINS_DIR_NAME
    except OSError:
        # An unwritable home directory just means no user-level plugins.
        user_dir = None
    planned: list[tuple[str, Path]] = (
        [(path.stem, path) for path in _plugin_files(user_dir)] if user_dir else []
    )
    project_dir = workspace_root / PROJECT_PLUGINS_DIR
    project_files = _plugin_files(project_dir)
    if settings["allow_project"]:
        planned.extend((path.stem, path) for path in project_files)
    else:
        result.skipped.extend(path.stem for path in project_files)

    disabled = set(settings["disabled"])
    for name, path in planned:
        if name in disabled:
            result.skipped.append(name)
            continue
        _load_one(name, path, console, bus, result)
    return result
