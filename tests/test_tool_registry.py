from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from kiwimatecoder import tools
from kiwimatecoder.tools.base import FunctionTool, ToolResult

BUILTIN_NAMES = [
    "read_file",
    "list_dir",
    "search",
    "load_skill",
    "write_file",
    "edit_file",
    "run_bash",
    "update_todos",
    "ask_user",
]


def make_tool(name: str, *, writes: bool = False) -> FunctionTool:
    def execute(args: dict[str, Any], session: Any) -> ToolResult:
        return ToolResult(content=f"{name} ran", ok=True)

    return FunctionTool(
        name=name,
        description=f"{name} test tool",
        parameters={"type": "object", "properties": {}},
        func=execute,
        writes=writes,
    )


@pytest.fixture(autouse=True)
def restore_registry() -> Iterator[None]:
    """Snapshot/restore the global registry so plugin tests stay isolated."""
    name_sources = {
        name: source for source, names in tools.TOOLS.sources().items() for name in names
    }
    original = list(tools.TOOLS.all())
    yield
    for name in list(tools.TOOLS):
        tools.TOOLS.unregister(name, force=True)
    for tool in original:
        tools.TOOLS.register(tool, source=name_sources.get(tool.name, "builtin"))


def test_tools_mapping_compatibility():
    assert tools.TOOLS["read_file"] is tools.get_tool("read_file")
    assert tools.TOOLS.get("nope") is None
    assert "read_file" in tools.TOOLS
    assert "nope" not in tools.TOOLS
    assert list(tools.TOOLS) == BUILTIN_NAMES
    assert len(tools.TOOLS) == len(BUILTIN_NAMES)
    assert [tool.name for tool in tools.TOOLS.values()] == BUILTIN_NAMES
    assert dict(tools.TOOLS.items())["search"] is tools.get_tool("search")


def test_register_get_schemas_dispatch(session):
    fake = make_tool("frobnicate")

    assert tools.register_tool(fake) is fake
    assert tools.get_tool("frobnicate") is fake
    assert tools.TOOLS["frobnicate"] is fake

    schema = tools.tool_schemas()[-1]["function"]
    assert schema["name"] == "frobnicate"
    assert "frobnicate" in {s["function"]["name"] for s in tools.tool_schemas()}

    result = tools.dispatch("frobnicate", {}, session)
    assert result.ok
    assert result.content == "frobnicate ran"


def test_duplicate_registration_raises():
    tools.register_tool(make_tool("dupe"))
    with pytest.raises(ValueError, match="already registered"):
        tools.register_tool(make_tool("dupe"))
    with pytest.raises(ValueError, match="already registered"):
        tools.register_tool(make_tool("read_file"))


def test_replace_swaps_tool_and_schema():
    original = make_tool("swappable")
    replacement = FunctionTool(
        name="swappable",
        description="replacement tool",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        func=lambda args, session: ToolResult(content="replaced", ok=True),
    )
    tools.register_tool(original)

    tools.TOOLS.register(replacement, source="plugin", replace=True)

    assert tools.get_tool("swappable") is replacement
    schema = next(
        s for s in tools.tool_schemas() if s["function"]["name"] == "swappable"
    )
    assert schema["function"]["description"] == "replacement tool"
    assert "x" in schema["function"]["parameters"]["properties"]


def test_unregister_plugin_tool(session):
    tools.register_tool(make_tool("temporary"), source="custom")

    assert tools.unregister_tool("temporary") is True
    assert "temporary" not in tools.TOOLS
    assert tools.get_tool("temporary") is None
    assert tools.unregister_tool("temporary") is False


def test_unregister_builtin_requires_force():
    assert tools.unregister_tool("read_file") is False
    assert "read_file" in tools.TOOLS
    assert tools.get_tool("read_file") is not None

    assert tools.unregister_tool("read_file", force=True) is True
    assert "read_file" not in tools.TOOLS
    assert tools.get_tool("read_file") is None


def test_unregister_unknown_returns_false():
    assert tools.unregister_tool("ghost") is False


def test_read_only_schemas_exclude_dynamic_writer(session):
    tools.register_tool(make_tool("peek", writes=False))
    tools.register_tool(make_tool("mutate", writes=True))

    read_only_names = {tool.name for tool in tools.read_only_tools()}
    assert "peek" in read_only_names
    assert "mutate" not in read_only_names

    schema_names = {
        s["function"]["name"] for s in tools.tool_schemas(read_only=True)
    }
    assert "peek" in schema_names
    assert "mutate" not in schema_names


def test_sources_groups_builtins_and_plugins():
    tools.register_tool(make_tool("one"), source="plugin")
    tools.register_tool(make_tool("two"), source="plugin")
    tools.register_tool(make_tool("other"), source="mcp")

    sources = tools.tool_sources()

    assert sources["builtin"] == sorted(BUILTIN_NAMES)
    assert sources["plugin"] == ["one", "two"]
    assert sources["mcp"] == ["other"]
    assert tools.TOOLS.sources() == sources


def test_registry_iteration_items_and_len_after_mutation():
    tools.register_tool(make_tool("alpha"))
    tools.register_tool(make_tool("beta"))

    assert list(tools.TOOLS)[-2:] == ["alpha", "beta"]
    assert dict(tools.TOOLS.items())["alpha"] is tools.get_tool("alpha")
    assert len(tools.TOOLS) == len(BUILTIN_NAMES) + 2
    assert tools.TOOLS.all()[-1].name == "beta"
