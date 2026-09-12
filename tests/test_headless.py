from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from kiwimatecoder import config, main
from kiwimatecoder.client import Done, TextDelta, ToolCallDelta, Usage

STREAM_CHAT = "kiwimatecoder.client.UnifiedClient.stream_chat"


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    """Point config storage at a temp dir and clear provider env vars."""
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for name in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def _key(monkeypatch, value: str = "test-key") -> None:
    monkeypatch.setattr(config, "get_key", lambda provider_id: value)


def _scripted_stream(rounds, calls=None):
    """Async stream generator yielding one scripted round per call."""
    state = {"n": 0}

    async def stream(*args, **kwargs):
        if calls is not None:
            calls.append(args)
        index = state["n"]
        state["n"] += 1
        chosen = rounds[min(index, len(rounds) - 1)]
        for event in chosen:
            yield event

    return stream


def test_text_output_stdout_and_tool_noise_stderr(tmp_path, monkeypatch):
    (tmp_path / "hello.txt").write_text("file content")
    monkeypatch.chdir(tmp_path)
    _key(monkeypatch)
    rounds = [
        [
            ToolCallDelta(
                index=0,
                id="c1",
                name="read_file",
                args_fragment='{"path": "hello.txt"}',
            ),
            Done(finish_reason="tool_calls"),
        ],
        [TextDelta(text="The file has: file content"), Done(finish_reason="stop")],
    ]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(STREAM_CHAT, _scripted_stream(rounds))
        result = CliRunner().invoke(main.app, ["-p", "Read hello.txt"])

    assert result.exit_code == 0
    assert "The file has: file content" in result.stdout
    assert "read_file" not in result.stdout
    assert "hello.txt" in result.stderr


def test_quiet_suppresses_tool_progress(tmp_path, monkeypatch):
    (tmp_path / "hello.txt").write_text("file content")
    monkeypatch.chdir(tmp_path)
    _key(monkeypatch)
    rounds = [
        [
            ToolCallDelta(
                index=0,
                id="c1",
                name="read_file",
                args_fragment='{"path": "hello.txt"}',
            ),
            Done(finish_reason="tool_calls"),
        ],
        [TextDelta(text="ok"), Done(finish_reason="stop")],
    ]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(STREAM_CHAT, _scripted_stream(rounds))
        result = CliRunner().invoke(main.app, ["-p", "Read hello.txt", "--quiet"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "ok"
    assert "read_file" not in result.stderr


def test_json_output_shape_and_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _key(monkeypatch)
    config.set_selected_model("test-model")
    events = [
        TextDelta(text="Hi "),
        TextDelta(text="there"),
        Usage(prompt_tokens=11, completion_tokens=7),
        Done(finish_reason="stop"),
    ]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(STREAM_CHAT, _scripted_stream([events]))
        result = CliRunner().invoke(
            main.app, ["-p", "hello", "--output-format", "json"]
        )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == {
        "result": "Hi there",
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        "cost_usd": None,
        "provider": "openrouter",
        "model": "test-model",
        "mode": "ask",
        "tools_used": [],
        "messages": 2,
        "success": True,
    }


def test_stream_json_one_object_per_line_end_with_result(tmp_path, monkeypatch):
    (tmp_path / "hello.txt").write_text("file content")
    monkeypatch.chdir(tmp_path)
    _key(monkeypatch)
    rounds = [
        [
            ToolCallDelta(
                index=0,
                id="c1",
                name="read_file",
                args_fragment='{"path": "hello.txt"}',
            ),
            Done(finish_reason="tool_calls"),
        ],
        [
            TextDelta(text="done"),
            Usage(prompt_tokens=3, completion_tokens=2),
            Done(finish_reason="stop"),
        ],
    ]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(STREAM_CHAT, _scripted_stream(rounds))
        result = CliRunner().invoke(
            main.app, ["-p", "read", "--output-format", "stream-json"]
        )

    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert lines
    records = [json.loads(line) for line in lines]
    types = [record["type"] for record in records]
    assert types[:2] == ["tool_start", "tool_end"]
    assert "text_delta" in types
    assert "usage" in types
    final = records[-1]
    assert final["type"] == "result"
    assert final["result"] == "done"
    assert final["usage"] == {"prompt_tokens": 3, "completion_tokens": 2}
    assert final["success"] is True


def test_default_denies_writes_and_yes_allows_them(tmp_path, monkeypatch):
    _key(monkeypatch)
    rounds = [
        [
            ToolCallDelta(
                index=0,
                id="c1",
                name="write_file",
                args_fragment='{"path": "out.txt", "content": "hello"}',
            ),
            Done(finish_reason="tool_calls"),
        ],
        [TextDelta(text="done"), Done(finish_reason="stop")],
    ]
    base = ["-p", "write out.txt", "--workspace", str(tmp_path)]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(STREAM_CHAT, _scripted_stream(rounds))
        result = CliRunner().invoke(main.app, base)

    assert result.exit_code == 0
    assert not (tmp_path / "out.txt").exists()
    assert "denied" in result.stderr

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(STREAM_CHAT, _scripted_stream(rounds))
        result = CliRunner().invoke(main.app, [*base, "--yes"])

    assert result.exit_code == 0
    assert (tmp_path / "out.txt").read_text() == "hello"


def test_plan_mode_blocks_writes(tmp_path, monkeypatch):
    _key(monkeypatch)
    rounds = [
        [
            ToolCallDelta(
                index=0,
                id="c1",
                name="write_file",
                args_fragment='{"path": "out.txt", "content": "hello"}',
            ),
            Done(finish_reason="tool_calls"),
        ],
        [TextDelta(text="blocked"), Done(finish_reason="stop")],
    ]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(STREAM_CHAT, _scripted_stream(rounds))
        result = CliRunner().invoke(
            main.app,
            ["-p", "write", "--workspace", str(tmp_path), "--mode", "plan"],
        )

    assert result.exit_code == 0
    assert not (tmp_path / "out.txt").exists()
    assert "plan (read-only)" in result.stderr


def test_print_dash_reads_prompt_from_stdin(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _key(monkeypatch)
    captured: list[list[dict[str, object]]] = []

    async def stream(*args, **kwargs):
        captured.append(args[1])
        yield TextDelta(text="ok")
        yield Done(finish_reason="stop")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(STREAM_CHAT, stream)
        result = CliRunner().invoke(main.app, ["-p", "-"], input="Hello from stdin")

    assert result.exit_code == 0
    assert "ok" in result.stdout
    user_messages = [
        message
        for message in captured[0]
        if message.get("role") == "user"
    ]
    assert user_messages[-1]["content"] == "Hello from stdin"


def test_max_turns_caps_a_runaway_tool_loop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _key(monkeypatch)
    calls: list[object] = []

    async def stream(*args, **kwargs):
        calls.append(args)
        yield ToolCallDelta(
            index=0,
            id=f"c{len(calls)}",
            name="list_dir",
            args_fragment='{"path": "."}',
        )
        yield Done(finish_reason="tool_calls")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(STREAM_CHAT, stream)
        result = CliRunner().invoke(
            main.app,
            ["-p", "loop", "--max-turns", "2", "--output-format", "json"],
        )

    # Two tool batches run, then the third model call is cut short.
    assert len(calls) == 3
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["success"] is False
    assert payload["messages"] == 6
    assert "maximum of 2" in result.stderr


def test_invalid_output_format_exits_2():
    result = CliRunner().invoke(main.app, ["-p", "hi", "--output-format", "yaml"])

    assert result.exit_code == 2
    assert "Invalid --output-format" in result.stderr


def test_invalid_mode_exits_2():
    result = CliRunner().invoke(main.app, ["-p", "hi", "--mode", "wild"])

    assert result.exit_code == 2
    assert "Unknown mode" in result.stderr


def test_unknown_provider_exits_2():
    result = CliRunner().invoke(main.app, ["-p", "hi", "--provider", "nope"])

    assert result.exit_code == 2
    assert "Unknown provider" in result.stderr


def test_missing_key_exits_1(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(main.app, ["-p", "hello"])

    assert result.exit_code == 1
    assert "No API key" in result.stderr


def test_workspace_flag_controls_relative_paths(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    _key(monkeypatch)
    rounds = [
        [
            ToolCallDelta(
                index=0,
                id="c1",
                name="write_file",
                args_fragment='{"path": "out.txt", "content": "hi"}',
            ),
            Done(finish_reason="tool_calls"),
        ],
        [TextDelta(text="done"), Done(finish_reason="stop")],
    ]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(STREAM_CHAT, _scripted_stream(rounds))
        result = CliRunner().invoke(
            main.app,
            ["-p", "write", "--workspace", str(workspace), "--yes"],
        )

    assert result.exit_code == 0
    assert (workspace / "out.txt").read_text() == "hi"
    assert not (elsewhere / "out.txt").exists()


def test_missing_workspace_exits_2(tmp_path):
    result = CliRunner().invoke(
        main.app, ["-p", "hi", "--workspace", str(tmp_path / "nope")]
    )

    assert result.exit_code == 2
    assert "not a directory" in result.stderr
