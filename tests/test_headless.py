from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from kiwimatecoder import catalog, config, headless, main
from kiwimatecoder.client import Done, TextDelta, ToolCallDelta, Usage
from kiwimatecoder.providers import REGISTRY

STREAM_CHAT = "kiwimatecoder.client.UnifiedClient.stream_chat"


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    """Point config storage at a temp dir and clear provider env vars.

    There are no default models, so the primary provider gets an explicit
    model choice; tests about a missing model clear it again.
    """
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for provider in REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)
    config.set_provider_model("openrouter", "test-model")


def _key(monkeypatch, value: str = "test-key") -> None:
    monkeypatch.setattr(config, "get_key", lambda provider_id: value)


def _install_fetch(monkeypatch, model_ids):
    """Patch the network fetch to return ``model_ids`` newest-first."""

    def fake_fetch(provider, api_key=None, **kwargs):
        return [
            catalog.RemoteModel(model_id, float(len(model_ids) - index))
            for index, model_id in enumerate(model_ids)
        ]

    monkeypatch.setattr(config.catalog, "fetch_models", fake_fetch)


def _forbid_fetch(monkeypatch):
    def fake_fetch(provider, api_key=None, **kwargs):
        raise AssertionError("the network must not be touched here")

    monkeypatch.setattr(config.catalog, "fetch_models", fake_fetch)


def _offline_fetch(monkeypatch):
    def fake_fetch(provider, api_key=None, **kwargs):
        raise catalog.CatalogFetchError("connection refused")

    monkeypatch.setattr(config.catalog, "fetch_models", fake_fetch)


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
    config.set_selected_model("json-model")
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
        "model": "json-model",
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


def test_usage_errors_win_over_a_missing_model(tmp_path):
    # Invalid usage (exit 2) is reported before the runtime model check (exit 1).
    config.set_provider_model("openrouter", None)

    result = CliRunner().invoke(
        main.app, ["-p", "hi", "--workspace", str(tmp_path / "nope")]
    )

    assert result.exit_code == 2
    assert "not a directory" in result.stderr
    assert "No model chosen" not in result.stderr


# ---------------------------------------------------------------------------
# Model choice (there are no default models)
# ---------------------------------------------------------------------------


def test_print_without_a_chosen_model_exits_1_before_any_network(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _key(monkeypatch)
    _forbid_fetch(monkeypatch)
    config.set_provider_model("openrouter", None)  # undo the fixture's choice
    streamed: list[object] = []

    async def stream(*args, **kwargs):
        streamed.append(args)
        yield Done(finish_reason="stop")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(STREAM_CHAT, stream)
        result = CliRunner().invoke(
            main.app, ["-p", "hello", "--output-format", "json"]
        )

    assert result.exit_code == 1
    assert "error: No model chosen for OpenRouter" in result.stderr
    assert "--model" in result.stderr
    # Like other runtime failures (e.g. a missing key), JSON consumers still
    # get a result record.
    record = json.loads(result.stdout)
    assert record == {
        "result": "",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "cost_usd": 0.0,
        "provider": "openrouter",
        "model": None,
        "mode": "ask",
        "tools_used": [],
        "messages": 0,
        "success": False,
    }
    assert streamed == []


def test_print_without_a_model_stream_json_and_text_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _key(monkeypatch)
    _forbid_fetch(monkeypatch)
    config.set_provider_model("openrouter", None)

    streamed = CliRunner().invoke(
        main.app,
        ["-p", "hello", "--output-format", "stream-json", "--mode", "plan"],
    )
    text = CliRunner().invoke(main.app, ["-p", "hello"])

    assert streamed.exit_code == 1
    (line,) = streamed.stdout.splitlines()
    record = json.loads(line)
    assert record["type"] == "result"
    assert record["success"] is False
    assert record["mode"] == "plan"
    assert text.exit_code == 1
    assert text.stdout == ""
    assert "No model chosen for OpenRouter" in text.stderr


def test_print_model_flag_supplies_the_missing_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _key(monkeypatch)
    config.set_provider_model("openrouter", None)
    models: list[str] = []

    async def stream(self, messages, tools, model):
        models.append(model)
        yield TextDelta(text="ok")
        yield Done(finish_reason="stop")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(STREAM_CHAT, stream)
        result = CliRunner().invoke(main.app, ["-p", "hello", "--model", "flag-model"])

    assert result.exit_code == 0
    assert models == ["flag-model"]


def test_print_provider_override_needs_a_model_for_that_provider(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _key(monkeypatch)
    _forbid_fetch(monkeypatch)

    # openrouter (the configured primary) has a model; deepseek does not.
    result = CliRunner().invoke(main.app, ["-p", "hello", "--provider", "deepseek"])

    assert result.exit_code == 1
    assert "No model chosen for DeepSeek" in result.stderr


def test_print_with_a_stopped_local_server_and_no_model_exits_1(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _offline_fetch(monkeypatch)
    streamed: list[object] = []

    async def stream(*args, **kwargs):
        streamed.append(args)
        yield Done(finish_reason="stop")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(STREAM_CHAT, stream)
        result = CliRunner().invoke(main.app, ["-p", "hello", "--provider", "ollama"])

    # Nothing chosen and nothing listed: the suggested models are not a fallback.
    assert result.exit_code == 1
    assert "No model chosen for Ollama (local)" in result.stderr
    assert streamed == []


def test_print_provider_and_model_flags_make_that_provider_primary(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _key(monkeypatch)
    config.set_active_providers(["openrouter", "openai"])
    config.set_provider_model("openrouter", "or-model")
    config.set_provider_model("openai", "openai-model")
    attempts: list[tuple[str, str]] = []

    async def stream(self, messages, tools, model):
        attempts.append((self.provider.id, model))
        yield TextDelta(text="ok")
        yield Done(finish_reason="stop")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(STREAM_CHAT, stream)
        result = CliRunner().invoke(
            main.app,
            [
                "-p",
                "hello",
                "--provider",
                "deepseek",
                "--model",
                "ds-model",
                "--output-format",
                "json",
            ],
        )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["provider"] == "deepseek"
    assert payload["model"] == "ds-model"
    assert attempts == [("deepseek", "ds-model")]
    # Nothing is persisted by a per-run override.
    assert config.get_active_provider_ids() == ["openrouter", "openai"]


def test_build_session_puts_an_overriding_provider_first(tmp_path):
    config.set_active_providers(["openrouter", "openai"])
    config.set_provider_model("openrouter", "or-model")

    session = headless.build_session(
        workspace=tmp_path, provider="deepseek", model="ds-model"
    )

    assert session.provider_id == "deepseek"
    assert session.model == "ds-model"
    assert session.active_provider_ids == ["deepseek", "openrouter", "openai"]

    # A provider already in the roster moves to the front, not duplicated.
    config.set_provider_model("openai", "openai-model")
    session = headless.build_session(workspace=tmp_path, provider="openai")

    assert session.active_provider_ids == ["openai", "openrouter"]
    assert session.model == "openai-model"


def test_build_session_uses_the_configured_roster_without_override(tmp_path):
    config.set_active_providers(["openrouter", "openai"])
    config.set_provider_model("openrouter", "or-model")

    session = headless.build_session(workspace=tmp_path)

    assert session.provider_id == "openrouter"
    assert session.model == "or-model"
    assert session.active_provider_ids == ["openrouter", "openai"]


def test_build_session_raises_model_not_chosen(tmp_path, monkeypatch):
    _forbid_fetch(monkeypatch)
    config.set_provider_model("openrouter", None)

    with pytest.raises(config.ModelNotChosenError, match="No model chosen for OpenRouter"):
        headless.build_session(workspace=tmp_path)
    # A ValueError, so callers that already catch ValueError keep working.
    with pytest.raises(ValueError):
        headless.build_session(workspace=tmp_path, provider="openai")

    assert headless.build_session(workspace=tmp_path, model="m").model == "m"


def test_build_session_uses_a_local_servers_first_model(tmp_path, monkeypatch):
    _install_fetch(monkeypatch, ["qwen3:8b", "llama3.1:8b"])

    session = headless.build_session(workspace=tmp_path, provider="ollama")

    assert session.model == "qwen3:8b"


def test_build_session_never_falls_back_to_suggested_local_models(
    tmp_path, monkeypatch
):
    # Offline: the suggested tuple would be a hidden default, so it is not used.
    _offline_fetch(monkeypatch)

    with pytest.raises(config.ModelNotChosenError, match="Ollama"):
        headless.build_session(workspace=tmp_path, provider="ollama")
