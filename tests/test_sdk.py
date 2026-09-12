from __future__ import annotations

from unittest.mock import patch

import pytest

from kiwimatecoder import config, sdk
from kiwimatecoder.client import Done, TextDelta, ToolCallDelta, Usage


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    """Point config storage at a temp dir and clear provider env vars."""
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _scripted(rounds):
    """Return a fake ``UnifiedClient.stream_chat`` yielding one round per call."""
    state = {"n": 0}

    async def stream(*args, **kwargs):
        index = state["n"]
        state["n"] += 1
        for event in rounds[min(index, len(rounds) - 1)]:
            yield event

    return stream


async def _run(rounds, **kwargs):
    """Patch the stream and key, then await the SDK entry point."""
    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy-key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=_scripted(rounds),
        ),
    ):
        return await sdk.run_agent(**kwargs)


@pytest.mark.anyio
async def test_run_agent_returns_text_usage_and_tools(tmp_path):
    (tmp_path / "hello.txt").write_text("file content")
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
            TextDelta(text="The file has: file content"),
            Usage(prompt_tokens=9, completion_tokens=4),
            Done(finish_reason="stop"),
        ],
    ]

    result = await _run(rounds, prompt="Read hello.txt", workspace=tmp_path)

    assert isinstance(result, sdk.RunResult)
    assert result.text == "The file has: file content"
    assert result.usage == {"prompt_tokens": 9, "completion_tokens": 4}
    assert result.tools_used == ["read_file"]
    assert result.messages == 4
    assert result.success is True
    assert result.error is None
    assert result.provider == "openrouter"
    assert result.mode == "ask"


@pytest.mark.anyio
async def test_on_event_collects_events(tmp_path):
    (tmp_path / "hello.txt").write_text("file content")
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
            TextDelta(text="ok"),
            Usage(prompt_tokens=2, completion_tokens=1),
            Done(finish_reason="stop"),
        ],
    ]
    seen: list[tuple[str, dict]] = []

    await _run(rounds, prompt="Read", workspace=tmp_path, on_event=lambda n, p: seen.append((n, p)))

    names = [name for name, _payload in seen]
    assert names == ["tool_start", "tool_end", "text_delta", "usage", "done"]
    by_name = dict(seen)
    assert by_name["tool_start"]["args"] == {"path": "hello.txt"}
    assert by_name["tool_end"]["ok"] is True
    assert by_name["tool_end"]["duration_ms"] >= 0
    assert by_name["done"]["reason"] == "stop"


@pytest.mark.anyio
async def test_max_turns_marks_failure(tmp_path):
    rounds = [
        [
            ToolCallDelta(
                index=0,
                id="c1",
                name="list_dir",
                args_fragment='{"path": "."}',
            ),
            Done(finish_reason="tool_calls"),
        ]
    ]

    result = await _run(rounds, prompt="loop", workspace=tmp_path, max_turns=2)

    assert result.success is False
    assert result.error is not None
    assert "maximum of 2" in result.error


@pytest.mark.anyio
async def test_confirm_allows_a_write_and_default_denies(tmp_path):
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

    denied = await _run(rounds, prompt="write", workspace=tmp_path)
    assert denied.success is True
    assert not (tmp_path / "out.txt").exists()

    prompts: list[str] = []

    def confirm(summary: str, preview: str | None) -> bool:
        prompts.append(summary)
        return True

    allowed = await _run(
        rounds, prompt="write", workspace=tmp_path, confirm=confirm
    )
    assert allowed.success is True
    assert (tmp_path / "out.txt").read_text() == "hello"
    assert prompts and "write_file" in prompts[0]


@pytest.mark.anyio
async def test_cost_computed_for_priced_model(tmp_path):
    rounds = [
        [
            TextDelta(text="hi"),
            Usage(prompt_tokens=1000, completion_tokens=500),
            Done(finish_reason="stop"),
        ]
    ]

    result = await _run(
        rounds, prompt="hi", workspace=tmp_path, model="claude-sonnet-5"
    )

    expected = (1000 * 3.0 + 500 * 15.0) / 1_000_000
    assert result.cost_usd == pytest.approx(expected)


@pytest.mark.anyio
async def test_cost_none_for_unknown_model(tmp_path):
    rounds = [
        [
            TextDelta(text="hi"),
            Usage(prompt_tokens=1000, completion_tokens=500),
            Done(finish_reason="stop"),
        ]
    ]

    result = await _run(
        rounds, prompt="hi", workspace=tmp_path, model="totally-unknown-model"
    )

    assert result.cost_usd is None


def test_run_agent_sync_works_outside_event_loop(tmp_path):
    rounds = [
        [TextDelta(text="sync ok"), Done(finish_reason="stop")]
    ]

    with (
        patch("kiwimatecoder.config.get_key", return_value="dummy-key"),
        patch(
            "kiwimatecoder.client.UnifiedClient.stream_chat",
            side_effect=_scripted(rounds),
        ),
    ):
        result = sdk.run_agent_sync("hi", workspace=tmp_path)

    assert result.text == "sync ok"
    assert result.success is True


@pytest.mark.anyio
async def test_run_agent_sync_refuses_inside_running_loop():
    with pytest.raises(RuntimeError, match="running event loop"):
        sdk.run_agent_sync("hi")
