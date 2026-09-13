"""Redacted session sharing: bundles, import, and the gh gist path."""

from __future__ import annotations

import io
import json
import types
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from kiwimatecoder import config, main, share
from kiwimatecoder.commands import (
    CommandResult,
    dispatch,
    has_command,
    slash_argument_completions,
    slash_command_completions,
)
from kiwimatecoder.session import load_session, save_session

SECRET = "sk-or-abcdefghijklmnopqrstuv"


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    return tmp_path


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def _output(console: Console) -> str:
    return console.file.getvalue()


def _patch_sessions(monkeypatch, tmp_path) -> Path:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("kiwimatecoder.session._sessions_dir", lambda: sessions_dir)
    return sessions_dir


def _share_files(session) -> list[Path]:
    return list(
        (session.workspace_root / ".kiwimatecoder" / "shares").glob("*.share.json")
    )


# ---------------------------------------------------------------------------
# build_share
# ---------------------------------------------------------------------------


def test_build_share_redacts_secrets_and_omits_tool_output(session):
    session.messages = [
        {"role": "user", "content": f"Deploy using {SECRET} now"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "run_bash",
                        "arguments": json.dumps(
                            {"cmd": f"curl -H 'Authorization: Bearer {SECRET}'"}
                        ),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": f"tool says {SECRET}"},
    ]

    bundle = share.build_share(session)

    encoded = json.dumps(bundle)
    assert SECRET not in encoded
    assert bundle["version"] == share.SHARE_VERSION
    assert bundle["provider"] == "openrouter"
    assert bundle["model"] == "test-model"
    assert bundle["mode"] == "ask"
    assert "[REDACTED]" in bundle["summary"]
    assert "[REDACTED]" in bundle["messages"][0]["content"]
    tool_call = bundle["messages"][1]["tool_calls"][0]
    assert tool_call["function"]["name"] == "run_bash"
    assert "[REDACTED]" in tool_call["function"]["arguments"]
    assert bundle["messages"][2]["content"] == share.OMITTED_TOOL_OUTPUT
    assert bundle["stats"]["messages"] == 3


def test_build_share_redacts_tool_arguments_without_tool_output(session):
    session.messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({"api_key": SECRET}),
                    },
                }
            ],
        }
    ]

    bundle = share.build_share(session)

    arguments = bundle["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert SECRET not in arguments
    assert "[REDACTED]" in arguments


def test_build_share_includes_truncated_tool_output_when_asked(session):
    session.messages = [
        {"role": "tool", "tool_call_id": "x", "content": "s" * 5000}
    ]

    bundle = share.build_share(session, include_tool_output=True)

    content = bundle["messages"][0]["content"]
    assert content != share.OMITTED_TOOL_OUTPUT
    assert "[truncated" in content
    assert len(content) <= share.MAX_TOOL_CHARS + 64


def test_build_share_omits_images(session):
    session.messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,QUJDRA=="},
                },
            ],
        }
    ]

    bundle = share.build_share(session)

    blocks = bundle["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "look"}
    assert blocks[1] == {"type": "text", "text": share.OMITTED_IMAGE}
    assert "QUJDRA==" not in json.dumps(bundle)


def test_build_share_caps_size_and_notes_truncation(session):
    session.messages = [
        {"role": "user", "content": f"message {index} " + "x" * share.MAX_MESSAGE_CHARS}
        for index in range(40)
    ]

    bundle = share.build_share(session)

    encoded = json.dumps(bundle, ensure_ascii=False).encode("utf-8")
    assert len(encoded) <= share.MAX_SHARE_BYTES
    assert bundle["truncated"] is True
    assert bundle["truncated_messages"] > 0
    assert bundle["summary"]
    assert bundle["messages"][0]["role"] != "tool"


def test_build_share_with_todos(session):
    session.todos = [{"id": "1", "content": "ship it", "status": "pending"}]

    bundle = share.build_share(session)

    assert bundle["todos"] == [{"id": "1", "content": "ship it", "status": "pending"}]


# ---------------------------------------------------------------------------
# write / load / import
# ---------------------------------------------------------------------------


def test_write_share_default_path_and_roundtrip(session):
    session.messages = [{"role": "user", "content": "add a login flow"}]

    path = share.write_share(session)

    assert path.parent == session.workspace_root / ".kiwimatecoder" / "shares"
    assert path.name.endswith(".share.json")
    loaded = share.load_share(path)
    assert loaded["messages"][0]["content"] == "add a login flow"
    assert loaded["summary"] == "add a login flow"
    assert loaded["version"] == share.SHARE_VERSION


def test_write_share_custom_dest_and_directory(session, tmp_path):
    session.messages = [{"role": "user", "content": "hello"}]

    dest = tmp_path / "out" / "bundle.json"
    path = share.write_share(session, dest=dest)
    assert path == dest
    assert dest.is_file()
    assert not (dest.parent / f".{dest.name}.tmp").exists()

    directory = tmp_path / "drop"
    directory.mkdir()
    path2 = share.write_share(session, dest=directory)
    assert path2.parent == directory
    assert path2.name.endswith(".share.json")


def test_load_share_validation(tmp_path):
    with pytest.raises(share.ShareError):
        share.load_share(tmp_path / "missing.json")

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(share.ShareError):
        share.load_share(bad)

    wrong_version = tmp_path / "version.json"
    wrong_version.write_text(json.dumps({"version": 99, "messages": []}))
    with pytest.raises(share.ShareError):
        share.load_share(wrong_version)

    no_messages = tmp_path / "empty.json"
    no_messages.write_text(json.dumps({"version": share.SHARE_VERSION}))
    with pytest.raises(share.ShareError):
        share.load_share(no_messages)

    minimal = tmp_path / "minimal.json"
    minimal.write_text(json.dumps({"version": share.SHARE_VERSION, "messages": []}))
    loaded = share.load_share(minimal)
    assert loaded["summary"] == ""
    assert loaded["todos"] == []
    assert loaded["stats"] == {}
    assert loaded["truncated"] is False
    assert loaded["mode"] == "ask"


def test_import_share_creates_loadable_session(session, monkeypatch, tmp_path):
    sessions_dir = _patch_sessions(monkeypatch, tmp_path)
    session.messages = [{"role": "user", "content": "remember this"}]

    bundle = share.write_share(session, dest=tmp_path / "b.share.json")
    saved = share.import_share(bundle, name="from-team")

    assert saved == sessions_dir / "from-team.json"
    loaded = load_session("from-team")
    assert loaded.provider_id == "openrouter"
    assert loaded.model == "test-model"
    assert loaded.messages[0]["content"] == "remember this"


def test_import_share_default_name(session, monkeypatch, tmp_path):
    _patch_sessions(monkeypatch, tmp_path)
    session.messages = [{"role": "user", "content": "hi"}]

    bundle = share.write_share(session, dest=tmp_path / "b.share.json")
    saved = share.import_share(bundle)

    assert saved.name.startswith("shared_")
    assert saved.is_file()


# ---------------------------------------------------------------------------
# gist integration
# ---------------------------------------------------------------------------


def test_gist_args_exact(tmp_path):
    path = tmp_path / "b.share.json"

    assert share.gist_args(path) == [
        "gh",
        "gist",
        "create",
        "--public=false",
        "--desc",
        "kiwimatecoder session share",
        str(path),
    ]


def test_share_to_gist_unavailable(session, monkeypatch):
    monkeypatch.setattr(share.forge, "cli_available", lambda command: False)

    ok, detail = share.share_to_gist(session)

    assert ok is False
    assert "gh" in detail
    assert "auth login" in detail


def test_share_to_gist_success(session, monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return types.SimpleNamespace(
            returncode=0, stdout="https://gist.github.com/abc123\n", stderr=""
        )

    monkeypatch.setattr(share.forge, "cli_available", lambda command: True)
    monkeypatch.setattr(share.subprocess, "run", fake_run)

    dest = tmp_path / "g.share.json"
    ok, detail = share.share_to_gist(session, dest=dest, include_tool_output=True)

    assert ok is True
    assert detail == "https://gist.github.com/abc123"
    assert calls[0] == share.gist_args(dest)
    assert dest.is_file()


def test_share_to_gist_failure_reports_stderr(session, monkeypatch, tmp_path):
    def fake_run(argv, **kwargs):
        return types.SimpleNamespace(
            returncode=1, stdout="", stderr="gh: not logged in"
        )

    monkeypatch.setattr(share.forge, "cli_available", lambda command: True)
    monkeypatch.setattr(share.subprocess, "run", fake_run)

    ok, detail = share.share_to_gist(session, dest=tmp_path / "g.share.json")

    assert ok is False
    assert "not logged in" in detail


# ---------------------------------------------------------------------------
# slash commands
# ---------------------------------------------------------------------------


def test_slash_share_registered_and_completes():
    assert has_command("share")

    names = {name for name, _ in slash_command_completions("sha")}
    assert "/share" in names

    actions = {value for value, _ in slash_argument_completions("share", "")}
    assert {"gist", "import"} <= actions


def test_slash_share_writes_bundle(session):
    session.messages = [{"role": "user", "content": "hello"}]
    console = _console()

    assert dispatch("/share", session, console) == CommandResult.CONTINUE

    assert "share bundle" in _output(console).lower()
    files = _share_files(session)
    assert len(files) == 1
    assert json.loads(files[0].read_text())["messages"][0]["content"] == "hello"


def test_slash_share_include_tool_output(session):
    session.messages = [
        {"role": "tool", "tool_call_id": "1", "content": "result " + "z" * 4000}
    ]
    console = _console()

    dispatch("/share --include-tool-output", session, console)

    files = _share_files(session)
    bundle = json.loads(files[0].read_text())
    assert bundle["messages"][0]["content"] != share.OMITTED_TOOL_OUTPUT
    assert "[truncated" in bundle["messages"][0]["content"]


def test_slash_share_gist_reports_failure(session, monkeypatch):
    monkeypatch.setattr(
        share, "share_to_gist", lambda *args, **kwargs: (False, "gh exploded")
    )
    console = _console()

    dispatch("/share gist", session, console)

    assert "gh exploded" in _output(console)


def test_slash_share_gist_success(session, monkeypatch):
    monkeypatch.setattr(
        share,
        "share_to_gist",
        lambda *args, **kwargs: (True, "https://gist.github.com/x"),
    )
    console = _console()

    dispatch("/share gist", session, console)

    assert "https://gist.github.com/x" in _output(console)


def test_slash_share_import(session, monkeypatch, tmp_path):
    sessions_dir = _patch_sessions(monkeypatch, tmp_path)
    session.messages = [{"role": "user", "content": "hi there"}]
    bundle = share.write_share(session, dest=tmp_path / "b.share.json")
    console = _console()

    dispatch(f"/share import {bundle}", session, console)

    output = _output(console)
    assert "Imported session" in output
    assert list(sessions_dir.glob("shared_*.json"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_share_create_writes_bundle(session, monkeypatch, tmp_path):
    _patch_sessions(monkeypatch, tmp_path)
    session.messages = [{"role": "user", "content": "hello cli"}]
    save_session(session, "last")
    dest = tmp_path / "bundle.json"

    result = CliRunner().invoke(
        main.app, ["share", "create", "--output", str(dest)]
    )

    assert result.exit_code == 0
    assert dest.is_file()
    assert "share bundle" in result.stdout.lower()


def test_cli_share_create_gist_unavailable_exits_1(
    session, monkeypatch, tmp_path
):
    _patch_sessions(monkeypatch, tmp_path)
    session.messages = [{"role": "user", "content": "hello"}]
    save_session(session, "last")
    monkeypatch.setattr(share.forge, "cli_available", lambda command: False)

    result = CliRunner().invoke(main.app, ["share", "create", "--gist"])

    assert result.exit_code == 1
    assert "gh" in result.stdout


def test_cli_share_import_prints_session_name(session, monkeypatch, tmp_path):
    sessions_dir = _patch_sessions(monkeypatch, tmp_path)
    session.messages = [{"role": "user", "content": "hi"}]
    bundle = share.write_share(session, dest=tmp_path / "b.share.json")

    result = CliRunner().invoke(
        main.app, ["share", "import", str(bundle), "--name", "team-session"]
    )

    assert result.exit_code == 0
    assert "team-session" in result.stdout
    assert (sessions_dir / "team-session.json").is_file()


def test_cli_share_import_missing_file_exits_1(tmp_path):
    result = CliRunner().invoke(
        main.app, ["share", "import", str(tmp_path / "missing.json")]
    )

    assert result.exit_code == 1
    assert "not found" in result.stdout.lower()


# ---------------------------------------------------------------------------
# config team CLI / slash
# ---------------------------------------------------------------------------


def test_cli_config_team_roundtrip(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"default_mode": "plan"}))
    runner = CliRunner()

    result = runner.invoke(main.app, ["config", "team", "set-policy", str(policy)])
    assert result.exit_code == 0
    assert config.get_team()["policy_path"] == str(policy)

    result = runner.invoke(main.app, ["config", "team", "enforce", "on"])
    assert result.exit_code == 0
    assert config.get_team()["enforce"] is True

    result = runner.invoke(main.app, ["config", "team", "show"])
    assert result.exit_code == 0
    assert "default_mode" in result.stdout

    result = runner.invoke(main.app, ["config", "team", "enforce", "bogus"])
    assert result.exit_code == 1


def test_cli_config_team_set_policy_missing_file_exits_1(tmp_path):
    result = CliRunner().invoke(
        main.app, ["config", "team", "set-policy", str(tmp_path / "gone.json")]
    )

    assert result.exit_code == 1
    assert "does not exist" in result.stdout


def test_slash_config_team_show_set_enforce(session, tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"default_mode": "plan"}))
    console = _console()

    dispatch(f"/config team set-policy {policy}", session, console)
    assert config.get_team()["policy_path"] == str(policy)

    console = _console()
    dispatch("/config team show", session, console)
    assert "default_mode" in _output(console)
    assert "advisory" in _output(console)

    dispatch("/config team enforce on", session, _console())
    assert config.get_team()["enforce"] is True


def test_slash_config_team_missing_policy_reports_error(session, tmp_path):
    path = tmp_path / "gone.json"

    console = _console()
    dispatch(f"/config team set-policy {path}", session, console)

    assert "does not exist" in _output(console)


def test_slash_config_team_show_reports_issues(session, tmp_path):
    from kiwimatecoder import team

    policy = tmp_path / "policy.json"
    policy.write_text("{not json")
    config.set_team(policy_path=str(policy))
    team.clear_policy_cache()

    console = _console()
    dispatch("/config team show", session, console)

    assert "Cannot read team policy" in _output(console)
