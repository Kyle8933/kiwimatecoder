from __future__ import annotations

from kiwimatecoder.checkpoints import CheckpointStore
from kiwimatecoder.session import (
    Session,
    export_session_markdown,
    fork_session,
    list_saved_sessions,
    load_session,
)


def test_capture_and_restore_existing_file(tmp_path):
    (tmp_path / "a.txt").write_text("before")
    store = CheckpointStore(snapshot_root=tmp_path / "snaps")

    checkpoint = store.capture(tmp_path, ["a.txt"], "edit a.txt")
    (tmp_path / "a.txt").write_text("after")

    assert store.restore(tmp_path, checkpoint) == ["a.txt"]
    assert (tmp_path / "a.txt").read_text() == "before"


def test_restore_removes_file_that_did_not_exist(tmp_path):
    store = CheckpointStore(snapshot_root=tmp_path / "snaps")
    checkpoint = store.capture(tmp_path, ["new.txt"], "write new.txt")
    (tmp_path / "new.txt").write_text("created")

    store.restore(tmp_path, checkpoint)

    assert not (tmp_path / "new.txt").exists()


def test_restore_recreates_deleted_file(tmp_path):
    (tmp_path / "gone.txt").write_text("keep")
    store = CheckpointStore(snapshot_root=tmp_path / "snaps")
    checkpoint = store.capture(tmp_path, ["gone.txt"], "edit gone.txt")
    (tmp_path / "gone.txt").unlink()

    store.restore(tmp_path, checkpoint)

    assert (tmp_path / "gone.txt").read_text() == "keep"


def test_capture_skips_outside_workspace(tmp_path):
    store = CheckpointStore(snapshot_root=tmp_path / "snaps")

    checkpoint = store.capture(tmp_path, ["../escape.txt"], "bad")

    assert checkpoint.files == {}


def test_capture_deduplicates_paths(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    store = CheckpointStore(snapshot_root=tmp_path / "snaps")

    checkpoint = store.capture(tmp_path, ["a.txt", "a.txt"], "edit")

    assert checkpoint.paths == ["a.txt"]


def test_session_checkpoint_and_undo_multiple(tmp_path):
    session = Session(provider_id="openrouter", model="m", workspace_root=tmp_path)
    target = tmp_path / "f.txt"
    target.write_text("v0")

    session.checkpoint(["f.txt"], "op1")
    target.write_text("v1")
    session.checkpoint(["f.txt"], "op2")
    target.write_text("v2")

    restored = session.undo_checkpoints(2)

    assert [item.label for item in restored] == ["op1", "op2"]
    assert target.read_text() == "v0"
    assert session.checkpoints == []


def test_undo_with_no_checkpoints_is_noop(tmp_path):
    session = Session(provider_id="openrouter", model="m", workspace_root=tmp_path)

    assert session.undo_checkpoints() == []


def test_fork_session_writes_copy(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    monkeypatch.setattr("kiwimatecoder.session._sessions_dir", lambda: sessions_dir)
    session = Session(
        provider_id="anthropic",
        model="claude-sonnet-5",
        workspace_root=tmp_path,
        messages=[{"role": "user", "content": "hi"}],
    )

    path = fork_session(session, "branch")

    assert path.name == "branch.json"
    assert list_saved_sessions()[0]["name"] == "branch"
    assert load_session("branch", workspace_root=tmp_path).messages == session.messages


def test_session_to_dict_includes_format_version(tmp_path):
    session = Session(provider_id="openai", model="m", workspace_root=tmp_path)

    assert session.to_dict()["format_version"] == 2


def test_export_markdown_contains_turns_and_tools(tmp_path):
    session = Session(
        provider_id="openai",
        model="gpt-5.6-sol",
        workspace_root=tmp_path,
        messages=[
            {"role": "user", "content": "read it"},
            {
                "role": "assistant",
                "content": "sure",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "file contents"},
        ],
    )

    markdown = export_session_markdown(session)

    assert "# KiwiMateCoder session" in markdown
    assert "## User" in markdown
    assert "## Assistant" in markdown
    assert "read_file" in markdown
    assert "file contents" in markdown


def test_export_markdown_truncates_long_tool_output(tmp_path):
    session = Session(
        provider_id="openai",
        model="m",
        workspace_root=tmp_path,
        messages=[
            {"role": "tool", "tool_call_id": "c1", "content": "x" * 5000},
        ],
    )

    markdown = export_session_markdown(session, max_tool_chars=100)

    assert "[truncated 4900 chars]" in markdown
