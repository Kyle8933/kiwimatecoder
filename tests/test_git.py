from __future__ import annotations

import shutil
import subprocess
import types

import pytest

from kiwimatecoder import git as git_module
from kiwimatecoder import tools
from kiwimatecoder.tools.git import _git, _git_write

GIT = shutil.which("git")


def _fake_git(calls: list[list[str]], code: int = 0, output: str = ""):
    def fake(command, session, timeout=30):
        calls.append(command)
        return code, output

    return fake


# ---------------------------------------------------------------------------
# Argument builders
# ---------------------------------------------------------------------------


def test_status_args():
    assert git_module.status_args() == ["status", "--short"]
    assert git_module.status_args(short=False) == ["status"]


def test_diff_args():
    assert git_module.diff_args() == ["diff"]
    assert git_module.diff_args(staged=True, path="a.py", ref="HEAD~1") == [
        "diff",
        "--staged",
        "HEAD~1",
        "--",
        "a.py",
    ]
    assert git_module.diff_args(ref="main") == ["diff", "main"]
    assert git_module.diff_args(path="src") == ["diff", "--", "src"]


def test_log_args_clamps_limit_and_adds_path():
    assert git_module.log_args(5) == ["log", "-5", "--no-color", "--oneline"]
    assert git_module.log_args(0) == ["log", "-1", "--no-color", "--oneline"]
    assert git_module.log_args(500) == ["log", "-100", "--no-color", "--oneline"]
    assert git_module.log_args(3, path="src", oneline=False) == [
        "log",
        "-3",
        "--no-color",
        "--",
        "src",
    ]
    assert git_module.log_args("nope") == ["log", "-20", "--no-color", "--oneline"]


def test_show_and_branch_args():
    assert git_module.show_args("abc123") == ["show", "--no-color", "abc123"]
    assert git_module.show_args() == ["show", "--no-color", "HEAD"]
    assert git_module.branch_args() == ["branch", "--list", "--no-color"]


def test_valid_branch():
    assert git_module.valid_branch("feature/web-tools")
    assert git_module.valid_branch("fix_123")
    assert git_module.valid_branch("release/1.2.x")

    assert not git_module.valid_branch("")
    assert not git_module.valid_branch("-flag")
    assert not git_module.valid_branch("has space")
    assert not git_module.valid_branch("a..b")
    assert not git_module.valid_branch("a@{b}")
    assert not git_module.valid_branch("a//b")
    assert not git_module.valid_branch("feature/")
    assert not git_module.valid_branch("feature.")
    assert not git_module.valid_branch("feature.lock")


# ---------------------------------------------------------------------------
# Write-argument building and validation
# ---------------------------------------------------------------------------


def test_build_write_args_stage_and_unstage():
    assert git_module.build_write_args(
        {"action": "stage", "paths": ["a.py", "b.py"]}
    ) == ["add", "--", "a.py", "b.py"]
    assert git_module.build_write_args(
        {"action": "unstage", "paths": ["a.py"]}
    ) == ["restore", "--staged", "--", "a.py"]


def test_build_write_args_commit():
    assert git_module.build_write_args(
        {"action": "commit", "message": "fix bug"}
    ) == ["commit", "-m", "fix bug"]


def test_build_write_args_checkout_branch():
    assert git_module.build_write_args(
        {"action": "checkout_branch", "branch": "feature/x"}
    ) == ["checkout", "-b", "feature/x"]


@pytest.mark.parametrize(
    "args",
    [
        {"action": "commit", "message": "   "},
        {"action": "commit"},
        {"action": "checkout_branch", "branch": "-bad"},
        {"action": "checkout_branch", "branch": "a..b"},
        {"action": "checkout_branch"},
        {"action": "stage", "paths": []},
        {"action": "stage", "paths": [""]},
        {"action": "unstage", "paths": "nope"},
        {"action": "push"},
        {"action": ""},
    ],
)
def test_build_write_args_rejects_invalid(args):
    with pytest.raises(ValueError):
        git_module.build_write_args(args)


def test_preview_shows_exact_command(session):
    assert git_module.preview(
        {"action": "commit", "message": "fix bug"}, session
    ) == "git commit -m 'fix bug'"
    assert git_module.preview(
        {"action": "stage", "paths": ["a b.py"]}, session
    ) == "git add -- 'a b.py'"
    assert git_module.preview(
        {"action": "checkout_branch", "branch": "feature/x"}, session
    ) == "git checkout -b feature/x"
    assert "Invalid branch" in git_module.preview(
        {"action": "checkout_branch", "branch": "-x"}, session
    )


def test_git_write_preview_is_wired(session):
    assert tools.preview(
        "git_write", {"action": "commit", "message": "fix bug"}, session
    ) == "git commit -m 'fix bug'"
    assert tools.preview("git", {"action": "status"}, session) is None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _fake_run(calls: list[tuple[list[str], dict[str, object]]], code: int = 0, output: str = "ok"):
    def fake(command, **kwargs):
        calls.append((command, kwargs))
        return types.SimpleNamespace(returncode=code, stdout=output, stderr="")

    return fake


def test_run_git_uses_argv_without_shell(session, monkeypatch):
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr(git_module.subprocess, "run", _fake_run(calls))

    code, output = git_module.run_git(["status", "--short"], session)

    assert (code, output) == (0, "ok")
    command, kwargs = calls[0]
    assert command == ["git", "status", "--short"]
    assert kwargs["cwd"] == str(session.workspace_root)
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert "shell" not in kwargs


def test_run_git_merges_stderr(session, monkeypatch):
    def fake(command, **kwargs):
        return types.SimpleNamespace(returncode=1, stdout="out\n", stderr="err\n")

    monkeypatch.setattr(git_module.subprocess, "run", fake)

    code, output = git_module.run_git(["status"], session)

    assert code == 1
    assert "out" in output and "err" in output


def test_run_git_handles_missing_binary(session, monkeypatch):
    def fake(command, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(git_module.subprocess, "run", fake)

    code, output = git_module.run_git(["status"], session)

    assert code == 127
    assert "not installed" in output


@pytest.mark.skipif(GIT is None, reason="git is not installed")
def test_run_git_in_real_repository(session):
    subprocess.run(
        ["git", "init", "-q"], cwd=session.workspace_root, check=True, capture_output=True
    )

    code, output = git_module.run_git(["status", "--short"], session)

    assert code == 0
    assert output.strip() == ""


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_git_and_git_write_schemas():
    assert tools.get_tool("git") is not None
    assert tools.get_tool("git").writes is False
    assert tools.get_tool("git").runs is False
    assert tools.get_tool("git").needs_approval is False

    assert tools.get_tool("git_write") is not None
    assert tools.get_tool("git_write").writes is True
    assert tools.get_tool("git_write").needs_approval is True

    read_only_names = {
        schema["function"]["name"] for schema in tools.tool_schemas(read_only=True)
    }
    assert "git" in read_only_names
    assert "git_write" not in read_only_names


def test_git_tool_rejects_unknown_action(session):
    result = _git({"action": "push"}, session)

    assert not result.ok
    assert "Unknown git action" in result.content


def test_git_tool_runs_status(session, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(git_module, "run_git", _fake_git(calls, output=" M a.py"))

    result = _git({"action": "status"}, session)

    assert result.ok
    assert calls == [["status", "--short"]]
    assert "M a.py" in result.content
    assert "[exit code: 0]" in result.content


def test_git_tool_log_limit_validation(session, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(git_module, "run_git", _fake_git(calls))

    result = _git({"action": "log", "limit": 500}, session)

    assert result.ok
    assert calls == [["log", "-100", "--no-color", "--oneline"]]

    bad = _git({"action": "log", "limit": "lots"}, session)
    assert not bad.ok
    assert "'limit' must be an integer" in bad.content


def test_git_tool_truncates_output(session, monkeypatch):
    monkeypatch.setattr(
        git_module,
        "run_git",
        _fake_git([], output="q" * (git_module.MAX_OUTPUT + 100)),
    )

    result = _git({"action": "diff"}, session)

    assert result.ok
    assert "[output truncated]" in result.content
    assert result.content.count("q") == git_module.MAX_OUTPUT


def test_git_write_tool_never_runs_invalid_commit(session, monkeypatch):
    called: list[list[str]] = []
    monkeypatch.setattr(git_module, "run_git", _fake_git(called))

    result = _git_write({"action": "commit", "message": ""}, session)

    assert not result.ok
    assert "message" in result.content
    assert called == []


def test_git_write_tool_stages_paths(session, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(git_module, "run_git", _fake_git(calls))

    result = _git_write({"action": "stage", "paths": ["a.py"]}, session)

    assert result.ok
    assert calls == [["add", "--", "a.py"]]
