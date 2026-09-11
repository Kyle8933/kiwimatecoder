from __future__ import annotations

import types

import pytest

from kiwimatecoder import forge as forge_module
from kiwimatecoder import tools
from kiwimatecoder.tools.forge import _forge, _forge_write

REMOTE_GITHUB = "git@github.com:owner/repo.git"
REMOTE_GITLAB = "https://gitlab.com/group/project.git"


def _patch_forge(
    monkeypatch,
    *,
    remote: str | None = REMOTE_GITHUB,
    available: bool = True,
    result: tuple[int, str] = (0, "ok"),
):
    calls: list[tuple[str, list[str], object]] = []
    monkeypatch.setattr(forge_module, "current_remote_url", lambda session: remote)
    monkeypatch.setattr(forge_module, "cli_available", lambda command: available)

    def fake_run(cli, args, session, timeout=60):
        calls.append((cli, args, timeout))
        return result

    monkeypatch.setattr(forge_module, "run_forge", fake_run)
    return calls


# ---------------------------------------------------------------------------
# detect_forge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "remote,expected",
    [
        ("git@github.com:owner/repo.git", "github"),
        ("https://github.com/owner/repo.git", "github"),
        ("https://www.github.com/owner/repo", "github"),
        ("ssh://git@github.com/owner/repo.git", "github"),
        ("git@gitlab.com:group/project.git", "gitlab"),
        ("https://gitlab.com/group/project.git", "gitlab"),
        ("ssh://git@gitlab.com:group/project.git", "gitlab"),
        ("git@bitbucket.org:owner/repo.git", None),
        ("https://git.example.com/owner/repo.git", None),
        ("https://github.example.com/owner/repo.git", None),
        ("", None),
        (None, None),
    ],
)
def test_detect_forge(remote, expected):
    assert forge_module.detect_forge(remote) == expected


def test_forge_cli_mapping():
    assert forge_module.forge_cli("github") == "gh"
    assert forge_module.forge_cli("gitlab") == "glab"
    assert forge_module.forge_cli("bitbucket") is None
    assert forge_module.forge_cli(None) is None


def test_cli_available(monkeypatch):
    monkeypatch.setattr(forge_module.shutil, "which", lambda command: f"/usr/bin/{command}")
    assert forge_module.cli_available("gh") is True

    monkeypatch.setattr(forge_module.shutil, "which", lambda command: None)
    assert forge_module.cli_available("gh") is False


# ---------------------------------------------------------------------------
# Argument builders
# ---------------------------------------------------------------------------


def test_pr_builders_github():
    assert forge_module.pr_list_args() == ["pr", "list"]
    assert forge_module.pr_view_args(42) == ["pr", "view", "42"]
    assert forge_module.pr_create_args("Add thing", "Body", "main", True) == [
        "pr",
        "create",
        "--title",
        "Add thing",
        "--body",
        "Body",
        "--base",
        "main",
        "--draft",
    ]
    assert forge_module.pr_create_args("Add thing") == [
        "pr",
        "create",
        "--title",
        "Add thing",
        "--body",
        "",
    ]


def test_pr_builders_gitlab_use_mr_and_glab_flags():
    assert forge_module.pr_list_args(forge="gitlab") == ["mr", "list"]
    assert forge_module.pr_view_args(7, forge="gitlab") == ["mr", "view", "7"]
    assert forge_module.pr_create_args(
        "Add thing", "Body", "main", False, forge="gitlab"
    ) == [
        "mr",
        "create",
        "--title",
        "Add thing",
        "--description",
        "Body",
        "--target-branch",
        "main",
    ]


def test_issue_builders():
    assert forge_module.issue_list_args() == ["issue", "list"]
    assert forge_module.issue_view_args(9) == ["issue", "view", "9"]
    assert forge_module.issue_create_args("Bug", "Details") == [
        "issue",
        "create",
        "--title",
        "Bug",
        "--body",
        "Details",
    ]
    assert forge_module.issue_create_args("Bug", "Details", forge="gitlab") == [
        "issue",
        "create",
        "--title",
        "Bug",
        "--description",
        "Details",
    ]


def test_build_write_args_validation():
    assert forge_module.build_write_args(
        "github", {"action": "pr_create", "title": "T"}
    ) == ["pr", "create", "--title", "T", "--body", ""]
    assert forge_module.build_write_args(
        "gitlab", {"action": "issue_create", "title": "T", "body": "B"}
    ) == ["issue", "create", "--title", "T", "--description", "B"]

    with pytest.raises(ValueError, match="title"):
        forge_module.build_write_args("github", {"action": "pr_create", "title": " "})
    with pytest.raises(ValueError, match="title"):
        forge_module.build_write_args(
            "github", {"action": "issue_create", "title": ""}
        )
    with pytest.raises(ValueError, match="Unknown forge_write action"):
        forge_module.build_write_args("github", {"action": "pr_merge"})


def test_run_forge_uses_argv_without_shell(monkeypatch, session):
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake(command, **kwargs):
        calls.append((command, kwargs))
        return types.SimpleNamespace(returncode=0, stdout="listed", stderr="")

    monkeypatch.setattr(forge_module.subprocess, "run", fake)

    code, output = forge_module.run_forge("gh", ["pr", "list"], session)

    assert (code, output) == (0, "listed")
    command, kwargs = calls[0]
    assert command == ["gh", "pr", "list"]
    assert kwargs["cwd"] == str(session.workspace_root)
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert "shell" not in kwargs


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_forge_tool_schemas():
    assert tools.get_tool("forge").writes is False
    assert tools.get_tool("forge").runs is False
    assert tools.get_tool("forge").needs_approval is False

    assert tools.get_tool("forge_write").writes is True
    assert tools.get_tool("forge_write").needs_approval is True

    read_only_names = {
        schema["function"]["name"] for schema in tools.tool_schemas(read_only=True)
    }
    assert "forge" in read_only_names
    assert "forge_write" not in read_only_names


def test_forge_tool_without_remote(session, monkeypatch):
    _patch_forge(monkeypatch, remote=None)

    result = _forge({"action": "pr_list"}, session)

    assert not result.ok
    assert "remote 'origin'" in result.content


def test_forge_tool_rejects_unknown_host(session, monkeypatch):
    _patch_forge(monkeypatch, remote="https://bitbucket.org/owner/repo.git")

    result = _forge({"action": "pr_list"}, session)

    assert not result.ok
    assert "not a supported forge" in result.content


def test_forge_tool_requires_cli(session, monkeypatch):
    _patch_forge(monkeypatch, available=False)

    result = _forge({"action": "pr_list"}, session)

    assert not result.ok
    assert "'gh' CLI is required" in result.content


def test_forge_tool_pr_list_uses_gh(session, monkeypatch):
    calls = _patch_forge(monkeypatch, result=(0, "1  open  Fix bug"))

    result = _forge({"action": "pr_list"}, session)

    assert result.ok
    assert calls == [("gh", ["pr", "list"], 60)]
    assert "Fix bug" in result.content


def test_forge_tool_gitlab_uses_glab_mr(session, monkeypatch):
    calls = _patch_forge(monkeypatch, remote=REMOTE_GITLAB)

    _forge({"action": "pr_view", "number": 3}, session)

    assert calls == [("glab", ["mr", "view", "3"], 60)]


def test_forge_tool_view_requires_number(session, monkeypatch):
    calls = _patch_forge(monkeypatch)

    result = _forge({"action": "pr_view"}, session)

    assert not result.ok
    assert "'number' must be an integer" in result.content
    assert calls == []


def test_forge_tool_issue_view_requires_positive_number(session, monkeypatch):
    _patch_forge(monkeypatch)

    result = _forge({"action": "issue_view", "number": 0}, session)

    assert not result.ok
    assert "positive" in result.content


def test_forge_tool_rejects_unknown_action(session, monkeypatch):
    _patch_forge(monkeypatch)

    result = _forge({"action": "pr_merge"}, session)

    assert not result.ok
    assert "Unknown forge action" in result.content


def test_forge_write_creates_pr(session, monkeypatch):
    calls = _patch_forge(monkeypatch)

    result = _forge_write(
        {"action": "pr_create", "title": "Add x", "body": "why", "base": "main"},
        session,
    )

    assert result.ok
    assert calls == [
        (
            "gh",
            ["pr", "create", "--title", "Add x", "--body", "why", "--base", "main"],
            60,
        )
    ]


def test_forge_write_requires_title(session, monkeypatch):
    calls = _patch_forge(monkeypatch)

    result = _forge_write({"action": "issue_create", "body": "no title"}, session)

    assert not result.ok
    assert "title" in result.content
    assert calls == []


def test_forge_write_preview_is_wired(session, monkeypatch):
    _patch_forge(monkeypatch)

    text = tools.preview(
        "forge_write",
        {"action": "pr_create", "title": "Add x", "base": "main"},
        session,
    )

    assert text == "gh pr create --title 'Add x' --body '' --base main"
