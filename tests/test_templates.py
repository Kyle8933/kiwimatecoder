from __future__ import annotations

import pytest

from kiwimatecoder import config
from kiwimatecoder.templates import (
    ARGUMENTS_TOKEN,
    MAX_TEMPLATE_BYTES,
    discover_templates,
    find_template,
    render_template,
)


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "CONFIG_FILE", config_dir / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", config_dir / "config")
    return tmp_path


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_discover_templates_from_workspace_and_user(tmp_path):
    workspace = tmp_path / "ws"
    _write(workspace / ".kiwimatecoder" / "commands" / "review.md", "# Review\nLook closely.")
    _write(tmp_path / "config" / "commands" / "explain.md", "Explain $ARGUMENTS")

    found = discover_templates(workspace)

    assert list(found) == ["explain", "review"]
    assert found["review"].description == "Review"
    assert found["explain"].body == "Explain $ARGUMENTS"


def test_workspace_template_wins_name_collision(tmp_path):
    workspace = tmp_path / "ws"
    _write(workspace / ".kiwimatecoder" / "commands" / "review.md", "from workspace")
    _write(tmp_path / "config" / "commands" / "review.md", "from user")

    found = discover_templates(workspace)

    assert found["review"].body == "from workspace"
    assert str(found["review"].path).startswith(str(workspace))


def test_invalid_names_are_skipped(tmp_path):
    workspace = tmp_path / "ws"
    commands = workspace / ".kiwimatecoder" / "commands"
    _write(commands / "bad name.md", "space in name")
    _write(commands / "!bang.md", "leading punctuation")
    _write(commands / ("x" * 41 + ".md"), "too long")
    _write(commands / "Good-Name.md", "valid after lowercasing")

    found = discover_templates(workspace)

    assert list(found) == ["good-name"]


def test_name_uppercase_is_normalized(tmp_path):
    workspace = tmp_path / "ws"
    _write(workspace / ".kiwimatecoder" / "commands" / "Review.md", "hi")

    assert find_template("review", workspace) is not None
    assert find_template("REVIEW", workspace) is not None


def test_description_extraction(tmp_path):
    workspace = tmp_path / "ws"
    commands = workspace / ".kiwimatecoder" / "commands"
    _write(commands / "hashed.md", "\n\n## Heading here\nBody")
    _write(commands / "empty-heading.md", "###\n\nreal line")
    _write(commands / "blank.md", "\n\n")

    found = discover_templates(workspace)

    assert found["hashed"].description == "Heading here"
    assert found["empty-heading"].description == "real line"
    assert found["blank"].description == "blank"


def test_body_is_capped_at_16kb(tmp_path):
    workspace = tmp_path / "ws"
    _write(
        workspace / ".kiwimatecoder" / "commands" / "big.md",
        "A" * (MAX_TEMPLATE_BYTES + 1000),
    )

    found = discover_templates(workspace)

    assert len(found["big"].body) == MAX_TEMPLATE_BYTES


def test_binary_template_is_skipped(tmp_path):
    workspace = tmp_path / "ws"
    path = workspace / ".kiwimatecoder" / "commands" / "bin.md"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\x00\x01binary")

    assert discover_templates(workspace) == {}


def test_render_template_replaces_every_arguments_token():
    body = f"Do {ARGUMENTS_TOKEN} then {ARGUMENTS_TOKEN}."

    assert render_template(body, "this") == "Do this then this."
    assert render_template(body, "") == "Do  then ."


def test_render_template_appends_arguments_when_missing():
    assert render_template("Do the thing.", "quickly") == (
        "Do the thing.\n\nArguments: quickly"
    )
    assert render_template("Do the thing.", "   ") == "Do the thing."


def test_find_template_unknown_returns_none(tmp_path):
    assert find_template("nope", tmp_path) is None
    assert find_template("", tmp_path) is None
