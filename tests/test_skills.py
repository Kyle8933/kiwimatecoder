from __future__ import annotations

import pytest

from kiwimatecoder import config, tools
from kiwimatecoder.skills import (
    MAX_LISTING_BYTES,
    MAX_SKILL_BYTES,
    discover_skills,
    load_skill,
    skills_section,
)

BODY_SENTINEL = "SECRET-SKILL-BODY-CONTENT"


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "CONFIG_FILE", config_dir / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", config_dir / "config")
    return tmp_path


def _write_skill(skills_dir, name, text):
    path = skills_dir / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_discover_skills_sorted_and_deduplicated(tmp_path):
    workspace = tmp_path / "ws"
    workspace_skills = workspace / ".kiwimatecoder" / "skills"
    user_skills = tmp_path / "config" / "skills"
    _write_skill(user_skills, "alpha", "# Alpha skill")
    _write_skill(workspace_skills, "beta", "# Beta skill")
    _write_skill(workspace_skills, "alpha", "# Workspace alpha")
    _write_skill(user_skills, "gamma", "# Gamma skill")

    found = discover_skills(workspace)

    assert [skill.name for skill in found] == ["alpha", "beta", "gamma"]
    assert found[0].description == "Workspace alpha"


def test_bare_skill_md_is_ignored(tmp_path):
    workspace = tmp_path / "ws"
    skills_dir = workspace / ".kiwimatecoder" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("# Not a skill directory")
    _write_skill(skills_dir, "real", "# Real skill")

    found = discover_skills(workspace)

    assert [skill.name for skill in found] == ["real"]


def test_description_strips_hashes_and_falls_back(tmp_path):
    workspace = tmp_path / "ws"
    skills_dir = workspace / ".kiwimatecoder" / "skills"
    _write_skill(skills_dir, "hashed", "\n\n## Do the thing\nBody")
    _write_skill(skills_dir, "blank", "\n\n")
    _write_skill(skills_dir, "long", "x" * 500)

    found = {skill.name: skill for skill in discover_skills(workspace)}

    assert found["hashed"].description == "Do the thing"
    assert found["blank"].description == "blank"
    assert len(found["long"].description) == 200


def test_load_skill_caps_and_notes_truncation(tmp_path):
    workspace = tmp_path / "ws"
    _write_skill(
        workspace / ".kiwimatecoder" / "skills",
        "big",
        "A" * (MAX_SKILL_BYTES + 500),
    )

    content = load_skill("big", workspace)

    assert content is not None
    assert content.startswith("A" * 100)
    assert "[truncated" in content
    assert str(MAX_SKILL_BYTES) in content


def test_load_skill_unknown_returns_none(tmp_path):
    workspace = tmp_path / "ws"
    _write_skill(workspace / ".kiwimatecoder" / "skills", "known", "# Known")

    assert load_skill("missing", workspace) is None
    assert load_skill("", workspace) is None


def test_binary_skill_is_skipped(tmp_path):
    workspace = tmp_path / "ws"
    path = workspace / ".kiwimatecoder" / "skills" / "bin" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\x00\x01binary")

    assert discover_skills(workspace) == []
    assert load_skill("bin", workspace) is None


def test_load_skill_tool_returns_content(session):
    _write_skill(
        session.workspace_root / ".kiwimatecoder" / "skills",
        "pdf",
        "# PDF\nUse pypdf.\n\n" + BODY_SENTINEL,
    )

    result = tools.dispatch("load_skill", {"name": "pdf"}, session)

    assert result.ok
    assert BODY_SENTINEL in result.content


def test_load_skill_tool_reports_unknown(session):
    _write_skill(session.workspace_root / ".kiwimatecoder" / "skills", "pdf", "# PDF")

    result = tools.dispatch("load_skill", {"name": "nope"}, session)
    blank = tools.dispatch("load_skill", {}, session)

    assert not result.ok
    assert "Unknown skill 'nope'" in result.content
    assert not blank.ok
    assert "'name' is required" in blank.content


def test_load_skill_tool_is_read_only_and_advertised():
    tool = tools.get_tool("load_skill")

    assert tool is not None
    assert tool.needs_approval is False
    read_only_names = {
        schema["function"]["name"] for schema in tools.tool_schemas(read_only=True)
    }
    assert "load_skill" in read_only_names
    assert "load_skill" in tools.tool_sources()["builtin"]


def test_skills_section_lists_names_and_descriptions_only(tmp_path):
    workspace = tmp_path / "ws"
    _write_skill(
        workspace / ".kiwimatecoder" / "skills",
        "pdf",
        f"# PDF Handling\nUse pypdf, never shell out.\n\n{BODY_SENTINEL}",
    )

    section = skills_section(workspace)

    assert "Available skills" in section
    assert "load_skill(name)" in section
    assert "- pdf: PDF Handling" in section
    assert BODY_SENTINEL not in section


def test_skills_section_caps_the_listing(tmp_path):
    workspace = tmp_path / "ws"
    skills_dir = workspace / ".kiwimatecoder" / "skills"
    for index in range(40):
        _write_skill(skills_dir, f"skill-{index:02d}", f"# {'d' * 150}")

    section = skills_section(workspace)

    assert "more skill(s) omitted" in section
    assert len(section.encode("utf-8")) <= MAX_LISTING_BYTES + 200


def test_skills_section_empty_without_skills(tmp_path):
    assert skills_section(tmp_path) == ""
