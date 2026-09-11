from __future__ import annotations

from pathlib import Path

from kiwimatecoder.instructions import instructions_section, load_instructions
from kiwimatecoder.prompts import build_system_prompt
from kiwimatecoder.session import Session


def test_load_instructions_reads_agents_md(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# Rules\nRun pytest.")

    loaded = load_instructions(tmp_path)

    assert loaded[0][0] == "AGENTS.md"
    assert "Run pytest." in loaded[0][1]


def test_load_instructions_missing_is_empty(tmp_path):
    assert load_instructions(tmp_path) == []


def test_load_instructions_prefers_agents_over_claude(tmp_path):
    (tmp_path / "AGENTS.md").write_text("from agents")
    (tmp_path / "CLAUDE.md").write_text("from claude")

    loaded = load_instructions(tmp_path)

    assert [name for name, _ in loaded] == ["AGENTS.md", "CLAUDE.md"]


def test_load_instructions_skips_binary(tmp_path):
    (tmp_path / "AGENTS.md").write_bytes(b"\x00\x01binary")

    assert load_instructions(tmp_path) == []


def test_load_instructions_enforces_budget(tmp_path):
    (tmp_path / "AGENTS.md").write_text("x" * 500)

    loaded = load_instructions(tmp_path, total_budget=100)

    assert len(loaded) == 1
    assert "truncated" in loaded[0][1]


def test_instructions_section_renders_marker(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Use tabs.")

    section = instructions_section(tmp_path)

    assert 'path="AGENTS.md"' in section
    assert "Use tabs." in section


def test_build_system_prompt_includes_project_instructions(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Always run ruff.")
    session = Session(provider_id="openrouter", model="m", workspace_root=tmp_path)

    prompt = build_system_prompt(session)["content"]

    assert "Always run ruff." in prompt
    assert "Project instructions follow" in prompt


def test_build_system_prompt_applies_output_style_and_custom_prompt(tmp_path):
    session = Session(
        provider_id="openrouter",
        model="m",
        workspace_root=tmp_path,
        output_style="concise",
        custom_system_prompt="Use British spelling.",
    )

    prompt = build_system_prompt(session)["content"]

    assert "Be terse" in prompt
    assert "Use British spelling." in prompt


def test_build_system_prompt_default_style_has_no_style_block(tmp_path):
    session = Session(
        provider_id="openrouter",
        model="m",
        workspace_root=Path(tmp_path),
    )

    prompt = build_system_prompt(session)["content"]

    assert "Output style:" not in prompt
