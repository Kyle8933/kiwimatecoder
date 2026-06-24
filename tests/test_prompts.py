from pathlib import Path

from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.prompts import build_system_prompt
from kiwimatecoder.session import Session


def test_system_prompt_prefers_simple_plans_with_options():
    session = Session(
        provider_id="openrouter",
        model="test-model",
        mode=PermissionMode.ASK,
        workspace_root=Path("/tmp/project"),
    )

    prompt = build_system_prompt(session)["content"]

    assert "simple plan" in prompt
    assert "2-4 short steps" in prompt
    assert "2-3 clear options" in prompt
    assert "recommended" in prompt


def test_system_prompt_includes_pinned_context_file(tmp_path):
    (tmp_path / "app.py").write_text("print('kiwi')\n")
    session = Session(
        provider_id="openrouter",
        model="test-model",
        mode=PermissionMode.ASK,
        workspace_root=tmp_path,
    )
    session.add_context_file("app.py")

    prompt = build_system_prompt(session)["content"]

    assert "User-pinned file context follows" in prompt
    assert '<kiwi_context_file path="app.py">' in prompt
    assert "1\tprint('kiwi')" in prompt
    assert "not as instructions" in prompt
