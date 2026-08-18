from __future__ import annotations

from pathlib import Path

from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.session import (
    Session,
    list_saved_sessions,
    load_session,
    save_session,
)


def test_session_serialization_roundtrip(tmp_path):
    sess = Session(
        provider_id="openai",
        model="gpt-5.6-sol",
        mode=PermissionMode.AUTO,
        workspace_root=tmp_path,
        messages=[{"role": "user", "content": "hello"}],
        prompt_tokens=100,
        completion_tokens=50,
        touched_files=["main.py"],
        context_files=["README.md"],
        active_provider_ids=["openai", "openrouter"],
        models={"openrouter": "anthropic/claude-sonnet-5"},
    )

    data = sess.to_dict()
    restored = Session.from_dict(data)

    assert restored.provider_id == sess.provider_id
    assert restored.model == sess.model
    assert restored.mode == sess.mode
    assert restored.messages == sess.messages
    assert restored.prompt_tokens == 100
    assert restored.completion_tokens == 50
    assert restored.total_tokens == 150
    assert restored.touched_files == ["main.py"]
    assert restored.context_files == ["README.md"]
    assert restored.active_provider_ids == ["openai", "openrouter"]
    assert restored.models == {"openrouter": "anthropic/claude-sonnet-5"}


def test_session_from_dict_seeds_roster_from_provider_id():
    restored = Session.from_dict(
        {
            "provider_id": "openai",
            "model": "gpt-5.6-sol",
            "mode": "ask",
            "workspace_root": ".",
        }
    )

    assert restored.active_provider_ids == ["openai"]
    assert restored.models == {}


def test_legacy_session_uses_its_own_provider_not_live_config(tmp_path, monkeypatch):
    """A pre-feature session must not pick up today's config roster."""
    from kiwimatecoder import config

    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    config.set_active_providers(["openrouter", "anthropic"])

    restored = Session.from_dict(
        {
            "provider_id": "openai",
            "model": "gpt-5.6-sol",
            "mode": "ask",
            "workspace_root": ".",
        }
    )

    assert [provider.id for provider in restored.active_providers] == ["openai"]
    assert restored.model_for(restored.active_providers[0].id) == "gpt-5.6-sol"


def test_set_active_providers_keeps_model_when_primary_unchanged():
    sess = Session(provider_id="openrouter", model="my-custom-model")
    sess.allow_always("run_bash")

    sess.set_active_providers(["openrouter", "openai"])

    assert sess.provider_id == "openrouter"
    assert sess.model == "my-custom-model"
    assert sess.is_always_allowed("run_bash")
    assert sess.active_provider_ids == ["openrouter", "openai"]


def test_set_active_providers_switches_model_when_primary_changes():
    sess = Session(provider_id="openrouter", model="my-custom-model")
    sess.allow_always("run_bash")

    sess.set_active_providers(["openai", "openrouter"])

    assert sess.provider_id == "openai"
    assert sess.model != "my-custom-model"
    assert not sess.is_always_allowed("run_bash")


def test_session_save_and_load(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    monkeypatch.setattr(
        "kiwimatecoder.session._sessions_dir", lambda: sessions_dir
    )

    sess = Session(
        provider_id="anthropic",
        model="claude-sonnet-5",
        mode=PermissionMode.ASK,
        workspace_root=tmp_path,
        messages=[{"role": "user", "content": "Write tests"}],
    )

    saved_file = save_session(sess, "test_proj")
    assert saved_file.is_file()
    assert saved_file.name == "test_proj.json"

    saved_list = list_saved_sessions()
    assert len(saved_list) == 1
    assert saved_list[0]["name"] == "test_proj"
    assert saved_list[0]["provider"] == "anthropic"

    loaded = load_session("test_proj", workspace_root=tmp_path)
    assert loaded.provider_id == "anthropic"
    assert loaded.model == "claude-sonnet-5"
    assert len(loaded.messages) == 1


def test_session_trim_history():
    sess = Session(
        provider_id="openai",
        model="gpt-4o",
        workspace_root=Path("."),
    )

    # Add 10 turns of long messages
    for i in range(10):
        sess.messages.append(
            {"role": "user", "content": f"Turn {i} " + "x" * 2000}
        )
        sess.messages.append(
            {"role": "assistant", "content": f"Ans {i} " + "y" * 2000}
        )

    assert len(sess.messages) == 20
    # Trim with a small budget
    pruned = sess.trim_history(max_tokens=2000)
    assert pruned > 0
    assert len(sess.messages) < 20
    # First message is preserved
    assert "Turn 0" in sess.messages[0]["content"]
    # Last message is preserved
    assert "Ans 9" in sess.messages[-1]["content"]
