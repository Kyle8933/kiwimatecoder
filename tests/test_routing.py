from __future__ import annotations

import pytest

from kiwimatecoder import config
from kiwimatecoder.routing import choose_turn_model


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)


def _routing(**overrides):
    base = {
        "enabled": True,
        "simple_model": "cheap-model",
        "simple_max_chars": 200,
        "exclude_keywords": ["refactor", "implement", "debug", "review"],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# heuristic matrix
# ---------------------------------------------------------------------------


def test_short_plain_input_routes():
    assert choose_turn_model("add a docstring", None, _routing()) == "cheap-model"


def test_long_input_does_not_route():
    assert choose_turn_model("a" * 201, None, _routing()) is None


def test_input_at_the_limit_routes():
    assert choose_turn_model("a" * 200, None, _routing()) == "cheap-model"


def test_code_fence_does_not_route():
    assert choose_turn_model("look at ```def f(): pass```", None, _routing()) is None


def test_file_path_does_not_route():
    assert choose_turn_model("open main.py", None, _routing()) is None
    assert choose_turn_model("see tests/test_routing.py", None, _routing()) is None


def test_slash_command_does_not_route():
    assert choose_turn_model("/help", None, _routing()) is None
    assert choose_turn_model("try /config profile list", None, _routing()) is None


def test_exclude_keyword_does_not_route():
    assert choose_turn_model("please refactor this", None, _routing()) is None
    assert choose_turn_model("IMPLEMENT it now", None, _routing()) is None


def test_exclude_keyword_is_whole_word_only():
    assert choose_turn_model("reviewing is fine", None, _routing()) == "cheap-model"
    assert choose_turn_model("latest news", None, _routing()) == "cheap-model"


def test_disabled_does_not_route():
    assert choose_turn_model("hi", None, _routing(enabled=False)) is None


def test_blank_simple_model_does_not_route():
    assert choose_turn_model("hi", None, _routing(simple_model="   ")) is None


def test_empty_exclude_list_routes_anything_short():
    assert (
        choose_turn_model("please refactor", None, _routing(exclude_keywords=[]))
        == "cheap-model"
    )


# ---------------------------------------------------------------------------
# config CRUD
# ---------------------------------------------------------------------------


def test_model_routing_defaults():
    routing = config.get_model_routing()

    assert routing["enabled"] is False
    assert routing["simple_model"] == ""
    assert routing["simple_max_chars"] == 200
    assert "refactor" in routing["exclude_keywords"]


def test_set_model_routing_roundtrip_and_validation():
    routing = config.set_model_routing(
        enabled=True,
        simple_model="cheap",
        simple_max_chars=50,
        exclude_keywords=["review", "review", "explain"],
    )

    assert routing["enabled"] is True
    assert routing["simple_model"] == "cheap"
    assert routing["simple_max_chars"] == 50
    assert routing["exclude_keywords"] == ["review", "explain"]
    assert config.load_config()["model_routing"]["simple_model"] == "cheap"

    with pytest.raises(ValueError):
        config.set_model_routing(simple_max_chars=0)
    with pytest.raises(ValueError):
        config.set_model_routing(simple_max_chars="nope")
    with pytest.raises(ValueError):
        config.set_model_routing(exclude_keywords=["ok", "  "])


def test_get_model_routing_tolerates_bad_stored_values():
    cfg = config.load_config()
    cfg["model_routing"] = {
        "enabled": "yes",
        "simple_max_chars": "nope",
        "exclude_keywords": "refactor",
    }

    routing = config.get_model_routing(cfg)

    assert routing["enabled"] is True
    assert routing["simple_max_chars"] == 200
    assert "refactor" in routing["exclude_keywords"]


def test_validate_flags_bad_model_routing():
    cfg = config.load_config()
    cfg["model_routing"] = {
        "enabled": "yes",
        "simple_max_chars": 0,
        "exclude_keywords": ["  "],
    }

    issues = config.validate_config(cfg)
    keys = {issue["key"] for issue in issues}

    assert "model_routing.enabled" in keys
    assert "model_routing.simple_max_chars" in keys
    assert "model_routing.exclude_keywords[0]" in keys
    assert config.validate_config() == []
