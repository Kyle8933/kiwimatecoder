from __future__ import annotations

import io

import pytest

from kiwimatecoder import config, ui


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)


# ---------------------------------------------------------------------------
# Color resolution
# ---------------------------------------------------------------------------


def test_resolve_color_defaults_to_true():
    assert ui.resolve_color() is True


def test_resolve_color_no_color_env_disables(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    assert ui.resolve_color() is False


def test_resolve_color_empty_no_color_is_ignored(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "")

    assert ui.resolve_color() is True


def test_resolve_color_config_always_beats_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    assert ui.resolve_color({"ui": {"color": "always"}}) is True


def test_resolve_color_force_color_enables(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")

    assert ui.resolve_color() is True


def test_resolve_color_config_never_beats_force_color(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")

    assert ui.resolve_color({"ui": {"color": "never"}}) is False


# ---------------------------------------------------------------------------
# Console construction
# ---------------------------------------------------------------------------


def test_make_console_defaults_to_color():
    console = ui.make_console(file=io.StringIO(), force_terminal=True)

    assert console.no_color is False


def test_make_console_no_color_env(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    console = ui.make_console(file=io.StringIO(), force_terminal=True)

    assert console.no_color is True


def test_make_console_accepts_explicit_override(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    console = ui.make_console(
        file=io.StringIO(), force_terminal=True, no_color=False
    )

    assert console.no_color is False


# ---------------------------------------------------------------------------
# Glyphs and themes
# ---------------------------------------------------------------------------


def test_glyph_unicode_defaults():
    assert ui.glyph("check") == "✓"
    assert ui.glyph("cross") == "✗"
    assert ui.glyph("blocked") == "⊘"
    assert ui.glyph("folder") == "📁"
    assert ui.glyph("bullet") == "•"


def test_glyph_ascii_map():
    cfg = {"ui": {"ascii": True}}

    assert ui.glyph("check", cfg) == "[ok]"
    assert ui.glyph("cross", cfg) == "[fail]"
    assert ui.glyph("blocked", cfg) == "[blocked]"
    assert ui.glyph("folder", cfg) == ""
    assert ui.glyph("bullet", cfg) == "-"


def test_glyph_unknown_name_falls_back_to_name():
    assert ui.glyph("mystery") == "mystery"
    assert ui.glyph("mystery", {"ui": {"ascii": True}}) == "mystery"


def test_glyphs_returns_active_map():
    assert ui.glyphs() == ui.UNICODE_GLYPHS
    assert ui.glyphs({"ui": {"ascii": True}}) == ui.ASCII_GLYPHS


def test_glyphs_returns_a_copy():
    active = ui.glyphs()
    active["check"] = "mutated"

    assert ui.glyph("check") == "✓"


def test_theme_accent_mapping():
    assert ui.theme_accent() == "green"
    assert ui.theme_accent({"ui": {"theme": "ocean"}}) == "cyan"
    assert ui.theme_accent({"ui": {"theme": "magenta"}}) == "magenta"
    assert ui.theme_accent({"ui": {"theme": "mono"}}) == "white"
    assert ui.theme_accent({"ui": {"theme": "neon"}}) == "green"


# ---------------------------------------------------------------------------
# Config roundtrip and tolerance
# ---------------------------------------------------------------------------


def test_get_ui_defaults_when_unset():
    assert config.get_ui() == ui.UI_DEFAULTS


def test_set_ui_roundtrip():
    updated = config.set_ui(
        color="never", output_mode="compact", ascii=True, theme="ocean", locale="de"
    )

    assert updated == {
        "color": "never",
        "output_mode": "compact",
        "ascii": True,
        "theme": "ocean",
        "locale": "de",
        "keybindings": "emacs",
        "notify": "off",
        "notify_after_seconds": 20,
        "spinner": "auto",
    }
    assert config.get_ui() == updated
    assert config.load_config()["ui"] == updated


def test_set_ui_omitted_values_are_unchanged():
    config.set_ui(theme="mono")

    assert config.set_ui(color="always")["theme"] == "mono"


def test_set_ui_rejects_invalid_values():
    with pytest.raises(ValueError):
        config.set_ui(color="rainbow")
    with pytest.raises(ValueError):
        config.set_ui(output_mode="loud")
    with pytest.raises(ValueError):
        config.set_ui(theme="neon")
    with pytest.raises(ValueError):
        config.set_ui(ascii="yes")  # type: ignore[arg-type]

    assert config.get_ui() == ui.UI_DEFAULTS


def test_get_ui_falls_back_on_bad_stored_values():
    cfg = config.load_config()
    cfg["ui"] = {
        "color": "rainbow",
        "output_mode": "loud",
        "ascii": "yes",
        "theme": "neon",
    }
    config.save_config(cfg)

    assert config.get_ui() == ui.UI_DEFAULTS


def test_get_ui_tolerates_corrupt_section():
    cfg = config.load_config()
    cfg["ui"] = "garbage"
    config.save_config(cfg)

    assert config.get_ui() == ui.UI_DEFAULTS


# ---------------------------------------------------------------------------
# Keybindings, notifications, and locale
# ---------------------------------------------------------------------------


def test_set_ui_keybindings_and_notify_roundtrip():
    updated = config.set_ui(
        keybindings="vim", notify="bell", notify_after_seconds=5
    )

    assert updated["keybindings"] == "vim"
    assert updated["notify"] == "bell"
    assert updated["notify_after_seconds"] == 5
    assert config.get_ui()["keybindings"] == "vim"


def test_set_ui_rejects_invalid_keybindings_and_notify():
    with pytest.raises(ValueError):
        config.set_ui(keybindings="nano")
    with pytest.raises(ValueError):
        config.set_ui(notify="loud")
    with pytest.raises(ValueError):
        config.set_ui(notify_after_seconds=-1)
    with pytest.raises(ValueError):
        config.set_ui(notify_after_seconds="soon")

    assert config.get_ui() == ui.UI_DEFAULTS


def test_get_ui_falls_back_on_bad_keybindings_and_notify():
    cfg = config.load_config()
    cfg["ui"] = {
        "keybindings": "nano",
        "notify": "loud",
        "notify_after_seconds": -5,
    }
    config.save_config(cfg)

    effective = config.get_ui()
    assert effective["keybindings"] == "emacs"
    assert effective["notify"] == "off"
    assert effective["notify_after_seconds"] == 20


def test_vi_mode_selection_helper():
    assert ui.vi_mode_enabled() is False
    assert ui.vi_mode_enabled({"ui": {"keybindings": "vim"}}) is True
    assert ui.vi_mode_enabled({"ui": {"keybindings": "emacs"}}) is False
    assert ui.vi_mode_enabled({"ui": {"keybindings": "nano"}}) is False


def test_spinner_config_and_helper():
    assert ui.spinner_enabled() is True
    assert ui.spinner_enabled({"ui": {"spinner": "auto"}}) is True
    assert ui.spinner_enabled({"ui": {"spinner": "on"}}) is True
    assert ui.spinner_enabled({"ui": {"spinner": "off"}}) is False

    updated = config.set_ui(spinner="off")
    assert updated["spinner"] == "off"
    assert config.get_ui()["spinner"] == "off"

    with pytest.raises(ValueError):
        config.set_ui(spinner="sometimes")

    cfg = config.load_config()
    cfg["ui"] = {"spinner": "sometimes"}
    config.save_config(cfg)
    assert config.get_ui()["spinner"] == "auto"


def test_apply_plain_mode_runtime_overrides():
    ui.apply_plain_mode(True)
    try:
        assert ui.runtime_overrides()["ascii"] is True
        assert ui.glyph("check") == "[ok]"
        assert ui.resolve_color() is False
        assert ui.spinner_enabled() is False
    finally:
        ui.apply_plain_mode(False)

    assert ui.runtime_overrides() == {}
    assert ui.glyph("check") == "✓"
