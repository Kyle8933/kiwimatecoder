from __future__ import annotations

import io

import pytest
from rich.console import Console
from typer.testing import CliRunner

from kiwimatecoder import config, i18n, main
from kiwimatecoder.commands import dispatch
from kiwimatecoder.repl import _banner

LOCALE_KEYS = set(i18n._CATALOGS["en"])


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    monkeypatch.delenv(i18n.LOCALE_ENV, raising=False)
    i18n.set_locale(i18n.DEFAULT_LOCALE)
    yield
    i18n.set_locale(i18n.DEFAULT_LOCALE)


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def _output(console: Console) -> str:
    return console.file.getvalue()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Catalog basics
# ---------------------------------------------------------------------------


def test_available_locales_and_default():
    assert i18n.available_locales() == ("en", "de", "es")
    assert i18n.DEFAULT_LOCALE == "en"
    assert i18n.get_locale() == "en"


def test_set_locale_normalizes_and_rejects_invalid():
    assert i18n.set_locale("de-DE") == "de"
    assert i18n.get_locale() == "de"
    assert i18n.set_locale("fr") == "en"
    assert i18n.get_locale() == "en"


def test_t_substitutes_format_values():
    assert i18n.t("error.unknown_command", name="costs") == (
        "Unknown command '/costs'. Try /help."
    )
    i18n.set_locale("de")
    assert i18n.t("error.unknown_command", name="costs") == (
        "Unbekannter Befehl '/costs'. Versuche /help."
    )


def test_t_falls_back_to_english_then_key(monkeypatch):
    monkeypatch.delitem(i18n._CATALOGS["de"], "banner.help_hint")
    i18n.set_locale("de")

    assert i18n.t("banner.help_hint") == i18n._CATALOGS["en"]["banner.help_hint"]
    assert i18n.t("no.such.key") == "no.such.key"


def test_t_returns_template_on_bad_format_arguments():
    assert i18n.t("error.unknown_command", wrong="x") == (
        "Unknown command '/{name}'. Try /help."
    )


def test_german_and_spanish_cover_every_english_key():
    assert set(i18n._CATALOGS["de"]) == LOCALE_KEYS
    assert set(i18n._CATALOGS["es"]) == LOCALE_KEYS


# ---------------------------------------------------------------------------
# Locale resolution precedence
# ---------------------------------------------------------------------------


def test_resolve_locale_prefers_config_then_env_then_default(monkeypatch):
    assert i18n.resolve_locale({"ui": {}}) == "en"

    monkeypatch.setenv(i18n.LOCALE_ENV, "es")
    assert i18n.resolve_locale({"ui": {}}) == "es"
    assert i18n.resolve_locale({"ui": {"locale": "de"}}) == "de"

    # An invalid explicit value is ignored and the env (then default) wins.
    assert i18n.resolve_locale({"ui": {"locale": "fr"}}) == "es"
    monkeypatch.setenv(i18n.LOCALE_ENV, "kl")
    assert i18n.resolve_locale({"ui": {"locale": "fr"}}) == "en"


def test_apply_config_locale_sets_the_global_locale():
    config.set_ui(locale="es")

    assert i18n.apply_config_locale() == "es"
    assert i18n.get_locale() == "es"


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


def test_get_ui_locale_defaults_and_normalizes():
    assert config.get_ui()["locale"] == "en"

    cfg = config.load_config()
    cfg["ui"] = {"locale": "de-DE"}
    config.save_config(cfg)

    assert config.get_ui()["locale"] == "de"


def test_set_ui_locale_roundtrip_and_rejection():
    updated = config.set_ui(locale="de")

    assert updated["locale"] == "de"
    assert config.get_ui()["locale"] == "de"

    with pytest.raises(ValueError):
        config.set_ui(locale="fr")


def test_validate_config_flags_unknown_locale():
    cfg = config.load_config()
    cfg["ui"] = {"locale": "kl"}

    issues = config.validate_config(cfg)

    assert any(
        issue["level"] == "error" and issue["key"] == "ui.locale"
        for issue in issues
    )


def test_config_ui_locale_cli_roundtrip_and_rejection():
    runner = CliRunner()

    result = runner.invoke(main.app, ["config", "ui", "locale", "de"])
    assert result.exit_code == 0
    assert config.get_ui()["locale"] == "de"

    result = runner.invoke(main.app, ["config", "ui", "locale", "fr"])
    assert result.exit_code == 1
    assert config.get_ui()["locale"] == "de"


def test_config_ui_locale_slash_command(session):
    console = _console()

    dispatch("/config ui locale es", session, console)
    assert config.get_ui()["locale"] == "es"
    assert i18n.get_locale() == "es"

    dispatch("/config ui locale fr", session, console)
    assert config.get_ui()["locale"] == "es"
    assert "Unknown locale" in _output(console)


# ---------------------------------------------------------------------------
# Translated rendering
# ---------------------------------------------------------------------------


def test_banner_help_hint_translates(session):
    i18n.set_locale("de")

    rendered = str(_banner(session).renderable)

    assert "Tippe /help" in rendered
    assert "Type /help" not in rendered


def test_help_group_titles_translate(session):
    i18n.set_locale("de")
    console = _console()

    dispatch("/help", session, console)

    output = _output(console)
    assert "Sitzung" in output
    assert "Konfiguration" in output


def test_unknown_command_error_translates(session):
    i18n.set_locale("es")
    console = _console()

    dispatch("/no-such-thing", session, console)

    assert "Comando desconocido" in _output(console)
