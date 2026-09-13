"""Internationalization for user-facing strings (incremental migration).

The catalog covers a representative subset of the CLI/REPL: the banner hint,
prompt and approval labels, common errors, help-group titles, and key CLI
messages. Migration is **incremental** — untranslated strings stay in English
and are wired through :func:`t` as they are touched. English is the reference
catalog; a locale may omit a key, in which case :func:`t` falls back to English
and finally to the key itself.

Locale resolution precedence is:

1. an explicit ``ui.locale`` value in the config,
2. the ``KIWIMATECODER_LANG`` environment variable,
3. :data:`DEFAULT_LOCALE` (``"en"``).

Invalid values are ignored at every step rather than raising.
"""

from __future__ import annotations

import os
from typing import Any

LOCALES: tuple[str, ...] = ("en", "de", "es")
DEFAULT_LOCALE = "en"
LOCALE_ENV = "KIWIMATECODER_LANG"

_EN: dict[str, str] = {
    # Banner and prompts
    "banner.help_hint": (
        "Type /help for commands · Alt+Enter for newline · Ctrl-C cancels · "
        "Ctrl-D exits"
    ),
    "banner.pinned": "{count} pinned",
    "banner.dry_run": "dry-run",
    "prompt.steering": "(steering — Enter to send, Ctrl-C to cancel turn)",
    "prompt.goodbye": "Goodbye!",
    "prompt.interrupted": "Interrupted.",
    "prompt.command_queued": "Command queued; it will run after this turn.",
    "prompt.turn_finished": "Turn finished in {seconds}s",
    # Approvals
    "approval.allow": "Allow?",
    "approval.denied": "Denied.",
    "approval.approve": "Approve: {summary}",
    "approval.approve_change": "Approve Change: {summary}",
    "approval.approve_shell": "Approve Shell: {summary}",
    "approval.always_saved": (
        "'{tool}' will be allowed in future sessions "
        "(remove with /config permissions remove {tool})."
    ),
    "approval.review_hunks": "Review hunks:",
    "approval.apply_hunks": "Apply which hunks? (1,3 / 1-2 / all / none):",
    "approval.retry_selection": "Unrecognized selection — try again.",
    "approval.too_many_attempts": "Too many invalid attempts; denied.",
    # Errors
    "error.unknown_command": "Unknown command '/{name}'. Try /help.",
    "error.no_key": (
        "No API key for {provider}. Set one with "
        "`config set-key --provider {provider_id} <KEY>` or the {env} "
        "environment variable."
    ),
    "error.no_models": "No models are visible for {provider}.",
    "error.no_model_matches": (
        "No models matching '{query}' for {provider} ({provider_id})."
    ),
    "error.unknown_config_action": "Unknown provider config action. Try /config help.",
    "error.unknown_profile": "Unknown profile '{name}'.",
    # Help
    "help.group.session": "Session",
    "help.group.model": "Model & provider",
    "help.group.context": "Context",
    "help.group.configuration": "Configuration",
    "help.group.custom": "Custom commands",
    "help.tip": "Tip: no API key yet? From the shell run `kiwimatecoder setup`.",
    # CLI messages
    "cli.no_tty": (
        'Interactive session needs a TTY. Use: kiwimatecoder -p "..." or '
        "echo ... | kiwimatecoder -p -"
    ),
    "cli.session_resumed": "Resumed session '{name}'",
    # Config
    "config.ui.updated": (
        "UI updated: theme={theme}, color={color}, output={output}, "
        "ascii={ascii}, locale={locale}"
    ),
}

_DE: dict[str, str] = {
    "banner.help_hint": (
        "Tippe /help für Befehle · Alt+Enter für Zeilenumbruch · "
        "Strg-C bricht ab · Strg-D beendet"
    ),
    "banner.pinned": "{count} angeheftet",
    "banner.dry_run": "Probelauf",
    "prompt.steering": (
        "(Steuerung — Enter zum Senden, Strg-C bricht den Zug ab)"
    ),
    "prompt.goodbye": "Auf Wiedersehen!",
    "prompt.interrupted": "Abgebrochen.",
    "prompt.command_queued": "Befehl in Warteschlange; er läuft nach diesem Zug.",
    "prompt.turn_finished": "Zug nach {seconds}s beendet",
    "approval.allow": "Erlauben?",
    "approval.denied": "Abgelehnt.",
    "approval.approve": "Genehmigen: {summary}",
    "approval.approve_change": "Änderung genehmigen: {summary}",
    "approval.approve_shell": "Shell-Befehl genehmigen: {summary}",
    "approval.always_saved": (
        "'{tool}' wird künftig erlaubt "
        "(entfernen mit /config permissions remove {tool})."
    ),
    "approval.review_hunks": "Änderungsblöcke prüfen:",
    "approval.apply_hunks": (
        "Welche Blöcke anwenden? (1,3 / 1-2 / all / none):"
    ),
    "approval.retry_selection": "Auswahl nicht erkannt — bitte erneut versuchen.",
    "approval.too_many_attempts": "Zu viele ungültige Versuche; abgelehnt.",
    "error.unknown_command": "Unbekannter Befehl '/{name}'. Versuche /help.",
    "error.no_key": (
        "Kein API-Schlüssel für {provider}. Setze einen mit "
        "`config set-key --provider {provider_id} <KEY>` oder der "
        "Umgebungsvariablen {env}."
    ),
    "error.no_models": "Keine Modelle für {provider} sichtbar.",
    "error.no_model_matches": (
        "Keine Modelle für '{query}' bei {provider} ({provider_id}) gefunden."
    ),
    "error.unknown_config_action": (
        "Unbekannte Provider-Konfigurationsaktion. Versuche /config help."
    ),
    "error.unknown_profile": "Unbekanntes Profil '{name}'.",
    "help.group.session": "Sitzung",
    "help.group.model": "Modell & Anbieter",
    "help.group.context": "Kontext",
    "help.group.configuration": "Konfiguration",
    "help.group.custom": "Eigene Befehle",
    "help.tip": (
        "Tipp: Noch kein API-Schlüssel? Führe `kiwimatecoder setup` aus."
    ),
    "cli.no_tty": (
        "Interaktive Sitzung benötigt ein TTY. Nutze: kiwimatecoder -p \"...\" "
        "oder echo ... | kiwimatecoder -p -"
    ),
    "cli.session_resumed": "Sitzung '{name}' fortgesetzt",
    "config.ui.updated": (
        "UI aktualisiert: theme={theme}, color={color}, output={output}, "
        "ascii={ascii}, locale={locale}"
    ),
}

_ES: dict[str, str] = {
    "banner.help_hint": (
        "Escribe /help para ver los comandos · Alt+Enter para nueva línea · "
        "Ctrl-C cancela · Ctrl-D sale"
    ),
    "banner.pinned": "{count} fijados",
    "banner.dry_run": "simulación",
    "prompt.steering": (
        "(dirección — Enter para enviar, Ctrl-C para cancelar el turno)"
    ),
    "prompt.goodbye": "¡Adiós!",
    "prompt.interrupted": "Interrumpido.",
    "prompt.command_queued": "Comando en cola; se ejecutará después de este turno.",
    "prompt.turn_finished": "Turno terminado en {seconds}s",
    "approval.allow": "¿Permitir?",
    "approval.denied": "Denegado.",
    "approval.approve": "Aprobar: {summary}",
    "approval.approve_change": "Aprobar cambio: {summary}",
    "approval.approve_shell": "Aprobar comando: {summary}",
    "approval.always_saved": (
        "'{tool}' se permitirá en futuras sesiones "
        "(elimínalo con /config permissions remove {tool})."
    ),
    "approval.review_hunks": "Revisar bloques:",
    "approval.apply_hunks": "¿Qué bloques aplicar? (1,3 / 1-2 / all / none):",
    "approval.retry_selection": "Selección no reconocida — inténtalo de nuevo.",
    "approval.too_many_attempts": "Demasiados intentos no válidos; denegado.",
    "error.unknown_command": "Comando desconocido '/{name}'. Prueba /help.",
    "error.no_key": (
        "No hay clave de API para {provider}. Configura una con "
        "`config set-key --provider {provider_id} <KEY>` o la variable de "
        "entorno {env}."
    ),
    "error.no_models": "No hay modelos visibles para {provider}.",
    "error.no_model_matches": (
        "No hay modelos que coincidan con '{query}' para {provider} "
        "({provider_id})."
    ),
    "error.unknown_config_action": (
        "Acción de configuración de proveedor desconocida. Prueba /config help."
    ),
    "error.unknown_profile": "Perfil desconocido '{name}'.",
    "help.group.session": "Sesión",
    "help.group.model": "Modelo y proveedor",
    "help.group.context": "Contexto",
    "help.group.configuration": "Configuración",
    "help.group.custom": "Comandos personalizados",
    "help.tip": (
        "Consejo: ¿aún no tienes clave de API? Ejecuta `kiwimatecoder setup`."
    ),
    "cli.no_tty": (
        "La sesión interactiva necesita una TTY. Usa: kiwimatecoder -p \"...\" "
        "o echo ... | kiwimatecoder -p -"
    ),
    "cli.session_resumed": "Sesión '{name}' reanudada",
    "config.ui.updated": (
        "UI actualizada: theme={theme}, color={color}, output={output}, "
        "ascii={ascii}, locale={locale}"
    ),
}

_CATALOGS: dict[str, dict[str, str]] = {"en": _EN, "de": _DE, "es": _ES}

_current_locale: str = DEFAULT_LOCALE


def available_locales() -> tuple[str, ...]:
    """Return the locales with a catalog, English first."""
    return LOCALES


def normalize_locale(value: object) -> str | None:
    """Return the supported base locale for ``value``, or None when unknown.

    Accepts ``en``, ``en_US``, and ``en-US`` (case-insensitive).
    """
    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        return None
    base = text.split("_", 1)[0]
    return base if base in LOCALES else None


def resolve_locale(cfg: dict[str, Any] | None = None) -> str:
    """Resolve the locale from config, then the environment, then the default."""
    if cfg is None:
        from kiwimatecoder.config import load_config

        cfg = load_config()
    if isinstance(cfg, dict):
        stored = cfg.get("ui") or {}
        if isinstance(stored, dict):
            explicit = normalize_locale(stored.get("locale"))
            if explicit is not None:
                return explicit
    from_env = normalize_locale(os.environ.get(LOCALE_ENV))
    if from_env is not None:
        return from_env
    return DEFAULT_LOCALE


def set_locale(locale: object) -> str:
    """Set the active locale, falling back to the default for invalid values."""
    global _current_locale
    _current_locale = normalize_locale(locale) or DEFAULT_LOCALE
    return _current_locale


def apply_config_locale(cfg: dict[str, Any] | None = None) -> str:
    """Resolve and activate the locale from config/env; returns the locale."""
    return set_locale(resolve_locale(cfg))


def get_locale() -> str:
    """Return the active locale."""
    return _current_locale


def _lookup(key: str) -> str:
    catalog = _CATALOGS.get(_current_locale, {})
    template = catalog.get(key)
    if template is None:
        template = _CATALOGS[DEFAULT_LOCALE].get(key)
    if template is None:
        return key
    return template


def t(key: str, **kwargs: Any) -> str:
    """Translate ``key`` and substitute ``kwargs`` with ``str.format``.

    Missing keys fall back to English and then to the key itself; malformed
    format values return the raw template rather than raising.
    """
    template = _lookup(key)
    if not kwargs:
        return template
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError, ValueError):
        return template
