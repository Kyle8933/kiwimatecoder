"""The bare ``/config`` menu edits settings instead of just printing them.

Every entry in the menu used to re-run ``/config <section>`` with no action,
which each handler reads as "show". These tests drive the real ``dispatch``
with a scripted selector and scripted typed input.
"""

import io

import pytest
from rich.console import Console

from kiwimatecoder import catalog, config, i18n
from kiwimatecoder.commands import (
    _CONFIG_SECTIONS,
    _MENU_DONE,
    CommandResult,
    MultiSelectionPrompt,
    SelectionPrompt,
    dispatch,
)
from kiwimatecoder.providers import REGISTRY


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=140)


def _output(console: Console) -> str:
    return console.file.getvalue()  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    for provider in REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)
    monkeypatch.delenv("LOCAL_API_KEY", raising=False)
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    # `/config ui locale` switches the process-wide language; put it back so
    # later tests (in any file) still see English.
    i18n.set_locale(i18n.DEFAULT_LOCALE)
    yield
    i18n.set_locale(i18n.DEFAULT_LOCALE)


class Script:
    """A selector that plays back answers and records every prompt it was shown.

    Once the answers run out it returns ``None`` (the user pressing Esc), so a
    menu that loops always terminates instead of hanging the test.
    """

    def __init__(self, *answers: str | None) -> None:
        self._answers = iter(answers)
        self.prompts: list[SelectionPrompt] = []

    def __call__(self, prompt: SelectionPrompt) -> str | None:
        self.prompts.append(prompt)
        return next(self._answers, None)

    def titles(self) -> list[str]:
        return [prompt.title for prompt in self.prompts]

    def options(self, index: int = -1) -> list[str]:
        return [option.value for option in self.prompts[index].options]


class Typed:
    """A ``prompt_input`` that plays back typed lines and records the calls."""

    def __init__(self, *lines: str) -> None:
        self._lines = iter(lines)
        self.calls = 0

    def __call__(self, _message: str) -> str:
        self.calls += 1
        try:
            return next(self._lines)
        except StopIteration:
            raise EOFError from None


def _run(session, selector, typed=None, multi=None, console=None) -> Console:
    console = console or _console()
    result = dispatch(
        "/config",
        session,
        console,
        selector=selector,
        prompt_input=typed,
        multi_selector=multi,
    )
    assert result == CommandResult.CONTINUE
    return console


def _install_fetch(monkeypatch, model_ids):
    def fake_fetch(provider, api_key=None, **kwargs):
        return [
            catalog.RemoteModel(model_id, float(len(model_ids) - index))
            for index, model_id in enumerate(model_ids)
        ]

    monkeypatch.setattr(config.catalog, "fetch_models", fake_fetch)


# ---------------------------------------------------------------------------
# The complaint: entries that only gave information
# ---------------------------------------------------------------------------


def _menu_sections(session) -> list[str]:
    probe = Script(None)
    _run(session, probe)
    return probe.options(0)


def test_menu_lists_every_section_this_suite_knows_about(session):
    sections = set(_menu_sections(session))

    # If a section is added to the menu it must be added to the checks below.
    assert sections == {
        "show", "providers", "keys", "model", "models", "mode", "permissions",
        "commands", "trust", "verify", "budget", "subagents", "sampling", "browser",
        "shell", "sandbox", "remote", "acp", "web", "network", "vision", "media",
        "telemetry", "style", "ui", "prompt", "profile", "team", "cache", "help",
    }


@pytest.mark.parametrize(
    "section",
    [
        "providers", "model", "models", "mode", "commands", "trust", "verify",
        "budget", "subagents", "sampling", "browser", "shell", "sandbox", "remote",
        "acp", "web", "network", "vision", "media", "telemetry", "style", "ui",
        "prompt", "profile", "team", "cache",
    ],
)
def test_picking_a_section_asks_what_to_change(section, session, monkeypatch):
    # An editable entry must ask for something after it is picked, either
    # another choice or typed text; printing a status line is not enough.
    _install_fetch(monkeypatch, ["vendor/one", "vendor/two"])
    config.set_key("openrouter", "sk-test")
    script = Script(section)
    typed = Typed()

    console = _run(session, script, typed)

    assert len(script.prompts) >= 2 or typed.calls >= 1, (
        f"{section!r} only printed: {_output(console)!r}"
    )


def test_permissions_entry_offers_each_approved_tool(session):
    session.always_allowed.update({"run_bash", "write_file"})
    script = Script("permissions")

    _run(session, script)

    assert script.options() == [
        "remove:run_bash", "remove:write_file", "clear", "__done__",
    ]


def test_permissions_entry_with_nothing_approved_explains_how_to_add(session):
    console = _run(session, Script("permissions"))

    assert "No persisted tool approvals" in _output(console)


def test_show_and_help_still_only_print(session):
    for entry, expected in (("show", "Active providers"), ("help", "/config commands")):
        script = Script(entry)
        console = _run(session, script)
        assert len(script.prompts) == 1
        assert expected in _output(console)


def test_setting_keys_are_unique_within_each_section():
    # The editor looks a picked entry up by key, so a repeat would silently
    # shadow another setting.
    for name, section in _CONFIG_SECTIONS.items():
        keys = [setting.key for setting in section.settings]
        assert len(keys) == len(set(keys)), name
        assert _MENU_DONE not in keys, name


def test_typed_input_falls_back_to_the_console_when_no_reader_is_given(session):
    # The REPL passes prompt_input=None; typed values then come from console.input.
    console = _console()
    asked: list[str] = []

    def console_input(prompt: str = "") -> str:
        asked.append(prompt)
        return "1792x1024"

    console.input = console_input  # type: ignore[method-assign]

    _run(session, Script("media", "size"), console=console)

    assert asked == ["value> "]
    assert config.get_media()["size"] == "1792x1024"


# ---------------------------------------------------------------------------
# Single-setting sections: pick, then the value is saved
# ---------------------------------------------------------------------------


def test_trust_asks_on_or_off_and_saves(session):
    script = Script("trust", "on")

    console = _run(session, script)

    assert session.trusted_workspace is True
    assert config.load_config()["trusted_workspace"] is True
    assert script.options() == ["on", "off"]
    assert "Trusted workspace on" in _output(console)


def test_value_prompt_starts_on_the_current_value(session):
    session.trusted_workspace = True
    script = Script("trust", "off")

    _run(session, script)

    assert script.prompts[-1].selected == "on"
    assert session.trusted_workspace is False


def test_cache_toggle_saves(session):
    _run(session, Script("cache", "on"))

    assert config.get_prompt_cache() is True


def test_style_choice_updates_the_session(session):
    script = Script("style", "concise")

    _run(session, script)

    assert script.options() == list(config.OUTPUT_STYLES)
    assert session.output_style == "concise"
    assert config.get_output_style() == "concise"


def test_acp_timeout_is_typed(session):
    typed = Typed("45")

    console = _run(session, Script("acp"), typed)

    assert config.get_acp()["permission_timeout"] == 45
    assert "45" in _output(console)


def test_system_prompt_set_and_clear(session):
    _run(session, Script("prompt", "set"), Typed("Always answer in te reo."))
    assert session.custom_system_prompt == "Always answer in te reo."
    assert config.get_system_prompt() == "Always answer in te reo."

    _run(session, Script("prompt", "clear", "yes"))
    assert session.custom_system_prompt is None
    assert config.get_system_prompt() is None


def test_system_prompt_clear_can_be_declined(session):
    session.custom_system_prompt = "keep me"
    config.set_system_prompt("keep me")

    _run(session, Script("prompt", "clear", "no"))

    assert config.get_system_prompt() == "keep me"


def test_mode_set_then_reset(session):
    script = Script("mode", "set", "plan")
    _run(session, script)
    assert config.get_default_mode() == "plan"
    assert script.options() == ["ask", "auto-accept", "plan"]

    console = _run(session, Script("mode", "reset"))
    assert config.get_default_mode() == "ask"
    assert "reset" in _output(console)


# ---------------------------------------------------------------------------
# Multi-setting sections: a list with current values, repeated until Done
# ---------------------------------------------------------------------------


def test_media_menu_changes_several_settings_in_one_visit(session):
    script = Script(
        "media",
        "enable", "on",
        "size",
        "video-model",
        "video-model",
        "provider", "openrouter",
        "__done__",
    )
    typed = Typed("1792x1024", "vendor/video-1", "clear")

    console = _run(session, script, typed)

    media = config.get_media()
    assert media["enabled"] is True
    assert media["size"] == "1792x1024"
    assert media["provider"] == "openrouter"
    # Typing "clear" unset the video model that was just set.
    assert media["video_model"] == ""
    assert "Media generation enabled" in _output(console)


def test_media_list_shows_current_values_and_a_done_entry(session):
    config.set_media(model="my-image-model", size="512x512")
    script = Script("media")

    _run(session, script)

    labels = {o.value: o.label for o in script.prompts[1].options}
    assert labels["model"] == "Image model — my-image-model"
    assert labels["size"] == "Image size — 512x512"
    assert labels["video-model"] == "Video model — none chosen"
    assert "__done__" in labels


def test_media_invalid_value_reports_the_error_and_keeps_the_old_one(session):
    console = _run(session, Script("media", "size"), Typed("big"))

    assert "WxH" in _output(console)
    assert config.get_media()["size"] == "1024x1024"


def test_menu_returns_to_the_list_after_a_change(session):
    script = Script("ui", "theme", "ocean", "ascii", "on", "__done__")

    _run(session, script)

    # top menu, ui list, theme value, ui list, ascii value, ui list.
    assert script.titles() == [
        "Configure KiwiMateCoder",
        "Appearance and interface",
        "Theme",
        "Appearance and interface",
        "ASCII only",
        "Appearance and interface",
    ]
    assert config.get_ui()["theme"] == "ocean"
    assert config.get_ui()["ascii"] is True


def test_ui_choices_are_the_real_options(session):
    script = Script("ui", "locale", "de")

    _run(session, script)

    # prompts: top menu, ui list, then the locale value prompt.
    assert script.options(2) == ["en", "de", "es"]
    assert config.get_ui()["locale"] == "de"


def test_ui_text_setting(session):
    _run(session, Script("ui", "notify-after"), Typed("90"))

    assert config.get_ui()["notify_after_seconds"] == 90


def test_escape_at_the_value_step_changes_nothing(session):
    before = config.get_ui()["theme"]

    _run(session, Script("ui", "theme", None))

    assert config.get_ui()["theme"] == before


def test_escape_in_the_section_list_leaves_cleanly(session):
    console = _run(session, Script("media", None))

    assert config.get_media()["enabled"] is False
    assert "Traceback" not in _output(console)


def test_empty_typed_value_changes_nothing(session):
    console = _run(session, Script("media", "model"), Typed(""))

    assert config.get_media()["model"] == "gpt-image-1"
    assert "nothing changed" in _output(console)


def test_cancelling_typed_input_changes_nothing(session):
    console = _run(session, Script("media", "model"), Typed())  # raises EOFError

    assert config.get_media()["model"] == "gpt-image-1"
    assert "Cancelled" in _output(console)


def test_typed_values_with_spaces_and_quotes_survive(session):
    _run(session, Script("media", "output-dir"), Typed("my media/it's here"))

    assert config.get_media()["output_dir"] == "my media/it's here"


def test_network_proxy_set_and_clear(session):
    _run(session, Script("network", "proxy"), Typed("http://proxy.nz:3128"))
    assert config.get_network()["proxy"] == "http://proxy.nz:3128"

    _run(session, Script("network", "proxy"), Typed("clear"))
    assert config.get_network()["proxy"] == ""


def test_network_offline_toggle(session):
    _run(session, Script("network", "offline", "on"))

    assert config.get_network()["offline"] is True


def test_sampling_values_set_and_clear_individually(session):
    script = Script(
        "sampling",
        "temperature", "top_p", "reasoning_effort", "high",
        "temperature",
        "__done__",
    )
    typed = Typed("0.3", "0.8", "clear")

    _run(session, script, typed)

    # temperature was set then cleared; top_p and effort remain.
    assert config.get_sampling() == {"top_p": 0.8, "reasoning_effort": "high"}


def test_sampling_reset_clears_everything(session):
    config.set_sampling({"temperature": 0.4, "max_tokens": 2000})

    _run(session, Script("sampling", "reset"))

    assert config.get_sampling() == {}


def test_sampling_choice_with_no_current_value_still_opens(session):
    # "provider default" is a label, not an option; it must not be passed to
    # the widget as the cursor position.
    script = Script("sampling", "reasoning_effort", "low")

    _run(session, script)

    # prompts: top menu, sampling list, then the reasoning-effort value prompt.
    assert script.prompts[2].selected is None
    assert config.get_sampling() == {"reasoning_effort": "low"}
    # Back on the list, the cursor stays on the setting that was just changed.
    assert script.prompts[3].selected == "reasoning_effort"


def test_browser_toggles_and_timeout(session):
    script = Script("browser", "enable", "on", "headless", "off", "timeout")

    _run(session, script, Typed("15000"))

    browser = config.get_browser()
    assert browser["enabled"] is True
    assert browser["headless"] is False
    assert browser["timeout_ms"] == 15000


def test_shell_vision_and_web_numbers(session):
    _run(session, Script("shell", "max-jobs"), Typed("3"))
    _run(session, Script("vision", "max-images"), Typed("2"))
    _run(session, Script("web", "allow-local", "on"))

    assert config.get_shell_config()["max_jobs"] == 3
    assert config.get_vision()["max_images_per_turn"] == 2
    assert config.get_web()["allow_local"] is True


def test_telemetry_level_is_a_choice(session):
    script = Script("telemetry", "level", "error")

    _run(session, script)

    assert script.options(2) == list(config.TELEMETRY_LEVELS)
    assert config.get_telemetry()["level"] == "error"


def test_remote_enable_without_a_host_explains_why_and_stays_off(session):
    console = _run(session, Script("remote", "enable", "on"))

    assert "host is required" in _output(console)
    assert config.get_remote()["enabled"] is False


def test_remote_host_user_port(session):
    script = Script("remote", "host", "user", "port")

    _run(session, script, Typed("build.example.nz", "kiwi", "2222"))

    remote = config.get_remote()
    assert (remote["host"], remote["user"], remote["port"]) == (
        "build.example.nz", "kiwi", 2222,
    )


def test_team_policy_path_and_enforce(session, tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text("{}")

    _run(session, Script("team", "policy"), Typed(str(policy)))
    _run(session, Script("team", "enforce", "on"))

    assert config.get_team() == {"policy_path": str(policy), "enforce": True}

    _run(session, Script("team", "policy"), Typed("clear"))
    assert config.get_team()["policy_path"] == ""


# ---------------------------------------------------------------------------
# verify, budget, subagents, sandbox
# ---------------------------------------------------------------------------


def test_verify_command_is_typed_and_keeps_its_spaces_and_operators(session):
    console = _run(session, Script("verify", "set"), Typed("pytest -q && ruff check ."))

    assert config.get_verify_command() == "pytest -q && ruff check ."
    assert session.verify_command == "pytest -q && ruff check ."
    assert "Auto-verify command set" in _output(console)


def test_verify_can_be_turned_off(session):
    config.set_verify_command("pytest")
    session.verify_command = "pytest"

    console = _run(session, Script("verify", "clear"))

    assert config.get_verify_command() == ""
    assert session.verify_command == ""
    assert "Auto-verify disabled" in _output(console)


def test_verify_list_shows_the_current_command(session):
    session.verify_command = "pytest -q"
    script = Script("verify")

    _run(session, script)

    labels = {o.value: o.label for o in script.prompts[1].options}
    assert labels["set"] == "Set the command — pytest -q"


def test_budget_shows_no_limit_then_the_limits(session):
    script = Script("budget")
    _run(session, script)
    labels = {o.value: o.label for o in script.prompts[1].options}
    assert labels["tokens"] == "Token limit — no limit"
    assert labels["cost"] == "Cost limit — no limit"

    config.set_budget(max_tokens="50000", max_cost_usd="2.5")
    script = Script("budget")
    _run(session, script)
    labels = {o.value: o.label for o in script.prompts[1].options}
    assert labels["tokens"] == "Token limit — 50,000 tokens"
    assert labels["cost"] == "Cost limit — $2.5"


def test_budget_set_each_limit_and_clear_one(session):
    script = Script("budget", "tokens", "cost", "tokens", "__done__")
    typed = Typed("50000", "2.50", "clear")

    _run(session, script, typed)

    # The token limit was set, then cleared; the cost limit remains.
    assert config.get_budget() == {"max_cost_usd": 2.5}


def test_budget_invalid_value_is_reported(session):
    console = _run(session, Script("budget", "tokens"), Typed("lots"))

    assert config.get_budget() == {}
    assert _output(console).strip() != ""


def test_budget_remove_both_limits_needs_confirmation(session):
    config.set_budget(max_tokens="1000", max_cost_usd="1")

    _run(session, Script("budget", "clear", "no"))
    assert config.get_budget() == {"max_tokens": 1000, "max_cost_usd": 1.0}

    _run(session, Script("budget", "clear", "yes"))
    assert config.get_budget() == {}


def test_subagents_toggle_steps_and_model(session):
    script = Script("subagents", "enable", "off", "max-steps", "model", "__done__")
    typed = Typed("12", "vendor/small")

    _run(session, script, typed)

    subagents = config.get_subagents()
    assert subagents["enabled"] is False
    assert subagents["max_steps"] == 12
    assert subagents["model"] == "vendor/small"


def test_subagents_model_can_be_cleared(session):
    config.set_subagents(model="vendor/small")

    _run(session, Script("subagents", "model"), Typed("clear"))

    assert config.get_subagents()["model"] in ("", None)


def test_subagents_toggle_starts_on_the_current_value(session):
    script = Script("subagents", "enable", "on")

    _run(session, script)

    # Subagents default to on, so the cursor starts there.
    assert script.prompts[2].selected == "on"


def test_sandbox_toggles(session):
    script = Script("sandbox", "enable", "on", "network", "off", "__done__")

    _run(session, script)

    sandbox = config.get_sandbox()
    assert sandbox["enabled"] is True
    assert sandbox["network"] is False


def test_sandbox_add_and_remove_a_writable_path(session):
    script = Script("sandbox", "add-path", "add-path", "remove-path", "/data/cache")
    typed = Typed("/data/cache", "/data/out")

    _run(session, script, typed)

    assert config.get_sandbox()["extra_writable"] == ["/data/out"]
    # The remove list offered exactly what was set at that moment.
    assert script.options(4) == ["/data/cache", "/data/out"]


def test_sandbox_remove_path_with_none_set_says_so(session):
    script = Script("sandbox", "remove-path")

    console = _run(session, script)

    assert "No extra writable paths" in _output(console)
    # Only the top menu and the section list were shown; no empty picker.
    assert len(script.prompts) == 3


def test_sandbox_clear_paths_needs_confirmation(session):
    config.set_sandbox(extra_writable=["/a", "/b"])

    _run(session, Script("sandbox", "clear-paths", "no"))
    assert config.get_sandbox()["extra_writable"] == ["/a", "/b"]

    _run(session, Script("sandbox", "clear-paths", "yes"))
    assert config.get_sandbox()["extra_writable"] == []


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def test_providers_menu_lists_the_actions(session):
    script = Script("providers")

    _run(session, script)

    assert script.options() == ["use", "add", "edit", "remove", "list", "__done__"]


def test_add_edit_use_and_remove_a_custom_provider(session):
    typed = Typed(
        "local", "Local Models", "http://localhost:1234/v1", "local-code", "",
    )
    _run(session, Script("providers", "add"), typed)
    provider = config.get_provider_config("local")
    assert provider.name == "Local Models"
    assert config.get_provider_model("local") == "local-code"

    _run(session, Script("providers", "edit", "local", "name"), Typed("Local Two"))
    assert config.get_provider_config("local").name == "Local Two"

    _run(session, Script("providers", "use", "local"))
    assert session.provider_id == "local"
    assert session.model == "local-code"

    _run(session, Script("providers", "remove", "local", "yes"))
    with pytest.raises(KeyError):
        config.get_provider_config("local")


def test_add_provider_with_a_key_env(session):
    typed = Typed("lm", "LM", "http://localhost:1/v1", "m1", "LM_API_KEY")

    _run(session, Script("providers", "add"), typed)

    assert config.get_provider_config("lm").key_env == "LM_API_KEY"


def test_add_provider_can_be_cancelled_part_way(session):
    typed = Typed("half", "Half")  # then EOF at the base URL

    console = _run(session, Script("providers", "add"), typed)

    assert "Cancelled" in _output(console)
    with pytest.raises(KeyError):
        config.get_provider_config("half")


def test_remove_provider_can_be_declined(session):
    config.add_provider("local", "Local", "http://localhost:1234/v1", "local-code")

    _run(session, Script("providers", "remove", "local", "no"))

    assert config.get_provider_config("local").id == "local"


def test_only_custom_providers_are_offered_for_edit_and_remove(session):
    config.add_provider("local", "Local", "http://localhost:1234/v1", "local-code")
    script = Script("providers", "edit")

    _run(session, script)

    assert script.options() == ["local"]


def test_edit_and_remove_explain_when_there_are_no_custom_providers(session):
    console = _run(session, Script("providers", "edit"))
    assert "No custom providers" in _output(console)

    console = _run(session, Script("providers", "remove"))
    assert "No custom providers" in _output(console)


def test_edit_provider_clearable_field(session):
    config.add_provider(
        "az", "Azure", "https://az.example/v1", "gpt", None,
        key_header="api-key", key_prefix="", api_version="2024-06-01",
    )

    _run(session, Script("providers", "edit", "az", "api_version"), Typed("clear"))

    assert config.get_provider_config("az").api_version == ""


def test_use_provider_without_a_model_asks_for_one(session, monkeypatch):
    _install_fetch(monkeypatch, ["m/a", "m/b"])
    config.set_key("openai", "sk-x")
    script = Script("providers", "use", "openai", "m/b")

    _run(session, script)

    assert session.provider_id == "openai"
    assert config.get_provider_model("openai") == "m/b"


# ---------------------------------------------------------------------------
# Model and model visibility
# ---------------------------------------------------------------------------


def test_model_menu_chooses_from_the_catalog(session, monkeypatch):
    config.set_key("openrouter", "sk-test")
    _install_fetch(monkeypatch, ["vendor/new", "vendor/stable"])
    script = Script("model", "choose", "vendor/stable")

    _run(session, script)

    assert script.options() == ["vendor/new", "vendor/stable"]
    assert session.model == "vendor/stable"
    assert config.get_provider_model("openrouter") == "vendor/stable"


def test_model_menu_accepts_a_typed_id(session):
    _run(session, Script("model", "type"), Typed("some/unlisted-model"))

    assert session.model == "some/unlisted-model"
    assert config.get_provider_model("openrouter") == "some/unlisted-model"


def test_model_menu_forgets_the_choice(session):
    config.set_provider_model("openrouter", "vendor/x")

    _run(session, Script("model", "forget"))

    assert not config.get_provider_model("openrouter")
    assert session.model == ""


def test_models_refresh_from_the_menu(session, monkeypatch):
    config.set_key("openrouter", "sk-test")
    _install_fetch(monkeypatch, ["vendor/one", "vendor/two"])

    console = _run(session, Script("models", "refresh"))

    assert "vendor/two" in _output(console)
    assert config.list_visible_models("openrouter") == ["vendor/one", "vendor/two"]


def test_models_allow_uses_a_checklist_when_one_is_available(session, monkeypatch):
    config.set_key("openrouter", "sk-test")
    _install_fetch(monkeypatch, ["vendor/one", "vendor/two", "vendor/three"])
    _run(session, Script("models", "refresh"))
    seen: list[MultiSelectionPrompt] = []

    def checklist(prompt: MultiSelectionPrompt) -> list[str]:
        seen.append(prompt)
        return ["vendor/two"]

    _run(session, Script("models", "allow"), multi=checklist)

    assert [o.value for o in seen[0].options] == [
        "vendor/one", "vendor/two", "vendor/three",
    ]
    assert config.get_model_filter("openrouter") == {
        "mode": "allow", "models": ["vendor/two"],
    }


def test_models_deny_by_typing_ids_without_a_checklist(session):
    _run(session, Script("models", "deny"), Typed("vendor/a vendor/b"))

    assert config.get_model_filter("openrouter") == {
        "mode": "deny", "models": ["vendor/a", "vendor/b"],
    }


def test_models_empty_checklist_changes_nothing(session, monkeypatch):
    config.set_key("openrouter", "sk-test")
    _install_fetch(monkeypatch, ["vendor/one"])
    _run(session, Script("models", "refresh"))

    console = _run(session, Script("models", "allow"), multi=lambda prompt: [])

    assert config.get_model_filter("openrouter")["mode"] == "all"
    assert "nothing changed" in _output(console)


def test_models_clear_removes_the_filter(session):
    config.set_model_filter("openrouter", "deny", ["vendor/a"])

    _run(session, Script("models", "clear"))

    assert config.get_model_filter("openrouter")["mode"] == "all"


# ---------------------------------------------------------------------------
# Permissions, command rules, profiles
# ---------------------------------------------------------------------------


def test_remove_one_approved_tool(session):
    config.persist_always_allowed_tool("run_bash")
    config.persist_always_allowed_tool("write_file")
    session.always_allowed.update({"run_bash", "write_file"})

    _run(session, Script("permissions", "remove:run_bash"))

    assert session.always_allowed == {"write_file"}
    assert config.get_always_allowed_tools() == ["write_file"]


def test_clear_all_approved_tools_needs_confirmation(session):
    config.persist_always_allowed_tool("run_bash")
    session.always_allowed.add("run_bash")

    _run(session, Script("permissions", "clear", "no"))
    assert config.get_always_allowed_tools() == ["run_bash"]

    _run(session, Script("permissions", "clear", "yes"))
    assert config.get_always_allowed_tools() == []
    assert session.always_allowed == set()


def test_command_rules_add_remove_and_clear(session):
    script = Script("commands", "allow", "deny", "remove", "0", "clear", "yes")
    typed = Typed(r"^pytest\b", r"rm\s+-rf")

    _run(session, script, typed)

    # deny rules list first, so index 0 was the deny rule that got removed;
    # then everything left was cleared.
    assert config.get_command_rules() == {"allow": [], "deny": []}
    assert session.command_rules == {"allow": [], "deny": []}


def test_command_rule_regex_survives_quoting(session):
    _run(session, Script("commands", "allow"), Typed(r"^git (status|diff)\b"))

    assert config.get_command_rules()["allow"] == [r"^git (status|diff)\b"]
    assert session.command_rules["allow"] == [r"^git (status|diff)\b"]


def test_command_rules_invalid_regex_is_reported(session):
    console = _run(session, Script("commands", "deny"), Typed("("))

    assert config.get_command_rules()["deny"] == []
    assert _output(console).strip() != ""


def test_command_rules_remove_with_none_set(session):
    console = _run(session, Script("commands", "remove"))

    assert "No command rules" in _output(console)


def test_profile_save_use_show_and_remove(session):
    _run(session, Script("profile", "save"), Typed("work"))
    assert "work" in config.get_profiles()

    config.set_default_mode("plan")
    console = _run(session, Script("profile", "use", "work"))
    assert "Applied profile" in _output(console)

    console = _run(session, Script("profile", "show", "work"))
    assert "provider" in _output(console)

    _run(session, Script("profile", "remove", "work", "no"))
    assert "work" in config.get_profiles()
    _run(session, Script("profile", "remove", "work", "yes"))
    assert config.get_profiles() == {}


def test_profile_use_with_no_profiles_says_so(session):
    console = _run(session, Script("profile", "use"))

    assert "No profiles saved" in _output(console)


# ---------------------------------------------------------------------------
# What did not change
# ---------------------------------------------------------------------------


def test_typed_config_commands_still_print_without_opening_a_menu(session):
    script = Script("should-not-be-asked")
    console = _console()

    dispatch("/config media", session, console, selector=script)

    assert script.prompts == []
    assert "Media:" in _output(console)


def test_menu_without_a_selector_is_unchanged(session):
    console = _console()

    dispatch("/config", session, console)

    assert "Active providers" in _output(console)
