"""nan, inf and overflowing numbers in the config.

Two ways a number can be bad:

* typed (``/config budget cost nan``): ``nan <= 0`` is False and ``inf`` is
  positive, so both used to pass the range check, be saved as the non-standard
  JSON ``NaN``/``Infinity``, and make a limit that never trips;
* hand-edited (``"max_tokens": 1e999``): JSON parses that to ``inf`` and
  ``int(inf)`` raises ``OverflowError``, which no getter caught, so thirteen
  getters (and ``validate_config``, the command meant to diagnose such a file)
  crashed on it.
"""

from __future__ import annotations

import inspect
import json
import math

import pytest

from kiwimatecoder import config
from kiwimatecoder.providers import REGISTRY


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for provider in REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)


def _write(text: str) -> None:
    """Write the config file as raw text, the way a hand edit would."""
    config.CONFIG_FILE.write_text(text)


def _stored_text() -> str:
    return config.CONFIG_FILE.read_text() if config.CONFIG_FILE.exists() else ""


# ---------------------------------------------------------------------------
# Typed values: rejected, with a clear message, and never saved
# ---------------------------------------------------------------------------

NON_FINITE_COST = ["nan", "NaN", "inf", "-inf", "Infinity", "1e999", "-1e999"]


@pytest.mark.parametrize("value", NON_FINITE_COST)
def test_set_budget_rejects_a_non_finite_cost(value):
    with pytest.raises(ValueError, match="finite number"):
        config.set_budget(max_cost_usd=value)

    assert config.get_budget() == {}
    # Nothing was written that a strict JSON parser would choke on.
    assert "NaN" not in _stored_text()
    assert "Infinity" not in _stored_text()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_set_budget_rejects_a_non_finite_float_cost(value):
    with pytest.raises(ValueError, match="finite number"):
        config.set_budget(max_cost_usd=value)

    assert config.get_budget() == {}


@pytest.mark.parametrize(
    "value",
    ["nan", "inf", "1e999", "lots", "", "1,000", "2.5", float("nan"), float("inf")],
)
def test_set_budget_rejects_a_non_whole_token_count_with_a_clear_message(value):
    # A float inf used to raise OverflowError (not a ValueError, so the typed
    # command would not even have caught it); strings such as "nan" raised a raw
    # "invalid literal for int()".
    with pytest.raises(ValueError, match="whole number"):
        config.set_budget(max_tokens=value)

    assert config.get_budget() == {}


@pytest.mark.parametrize("value", ["lots", "$3", "1,5", ""])
def test_set_budget_cost_not_a_number_is_a_clear_error(value):
    with pytest.raises(ValueError, match="must be a number"):
        config.set_budget(max_cost_usd=value)


def test_the_error_names_the_rejected_value():
    with pytest.raises(ValueError) as caught:
        config.set_budget(max_cost_usd="nan")

    assert "'nan'" in str(caught.value)


def test_set_budget_still_accepts_ordinary_limits():
    assert config.set_budget(max_tokens="50000") == {"max_tokens": 50000}
    assert config.set_budget(max_cost_usd="2.50") == {
        "max_tokens": 50000,
        "max_cost_usd": 2.5,
    }
    assert config.set_budget(max_cost_usd=100) == {
        "max_tokens": 50000,
        "max_cost_usd": 100.0,
    }
    assert config.set_budget(max_cost_usd="1e2") == {
        "max_tokens": 50000,
        "max_cost_usd": 100.0,
    }
    json.loads(_stored_text())  # still valid, strict JSON


def test_a_rejected_value_leaves_the_existing_limits_alone():
    config.set_budget(max_tokens=1000, max_cost_usd=1.5)

    with pytest.raises(ValueError):
        config.set_budget(max_cost_usd="inf")

    assert config.get_budget() == {"max_tokens": 1000, "max_cost_usd": 1.5}


def test_a_profile_cannot_carry_a_non_finite_budget():
    # Profiles share the budget validation with set_budget.
    for bad in ("nan", "inf"):
        with pytest.raises(ValueError, match="finite number"):
            config.save_profile("work", {"budget": {"max_cost_usd": bad}})
    with pytest.raises(ValueError, match="whole number"):
        config.save_profile("work", {"budget": {"max_tokens": "inf"}})

    assert config.get_profiles() == {}


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_setters_raise_value_error_for_a_float_that_cannot_be_an_integer(value):
    # int(inf) raises OverflowError; callers (and the typed commands) only
    # catch ValueError.
    with pytest.raises(ValueError):
        config.set_subagents(max_steps=value)
    with pytest.raises(ValueError):
        config.set_browser(timeout_ms=value)
    with pytest.raises(ValueError):
        config.set_shell_config(max_jobs=value)
    with pytest.raises(ValueError):
        config.set_sampling({"max_tokens": value})
    with pytest.raises(ValueError):
        config.set_sampling({"temperature": value})


# ---------------------------------------------------------------------------
# Hand-edited values: ignored on read, reported by validate_config
# ---------------------------------------------------------------------------


def test_get_budget_ignores_unusable_stored_limits_but_keeps_the_good_one():
    _write('{"budget": {"max_tokens": 1e999, "max_cost_usd": 2.5}}')
    assert config.get_budget() == {"max_cost_usd": 2.5}

    _write('{"budget": {"max_tokens": 5000, "max_cost_usd": NaN}}')
    assert config.get_budget() == {"max_tokens": 5000}

    _write('{"budget": {"max_tokens": 5000, "max_cost_usd": Infinity}}')
    assert config.get_budget() == {"max_tokens": 5000}

    _write('{"budget": {"max_cost_usd": 1' + "0" * 400 + "}}")  # float() overflows
    assert config.get_budget() == {}


def test_get_subagents_falls_back_to_the_default_for_an_overflowing_step_limit():
    _write('{"subagents": {"max_steps": 1e999, "enabled": false}}')

    assert config.get_subagents() == {
        "enabled": False,
        "max_steps": config.SUBAGENT_MAX_STEPS_DEFAULT,
        "model": "",
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"budget": {"max_cost_usd": NaN}}', "finite number"),
        ('{"budget": {"max_cost_usd": Infinity}}', "finite number"),
        ('{"budget": {"max_tokens": 1e999}}', "whole number"),
        ('{"subagents": {"max_steps": 1e999}}', "subagents.max_steps"),
        ('{"sampling": {"temperature": 1e999}}', "temperature"),
    ],
)
def test_validate_config_reports_a_non_finite_value_instead_of_crashing(text, message):
    _write(text)

    issues = config.validate_config()

    assert any(
        issue["level"] == "error" and message in f"{issue['key']} {issue['message']}"
        for issue in issues
    ), issues


# ---------------------------------------------------------------------------
# Every section: no getter crashes or returns a non-finite number
# ---------------------------------------------------------------------------

# JSON literals as a hand edit would write them.
HAND_EDITED = ["1e999", "-1e999", "NaN", "Infinity", "-Infinity", "1" + "0" * 400]
# Sections with no *_DEFAULTS dict, or whose getter is not named after them.
EXTRA_KEYS = {
    "budget": ["max_tokens", "max_cost_usd"],
    "subagents": ["max_steps", "enabled", "model"],
    "sampling": ["temperature", "top_p", "max_tokens", "reasoning_effort"],
}
GETTER_ALIASES = {"shell": "shell_config"}


def _section_keys() -> dict[str, list[str]]:
    keys: dict[str, list[str]] = {}
    for name, value in vars(config).items():
        if name.endswith("_DEFAULTS") and isinstance(value, dict):
            keys[name[: -len("_DEFAULTS")].lower()] = list(value)
    keys.update(EXTRA_KEYS)
    return keys


def _getter(section: str):
    function = getattr(config, f"get_{GETTER_ALIASES.get(section, section)}", None)
    if function is None or list(inspect.signature(function).parameters) != ["cfg"]:
        return None
    return function


def _floats(value):
    if isinstance(value, float):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _floats(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _floats(item)


SECTIONS = sorted(section for section in _section_keys() if _getter(section))


def test_the_sweep_covers_the_sections_that_used_to_crash():
    # If a getter is renamed the sweep would silently skip it; fail instead.
    assert {
        "acp", "browser", "budget", "index", "lsp", "memory", "model_routing",
        "remote", "sampling", "shell", "subagents", "telemetry", "vision", "web",
    } <= set(SECTIONS)


@pytest.mark.parametrize("section", SECTIONS)
def test_no_getter_crashes_or_returns_a_non_finite_number(section):
    getter = _getter(section)
    problems = []
    for key in _section_keys()[section]:
        for literal in HAND_EDITED:
            _write('{"%s": {"%s": %s}}' % (section, key, literal))
            where = f"{section}.{key}={literal[:12]}"
            try:
                result = getter()
            except Exception as exc:  # noqa: BLE001 - any crash is the failure
                problems.append(f"{where}: get_{section}() raised {type(exc).__name__}")
                continue
            if any(not math.isfinite(number) for number in _floats(result)):
                problems.append(f"{where}: get_{section}() returned {result!r}")
            try:
                config.validate_config()
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{where}: validate_config() raised {type(exc).__name__}")

    assert not problems, "\n".join(problems)
