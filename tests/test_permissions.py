from kiwimatecoder.permissions import PermissionMode, gate
from kiwimatecoder.tools.read_file import read_file_tool
from kiwimatecoder.tools.run_bash import run_bash_tool
from kiwimatecoder.tools.write_file import write_file_tool


def _always(summary, preview):
    return True


def _never(summary, preview):
    return False


def test_reads_allowed_in_all_modes(session):
    for mode in PermissionMode:
        session.mode = mode
        assert gate(read_file_tool, {"path": "x"}, session, _never).allowed


def test_plan_mode_blocks_writes(session):
    session.mode = PermissionMode.PLAN
    decision = gate(write_file_tool, {"path": "x", "content": ""}, session, _always)
    assert not decision.allowed
    assert "plan" in decision.reason.lower()


def test_auto_mode_allows_without_prompt(session):
    session.mode = PermissionMode.AUTO
    # _never would reject if called; AUTO must not call confirm.
    assert gate(write_file_tool, {"path": "x", "content": ""}, session, _never).allowed


def test_ask_mode_consults_confirm(session):
    session.mode = PermissionMode.ASK
    assert gate(write_file_tool, {"path": "x", "content": ""}, session, _always).allowed
    assert not gate(write_file_tool, {"path": "x", "content": ""}, session, _never).allowed


def test_always_allowed_skips_confirm(session):
    session.mode = PermissionMode.ASK
    session.allow_always("write_file")
    assert gate(write_file_tool, {"path": "x", "content": ""}, session, _never).allowed


def test_provider_switch_keeps_always_allowed(session):
    """Approvals are persisted user preferences, so they survive a switch."""
    session.allow_always("run_bash")
    session.set_provider("openai")
    assert session.is_always_allowed("run_bash")
    assert session.model == "gpt-5.6-sol"


def test_mode_from_str_aliases():
    assert PermissionMode.from_str("auto") is PermissionMode.AUTO
    assert PermissionMode.from_str("read-only") is PermissionMode.PLAN


def test_command_deny_rule_blocks_even_in_auto(session):
    session.command_rules = {"allow": [], "deny": [r"rm\s+-rf"]}
    session.mode = PermissionMode.AUTO

    decision = gate(run_bash_tool, {"command": "rm -rf /"}, session, _always)

    assert not decision.allowed
    assert "deny rule" in decision.reason


def test_command_allow_rule_skips_prompt(session):
    session.command_rules = {"allow": [r"^pytest\b"], "deny": []}
    session.mode = PermissionMode.ASK

    assert gate(run_bash_tool, {"command": "pytest -q"}, session, _never).allowed


def test_command_allow_rule_still_prompts_others(session):
    session.command_rules = {"allow": [r"^pytest\b"], "deny": []}
    session.mode = PermissionMode.ASK

    assert not gate(run_bash_tool, {"command": "ls"}, session, _never).allowed


def test_command_allow_rule_does_not_override_plan(session):
    session.command_rules = {"allow": [r"^pytest\b"], "deny": []}
    session.mode = PermissionMode.PLAN

    decision = gate(run_bash_tool, {"command": "pytest -q"}, session, _always)

    assert not decision.allowed
    assert "plan" in decision.reason.lower()


def test_command_rules_do_not_affect_file_tools(session):
    session.command_rules = {"allow": [], "deny": ["."]}
    session.mode = PermissionMode.ASK

    assert gate(write_file_tool, {"path": "x", "content": ""}, session, _always).allowed
