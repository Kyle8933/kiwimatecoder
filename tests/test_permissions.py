from kiwimatecoder.permissions import Decision, PermissionMode, gate
from kiwimatecoder.tools.read_file import read_file_tool
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


def test_provider_switch_clears_always_allowed(session):
    session.allow_always("run_bash")
    session.set_provider("openai")
    assert not session.is_always_allowed("run_bash")
    assert session.model == "gpt-5.6-sol"


def test_mode_from_str_aliases():
    assert PermissionMode.from_str("auto") is PermissionMode.AUTO
    assert PermissionMode.from_str("read-only") is PermissionMode.PLAN
