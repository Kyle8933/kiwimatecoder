from pathlib import Path

from rich.console import Console

from kiwimatecoder import updater


def test_find_source_root_detects_project_root(tmp_path):
    package_dir = tmp_path / "kiwimatecoder"
    package_dir.mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='kiwimatecoder'\n")
    package_file = package_dir / "updater.py"
    package_file.write_text("")

    assert updater.find_source_root(package_file) == tmp_path


def test_build_source_install_command_uses_editable_source_root():
    root = Path("/tmp/kiwimatecoder")

    command = updater.build_source_install_command(root)

    assert command[1:5] == ["-m", "pip", "install", "--upgrade"]
    assert command[-2:] == ["-e", str(root)]


def test_build_update_command_uses_github_repo_url():
    command = updater.build_update_command()

    assert command[0]  # current interpreter
    assert command[1:4] == ["-m", "pip", "install"]
    assert "--upgrade" in command
    assert "--force-reinstall" in command
    assert command[-1] == f"git+{updater.GITHUB_REPO_URL}"
    # Match the known-working manual command shape:
    # pip install --upgrade --force-reinstall git+https://github.com/.../kiwimatecoder.git
    upgrade_idx = command.index("--upgrade")
    force_idx = command.index("--force-reinstall")
    assert upgrade_idx < force_idx < len(command) - 1


def test_build_update_command_pins_ref():
    command = updater.build_update_command("v0.2.0")

    assert command[-1] == f"git+{updater.GITHUB_REPO_URL}@v0.2.0"


def test_build_git_checkout_command_uses_plain_checkout_for_other_refs():
    root = Path("/tmp/kiwimatecoder")

    assert updater.build_git_checkout_command(root, "v0.2.0") == [
        "git",
        "-C",
        str(root),
        "checkout",
        "v0.2.0",
    ]


def test_build_git_checkout_command_fast_forwards_current_branch():
    root = Path("/tmp/kiwimatecoder")

    assert updater.build_git_checkout_command(
        root, "main", current_branch="main"
    ) == updater.build_git_pull_command(root)
    assert updater.build_git_checkout_command(root, "main") == [
        "git",
        "-C",
        str(root),
        "checkout",
        "main",
    ]


def _patch_git_helpers(monkeypatch, *, root, behind, shas, branch="main"):
    """Patch the Git helper functions used by run_update's Git branch."""
    monkeypatch.setattr(updater, "find_source_root", lambda: root)
    monkeypatch.setattr(updater, "_has_git_remote", lambda source_root: True)
    monkeypatch.setattr(updater, "_fetch", lambda source_root, console: 0)
    monkeypatch.setattr(updater, "_get_branch", lambda source_root: branch)
    monkeypatch.setattr(updater, "_commits_behind", lambda source_root, b: behind)
    sha_iter = iter(shas)
    monkeypatch.setattr(updater, "_get_short_sha", lambda source_root: next(sha_iter))


def _record_run(calls: list[list[str]]):
    def fake_run(command: list[str], console: object) -> int:
        calls.append(command)
        return 0

    return fake_run


def test_run_update_uses_git_source_checkout(monkeypatch):
    calls: list[list[str]] = []
    root = Path("/tmp/kiwimatecoder")

    _patch_git_helpers(monkeypatch, root=root, behind=2, shas=["abc1234", "def5678"])
    monkeypatch.setattr(updater, "_run", _record_run(calls))

    code = updater.run_update(Console(record=True))

    assert code == 0
    assert calls == [
        updater.build_git_pull_command(root),
        updater.build_source_install_command(root),
    ]


def test_run_update_falls_back_to_git_url(monkeypatch):
    calls: list[list[str]] = []

    monkeypatch.setattr(updater, "find_source_root", lambda: None)
    monkeypatch.setattr(updater, "_run", _record_run(calls))

    code = updater.run_update(Console(record=True))

    assert code == 0
    assert len(calls) == 1
    command = calls[0]
    assert command == updater.build_update_command()
    assert "--upgrade" in command
    assert "--force-reinstall" in command
    assert command[-1] == f"git+{updater.GITHUB_REPO_URL}"


def test_run_update_stops_when_git_pull_fails(monkeypatch):
    calls: list[list[str]] = []
    root = Path("/tmp/kiwimatecoder")

    _patch_git_helpers(monkeypatch, root=root, behind=2, shas=["abc1234", "abc1234"])

    def fake_run(command: list[str], console: object) -> int:
        calls.append(command)
        return 7

    monkeypatch.setattr(updater, "_run", fake_run)

    code = updater.run_update(Console(record=True))

    assert code == 7
    assert calls == [updater.build_git_pull_command(root)]


def test_run_update_already_up_to_date_skips_reinstall(monkeypatch):
    calls: list[list[str]] = []
    root = Path("/tmp/kiwimatecoder")

    _patch_git_helpers(monkeypatch, root=root, behind=0, shas=["abc1234"])
    monkeypatch.setattr(updater, "_run", _record_run(calls))

    console = Console(record=True)
    code = updater.run_update(console)

    assert code == 0
    assert calls == []
    output = console.export_text()
    assert "Already on the latest version" in output
    assert "abc1234" in output


def test_run_update_reports_old_to_new_sha(monkeypatch):
    calls: list[list[str]] = []
    root = Path("/tmp/kiwimatecoder")

    _patch_git_helpers(monkeypatch, root=root, behind=3, shas=["abc1234", "def5678"])
    monkeypatch.setattr(updater, "_run", _record_run(calls))

    console = Console(record=True)
    code = updater.run_update(console)

    assert code == 0
    output = console.export_text()
    assert "abc1234" in output
    assert "def5678" in output
    assert "Updated abc1234" in output
    assert "3 commit(s) behind origin/main" in output


def test_run_update_skips_reinstall_when_pull_changes_nothing(monkeypatch):
    calls: list[list[str]] = []
    root = Path("/tmp/kiwimatecoder")

    # behind unknown (fetch failed) -> proceed to pull, but sha unchanged afterwards.
    _patch_git_helpers(monkeypatch, root=root, behind=None, shas=["abc1234", "abc1234"])
    monkeypatch.setattr(updater, "_run", _record_run(calls))

    console = Console(record=True)
    code = updater.run_update(console)

    assert code == 0
    # pull ran, but install was skipped because the SHA did not change.
    assert calls == [updater.build_git_pull_command(root)]
    assert "Already on the latest version" in console.export_text()


def test_run_update_git_fallback_failure_prints_guidance(monkeypatch):
    monkeypatch.setattr(updater, "find_source_root", lambda: None)
    monkeypatch.setattr(updater, "_run", lambda command, console: 1)

    console = Console(record=True)
    code = updater.run_update(console)

    assert code == 1
    output = console.export_text()
    assert "not on PyPI" in output
    assert updater.GITHUB_REPO_URL in output
    assert "--force-reinstall" in output


def test_run_update_with_ref_checks_out_tag(monkeypatch):
    calls: list[list[str]] = []
    root = Path("/tmp/kiwimatecoder")

    _patch_git_helpers(monkeypatch, root=root, behind=None, shas=["abc1234", "def5678"])
    monkeypatch.setattr(updater, "_run", _record_run(calls))

    console = Console(record=True)
    code = updater.run_update(console, ref="v0.2.0")

    assert code == 0
    assert calls == [
        updater.build_git_checkout_command(root, "v0.2.0", current_branch="main"),
        updater.build_source_install_command(root),
    ]
    output = console.export_text()
    assert "v0.2.0" in output
    assert "abc1234" in output
    assert "def5678" in output


def test_run_update_with_ref_matching_branch_fast_forwards(monkeypatch):
    calls: list[list[str]] = []
    root = Path("/tmp/kiwimatecoder")

    _patch_git_helpers(monkeypatch, root=root, behind=2, shas=["abc1234", "def5678"])
    monkeypatch.setattr(updater, "_run", _record_run(calls))

    code = updater.run_update(Console(record=True), ref="main")

    assert code == 0
    assert calls == [
        updater.build_git_pull_command(root),
        updater.build_source_install_command(root),
    ]


def test_run_update_with_ref_on_current_branch_is_noop(monkeypatch):
    calls: list[list[str]] = []
    root = Path("/tmp/kiwimatecoder")

    _patch_git_helpers(monkeypatch, root=root, behind=0, shas=["abc1234"])
    monkeypatch.setattr(updater, "_run", _record_run(calls))

    console = Console(record=True)
    code = updater.run_update(console, ref="main")

    assert code == 0
    assert calls == []
    output = console.export_text()
    assert "Already on main" in output
    assert "abc1234" in output


def test_run_update_ref_checkout_failure_stops(monkeypatch):
    calls: list[list[str]] = []
    root = Path("/tmp/kiwimatecoder")

    _patch_git_helpers(monkeypatch, root=root, behind=None, shas=["abc1234"])

    def fake_run(command: list[str], console: object) -> int:
        calls.append(command)
        return 7

    monkeypatch.setattr(updater, "_run", fake_run)

    console = Console(record=True)
    code = updater.run_update(console, ref="v0.2.0")

    assert code == 7
    assert calls == [updater.build_git_checkout_command(root, "v0.2.0", current_branch="main")]
    assert "checking out v0.2.0" in console.export_text()


def test_run_update_packaged_install_pins_ref(monkeypatch):
    calls: list[list[str]] = []

    monkeypatch.setattr(updater, "find_source_root", lambda: None)
    monkeypatch.setattr(updater, "_run", _record_run(calls))

    code = updater.run_update(Console(record=True), ref="v0.2.0")

    assert code == 0
    assert calls == [updater.build_update_command("v0.2.0")]
    assert calls[0][-1] == f"git+{updater.GITHUB_REPO_URL}@v0.2.0"


def test_run_update_packaged_ref_failure_prints_pinned_guidance(monkeypatch):
    monkeypatch.setattr(updater, "find_source_root", lambda: None)
    monkeypatch.setattr(updater, "_run", lambda command, console: 1)

    console = Console(record=True)
    code = updater.run_update(console, ref="v0.2.0")

    assert code == 1
    assert f"git+{updater.GITHUB_REPO_URL}@v0.2.0" in console.export_text()


# ---------------------------------------------------------------------------
# Update check tests
# ---------------------------------------------------------------------------


def test_check_for_update_offline_mode(monkeypatch):
    monkeypatch.setattr("kiwimatecoder.network.offline_enabled", lambda: True)
    monkeypatch.setattr("kiwimatecoder.updater.load_update_state", lambda: {})

    info = updater.check_for_update()

    assert info.update_available is False
    assert info.error == "offline mode is enabled"


def test_check_for_update_source_checkout_up_to_date(monkeypatch, tmp_path):
    root = tmp_path / "kiwimatecoder"
    root.mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='kiwimatecoder'\n")

    monkeypatch.setattr(updater, "find_source_root", lambda: root)
    monkeypatch.setattr(updater, "_has_git_remote", lambda source_root: True)
    monkeypatch.setattr(updater, "_get_short_sha", lambda source_root: "abc1234")
    monkeypatch.setattr(updater, "_get_branch", lambda source_root: "main")
    monkeypatch.setattr(updater, "_get_remote_head_sha", lambda branch="main": "abc1234")
    monkeypatch.setattr(updater, "_commits_behind", lambda source_root, branch: 0)
    monkeypatch.setattr("kiwimatecoder.network.offline_enabled", lambda: False)
    monkeypatch.setattr("kiwimatecoder.updater.load_update_state", lambda: {})
    monkeypatch.setattr("kiwimatecoder.updater.save_update_state", lambda state: None)

    info = updater.check_for_update()

    assert info.update_available is False
    assert info.current_sha == "abc1234"
    assert info.latest_sha == "abc1234"
    assert info.error is None


def test_check_for_update_source_checkout_behind(monkeypatch, tmp_path):
    root = tmp_path / "kiwimatecoder"
    root.mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='kiwimatecoder'\n")

    monkeypatch.setattr(updater, "find_source_root", lambda: root)
    monkeypatch.setattr(updater, "_has_git_remote", lambda source_root: True)
    monkeypatch.setattr(updater, "_get_short_sha", lambda source_root: "abc1234")
    monkeypatch.setattr(updater, "_get_branch", lambda source_root: "main")
    monkeypatch.setattr(updater, "_get_remote_head_sha", lambda branch="main": "def5678")
    monkeypatch.setattr(updater, "_commits_behind", lambda source_root, branch: 3)
    monkeypatch.setattr("kiwimatecoder.network.offline_enabled", lambda: False)
    monkeypatch.setattr("kiwimatecoder.updater.load_update_state", lambda: {})
    monkeypatch.setattr("kiwimatecoder.updater.save_update_state", lambda state: None)

    info = updater.check_for_update()

    assert info.update_available is True
    assert info.current_sha == "abc1234"
    assert info.latest_sha == "def5678"
    assert info.commits_behind == 3


def test_check_for_update_remote_unreachable(monkeypatch, tmp_path):
    root = tmp_path / "kiwimatecoder"
    root.mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='kiwimatecoder'\n")

    monkeypatch.setattr(updater, "find_source_root", lambda: root)
    monkeypatch.setattr(updater, "_has_git_remote", lambda source_root: True)
    monkeypatch.setattr(updater, "_get_short_sha", lambda source_root: "abc1234")
    monkeypatch.setattr(updater, "_get_branch", lambda source_root: "main")
    monkeypatch.setattr(updater, "_get_remote_head_sha", lambda branch="main": None)
    monkeypatch.setattr("kiwimatecoder.network.offline_enabled", lambda: False)
    monkeypatch.setattr("kiwimatecoder.updater.load_update_state", lambda: {})

    info = updater.check_for_update()

    assert info.update_available is False
    assert info.error == "could not reach GitHub"


def test_check_for_update_uses_cache(monkeypatch):
    import time

    cached_state = {
        "last_check": time.time() - 3600,
        "update_available": True,
        "current_version": "0.1.0",
        "latest_version": "0.2.0",
        "current_sha": "abc1234",
        "latest_sha": "def5678",
        "commits_behind": 2,
    }
    monkeypatch.setattr("kiwimatecoder.network.offline_enabled", lambda: False)
    monkeypatch.setattr("kiwimatecoder.updater.load_update_state", lambda: cached_state)
    monkeypatch.setattr(
        "kiwimatecoder.config.get_update",
        lambda cfg=None: {"auto_check": True, "check_interval_hours": 24},
    )
    # Should not reach the network at all
    monkeypatch.setattr(
        updater, "find_source_root", lambda: (_ for _ in ()).throw(AssertionError("should not check"))
    )

    info = updater.check_for_update()

    assert info.update_available is True
    assert info.current_sha == "abc1234"
    assert info.latest_sha == "def5678"


def test_check_for_update_force_bypasses_cache(monkeypatch, tmp_path):
    import time

    root = tmp_path / "kiwimatecoder"
    root.mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='kiwimatecoder'\n")

    cached_state = {
        "last_check": time.time(),
        "update_available": False,
        "current_version": "0.1.0",
        "latest_version": "0.1.0",
    }
    monkeypatch.setattr("kiwimatecoder.network.offline_enabled", lambda: False)
    monkeypatch.setattr("kiwimatecoder.updater.load_update_state", lambda: cached_state)
    monkeypatch.setattr("kiwimatecoder.updater.save_update_state", lambda state: None)
    monkeypatch.setattr(updater, "find_source_root", lambda: root)
    monkeypatch.setattr(updater, "_has_git_remote", lambda source_root: True)
    monkeypatch.setattr(updater, "_get_short_sha", lambda source_root: "abc1234")
    monkeypatch.setattr(updater, "_get_branch", lambda source_root: "main")
    monkeypatch.setattr(updater, "_get_remote_head_sha", lambda branch="main": "abc1234")
    monkeypatch.setattr(updater, "_commits_behind", lambda source_root, branch: 0)

    info = updater.check_for_update(force=True)

    assert info.update_available is False
    assert info.current_sha == "abc1234"


def test_should_check_for_update_first_run(monkeypatch):
    monkeypatch.setattr(
        "kiwimatecoder.config.get_update",
        lambda cfg=None: {"auto_check": True, "check_interval_hours": 24},
    )
    monkeypatch.setattr("kiwimatecoder.updater.load_update_state", lambda: {})

    assert updater.should_check_for_update() is True


def test_should_check_for_update_respects_interval(monkeypatch):
    import time

    monkeypatch.setattr(
        "kiwimatecoder.config.get_update",
        lambda cfg=None: {"auto_check": True, "check_interval_hours": 24},
    )
    monkeypatch.setattr(
        "kiwimatecoder.updater.load_update_state",
        lambda: {"last_check": time.time() - 3600},
    )

    assert updater.should_check_for_update() is False


def test_should_check_for_update_disabled(monkeypatch):
    monkeypatch.setattr(
        "kiwimatecoder.config.get_update",
        lambda cfg=None: {"auto_check": False, "check_interval_hours": 24},
    )
    monkeypatch.setattr("kiwimatecoder.updater.load_update_state", lambda: {})

    assert updater.should_check_for_update() is False


def test_update_state_roundtrip(monkeypatch, tmp_path):
    state_file = tmp_path / "update_check.json"
    monkeypatch.setattr(updater, "_update_state_path", lambda: state_file)

    assert updater.load_update_state() == {}

    updater.save_update_state({"last_check": 123.0, "update_available": True})
    loaded = updater.load_update_state()
    assert loaded["last_check"] == 123.0
    assert loaded["update_available"] is True


def test_get_remote_head_sha_success(monkeypatch):
    class FakeCompleted:
        returncode = 0
        stdout = "abc1234567890abcdef1234567890abcdef12345678\trefs/heads/main\n"

    monkeypatch.setattr(
        "subprocess.run", lambda *args, **kwargs: FakeCompleted()
    )

    sha = updater._get_remote_head_sha("main")
    assert sha == "abc1234"


def test_get_remote_head_sha_failure(monkeypatch):
    class FakeCompleted:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(
        "subprocess.run", lambda *args, **kwargs: FakeCompleted()
    )

    assert updater._get_remote_head_sha("main") is None


# ---------------------------------------------------------------------------
# /update slash command tests
# ---------------------------------------------------------------------------


def test_slash_update_check_up_to_date(session, monkeypatch):
    import io

    from kiwimatecoder import updater as updater_mod
    from kiwimatecoder.commands import dispatch

    monkeypatch.setattr(
        updater_mod,
        "check_for_update",
        lambda **kwargs: updater_mod.UpdateInfo(
            update_available=False,
            current_version="0.1.0",
            latest_version="0.1.0",
            current_sha="abc1234",
            latest_sha="abc1234",
        ),
    )

    console = Console(file=io.StringIO(), force_terminal=False, width=120)
    result = dispatch("/update check", session, console)

    assert result == "continue"
    assert "up to date" in console.file.getvalue()


def test_slash_update_check_available(session, monkeypatch):
    import io

    from kiwimatecoder import updater as updater_mod
    from kiwimatecoder.commands import dispatch

    monkeypatch.setattr(
        updater_mod,
        "check_for_update",
        lambda **kwargs: updater_mod.UpdateInfo(
            update_available=True,
            current_version="0.1.0",
            latest_version="0.1.0",
            current_sha="abc1234",
            latest_sha="def5678",
            commits_behind=3,
        ),
    )

    console = Console(file=io.StringIO(), force_terminal=False, width=120)
    result = dispatch("/update check", session, console)

    assert result == "continue"
    output = console.file.getvalue()
    assert "Update available" in output
    assert "abc1234" in output
    assert "def5678" in output


def test_slash_update_check_error(session, monkeypatch):
    import io

    from kiwimatecoder import updater as updater_mod
    from kiwimatecoder.commands import dispatch

    monkeypatch.setattr(
        updater_mod,
        "check_for_update",
        lambda **kwargs: updater_mod.UpdateInfo(
            update_available=False,
            current_version="0.1.0",
            latest_version="0.1.0",
            error="could not reach GitHub",
        ),
    )

    console = Console(file=io.StringIO(), force_terminal=False, width=120)
    result = dispatch("/update check", session, console)

    assert result == "continue"
    assert "Could not check for updates" in console.file.getvalue()


def test_slash_update_auto_check_toggle(session, monkeypatch, tmp_path):
    import io

    from kiwimatecoder import config
    from kiwimatecoder.commands import dispatch

    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")

    console = Console(file=io.StringIO(), force_terminal=False, width=120)

    dispatch("/update auto-check off", session, console)
    assert config.get_update()["auto_check"] is False
    assert "disabled" in console.file.getvalue()

    console2 = Console(file=io.StringIO(), force_terminal=False, width=120)
    dispatch("/update auto-check on", session, console2)
    assert config.get_update()["auto_check"] is True
    assert "enabled" in console2.file.getvalue()

    console3 = Console(file=io.StringIO(), force_terminal=False, width=120)
    dispatch("/update auto-check", session, console3)
    assert "on" in console3.file.getvalue()


def test_slash_update_interval(session, monkeypatch, tmp_path):
    import io

    from kiwimatecoder import config
    from kiwimatecoder.commands import dispatch

    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")

    console = Console(file=io.StringIO(), force_terminal=False, width=120)

    dispatch("/update interval 12", session, console)
    assert config.get_update()["check_interval_hours"] == 12

    console2 = Console(file=io.StringIO(), force_terminal=False, width=120)
    dispatch("/update interval", session, console2)
    assert "12" in console2.file.getvalue()


def test_slash_update_runs_updater(session, monkeypatch):
    import io

    from kiwimatecoder import updater as updater_mod
    from kiwimatecoder.commands import dispatch

    called = []

    def fake_run_update(console=None, ref=None):
        called.append(ref)
        return 0

    monkeypatch.setattr(updater_mod, "run_update", fake_run_update)

    console = Console(file=io.StringIO(), force_terminal=False, width=120)
    result = dispatch("/update", session, console)

    assert result == "continue"
    assert called == [None]
    assert "Restart" in console.file.getvalue()


def test_slash_update_unknown_action(session):
    import io

    from kiwimatecoder.commands import dispatch

    console = Console(file=io.StringIO(), force_terminal=False, width=120)
    result = dispatch("/update bogus", session, console)

    assert result == "continue"
    assert "Unknown /update action" in console.file.getvalue()


def test_slash_config_update_show(session, monkeypatch, tmp_path):
    import io

    from kiwimatecoder import config
    from kiwimatecoder.commands import dispatch

    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")

    console = Console(file=io.StringIO(), force_terminal=False, width=120)
    result = dispatch("/config update show", session, console)

    assert result == "continue"
    output = console.file.getvalue()
    assert "auto-check" in output
    assert "interval" in output


def test_slash_config_update_roundtrip(session, monkeypatch, tmp_path):
    import io

    from kiwimatecoder import config
    from kiwimatecoder.commands import dispatch

    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")

    console = Console(file=io.StringIO(), force_terminal=False, width=120)

    dispatch("/config update auto-check off", session, console)
    assert config.get_update()["auto_check"] is False

    dispatch("/config update interval 6", session, console)
    assert config.get_update()["check_interval_hours"] == 6

    dispatch("/config update show", session, console)
    output = console.file.getvalue()
    assert "off" in output
    assert "6h" in output
