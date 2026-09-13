"""Tests for the eval harness: schema, discovery, evaluation, reports, and CLI."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kiwimatecoder import config, main, sdk
from kiwimatecoder.evals import cases as eval_cases
from kiwimatecoder.evals import runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _write_case(directory: Path, filename: str, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _case(**overrides) -> eval_cases.EvalCase:
    payload = {"name": "sample", "prompt": "do the thing", "expect": {}}
    payload.update(overrides)
    return eval_cases.parse_case(payload)


def _result(**overrides) -> sdk.RunResult:
    values = {
        "text": "",
        "usage": {},
        "cost_usd": None,
        "provider": "openrouter",
        "model": "test-model",
        "mode": "auto-accept",
        "tools_used": [],
        "messages": 0,
        "success": True,
        "error": None,
    }
    values.update(overrides)
    return sdk.RunResult(**values)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"prompt": "p", "expect": {}}, "'name'"),
        ({"name": "n", "expect": {}}, "'prompt'"),
        ({"name": "n", "prompt": "p"}, "expect"),
        ({"name": "n", "prompt": "p", "expect": {}, "extra": 1}, "unknown key"),
        ({"name": "n", "prompt": "p", "expect": {"nope": []}}, "unknown expect key"),
        ({"name": " ", "prompt": "p", "expect": {}}, "'name'"),
    ],
)
def test_parse_case_requires_expected_fields(payload, message):
    with pytest.raises(eval_cases.CaseError, match=message):
        eval_cases.parse_case(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"name": "n", "prompt": "p", "description": 5, "expect": {}}, "description"),
        ({"name": "n", "prompt": "p", "files": [], "expect": {}}, "'files'"),
        ({"name": "n", "prompt": "p", "files": {"a.txt": 5}, "expect": {}}, "string"),
        ({"name": "n", "prompt": "p", "expect": {"text_contains": "x"}}, "list of strings"),
        ({"name": "n", "prompt": "p", "expect": {"tools_called": ["ok", 3]}}, "non-empty"),
        ({"name": "n", "prompt": "p", "expect": {"success": "yes"}}, "boolean"),
        ({"name": "n", "prompt": "p", "expect": {"files": []}}, "object mapping"),
        ({"name": "n", "prompt": "p", "expect": {"files": {"a.txt": "x"}}}, "list"),
    ],
)
def test_parse_case_rejects_bad_types(payload, message):
    with pytest.raises(eval_cases.CaseError, match=message):
        eval_cases.parse_case(payload)


@pytest.mark.parametrize(
    "bad_path",
    ["../escape.txt", "/etc/passwd", "~/secret", "dir\\file.txt", ""],
)
def test_parse_case_rejects_unsafe_paths(bad_path):
    with pytest.raises(eval_cases.CaseError):
        eval_cases.parse_case(
            {"name": "n", "prompt": "p", "files": {bad_path: "x"}, "expect": {}}
        )
    with pytest.raises(eval_cases.CaseError):
        eval_cases.parse_case(
            {"name": "n", "prompt": "p", "expect": {"files": {bad_path: ["x"]}}}
        )


def test_load_case_reports_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(eval_cases.CaseError, match="invalid JSON"):
        eval_cases.load_case(path)


def test_load_case_reports_missing_file(tmp_path):
    with pytest.raises(eval_cases.CaseError, match="could not read"):
        eval_cases.load_case(tmp_path / "missing.json")


def test_parse_case_accepts_full_schema():
    case = eval_cases.parse_case(
        {
            "name": "full",
            "description": "all fields",
            "prompt": "do it",
            "files": {"pkg/mod.py": "x = 1\n"},
            "expect": {
                "text_contains": ["done"],
                "text_not_contains": ["error"],
                "tools_called": ["write_file"],
                "tools_not_called": ["run_bash"],
                "files": {"pkg/mod.py": ["x = 1"]},
                "success": True,
            },
        }
    )

    assert case.name == "full"
    assert case.description == "all fields"
    assert case.files == {"pkg/mod.py": "x = 1\n"}
    assert case.expect.text_contains == ("done",)
    assert case.expect.files == {"pkg/mod.py": ("x = 1",)}
    assert case.expect.success is True


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discover_cases_sorts_by_name(tmp_path):
    _write_case(tmp_path, "b.json", {"name": "zeta", "prompt": "p", "expect": {}})
    _write_case(tmp_path, "a.json", {"name": "alpha", "prompt": "p", "expect": {}})

    names = [case.name for case in eval_cases.discover_cases(tmp_path)]

    assert names == ["alpha", "zeta"]


def test_discover_cases_rejects_duplicate_names(tmp_path):
    _write_case(tmp_path, "a.json", {"name": "dup", "prompt": "p", "expect": {}})
    _write_case(tmp_path, "b.json", {"name": "dup", "prompt": "p", "expect": {}})

    with pytest.raises(eval_cases.CaseError, match="Duplicate"):
        eval_cases.discover_cases(tmp_path)


def test_discover_cases_missing_directory(tmp_path):
    with pytest.raises(eval_cases.CaseError, match="not found"):
        eval_cases.discover_cases(tmp_path / "nope")


def test_committed_example_cases_are_valid_and_sorted():
    discovered = eval_cases.discover_cases(ROOT / "evals" / "cases")

    assert len(discovered) >= 3
    assert discovered == sorted(discovered, key=lambda case: case.name.lower())
    assert all(case.prompt.strip() for case in discovered)


# ---------------------------------------------------------------------------
# Materialization and evaluation
# ---------------------------------------------------------------------------


def test_materialize_case_writes_nested_files(tmp_path):
    case = _case(files={"pkg/mod.py": "x = 1\n", "README.md": "hi\n"})

    written = runner.materialize_case(case, tmp_path)

    assert sorted(written) == ["README.md", "pkg/mod.py"]
    assert (tmp_path / "pkg" / "mod.py").read_text(encoding="utf-8") == "x = 1\n"
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "hi\n"


def test_evaluate_case_passes_when_all_expectations_hold(tmp_path):
    (tmp_path / "hello.txt").write_text("hello eval", encoding="utf-8")
    case = _case(
        expect={
            "text_contains": ["done"],
            "text_not_contains": ["failed"],
            "tools_called": ["write_file"],
            "tools_not_called": ["run_bash"],
            "files": {"hello.txt": ["hello eval"]},
            "success": True,
        }
    )
    result = _result(text="all done", tools_used=["write_file"])

    assert runner.evaluate_case(case, result, tmp_path) == []


def test_evaluate_case_reports_text_expectations(tmp_path):
    case = _case(expect={"text_contains": ["needle"], "text_not_contains": ["forbidden"]})

    problems = runner.evaluate_case(case, _result(text="the forbidden word"), tmp_path)

    assert any("text_contains" in problem and "needle" in problem for problem in problems)
    assert any("text_not_contains" in problem and "forbidden" in problem for problem in problems)


def test_evaluate_case_reports_tool_expectations(tmp_path):
    case = _case(expect={"tools_called": ["read_file"], "tools_not_called": ["run_bash"]})

    problems = runner.evaluate_case(case, _result(tools_used=["run_bash"]), tmp_path)

    assert any("read_file" in problem and "not used" in problem for problem in problems)
    assert any("run_bash" in problem and "was used" in problem for problem in problems)


def test_evaluate_case_reports_file_expectations(tmp_path):
    (tmp_path / "exists.txt").write_text("other", encoding="utf-8")
    case = _case(expect={"files": {"exists.txt": ["wanted"], "missing.txt": ["x"]}})

    problems = runner.evaluate_case(case, _result(), tmp_path)

    assert any("exists.txt" in problem and "does not contain" in problem for problem in problems)
    assert any("missing.txt" in problem and "was not created" in problem for problem in problems)


def test_evaluate_case_reports_success_mismatch(tmp_path):
    case = _case(expect={"success": True})

    problems = runner.evaluate_case(case, _result(success=False, error="boom"), tmp_path)

    assert problems == ["success: expected True, got False"]


def test_evaluate_case_flags_unexpected_run_error(tmp_path):
    case = _case(expect={})

    problems = runner.evaluate_case(case, _result(success=False, error="boom"), tmp_path)

    assert problems == ["run failed: boom"]


def test_evaluate_case_accepts_expected_failure(tmp_path):
    case = _case(expect={"success": False})

    assert runner.evaluate_case(case, _result(success=False, error="boom"), tmp_path) == []


# ---------------------------------------------------------------------------
# run_case / run_suite
# ---------------------------------------------------------------------------


def test_run_case_materializes_workspace_and_calls_sdk(monkeypatch):
    calls: dict = {}

    def fake_run(prompt, **kwargs):
        calls["prompt"] = prompt
        calls.update(kwargs)
        workspace = Path(kwargs["workspace"])
        (workspace / "hello.txt").write_text("hello from the eval", encoding="utf-8")
        return _result(text="created hello.txt", tools_used=["write_file"])

    monkeypatch.setattr(sdk, "run_agent_sync", fake_run)
    case = _case(
        name="create",
        prompt="create hello",
        files={"notes.txt": "start\n"},
        expect={
            "text_contains": ["created"],
            "tools_called": ["write_file"],
            "files": {"hello.txt": ["hello from the eval"], "notes.txt": ["start"]},
            "success": True,
        },
    )

    result = runner.run_case(case, provider="openrouter", model="test-model", max_turns=5)

    assert result.success is True
    assert result.reasons == []
    assert calls["prompt"] == "create hello"
    assert calls["provider"] == "openrouter"
    assert calls["model"] == "test-model"
    assert calls["mode"] == "auto-accept"
    assert calls["max_turns"] == 5
    assert calls["workspace"].name.startswith("kiwimatecoder-eval-")


def test_run_case_reports_agent_exception(monkeypatch):
    def fake_run(prompt, **kwargs):
        raise RuntimeError("missing API key")

    monkeypatch.setattr(sdk, "run_agent_sync", fake_run)

    result = runner.run_case(_case(expect={"success": True}))

    assert result.success is False
    assert result.error == "RuntimeError: missing API key"
    assert result.reasons == ["run failed: RuntimeError: missing API key"]


def test_run_case_times_out(monkeypatch):
    def fake_run(prompt, **kwargs):
        time.sleep(0.3)
        return _result()

    monkeypatch.setattr(sdk, "run_agent_sync", fake_run)

    result = runner.run_case(_case(expect={}), timeout=0.05)

    assert result.success is False
    assert result.error == "timeout"
    assert "timed out" in result.reasons[0]


def test_run_suite_builds_report_and_writes_json(tmp_path, monkeypatch):
    cases_dir = tmp_path / "cases"
    _write_case(cases_dir, "a.json", {"name": "alpha", "prompt": "p", "expect": {}})
    _write_case(cases_dir, "b.json", {"name": "beta", "prompt": "p", "expect": {}})

    def fake_run_case(case, **kwargs):
        return runner.EvalResult(
            name=case.name,
            success=case.name == "alpha",
            reasons=[] if case.name == "alpha" else ["nope"],
            duration_s=0.5,
        )

    monkeypatch.setattr(runner, "run_case", fake_run_case)
    report_path = tmp_path / "reports" / "eval.json"

    report = runner.run_suite(
        cases_dir, provider="openrouter", model="test-model", report_path=report_path
    )

    assert report.total == 2
    assert report.passed == 1
    assert report.failed == 1
    assert report.success is False
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["provider"] == "openrouter"
    assert payload["model"] == "test-model"
    assert (payload["total"], payload["passed"], payload["failed"]) == (2, 1, 1)
    assert payload["success"] is False
    assert [item["name"] for item in payload["results"]] == ["alpha", "beta"]
    assert payload["results"][1]["reasons"] == ["nope"]
    assert set(payload["results"][0]) >= {
        "name",
        "success",
        "reasons",
        "duration_s",
        "error",
        "tools_used",
        "text",
    }


def test_run_suite_filter_selects_matching_cases(tmp_path, monkeypatch):
    cases_dir = tmp_path / "cases"
    _write_case(cases_dir, "a.json", {"name": "alpha", "prompt": "p", "expect": {}})
    _write_case(cases_dir, "b.json", {"name": "beta", "prompt": "p", "expect": {}})
    seen: list[str] = []

    def fake_run_case(case, **kwargs):
        seen.append(case.name)
        return runner.EvalResult(name=case.name, success=True)

    monkeypatch.setattr(runner, "run_case", fake_run_case)

    report = runner.run_suite(cases_dir, filter="ALP")

    assert [result.name for result in report.results] == ["alpha"]
    assert seen == ["alpha"]


def test_run_suite_no_match_raises(tmp_path):
    cases_dir = tmp_path / "cases"
    _write_case(cases_dir, "a.json", {"name": "alpha", "prompt": "p", "expect": {}})

    with pytest.raises(eval_cases.CaseError, match="matched"):
        runner.run_suite(cases_dir, filter="zzz")


def test_run_suite_empty_directory_raises(tmp_path):
    cases_dir = tmp_path / "empty"
    cases_dir.mkdir()

    with pytest.raises(eval_cases.CaseError, match="No eval cases found"):
        runner.run_suite(cases_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _fake_report(success: bool) -> runner.EvalReport:
    reasons = [] if success else ["run failed: RuntimeError: missing API key"]
    return runner.EvalReport(
        results=[runner.EvalResult(name="alpha", success=success, reasons=reasons)],
        provider="openrouter",
        model="test-model",
        cases_dir="cases",
    )


def test_eval_list_prints_cases(tmp_path):
    cases_dir = tmp_path / "cases"
    _write_case(
        cases_dir,
        "a.json",
        {"name": "alpha", "description": "first case", "prompt": "p", "expect": {}},
    )

    result = CliRunner().invoke(main.app, ["eval", "list", "--dir", str(cases_dir)])

    assert result.exit_code == 0
    assert "alpha" in result.stdout
    assert "first case" in result.stdout


def test_eval_list_bad_dir_exits_2(tmp_path):
    result = CliRunner().invoke(
        main.app, ["eval", "list", "--dir", str(tmp_path / "nope")]
    )

    assert result.exit_code == 2
    assert "not found" in result.stdout


def test_eval_run_exits_0_when_all_pass(monkeypatch):
    monkeypatch.setattr(runner, "run_suite", lambda *args, **kwargs: _fake_report(True))

    result = CliRunner().invoke(main.app, ["eval", "run"])

    assert result.exit_code == 0
    assert "PASS" in result.stdout
    assert "1/1" in result.stdout


def test_eval_run_exits_1_on_failure_and_prints_key_guidance(monkeypatch):
    monkeypatch.setattr(runner, "run_suite", lambda *args, **kwargs: _fake_report(False))

    result = CliRunner().invoke(main.app, ["eval", "run"])

    assert result.exit_code == 1
    assert "FAIL" in result.stdout
    assert "provider-related" in result.stdout


def test_eval_run_json_dumps_report(monkeypatch):
    monkeypatch.setattr(runner, "run_suite", lambda *args, **kwargs: _fake_report(True))

    result = CliRunner().invoke(main.app, ["eval", "run", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert payload["results"][0]["name"] == "alpha"


def test_eval_run_invalid_input_exits_2(monkeypatch):
    def boom(*args, **kwargs):
        raise runner.CaseError("No eval cases found")

    monkeypatch.setattr(runner, "run_suite", boom)

    result = CliRunner().invoke(main.app, ["eval", "run"])

    assert result.exit_code == 2
    assert "No eval cases found" in result.stdout


def test_eval_run_unknown_provider_exits_2():
    result = CliRunner().invoke(
        main.app, ["eval", "run", "--provider", "does-not-exist"]
    )

    assert result.exit_code == 2


def test_eval_run_passes_options_to_suite(monkeypatch):
    captured: dict = {}

    def fake_suite(cases_dir=None, **kwargs):
        captured["cases_dir"] = cases_dir
        captured.update(kwargs)
        return _fake_report(True)

    monkeypatch.setattr(runner, "run_suite", fake_suite)

    result = CliRunner().invoke(
        main.app,
        [
            "eval",
            "run",
            "--dir",
            "somewhere",
            "--provider",
            "openrouter",
            "--model",
            "test-model",
            "--filter",
            "x",
            "--report",
            "rep.json",
        ],
    )

    assert result.exit_code == 0
    assert str(captured["cases_dir"]) == "somewhere"
    assert captured["provider"] == "openrouter"
    assert captured["model"] == "test-model"
    assert captured["filter"] == "x"
    assert str(captured["report_path"]) == "rep.json"
