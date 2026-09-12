"""String checks for the committed CI, pre-commit, and automation docs.

These intentionally avoid a YAML parser dependency: the goal is to catch a
missing or gutted asset, not to re-implement GitHub's workflow schema.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
PRE_COMMIT = ROOT / ".pre-commit-config.yaml"
CI_DOC = ROOT / "docs" / "ci.md"


def test_ci_workflow_exists_with_expected_jobs_and_commands():
    text = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "test:" in text
    assert "lint:" in text
    assert "types:" in text
    assert "pytest -q" in text
    assert "ruff check kiwimatecoder tests" in text
    assert "mypy kiwimatecoder" in text
    assert 'pip install -e ".[dev]"' in text
    assert "pull_request" in text
    assert "push" in text


def test_ci_workflow_matrix_covers_supported_pythons_and_oses():
    text = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "ubuntu-latest" in text
    assert "macos-latest" in text
    for version in ("3.10", "3.11", "3.12"):
        assert f'"{version}"' in text


def test_pre_commit_config_has_ruff_check_and_pre_push_pytest():
    text = PRE_COMMIT.read_text(encoding="utf-8")

    assert "ruff" in text
    assert "ruff-format" not in text
    assert "pytest" in text
    assert "language: system" in text
    assert "pre-push" in text
    assert "stages: [pre-push]" in text


def test_ci_docs_show_headless_review_example_and_exit_codes():
    text = CI_DOC.read_text(encoding="utf-8")

    assert 'kiwimatecoder -p "Review the staged diff for bugs"' in text
    assert "--output-format json" in text
    assert "OPENROUTER_API_KEY" in text
    assert "secrets.OPENROUTER_API_KEY" in text
    assert "| `0` |" in text
    assert "| `1` |" in text
    assert "| `2` |" in text
