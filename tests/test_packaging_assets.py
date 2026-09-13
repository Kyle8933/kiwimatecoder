"""String and smoke checks for the packaging assets.

No Docker, Homebrew, or YAML parser dependency: these catch a missing or
gutted asset. The SBOM script is executed because it is stdlib-only; nothing
here touches the network.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
INSTALL_SH = ROOT / "scripts" / "install.sh"
HOMEBREW = ROOT / "packaging" / "homebrew" / "kiwimatecoder.rb"
SBOM_SCRIPT = ROOT / "scripts" / "generate_sbom.py"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def test_dockerfile_installs_package_and_runs_nonroot():
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "FROM python:3.12-slim" in text
    assert "pip install --no-cache-dir ." in text
    assert "useradd" in text
    assert "USER kiwimate" in text
    assert 'ENTRYPOINT ["kiwimatecoder"]' in text


def test_dockerignore_excludes_dev_artifacts():
    entries = {
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    assert ".git" in entries
    assert "venv" in entries
    assert "__pycache__/" in entries
    assert "*.egg-info/" in entries


def test_install_script_is_posix_and_parses_with_sh_n():
    text = INSTALL_SH.read_text(encoding="utf-8")

    assert text.startswith("#!/bin/sh")
    assert "set -eu" in text
    assert "pipx" in text
    assert "pip install --user" in text
    assert "kiwimatecoder --version" in text
    assert "sys.version_info >= (3, 10)" in text
    assert "curl" not in text

    sh = shutil.which("sh")
    if sh is None:  # pragma: no cover - exercised by the non-blocking Windows job
        pytest.skip("POSIX sh is not available")
    result = subprocess.run(
        [sh, "-n", str(INSTALL_SH)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_homebrew_formula_is_a_virtualenv_template():
    text = HOMEBREW.read_text(encoding="utf-8")

    assert "class Kiwimatecoder < Formula" in text
    assert "Language::Python::Virtualenv" in text
    assert "virtualenv_install_with_resources" in text
    assert "https://github.com/Kyle8933/kiwimatecoder.git" in text
    assert "tag:" in text
    assert "revision:" in text
    assert "update-python-resources" in text
    assert "bump" in text


def test_generate_sbom_emits_cyclonedx_shape(tmp_path):
    target = tmp_path / "out" / "sbom.json"
    result = subprocess.run(
        [sys.executable, str(SBOM_SCRIPT), "--output", str(target)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    bom = json.loads(target.read_text(encoding="utf-8"))
    assert bom["bomFormat"] == "CycloneDX"
    assert bom["specVersion"]
    assert bom["version"] == 1
    assert bom["metadata"]["component"]["name"] == "kiwimatecoder"
    assert bom["metadata"]["component"]["type"] == "application"

    by_name = {component["name"]: component for component in bom["components"]}
    assert "kiwimatecoder" in by_name
    assert by_name["kiwimatecoder"]["version"]
    for dependency in ("typer", "rich", "httpx", "prompt_toolkit"):
        assert dependency in by_name, f"{dependency} missing from SBOM"
        assert by_name[dependency]["version"]
    names = [component["name"].lower() for component in bom["components"]]
    assert names == sorted(names)


def test_generate_sbom_is_deterministic(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    for target in (first, second):
        result = subprocess.run(
            [sys.executable, str(SBOM_SCRIPT), "--output", str(target)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_ci_workflow_has_sbom_and_windows_jobs():
    text = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "sbom:" in text
    assert "generate_sbom.py" in text
    assert "actions/upload-artifact" in text
    assert "windows-latest" in text
    assert "continue-on-error: true" in text
    assert "windows, non-blocking" in text
