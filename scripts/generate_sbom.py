#!/usr/bin/env python3
"""Generate a CycloneDX-shaped SBOM from installed package metadata.

Stdlib only: the package itself plus its direct runtime dependencies are read
from :mod:`importlib.metadata`, so the SBOM describes the environment that
actually has KiwiMateCoder installed. Output is deterministic (components are
sorted by name).

Usage:
    python scripts/generate_sbom.py [--output sbom.json]
"""

from __future__ import annotations

import argparse
import json
import re
from importlib import metadata
from pathlib import Path
from typing import Any

PACKAGE_NAME = "kiwimatecoder"
SPEC_VERSION = "1.5"
_NAME_SPLIT = re.compile(r"[<>=!~;\[\]\s(]")
_UNDERSCORE = re.compile(r"[-_.]+")


def _normalized(name: str) -> str:
    return _UNDERSCORE.sub("-", name).lower()


def direct_dependencies() -> list[str]:
    """Return the package's direct runtime dependency names, sorted.

    Extras (``browser``, ``dev``) are excluded because they are not part of a
    default install.
    """
    try:
        requirements = metadata.requires(PACKAGE_NAME) or []
    except metadata.PackageNotFoundError:
        return []
    names: set[str] = set()
    for raw in requirements:
        lowered = raw.lower()
        if "extra ==" in lowered or "extra==" in lowered:
            continue
        requirement = raw.split(";", 1)[0].strip()
        if not requirement:
            continue
        name = _NAME_SPLIT.split(requirement, 1)[0].strip()
        if name:
            names.add(name)
    return sorted(names, key=lambda name: name.lower())


def _license_name(meta: metadata.PackageMetadata) -> str | None:
    for key in ("License-Expression", "License"):
        value = (meta.get(key) or "").strip()
        if not value or value.lower() in {"unknown", "none"}:
            continue
        # Some packages embed the full license text; only a short first line
        # is a usable SBOM license name.
        first_line = next((line.strip() for line in value.splitlines() if line.strip()), "")
        if first_line and len(first_line) <= 80:
            return first_line
    for classifier in meta.get_all("Classifier") or []:
        if classifier.startswith("License ::") and not classifier.endswith(
            "License :: OSI Approved"
        ):
            return classifier.rsplit("::", 1)[-1].strip()
    return None


def _component(name: str) -> dict[str, Any] | None:
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
    component: dict[str, Any] = {
        "type": "library",
        "name": name,
        "version": version,
    }
    try:
        meta = metadata.metadata(name)
    except metadata.PackageNotFoundError:
        return component
    license_name = _license_name(meta)
    if license_name:
        component["licenses"] = [{"license": {"name": license_name}}]
    return component


def build_bom() -> dict[str, Any]:
    """Return the SBOM document for the current environment."""
    components: list[dict[str, Any]] = []
    package = _component(PACKAGE_NAME)
    if package is not None:
        components.append(package)
    for dependency in direct_dependencies():
        component = _component(dependency)
        if component is not None:
            components.append(component)
    components.sort(key=lambda item: str(item["name"]).lower())

    application: dict[str, Any] = package or {
        "type": "application",
        "name": PACKAGE_NAME,
    }
    application = {**application, "type": "application"}
    return {
        "bomFormat": "CycloneDX",
        "specVersion": SPEC_VERSION,
        "version": 1,
        "metadata": {
            "tools": [{"vendor": "kiwimatecoder", "name": "generate_sbom.py"}],
            "component": application,
        },
        "components": components,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="sbom.json",
        help="Where to write the SBOM JSON (default: sbom.json).",
    )
    args = parser.parse_args(argv)

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(build_bom(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
