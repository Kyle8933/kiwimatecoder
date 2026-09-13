"""Shared team policy overlays for multi-user repositories.

A team policy is a JSON file with the same shape as a project
``.kiwimatecoder.json`` that is layered on top of global and project config.
Policy values always win; when ``team.enforce`` is on, each policy key replaces
the merged value wholesale and a project overlay may not add or override any
key in the policy namespace (a cloned repository cannot widen or weaken the
team's policy).

There is **no server component**: the file is read from the local path the user
configured in ``team.policy_path`` and API keys are never read from it. A
missing or corrupt policy never breaks config loading; :func:`policy_issues`
reports it and ``config validate`` / ``doctor`` surface it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

# Config keys a shared policy is allowed to control. ``team`` itself is not a
# policy key: a policy cannot redirect or disable policy loading.
POLICY_KEYS: tuple[str, ...] = (
    "default_mode",
    "selected_provider",
    "active_providers",
    "selected_model",
    "model_filters",
    "command_rules",
    "tool_permissions",
    "sampling",
    "trusted_workspace",
    "output_style",
    "system_prompt",
    "verify_command",
    "budget",
    "mcp_servers",
    "plugins",
    "subagents",
    "sandbox",
    "remote",
    "network",
)

# Keys that are never policy-controlled: ``keys`` would carry secrets and
# ``version``/``team`` are config bookkeeping. They are ignored silently.
_IGNORED_KEYS = frozenset({"keys", "version", "team"})

# path -> (mtime, size, parsed policy); re-read only when the file changes.
_CACHE: dict[str, tuple[float, int, dict[str, Any]]] = {}


class PolicyError(Exception):
    """Raised when a configured policy file is missing or unreadable."""


def policy_keys() -> tuple[str, ...]:
    """Return the config keys a shared policy may control."""
    return POLICY_KEYS


def clear_policy_cache() -> None:
    """Drop the mtime cache (tests and callers that rewrite a policy file)."""
    _CACHE.clear()


def _team_section(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Return the normalized ``team`` section from ``cfg`` (or active config)."""
    if cfg is None:
        from kiwimatecoder import config as config_module

        cfg = config_module.load_config()
    stored = cfg.get("team") or {}
    if not isinstance(stored, dict):
        return {"policy_path": "", "enforce": False}
    path = stored.get("policy_path")
    enforce = stored.get("enforce")
    return {
        "policy_path": path.strip() if isinstance(path, str) else "",
        "enforce": enforce if isinstance(enforce, bool) else False,
    }


def _read_policy_file(path: Path) -> dict[str, Any]:
    """Read, validate, and filter one policy file (cached by mtime+size)."""
    try:
        stat = path.stat()
    except OSError as exc:
        raise PolicyError(f"Team policy file not found: {path}") from exc
    cache_key = str(path)
    cached = _CACHE.get(cache_key)
    if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
        return copy.deepcopy(cached[2])
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PolicyError(f"Cannot read team policy {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PolicyError(f"Team policy {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PolicyError(f"Team policy {path} must be a JSON object.")
    policy = {key: value for key, value in data.items() if key in POLICY_KEYS}
    _CACHE[cache_key] = (stat.st_mtime, stat.st_size, copy.deepcopy(policy))
    return policy


def load_policy(
    path: str | Path | None = None, cfg: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Load the shared policy, returning only policy-controlled keys.

    ``path`` defaults to ``team.policy_path`` in ``cfg`` (or the active config
    when ``cfg`` is None); an unset path yields an empty policy. Raises
    :class:`PolicyError` for a missing file, unreadable data, invalid JSON, or
    a document that is not a JSON object. ``keys`` is never read from a policy.
    """
    if path is None:
        resolved = _team_section(cfg)["policy_path"]
        if not resolved:
            return {}
        target = Path(resolved).expanduser()
    else:
        target = Path(path).expanduser()
    return _read_policy_file(target)


def policy_issues(cfg: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Return issues with the configured policy file.

    A missing or corrupt policy never breaks config loading, so this is how
    ``config validate`` and ``doctor`` surface it. Unknown policy keys are
    warnings because a newer release may add keys this one does not know.
    """
    path = _team_section(cfg)["policy_path"]
    if not path:
        return []
    target = Path(path).expanduser()
    if not target.is_file():
        return [
            {
                "level": "error",
                "key": "team.policy_path",
                "message": f"Team policy file not found: {path}",
            }
        ]
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return [
            {
                "level": "error",
                "key": "team.policy_path",
                "message": f"Cannot read team policy {path}: {exc}",
            }
        ]
    if not isinstance(raw, dict):
        return [
            {
                "level": "error",
                "key": "team.policy_path",
                "message": f"Team policy {path} must be a JSON object.",
            }
        ]
    issues: list[dict[str, str]] = []
    for key in sorted(set(raw) - set(POLICY_KEYS) - _IGNORED_KEYS):
        issues.append(
            {
                "level": "warning",
                "key": f"team.policy_path.{key}",
                "message": f"Unknown policy key '{key}' is ignored.",
            }
        )
    return issues


def _deep_overlay(cfg: dict[str, Any], overlay: dict[str, Any]) -> None:
    """Deep-merge ``overlay`` onto ``cfg`` (overlay values win per sub-key)."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            merged = dict(cfg[key])
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict) and isinstance(merged.get(sub_key), dict):
                    merged[sub_key] = {**merged[sub_key], **sub_value}
                else:
                    merged[sub_key] = sub_value
            cfg[key] = merged
        else:
            cfg[key] = value


def apply_policy(
    cfg: dict[str, Any],
    *,
    project: dict[str, Any] | None = None,
    global_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the configured policy to ``cfg`` in place and return it.

    Called by :func:`kiwimatecoder.config.load_config` after the project
    overlay so policy values win. With ``team.enforce`` on, every policy key
    replaces the merged value wholesale and project values for policy keys the
    policy does not define are rolled back to the global config, so a
    repository cannot widen or weaken the policy namespace. Failures are
    swallowed so a missing/corrupt file can never break config loading;
    :func:`policy_issues` reports them.
    """
    section = _team_section(cfg)
    path = section["policy_path"]
    if not path:
        return {}
    try:
        policy = load_policy(path, cfg)
    except PolicyError:
        return {}
    if not policy:
        return {}
    if section["enforce"]:
        if project:
            for key in POLICY_KEYS:
                if key not in project or key in policy:
                    continue
                if global_cfg is not None and key in global_cfg:
                    cfg[key] = copy.deepcopy(global_cfg[key])
                else:
                    cfg.pop(key, None)
            # A project selected_provider seeds active_providers before the
            # policy is applied; roll that seeding back too unless the policy
            # pins the roster (or a provider) itself.
            if (
                "selected_provider" in project
                and "active_providers" not in project
                and "selected_provider" not in policy
                and "active_providers" not in policy
            ):
                if global_cfg is not None and "active_providers" in global_cfg:
                    cfg["active_providers"] = copy.deepcopy(
                        global_cfg["active_providers"]
                    )
                else:
                    cfg.pop("active_providers", None)
        for key, value in policy.items():
            cfg[key] = copy.deepcopy(value)
    else:
        _deep_overlay(cfg, policy)
    return policy
