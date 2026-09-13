"""Opt-in cross-machine session sync through a user-provided folder.

There is no server component: a machine copies saved sessions into a shared
folder (Dropbox, iCloud Drive, a network mount, a git checkout) and other
machines copy them back. The feature is disabled by default and never touches
anything until it is explicitly enabled with a path.

The shared folder contains a ``kiwimatecoder-sessions/`` directory with the
session JSON files plus a ``manifest.json`` index. Each machine also keeps a
small local record of what it last pushed or pulled so a file that changed on
two machines at once can be detected instead of silently overwritten. Every
write is best-effort: unreadable or corrupt session files are skipped and
reported, never fatal.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kiwimatecoder import session as session_store
from kiwimatecoder.config import ensure_config_dir, get_sync

MANIFEST_VERSION = 1
SYNC_DIR_NAME = "kiwimatecoder-sessions"
MANIFEST_NAME = "manifest.json"
STATE_NAME = "sync_state.json"


class SyncError(Exception):
    """Raised when sync is unavailable or misconfigured."""


@dataclass
class SyncReport:
    """Outcome of one :func:`push` or :func:`pull`."""

    lines: list[str] = field(default_factory=list)
    pushed: int = 0
    pulled: int = 0
    conflicts: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"pushed {self.pushed}",
            f"pulled {self.pulled}",
            f"conflicts {self.conflicts}",
        ]
        if self.errors:
            parts.append(f"errors {len(self.errors)}")
        return "Sync complete: " + ", ".join(parts) + "."


@dataclass
class SyncStatus:
    """A read-only view of the local and shared session stores."""

    enabled: bool = False
    machine: str = ""
    root: Path | None = None
    local: int = 0
    remote: int = 0
    last_sync: str | None = None
    pending_pushes: list[str] = field(default_factory=list)
    pending_pulls: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.enabled:
            return (
                "Session sync is disabled (opt-in).\n"
                "Enable it with: kiwimatecoder sync enable <folder>"
            )
        lines = [
            f"Sync folder: {self.root}",
            f"Machine: {self.machine}",
            f"Local sessions: {self.local}",
            f"Remote sessions: {self.remote}",
            f"Last sync: {self.last_sync or 'never'}",
            f"Pending pushes: {len(self.pending_pushes)}",
            f"Pending pulls: {len(self.pending_pulls)}",
            f"Conflicts: {len(self.conflicts)}",
        ]
        if self.pending_pushes:
            lines.append("  push: " + ", ".join(self.pending_pushes))
        if self.pending_pulls:
            lines.append("  pull: " + ", ".join(self.pending_pulls))
        if self.conflicts:
            lines.append("  both changed: " + ", ".join(self.conflicts))
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            lines.extend(f"  {error}" for error in self.errors)
        return "\n".join(lines)


def _now() -> str:
    return datetime.datetime.now().isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as JSON next to ``path`` and atomically replace it."""
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _copy_file(source: Path, destination: Path) -> None:
    """Copy the raw bytes of one session file, replacing the destination."""
    data = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.tmp")
    temp.write_bytes(data)
    os.replace(temp, destination)


def _read_meta(path: Path) -> dict[str, str] | None:
    """Return ``{"sha1", "saved_at"}`` for a session file, or None if corrupt.

    Raises ``OSError`` when the file itself cannot be read; callers turn that
    into a report entry rather than a failure.
    """
    raw = path.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return {
        "sha1": hashlib.sha1(raw).hexdigest(),
        "saved_at": str(data.get("saved_at") or ""),
    }


def _safe_meta(path: Path) -> dict[str, str] | None:
    try:
        return _read_meta(path)
    except OSError:
        return None


def _record(store: dict[str, Any], name: str, meta: dict[str, str]) -> None:
    store[name] = {"saved_at": meta.get("saved_at", ""), "sha1": meta["sha1"]}


def _require_enabled(settings: dict[str, Any]) -> None:
    if not settings["enabled"]:
        raise SyncError(
            "Session sync is disabled. Enable it with: "
            "kiwimatecoder sync enable <folder>"
        )
    if not settings["path"]:
        raise SyncError(
            "Session sync has no folder configured. Run: "
            "kiwimatecoder sync enable <folder>"
        )


def sync_root(cfg: dict[str, Any] | None = None) -> Path:
    """Return (creating on demand) the shared ``kiwimatecoder-sessions/`` dir.

    Raises :class:`SyncError` when sync is disabled or has no usable folder.
    """
    settings = get_sync(cfg)
    _require_enabled(settings)
    root = Path(settings["path"]).expanduser() / SYNC_DIR_NAME
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SyncError(f"Could not create the sync folder {root}: {exc}") from exc
    return root


def _filtered(paths: Iterable[Path], include_autosave: bool) -> list[Path]:
    """Sort session paths and drop the autosave unless it is opted in."""
    result: list[Path] = []
    for path in sorted(paths, key=lambda item: item.name):
        if not path.is_file():
            continue
        if not include_autosave and path.stem == session_store.AUTOSAVE_NAME:
            continue
        result.append(path)
    return result


def _local_session_files(include_autosave: bool) -> list[Path]:
    sessions_dir = session_store._sessions_dir()
    return [
        path
        for path in _filtered(sessions_dir.glob("*.json"), include_autosave)
        if path.name != MANIFEST_NAME
    ]


def _remote_session_files(root: Path, include_autosave: bool) -> list[Path]:
    return [
        path
        for path in _filtered(root.glob("*.json"), include_autosave)
        if path.name != MANIFEST_NAME
    ]


def _load_manifest(root: Path) -> dict[str, Any]:
    """Load the shared manifest, tolerating absence and corruption."""
    try:
        data = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = None
    machine = ""
    last_sync: str | None = None
    files: dict[str, dict[str, str]] = {}
    if isinstance(data, dict) and data.get("version") == MANIFEST_VERSION:
        if isinstance(data.get("machine"), str):
            machine = data["machine"]
        if isinstance(data.get("last_sync"), str):
            last_sync = data["last_sync"]
        raw_files = data.get("files")
        if isinstance(raw_files, dict):
            for name, entry in raw_files.items():
                if not isinstance(entry, dict):
                    continue
                sha1 = entry.get("sha1")
                if not isinstance(sha1, str) or not sha1:
                    continue
                files[str(name)] = {
                    "saved_at": str(entry.get("saved_at") or ""),
                    "sha1": sha1,
                }
    return {
        "version": MANIFEST_VERSION,
        "machine": machine,
        "last_sync": last_sync,
        "files": files,
    }


def _state_file() -> Path:
    return ensure_config_dir() / STATE_NAME


def _load_state(machine: str) -> dict[str, Any]:
    """Load this machine's record of what it last pushed/pulled.

    A record written by a different machine (the config was copied, or the
    ``machine`` name changed) is discarded so it can never mislabel baselines.
    """
    try:
        data = json.loads(_state_file().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = None
    files: dict[str, dict[str, str]] = {}
    if isinstance(data, dict) and data.get("machine") == machine:
        raw_files = data.get("files")
        if isinstance(raw_files, dict):
            for name, entry in raw_files.items():
                if not isinstance(entry, dict):
                    continue
                sha1 = entry.get("sha1")
                if not isinstance(sha1, str) or not sha1:
                    continue
                files[str(name)] = {
                    "saved_at": str(entry.get("saved_at") or ""),
                    "sha1": sha1,
                }
    return {"version": 1, "machine": machine, "files": files}


def _conflict_name(name: str, machine: str, directory: Path) -> str:
    """Pick a free ``<name>__<machine>.json`` conflict filename."""
    stem = Path(name).stem
    slug = re.sub(r"[^\w\-]", "_", str(machine).strip()) or "machine"
    candidate = f"{stem}__{slug}.json"
    counter = 2
    while (directory / candidate).exists():
        candidate = f"{stem}__{slug}-{counter}.json"
        counter += 1
    return candidate


def status(cfg: dict[str, Any] | None = None) -> SyncStatus:
    """Compare the local and shared session stores without copying anything."""
    settings = get_sync(cfg)
    result = SyncStatus(enabled=settings["enabled"], machine=settings["machine"])
    if not settings["enabled"]:
        return result
    try:
        root = sync_root(cfg)
    except SyncError as exc:
        result.errors.append(str(exc))
        return result
    manifest = _load_manifest(root)
    state = _load_state(settings["machine"])
    result.root = root
    result.last_sync = manifest["last_sync"]

    local: dict[str, dict[str, str]] = {}
    for path in _local_session_files(settings["include_autosave"]):
        meta = _safe_meta(path)
        if meta is None:
            result.errors.append(f"Skipped unreadable session: {path.name}")
        else:
            local[path.name] = meta
    remote: dict[str, dict[str, str]] = {}
    for path in _remote_session_files(root, settings["include_autosave"]):
        meta = _safe_meta(path)
        if meta is None:
            result.errors.append(f"Skipped corrupt session: {path.name}")
        else:
            remote[path.name] = meta
    result.local = len(local)
    result.remote = len(remote)

    for name in sorted(local):
        local_meta = local[name]
        remote_meta = remote.get(name)
        if remote_meta is None:
            result.pending_pushes.append(name)
            continue
        if remote_meta["sha1"] == local_meta["sha1"]:
            continue
        base = state["files"].get(name)
        local_changed = base is None or base["sha1"] != local_meta["sha1"]
        remote_changed = base is None or base["sha1"] != remote_meta["sha1"]
        if base is not None and local_changed and remote_changed:
            result.conflicts.append(name)
        elif local_changed or (
            base is None
            and local_meta["saved_at"] >= remote_meta["saved_at"]
        ):
            result.pending_pushes.append(name)
        else:
            result.pending_pulls.append(name)
    for name in sorted(remote):
        if name not in local:
            result.pending_pulls.append(name)
    return result


def _finish(
    root: Path,
    manifest: dict[str, Any],
    state: dict[str, Any],
    settings: dict[str, Any],
    report: SyncReport,
    *,
    claim_machine: bool,
) -> None:
    """Persist the manifest and local baseline, reporting any write failure."""
    manifest["version"] = MANIFEST_VERSION
    if claim_machine or not manifest["machine"]:
        manifest["machine"] = settings["machine"]
    manifest["last_sync"] = _now()
    try:
        _write_json_atomic(root / MANIFEST_NAME, manifest)
    except OSError as exc:
        report.errors.append(f"Could not update the sync manifest: {exc}")
    try:
        _write_json_atomic(_state_file(), state)
    except OSError as exc:
        report.errors.append(f"Could not update the local sync state: {exc}")


def push(cfg: dict[str, Any] | None = None, force: bool = False) -> SyncReport:
    """Copy local sessions into the shared folder.

    A shared copy that changed elsewhere while the local copy also changed
    since this machine last synced is kept as both: the shared copy stays and
    the local one is uploaded as ``<name>__<machine>.json``. Otherwise the
    newer ``saved_at`` wins; ``force=True`` makes the local copy win
    unconditionally.
    """
    settings = get_sync(cfg)
    _require_enabled(settings)
    root = sync_root(cfg)
    manifest = _load_manifest(root)
    state = _load_state(settings["machine"])
    machine = settings["machine"]
    report = SyncReport()
    files = _local_session_files(settings["include_autosave"])
    if not files:
        report.lines.append("No local sessions to push.")
    for path in files:
        local_meta = _safe_meta(path)
        if local_meta is None:
            report.errors.append(f"Skipped unreadable session: {path.name}")
            continue
        remote = root / path.name
        remote_meta = _safe_meta(remote) if remote.is_file() else None
        if remote.is_file() and remote_meta is None:
            report.errors.append(f"Skipped corrupt remote session: {path.name}")
            continue
        try:
            if remote_meta is None:
                _copy_file(path, remote)
                _record(manifest["files"], path.name, local_meta)
                _record(state["files"], path.name, local_meta)
                report.pushed += 1
                report.lines.append(f"Pushed {path.name}")
                continue
            if remote_meta["sha1"] == local_meta["sha1"]:
                _record(manifest["files"], path.name, local_meta)
                _record(state["files"], path.name, local_meta)
                continue
            base = state["files"].get(path.name)
            local_changed = base is None or base["sha1"] != local_meta["sha1"]
            remote_changed = base is None or base["sha1"] != remote_meta["sha1"]
            if base is not None and local_changed and remote_changed and not force:
                conflict = _conflict_name(path.name, machine, root)
                _copy_file(path, root / conflict)
                _record(manifest["files"], conflict, local_meta)
                _record(state["files"], conflict, local_meta)
                _record(manifest["files"], path.name, remote_meta)
                _record(state["files"], path.name, remote_meta)
                report.conflicts += 1
                report.lines.append(
                    f"Conflict on {path.name}: kept the shared copy and "
                    f"pushed the local one as {conflict}"
                )
                continue
            if force or local_meta["saved_at"] >= remote_meta["saved_at"]:
                _copy_file(path, remote)
                _record(manifest["files"], path.name, local_meta)
                _record(state["files"], path.name, local_meta)
                report.pushed += 1
                report.lines.append(f"Pushed {path.name}")
            else:
                _record(manifest["files"], path.name, remote_meta)
                _record(state["files"], path.name, remote_meta)
                report.lines.append(f"Kept the shared {path.name} (newer)")
        except OSError as exc:
            report.errors.append(f"Could not sync {path.name}: {exc}")
    _finish(root, manifest, state, settings, report, claim_machine=True)
    return report


def pull(cfg: dict[str, Any] | None = None, force: bool = False) -> SyncReport:
    """Copy newer shared sessions into the local sessions directory.

    The both-changed rule matches :func:`push`, except the incoming file is
    renamed with the remote machine's suffix instead of overwriting the local
    one. ``force=True`` makes the shared copy win unconditionally.
    """
    settings = get_sync(cfg)
    _require_enabled(settings)
    root = sync_root(cfg)
    manifest = _load_manifest(root)
    state = _load_state(settings["machine"])
    machine = settings["machine"]
    origin = manifest["machine"] if manifest["machine"] not in {"", machine} else "remote"
    report = SyncReport()
    files = _remote_session_files(root, settings["include_autosave"])
    if not files:
        report.lines.append("No remote sessions to pull.")
    local_dir = session_store._sessions_dir()
    for remote in files:
        remote_meta = _safe_meta(remote)
        if remote_meta is None:
            report.errors.append(f"Skipped corrupt remote session: {remote.name}")
            continue
        local = local_dir / remote.name
        local_meta = _safe_meta(local) if local.is_file() else None
        if local.is_file() and local_meta is None:
            report.errors.append(f"Skipped unreadable session: {remote.name}")
            continue
        try:
            if local_meta is None:
                _copy_file(remote, local)
                _record(manifest["files"], remote.name, remote_meta)
                _record(state["files"], remote.name, remote_meta)
                report.pulled += 1
                report.lines.append(f"Pulled {remote.name}")
                continue
            if local_meta["sha1"] == remote_meta["sha1"]:
                _record(manifest["files"], remote.name, remote_meta)
                _record(state["files"], remote.name, remote_meta)
                continue
            base = state["files"].get(remote.name)
            local_changed = base is None or base["sha1"] != local_meta["sha1"]
            remote_changed = base is None or base["sha1"] != remote_meta["sha1"]
            if base is not None and local_changed and remote_changed and not force:
                conflict = _conflict_name(remote.name, origin, local_dir)
                _copy_file(remote, local_dir / conflict)
                _record(state["files"], conflict, remote_meta)
                _record(state["files"], remote.name, local_meta)
                _record(manifest["files"], remote.name, remote_meta)
                report.conflicts += 1
                report.lines.append(
                    f"Conflict on {remote.name}: kept the local copy and "
                    f"pulled the shared one as {conflict}"
                )
                continue
            if force or remote_meta["saved_at"] > local_meta["saved_at"]:
                _copy_file(remote, local)
                _record(state["files"], remote.name, remote_meta)
                _record(manifest["files"], remote.name, remote_meta)
                report.pulled += 1
                report.lines.append(f"Pulled {remote.name}")
            else:
                _record(state["files"], remote.name, local_meta)
                report.lines.append(f"Kept the local {remote.name} (newer)")
        except OSError as exc:
            report.errors.append(f"Could not sync {remote.name}: {exc}")
    _finish(root, manifest, state, settings, report, claim_machine=False)
    return report
