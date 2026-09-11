"""Per-tool file snapshots backing ``/undo`` and ``/rewind``.

Before a mutating tool runs, the target files are copied into a per-session
temporary directory. A checkpoint records where each pre-change snapshot lives
(``None`` means the file did not exist yet, so undo removes it). Restoring is
best-effort and never raises on a single unreadable file.
"""

from __future__ import annotations

import datetime
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


def _resolve(relative: str, workspace_root: Path) -> Path | None:
    """Resolve inside the workspace without a circular import at module load."""
    from kiwimatecoder.tools.paths import PathError, resolve_in_workspace

    try:
        return resolve_in_workspace(relative, workspace_root)
    except PathError:
        return None


@dataclass
class Checkpoint:
    """A snapshot of files captured before one mutating tool call."""

    id: int
    label: str
    created_at: str
    # Workspace-relative path -> snapshot file path, or None when absent.
    files: dict[str, str | None] = field(default_factory=dict)

    @property
    def paths(self) -> list[str]:
        return list(self.files)


class CheckpointStore:
    """Captures and restores file snapshots for one session."""

    def __init__(self, snapshot_root: Path | None = None) -> None:
        self._root = snapshot_root or Path(
            tempfile.mkdtemp(prefix="kiwimatecoder-checkpoints-")
        )
        self._counter = 0

    def capture(
        self, workspace_root: Path, paths: list[str], label: str
    ) -> Checkpoint:
        """Snapshot ``paths`` as they are right now and return the checkpoint."""
        self._counter += 1
        files: dict[str, str | None] = {}
        for raw in paths:
            relative = str(raw).strip()
            if not relative or relative in files:
                continue
            target = _resolve(relative, workspace_root)
            if target is None:
                continue
            if target.is_file():
                dest = self._root / str(self._counter) / f"{len(files)}.snap"
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copyfile(target, dest)
                except OSError:
                    continue
                files[relative] = str(dest)
            else:
                files[relative] = None
        return Checkpoint(
            id=self._counter,
            label=label,
            created_at=datetime.datetime.now().isoformat(),
            files=files,
        )

    def restore(self, workspace_root: Path, checkpoint: Checkpoint) -> list[str]:
        """Restore the checkpoint's files; returns the paths actually changed."""
        restored: list[str] = []
        for relative, snapshot in checkpoint.files.items():
            target = _resolve(relative, workspace_root)
            if target is None:
                continue
            if snapshot is None:
                try:
                    if target.exists():
                        target.unlink()
                        restored.append(relative)
                except OSError:
                    continue
            else:
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(snapshot, target)
                    restored.append(relative)
                except OSError:
                    continue
        return restored
