"""Versioned JSON persistence for the codebase index.

Each workspace gets its own store at
``~/.kiwimatecoder/index/<sha1(root)[:16]>.json``. The store is written
atomically and every read tolerates a missing, truncated, or otherwise corrupt
file by returning an empty store, since the index is a cache that can always be
rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kiwimatecoder import config
from kiwimatecoder.tools.paths import atomic_write_text

INDEX_VERSION = 1
INDEX_DIR_NAME = "index"


def store_path(workspace_root: Path) -> Path:
    """Return the store path for ``workspace_root`` (directory may not exist)."""
    root = Path(workspace_root).resolve()
    digest = hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:16]
    return config.ensure_config_dir() / INDEX_DIR_NAME / f"{digest}.json"


def empty_store(workspace_root: Path) -> dict[str, Any]:
    """Return a fresh, empty store for ``workspace_root``."""
    root = Path(workspace_root).resolve()
    return {
        "version": INDEX_VERSION,
        "root": str(root),
        "files": {},
        "df": {},
        "built_at": None,
    }


def compute_df(files: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Compute the document-frequency map for a ``rel -> entry`` mapping."""
    df: dict[str, int] = {}
    for entry in files.values():
        tokens = entry.get("tokens")
        if not isinstance(tokens, dict):
            continue
        for term in tokens:
            df[str(term)] = df.get(str(term), 0) + 1
    return df


def _clean_tokens(raw: object) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    tokens: dict[str, int] = {}
    for term, count in raw.items():
        try:
            value = int(count)
        except (TypeError, ValueError):
            continue
        if value > 0:
            tokens[str(term)] = value
    return tokens


def _clean_chunks(raw: object) -> list[dict[str, Any]] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        return None
    chunks: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            return None
        try:
            start = int(item.get("start") or 1)
        except (TypeError, ValueError):
            start = 1
        chunks.append({"start": max(1, start), "text": item["text"]})
    return chunks


def _clean_vectors(raw: object, expected: int) -> list[list[float]] | None:
    if raw is None or expected <= 0:
        return None
    if not isinstance(raw, list) or len(raw) != expected:
        return None
    vectors: list[list[float]] = []
    for vector in raw:
        if not isinstance(vector, list) or not vector:
            return None
        try:
            vectors.append([float(value) for value in vector])
        except (TypeError, ValueError):
            return None
    return vectors


def _clean_entry(raw: object) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    tokens = raw.get("tokens")
    if not isinstance(tokens, dict):
        return None
    try:
        mtime = float(raw.get("mtime") or 0.0)
        size = int(raw.get("size") or 0)
    except (TypeError, ValueError):
        return None
    entry: dict[str, Any] = {
        "mtime": mtime,
        "size": size,
        "tokens": _clean_tokens(tokens),
    }
    chunks = _clean_chunks(raw.get("chunks"))
    if chunks is not None:
        entry["chunks"] = chunks
        vectors = _clean_vectors(raw.get("vectors"), len(chunks))
        if vectors is not None:
            entry["vectors"] = vectors
    return entry


def load_store(workspace_root: Path) -> dict[str, Any]:
    """Load the store for ``workspace_root``, tolerating any corruption.

    A missing file, invalid JSON, version mismatch, or wrong workspace root all
    yield an :func:`empty_store` result so callers can simply rebuild.
    """
    root = Path(workspace_root).resolve()
    path = store_path(root)
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return empty_store(root)
    if not isinstance(stored, dict) or stored.get("version") != INDEX_VERSION:
        return empty_store(root)
    if str(stored.get("root") or "") != str(root):
        return empty_store(root)
    raw_files = stored.get("files")
    if not isinstance(raw_files, dict):
        return empty_store(root)

    files: dict[str, dict[str, Any]] = {}
    for rel, entry in raw_files.items():
        if not isinstance(rel, str):
            continue
        cleaned = _clean_entry(entry)
        if cleaned is not None:
            files[rel] = cleaned

    df: dict[str, int] = {}
    raw_df = stored.get("df")
    if isinstance(raw_df, dict):
        for term, count in raw_df.items():
            try:
                value = int(count)
            except (TypeError, ValueError):
                continue
            if value > 0:
                df[str(term)] = value
    if not df and files:
        df = compute_df(files)

    built_at = stored.get("built_at")
    if not isinstance(built_at, (int, float)):
        built_at = None
    return {
        "version": INDEX_VERSION,
        "root": str(root),
        "files": files,
        "df": df,
        "built_at": built_at,
    }


def save_store(workspace_root: Path, store: dict[str, Any]) -> None:
    """Write ``store`` atomically, stamping version, root, and build time."""
    root = Path(workspace_root).resolve()
    path = store_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    store["version"] = INDEX_VERSION
    store["root"] = str(root)
    store["built_at"] = time.time()
    atomic_write_text(path, json.dumps(store, separators=(",", ":")))


def clear_store(workspace_root: Path) -> bool:
    """Delete the store for ``workspace_root``; returns whether it existed."""
    path = store_path(workspace_root)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError:
        return False
    return True


@dataclass(frozen=True)
class IndexStatus:
    """Human-facing snapshot of the index for one workspace."""

    enabled: bool
    path: str
    files: int
    terms: int
    size_bytes: int
    stale: int
    embeddings: bool
    built_at: float | None


def index_status(
    workspace_root: Path, *, cfg: dict[str, Any] | None = None
) -> IndexStatus:
    """Summarize the store, counting indexed files whose stat changed."""
    settings = config.get_index(cfg)
    root = Path(workspace_root).resolve()
    store = load_store(root)
    files = store["files"]
    stale = 0
    for rel, entry in files.items():
        try:
            stat = (root / rel).stat()
        except OSError:
            stale += 1
            continue
        if stat.st_mtime != entry.get("mtime") or stat.st_size != entry.get("size"):
            stale += 1
    path = store_path(root)
    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = 0
    embeddings = settings["embeddings"]
    return IndexStatus(
        enabled=bool(settings["enabled"]),
        path=str(path),
        files=len(files),
        terms=len(store["df"]),
        size_bytes=size_bytes,
        stale=stale,
        embeddings=bool(embeddings["provider"] and embeddings["model"]),
        built_at=store.get("built_at"),
    )
