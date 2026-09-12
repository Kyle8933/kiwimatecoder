"""Incremental, gitignore-aware builder for the codebase index.

A build walks the workspace using the same ignore rules as the file tools,
compares each file's ``mtime``/``size`` against the stored entry, and only
re-tokenizes files that are new or changed. Deleted files are dropped. Large
files are chunked (with overlap) so embeddings can be stored per chunk when an
embedding provider is configured.
"""

from __future__ import annotations

import os
import re
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kiwimatecoder import config
from kiwimatecoder.index import embeddings
from kiwimatecoder.index.store import load_store, save_store
from kiwimatecoder.tools.paths import get_workspace_ignore

MIN_TOKEN_LENGTH = 2
CHUNK_LINES = 60
CHUNK_OVERLAP = 10
HEAD_BYTES = 1024

Embedder = Callable[[list[str]], list[list[float]]]

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_BUILD_LOCK = threading.Lock()


@dataclass(frozen=True)
class BuildStats:
    """What one incremental build changed."""

    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    files: int = 0
    terms: int = 0


def tokenize(text: str) -> list[str]:
    """Split text into lowercase index terms.

    Runs of alphanumerics and underscores are kept, camelCase and snake_case
    are split into their parts, and tokens shorter than
    :data:`MIN_TOKEN_LENGTH` are dropped.
    """
    tokens: list[str] = []
    for word in _WORD_RE.findall(text):
        for part in _CAMEL_RE.split(word):
            for piece in part.split("_"):
                piece = piece.lower()
                if len(piece) >= MIN_TOKEN_LENGTH:
                    tokens.append(piece)
    return tokens


def count_tokens(text: str) -> dict[str, int]:
    """Return term -> frequency for ``text``."""
    return dict(Counter(tokenize(text)))


def chunk_text(
    text: str, *, size: int = CHUNK_LINES, overlap: int = CHUNK_OVERLAP
) -> list[tuple[int, str]]:
    """Split text into overlapping line windows for embeddings.

    Returns 1-based ``(start_line, chunk)`` pairs; a file shorter than
    ``size`` lines yields a single chunk and empty text yields none.
    """
    lines = text.splitlines()
    if not lines:
        return []
    step = max(1, size - overlap)
    chunks: list[tuple[int, str]] = []
    start = 0
    while start < len(lines):
        window = lines[start : start + size]
        if not window:
            break
        chunks.append((start + 1, "\n".join(window)))
        if start + size >= len(lines):
            break
        start += step
    return chunks


def embeddings_configured(settings: dict[str, Any] | None = None) -> bool:
    """Whether a provider and model are configured for embeddings."""
    settings = settings or config.get_index()
    embeddings_settings = settings.get("embeddings") or {}
    return bool(
        str(embeddings_settings.get("provider") or "").strip()
        and str(embeddings_settings.get("model") or "").strip()
    )


def _collect_files(root: Path, max_files: int) -> tuple[list[Path], bool]:
    """Collect indexable files, honoring ignore rules and the file cap.

    Returns ``(files, capped)``; ``capped`` is True when the walk stopped at
    ``max_files`` with potentially more files still unseen.
    """
    ignore = get_workspace_ignore(root)
    collected: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(
        root, topdown=True, followlinks=False
    ):
        dir_p = Path(dirpath)
        dirnames[:] = [
            name for name in dirnames if not ignore.is_ignored(dir_p / name, True)
        ]
        dirnames.sort()
        for name in sorted(filenames):
            file_p = dir_p / name
            if ignore.is_ignored(file_p, is_dir=False):
                continue
            collected.append(file_p)
            if len(collected) >= max_files:
                return collected, True
    return collected, False


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(HEAD_BYTES)
    except OSError:
        return True
    return b"\x00" in head


def _remove_entry(
    files: dict[str, dict[str, Any]], df: dict[str, int], rel: str
) -> None:
    entry = files.pop(rel, None)
    if entry is None:
        return
    tokens = entry.get("tokens")
    if not isinstance(tokens, dict):
        return
    for term in tokens:
        remaining = df.get(term, 0) - 1
        if remaining > 0:
            df[term] = remaining
        else:
            df.pop(term, None)


def _add_df(df: dict[str, int], tokens: dict[str, int]) -> None:
    for term in tokens:
        df[term] = df.get(term, 0) + 1


def _has_vectors(entry: dict[str, Any]) -> bool:
    vectors = entry.get("vectors")
    chunks = entry.get("chunks")
    return (
        isinstance(vectors, list)
        and isinstance(chunks, list)
        and len(vectors) == len(chunks)
        and len(vectors) > 0
    )


def _entry_chunks(entry: dict[str, Any]) -> list[tuple[int, str]] | None:
    chunks = entry.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        return None
    result: list[tuple[int, str]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
            return None
        try:
            start = int(chunk.get("start") or 1)
        except (TypeError, ValueError):
            start = 1
        result.append((max(1, start), chunk["text"]))
    return result


def _file_chunks(path: Path) -> list[tuple[int, str]] | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:HEAD_BYTES]:
        return None
    return chunk_text(data.decode("utf-8", "replace"))


def _make_embedder(settings: dict[str, Any]) -> Embedder:
    embedding_settings = settings["embeddings"]
    provider = str(embedding_settings["provider"])
    model = str(embedding_settings["model"])
    batch_size = int(embedding_settings["batch_size"])

    def embedder(texts: list[str]) -> list[list[float]]:
        return embeddings.embed_texts(
            texts, provider, model, batch_size=batch_size
        )

    return embedder


def _assign_vectors(
    files: dict[str, dict[str, Any]],
    pending: list[tuple[str, list[tuple[int, str]]]],
    embed: Embedder,
) -> None:
    """Embed pending chunks and attach vectors, silently falling back."""
    texts = [text for _rel, chunks in pending for _start, text in chunks]
    try:
        vectors = embed(texts)
    except Exception:
        vectors = []
    if len(vectors) != len(texts):
        for rel, _chunks in pending:
            files[rel].pop("vectors", None)
        return
    offset = 0
    for rel, chunks in pending:
        count = len(chunks)
        files[rel]["vectors"] = vectors[offset : offset + count]
        offset += count


def build_index(
    workspace_root: Path,
    *,
    cfg: dict[str, Any] | None = None,
    embedder: Embedder | None = None,
) -> BuildStats:
    """Incrementally update the index for ``workspace_root``.

    ``embedder`` overrides the configured embedding client (used by tests and
    callers that already hold vectors). Embedding failures are swallowed: the
    index simply stays lexical for those files. Never raises for a file that
    disappears or cannot be read.
    """
    root = Path(workspace_root).resolve()
    settings = config.get_index(cfg)
    if not settings["enabled"]:
        return BuildStats()
    with _BUILD_LOCK:
        return _build(root, settings, embedder)


def _build(
    root: Path,
    settings: dict[str, Any],
    embedder: Embedder | None,
) -> BuildStats:
    store = load_store(root)
    files: dict[str, dict[str, Any]] = store["files"]
    df: dict[str, int] = store["df"]

    embed = embedder
    if embed is None and embeddings_configured(settings):
        embed = _make_embedder(settings)

    max_files = int(settings["max_files"])
    max_file_bytes = int(settings["max_file_bytes"])
    collected, capped = _collect_files(root, max_files)

    added = updated = removed = unchanged = 0
    visited: set[str] = set()
    fresh_chunks: dict[str, list[tuple[int, str]]] = {}

    for file_p in collected:
        rel = file_p.relative_to(root).as_posix()
        visited.add(rel)
        try:
            stat = file_p.stat()
        except OSError:
            continue
        if stat.st_size > max_file_bytes:
            if rel in files:
                _remove_entry(files, df, rel)
                removed += 1
            continue

        old = files.get(rel)
        if (
            old is not None
            and old.get("mtime") == stat.st_mtime
            and old.get("size") == stat.st_size
            and isinstance(old.get("tokens"), dict)
        ):
            unchanged += 1
            continue

        if _is_binary(file_p):
            if old is not None:
                _remove_entry(files, df, rel)
                removed += 1
            continue

        try:
            text = file_p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        if old is not None:
            _remove_entry(files, df, rel)
            updated += 1
        else:
            added += 1

        tokens = count_tokens(text)
        entry: dict[str, Any] = {
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "tokens": tokens,
        }
        if embed is not None:
            chunks = chunk_text(text)
            entry["chunks"] = [
                {"start": start, "text": chunk} for start, chunk in chunks
            ]
            fresh_chunks[rel] = chunks
        files[rel] = entry
        _add_df(df, tokens)

    if not capped:
        # Only prune files we actually visited: when the file cap stopped the
        # walk early, untouched entries may still exist further down the tree.
        for rel in [name for name in files if name not in visited]:
            _remove_entry(files, df, rel)
            removed += 1

    pending: list[tuple[str, list[tuple[int, str]]]] = []
    if embed is not None:
        for rel, entry in files.items():
            if _has_vectors(entry):
                continue
            stored_chunks: list[tuple[int, str]] | None = fresh_chunks.get(
                rel
            ) or _entry_chunks(entry)
            if stored_chunks is None:
                stored_chunks = _file_chunks(root / rel)
                if stored_chunks is None:
                    continue
                entry["chunks"] = [
                    {"start": start, "text": chunk}
                    for start, chunk in stored_chunks
                ]
            if not stored_chunks:
                continue
            pending.append((rel, stored_chunks))
        if pending:
            _assign_vectors(files, pending, embed)

    if added or updated or removed or pending or store.get("built_at") is None:
        try:
            save_store(root, store)
        except OSError:
            pass  # the index is a cache; never fail a build over a write
    return BuildStats(
        added=added,
        updated=updated,
        removed=removed,
        unchanged=unchanged,
        files=len(files),
        terms=len(df),
    )
