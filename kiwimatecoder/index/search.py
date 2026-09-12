"""BM25 ranking over the local code index, with optional vector blending.

The lexical ranker is a pure function over ``{path: {term: count}}`` maps so it
can be unit-tested on a tiny corpus. Optional embeddings only reweight files
that already carry vectors; if the embedding call fails for any reason the
lexical scores are returned unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kiwimatecoder import config
from kiwimatecoder.index import embeddings
from kiwimatecoder.index.builder import Embedder, embeddings_configured, tokenize
from kiwimatecoder.index.store import load_store

BM25_K1 = 1.5
BM25_B = 0.75
SNIPPET_LINES = 6
VECTOR_WEIGHT = 0.5
DEFAULT_LIMIT = 10
MAX_LIMIT = 50


@dataclass(frozen=True)
class IndexHit:
    """One ranked file with the best matching snippet."""

    path: str
    score: float
    snippet: str
    start_line: int


def bm25_scores(
    query_terms: Sequence[str],
    docs: Mapping[str, Mapping[str, int]],
    df: Mapping[str, int],
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> dict[str, float]:
    """Score ``docs`` against ``query_terms`` with BM25.

    Only documents sharing at least one query term are returned. ``df`` is the
    document frequency of each term across the whole index.
    """
    total = len(docs)
    if total == 0 or not query_terms:
        return {}
    lengths = {path: sum(counts.values()) for path, counts in docs.items()}
    avgdl = sum(lengths.values()) / total
    if avgdl <= 0:
        return {}
    scores: dict[str, float] = {}
    for path, counts in docs.items():
        score = 0.0
        length = lengths[path]
        for term in query_terms:
            frequency = counts.get(term, 0)
            if frequency <= 0:
                continue
            doc_freq = max(0, min(int(df.get(term, 0)), total))
            idf = math.log(1.0 + (total - doc_freq + 0.5) / (doc_freq + 0.5))
            denominator = frequency + k1 * (1.0 - b + b * length / avgdl)
            score += idf * (frequency * (k1 + 1.0)) / denominator
        if score > 0.0:
            scores[path] = score
    return scores


def best_snippet(
    lines: Sequence[str],
    query_terms: Sequence[str],
    *,
    window: int = SNIPPET_LINES,
) -> tuple[str, int]:
    """Return the ``window``-line snippet with the most query terms.

    The returned line number is 1-based. When nothing matches, the first
    non-blank line is used.
    """

    def first_content_line() -> tuple[str, int]:
        for index, line in enumerate(lines):
            if line.strip():
                return line.strip(), index + 1
        return (lines[0].strip() if lines else "", 1)

    if not lines:
        return "", 1
    terms = set(query_terms)
    if not terms:
        return first_content_line()
    line_scores = [len(set(tokenize(line)) & terms) for line in lines]
    best_start = 0
    best_total = -1
    for start in range(len(lines)):
        total = sum(line_scores[start : start + max(1, window)])
        if total > best_total:
            best_total = total
            best_start = start
    if best_total <= 0:
        return first_content_line()
    snippet = "\n".join(lines[best_start : best_start + max(1, window)]).strip()
    return snippet, best_start + 1


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _vector_similarity(entry: dict[str, Any], query_vector: list[float]) -> float | None:
    vectors = entry.get("vectors")
    chunks = entry.get("chunks")
    if (
        not isinstance(vectors, list)
        or not isinstance(chunks, list)
        or len(vectors) != len(chunks)
        or not vectors
    ):
        return None
    best = 0.0
    for vector in vectors:
        if isinstance(vector, list):
            best = max(best, max(0.0, embeddings.cosine(query_vector, vector)))
    return best


def entry_has_vectors(entry: dict[str, Any]) -> bool:
    """Whether an entry carries one vector per stored chunk."""
    vectors = entry.get("vectors")
    chunks = entry.get("chunks")
    return (
        isinstance(vectors, list)
        and isinstance(chunks, list)
        and len(vectors) == len(chunks)
        and len(vectors) > 0
    )


def search_index(
    query: str,
    *,
    limit: int = DEFAULT_LIMIT,
    workspace_root: Path,
    embedder: Embedder | None = None,
) -> list[IndexHit]:
    """Rank indexed files against ``query`` and return up to ``limit`` hits.

    Lexical BM25 drives the ranking. When embeddings are configured (or an
    ``embedder`` is supplied) and the query embeds successfully, each file's
    normalized lexical score is blended 50/50 with its best chunk similarity;
    files without vectors keep their lexical score. Any embedding failure is
    ignored.
    """
    settings = config.get_index()
    if not settings["enabled"]:
        return []
    try:
        limit = max(1, min(int(limit), MAX_LIMIT))
    except (TypeError, ValueError):
        limit = DEFAULT_LIMIT
    query_terms = tokenize(query)
    if not query_terms:
        return []

    root = Path(workspace_root).resolve()
    store = load_store(root)
    files: dict[str, dict[str, Any]] = store["files"]
    if not files:
        return []
    docs = {
        rel: entry.get("tokens", {})
        for rel, entry in files.items()
        if isinstance(entry.get("tokens"), dict)
    }
    scores = bm25_scores(query_terms, docs, store["df"])
    if not scores:
        return []

    embed = embedder
    if embed is None and embeddings_configured(settings):
        vector_targets = [rel for rel in scores if entry_has_vectors(files[rel])]
        if vector_targets:
            embed = _query_embedder(settings)
    if embed is not None:
        try:
            vectors = embed([query])
        except Exception:
            vectors = []
        if vectors:
            scores = _blend(scores, files, vectors[0])

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
    hits: list[IndexHit] = []
    for rel, score in ranked:
        lines = _read_lines(root / rel)
        snippet, start = best_snippet(lines, query_terms)
        hits.append(
            IndexHit(
                path=rel, score=round(score, 6), snippet=snippet, start_line=start
            )
        )
    return hits


def _query_embedder(settings: dict[str, Any]) -> Embedder:
    embedding_settings = settings["embeddings"]
    provider = str(embedding_settings["provider"])
    model = str(embedding_settings["model"])
    batch_size = int(embedding_settings["batch_size"])

    def embedder(texts: list[str]) -> list[list[float]]:
        return embeddings.embed_texts(
            texts, provider, model, batch_size=batch_size
        )

    return embedder


def _blend(
    scores: dict[str, float],
    files: dict[str, dict[str, Any]],
    query_vector: list[float],
) -> dict[str, float]:
    """Blend lexical scores with vector similarity for vector-bearing files."""
    largest = max(scores.values())
    if largest <= 0:
        return scores
    blended: dict[str, float] = {}
    for rel, score in scores.items():
        lexical = score / largest
        similarity = _vector_similarity(files[rel], query_vector)
        if similarity is None:
            blended[rel] = lexical
        else:
            blended[rel] = (1.0 - VECTOR_WEIGHT) * lexical + VECTOR_WEIGHT * similarity
    return blended
