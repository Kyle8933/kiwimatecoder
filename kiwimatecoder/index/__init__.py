"""Incremental codebase index and semantic search.

The package is split into four modules:

- :mod:`kiwimatecoder.index.store` — versioned JSON persistence under
  ``~/.kiwimatecoder/index/``.
- :mod:`kiwimatecoder.index.builder` — gitignore-aware incremental indexing.
- :mod:`kiwimatecoder.index.search` — BM25 ranking plus optional vectors.
- :mod:`kiwimatecoder.index.embeddings` — optional OpenAI-compatible
  ``/embeddings`` support.

Import submodules directly (``from kiwimatecoder.index import builder``) to
keep the import graph acyclic.
"""
