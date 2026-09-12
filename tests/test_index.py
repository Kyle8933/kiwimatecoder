from __future__ import annotations

import io
import json
import os

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

from kiwimatecoder import config, main, tools
from kiwimatecoder.commands import CommandResult, dispatch
from kiwimatecoder.index import builder, embeddings
from kiwimatecoder.index import search as index_search
from kiwimatecoder.index import store as store_module
from kiwimatecoder.providers import REGISTRY
from kiwimatecoder.tools.search import _search


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    """Keep the index store and config out of the workspace."""
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "home")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "home" / "config.json")
    monkeypatch.setattr(
        config, "LEGACY_CONFIG_FILE", tmp_path / "home" / "config"
    )
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for provider in REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def _output(console: Console) -> str:
    return console.file.getvalue()


def _vector_handler(calls: list[httpx.Request], *, fail: bool = False):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if fail:
            return httpx.Response(500, text="boom", request=request)
        payload = json.loads(request.content)
        data = []
        for index, text in enumerate(payload.get("input", [])):
            vector = [1.0, 0.0] if "alpha" in text else [0.0, 1.0]
            data.append({"index": index, "embedding": vector})
        return httpx.Response(200, json={"data": data}, request=request)

    return handler


# ---------------------------------------------------------------------------
# Tokenizer and chunking
# ---------------------------------------------------------------------------


def test_tokenize_splits_camel_and_snake_case():
    tokens = builder.tokenize("getUserName snake_case HTTPResponse")

    assert tokens == ["get", "user", "name", "snake", "case", "http", "response"]


def test_tokenize_lowercases_and_drops_short_tokens():
    tokens = builder.tokenize("A x ID db-42 kiwi_code")

    assert tokens == ["id", "db", "42", "kiwi", "code"]


def test_chunk_text_short_file_is_one_chunk():
    assert builder.chunk_text("one\ntwo") == [(1, "one\ntwo")]


def test_chunk_text_splits_with_overlap():
    text = "\n".join(f"line {i}" for i in range(1, 26))

    chunks = builder.chunk_text(text, size=10, overlap=2)

    assert [start for start, _chunk in chunks] == [1, 9, 17]
    assert "line 9" in chunks[1][1]
    assert "line 10" in chunks[1][1]
    assert "line 25" in chunks[-1][1]


def test_chunk_text_empty_is_empty():
    assert builder.chunk_text("") == []


# ---------------------------------------------------------------------------
# Incremental builds
# ---------------------------------------------------------------------------


def test_incremental_build_add_update_delete(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    first = root / "first.py"
    first.write_text("def alpha(): pass\n")

    stats = builder.build_index(root)
    assert (stats.added, stats.updated, stats.removed, stats.unchanged) == (
        1,
        0,
        0,
        0,
    )
    assert stats.files == 1

    stats = builder.build_index(root)
    assert (stats.added, stats.updated, stats.removed, stats.unchanged) == (
        0,
        0,
        0,
        1,
    )

    (root / "second.py").write_text("def beta(): pass\n")
    stats = builder.build_index(root)
    assert stats.added == 1
    assert stats.files == 2

    # A same-size edit is still picked up because the mtime moved.
    first.write_text("def gamma(): pass\n")
    moved = first.stat().st_mtime + 5
    os.utime(first, (moved, moved))
    stats = builder.build_index(root)
    assert stats.updated == 1
    assert stats.unchanged == 1

    (root / "second.py").unlink()
    stats = builder.build_index(root)
    assert stats.removed == 1
    assert stats.files == 1


def test_unchanged_files_are_not_retokenized(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("def alpha(): pass\n")
    builder.build_index(root)

    calls: list[str] = []
    original = builder.count_tokens

    def spy(text: str):
        calls.append(text)
        return original(text)

    monkeypatch.setattr(builder, "count_tokens", spy)
    stats = builder.build_index(root)

    assert stats.unchanged == 1
    assert calls == []


def test_gitignore_and_default_skips_are_respected(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".gitignore").write_text("ignored.py\nbuild/\n")
    (root / "kept.py").write_text("keep me\n")
    (root / "ignored.py").write_text("ignore me\n")
    (root / "build").mkdir()
    (root / "build" / "out.py").write_text("skip\n")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "dep.py").write_text("skip\n")

    builder.build_index(root)
    files = store_module.load_store(root)["files"]

    assert "kept.py" in files
    assert "ignored.py" not in files
    assert not any(path.startswith("build/") for path in files)
    assert not any(path.startswith("node_modules/") for path in files)


def test_binary_files_are_skipped_and_removed_when_binary(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "text.py").write_text("alpha content\n")
    (root / "blob.dat").write_bytes(b"\x00\x01\x02binary")

    builder.build_index(root)
    files = store_module.load_store(root)["files"]
    assert "text.py" in files
    assert "blob.dat" not in files

    (root / "text.py").write_bytes(b"\x00binary now")
    moved = (root / "text.py").stat().st_mtime + 5
    os.utime(root / "text.py", (moved, moved))
    stats = builder.build_index(root)

    assert stats.removed == 1
    assert "text.py" not in store_module.load_store(root)["files"]


def test_max_files_and_max_file_bytes_caps(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    for name in ("a.py", "b.py", "c.py"):
        (root / name).write_text("kiwi code\n")

    config.set_index(max_files=2)
    stats = builder.build_index(root)
    assert stats.files == 2
    assert stats.added == 2

    config.set_index(max_files=10, max_file_bytes=1024)
    (root / "big.py").write_text("x" * 2000)
    builder.build_index(root)
    files = store_module.load_store(root)["files"]
    assert "c.py" in files
    assert "big.py" not in files


def test_oversized_existing_file_is_removed(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    target = root / "grows.py"
    target.write_text("small\n")
    builder.build_index(root)
    assert "grows.py" in store_module.load_store(root)["files"]

    config.set_index(max_file_bytes=1024)
    target.write_text("x" * 2000)
    stats = builder.build_index(root)

    assert stats.removed == 1
    assert "grows.py" not in store_module.load_store(root)["files"]


def test_disabled_index_build_is_a_noop(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("alpha\n")
    config.set_index(enabled=False)

    stats = builder.build_index(root)

    assert stats == builder.BuildStats()
    assert not store_module.store_path(root).exists()


# ---------------------------------------------------------------------------
# Store tolerance and status
# ---------------------------------------------------------------------------


def test_corrupt_store_is_tolerated_and_rebuilt(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("hello world\n")
    builder.build_index(root)

    path = store_module.store_path(root)
    path.write_text("{not json", encoding="utf-8")
    assert store_module.load_store(root)["files"] == {}

    stats = builder.build_index(root)
    assert stats.added == 1
    assert "a.py" in store_module.load_store(root)["files"]


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "a", "dict"],
        {"version": 999, "root": "/tmp", "files": {}},
        {"version": store_module.INDEX_VERSION, "root": "/tmp", "files": "nope"},
        {
            "version": store_module.INDEX_VERSION,
            "root": "wrong-root",
            "files": {"a.py": {"mtime": 1, "size": 1, "tokens": {"x": 1}}},
        },
    ],
)
def test_malformed_store_shapes_are_tolerated(tmp_path, payload):
    root = tmp_path / "proj"
    root.mkdir()
    path = store_module.store_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

    store = store_module.load_store(root)

    assert store["files"] == {}


def test_index_status_reports_stale_files_and_embeddings(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    target = root / "a.py"
    target.write_text("alpha content\n")
    builder.build_index(root)

    status = store_module.index_status(root)
    assert status.enabled is True
    assert status.files == 1
    assert status.terms >= 1
    assert status.stale == 0
    assert status.embeddings is False

    moved = target.stat().st_mtime + 5
    os.utime(target, (moved, moved))
    assert store_module.index_status(root).stale == 1

    config.set_index(embed_provider="openai", embed_model="embed-model")
    assert store_module.index_status(root).embeddings is True


def test_clear_store_returns_whether_it_existed(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("alpha\n")
    builder.build_index(root)

    assert store_module.clear_store(root) is True
    assert store_module.clear_store(root) is False


# ---------------------------------------------------------------------------
# BM25 ranking and snippets
# ---------------------------------------------------------------------------


def test_bm25_ranks_more_relevant_documents_first():
    docs = {
        "a": {"kiwi": 3, "code": 1},
        "b": {"kiwi": 1, "code": 1},
        "c": {"unrelated": 1},
    }
    df = {"kiwi": 2, "code": 2, "unrelated": 1}

    scores = index_search.bm25_scores(["kiwi"], docs, df)

    assert scores["a"] > scores["b"] > 0
    assert "c" not in scores


def test_bm25_returns_nothing_for_unmatched_query():
    assert index_search.bm25_scores(
        ["ghost"], {"a": {"kiwi": 1}}, {"kiwi": 1}
    ) == {}


def test_best_snippet_finds_the_matching_window():
    lines = [f"line {i}" for i in range(1, 21)]
    lines[12] = "def kiwi_code():"
    lines[13] = "    return kiwi_code()"

    snippet, start = index_search.best_snippet(lines, ["kiwi", "code"], window=3)

    assert "kiwi_code" in snippet
    assert start == 12


def test_best_snippet_falls_back_to_first_content_line():
    assert index_search.best_snippet(["", "  ", "hello"], ["zzz"]) == (
        "hello",
        3,
    )


def test_search_index_ranks_matching_files(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "auth.py").write_text(
        "def authenticate(token):\n    return token\n"
    )
    (root / "auth_docs.md").write_text("authenticate token notes\n" * 5)
    (root / "misc.py").write_text("print('nothing to see')\n")
    builder.build_index(root)

    hits = index_search.search_index(
        "authenticate token", limit=2, workspace_root=root
    )

    assert {hit.path for hit in hits} == {"auth.py", "auth_docs.md"}
    assert all(hit.start_line >= 1 and hit.snippet for hit in hits)


def test_search_index_returns_empty_when_disabled(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("alpha\n")
    builder.build_index(root)
    config.set_index(enabled=False)

    assert index_search.search_index("alpha", workspace_root=root) == []


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def test_embed_texts_batches_requests_and_orders_vectors():
    config.set_key("openai", "sk-test")
    calls: list[httpx.Request] = []

    vectors = embeddings.embed_texts(
        ["alpha one", "beta two", "alpha three"],
        "openai",
        "embed-model",
        batch_size=2,
        transport=httpx.MockTransport(_vector_handler(calls)),
    )

    assert vectors == [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    assert len(calls) == 2
    assert calls[0].url.path.endswith("/embeddings")
    assert calls[0].headers["authorization"] == "Bearer sk-test"
    assert json.loads(calls[0].content)["model"] == "embed-model"


def test_embed_texts_wraps_http_failures():
    config.set_key("openai", "sk-test")

    with pytest.raises(embeddings.EmbeddingError, match="HTTP 500"):
        embeddings.embed_texts(
            ["alpha"],
            "openai",
            "embed-model",
            transport=httpx.MockTransport(_vector_handler([], fail=True)),
        )


def test_embed_texts_requires_provider_and_model():
    with pytest.raises(embeddings.EmbeddingError, match="provider and a model"):
        embeddings.embed_texts(["alpha"], "", "")


def test_embed_texts_rejects_non_openai_provider():
    with pytest.raises(embeddings.EmbeddingError, match="not OpenAI-compatible"):
        embeddings.embed_texts(["alpha"], "anthropic", "embed-model")


def test_embed_texts_rejects_missing_key():
    with pytest.raises(embeddings.EmbeddingError, match="No API key"):
        embeddings.embed_texts(
            ["alpha"],
            "openai",
            "embed-model",
            transport=httpx.MockTransport(_vector_handler([])),
        )


def test_cosine_similarity():
    assert embeddings.cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert embeddings.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert embeddings.cosine([], [1.0]) == 0.0


def test_vectors_stored_and_blended_into_ranking(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "alpha.py").write_text("shared alpha\n")
    (root / "beta.py").write_text("shared shared shared\n")

    def embedder(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts]

    builder.build_index(root, embedder=embedder)
    stored = store_module.load_store(root)["files"]
    assert stored["alpha.py"]["vectors"] == [[1.0, 0.0]]
    assert stored["beta.py"]["vectors"] == [[0.0, 1.0]]

    # Lexically "beta.py" wins (more occurrences of "shared"); a query vector
    # aligned with alpha.py's stored vector flips the ranking.
    lexical = index_search.search_index(
        "shared", workspace_root=root, embedder=lambda texts: [[0.0, 1.0]]
    )
    blended = index_search.search_index(
        "shared", workspace_root=root, embedder=lambda texts: [[1.0, 0.0]]
    )

    assert lexical[0].path == "beta.py"
    assert blended[0].path == "alpha.py"


def test_config_embeddings_use_mock_transport_end_to_end(monkeypatch, tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "alpha.py").write_text("shared alpha\n")
    config.set_index(embed_provider="openai", embed_model="embed-model")
    config.set_key("openai", "sk-test")
    transport = httpx.MockTransport(_vector_handler([]))
    real_embed = embeddings.embed_texts

    def routed(texts, provider, model, **kwargs):
        return real_embed(texts, provider, model, transport=transport, **kwargs)

    monkeypatch.setattr(embeddings, "embed_texts", routed)
    stats = builder.build_index(root)

    assert stats.added == 1
    stored = store_module.load_store(root)["files"]
    assert stored["alpha.py"]["vectors"] == [[1.0, 0.0]]


def test_embedding_failure_falls_back_to_lexical(monkeypatch, tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "alpha.py").write_text("alpha content\n")
    config.set_index(embed_provider="openai", embed_model="embed-model")
    config.set_key("openai", "sk-test")

    def failing(texts, provider, model, **kwargs):
        raise embeddings.EmbeddingError("boom")

    monkeypatch.setattr(embeddings, "embed_texts", failing)

    stats = builder.build_index(root)
    assert stats.added == 1
    assert "vectors" not in store_module.load_store(root)["files"]["alpha.py"]

    hits = index_search.search_index("alpha", workspace_root=root)
    assert [hit.path for hit in hits] == ["alpha.py"]


# ---------------------------------------------------------------------------
# search tool integration
# ---------------------------------------------------------------------------


def test_search_tool_semantic_returns_hits(session):
    (session.workspace_root / "auth.py").write_text(
        "def authenticate(token):\n    return validate_token(token)\n"
    )
    (session.workspace_root / "unrelated.py").write_text("print('nothing')\n")

    result = _search({"pattern": "authenticate token", "mode": "semantic"}, session)

    assert result.ok
    assert "auth.py:" in result.content
    assert "authenticate" in result.content
    assert "unrelated.py" not in result.content


def test_search_tool_semantic_auto_builds_index(session):
    (session.workspace_root / "a.py").write_text("kiwi alpha\n")

    result = _search({"pattern": "kiwi", "mode": "semantic"}, session)

    assert result.ok
    assert store_module.store_path(session.workspace_root).exists()


def test_search_tool_semantic_disabled_is_friendly(session):
    config.set_index(enabled=False)

    result = _search({"pattern": "kiwi", "mode": "semantic"}, session)

    assert not result.ok
    assert "disabled" in result.content


def test_search_tool_semantic_rejects_bad_limit(session):
    (session.workspace_root / "a.py").write_text("kiwi alpha\n")

    result = _search(
        {"pattern": "kiwi", "mode": "semantic", "limit": "many"}, session
    )

    assert not result.ok
    assert "'limit' must be an integer" in result.content


def test_search_tool_semantic_respects_path(session):
    sub = session.workspace_root / "sub"
    sub.mkdir()
    (sub / "alpha.py").write_text("alpha content\n")
    (session.workspace_root / "beta.py").write_text("alpha content\n")

    result = _search(
        {"pattern": "alpha", "mode": "semantic", "path": "sub"}, session
    )

    assert result.ok
    assert "sub/alpha.py" in result.content
    assert "beta.py" not in result.content


def test_search_tool_grep_still_works(session):
    (session.workspace_root / "f.py").write_text("def foo():\n    return 1\n")

    result = _search({"pattern": "def foo", "mode": "grep"}, session)

    assert result.ok
    assert "f.py:1" in result.content


def test_search_tool_schema_exposes_semantic_mode():
    tool = tools.get_tool("search")

    assert tool is not None
    assert "semantic" in tool.parameters["properties"]["mode"]["enum"]
    assert "semantic" in tool.description
    assert tool.writes is False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_get_index_defaults():
    assert config.get_index() == {
        "enabled": True,
        "max_files": 5000,
        "max_file_bytes": 262144,
        "embeddings": {"provider": "", "model": "", "batch_size": 32},
    }


def test_set_index_roundtrip():
    config.set_index(
        enabled=False,
        max_files=10,
        max_file_bytes=2048,
        embed_provider="openai",
        embed_model="embed-model",
        embed_batch_size=4,
    )

    settings = config.get_index()
    assert settings["enabled"] is False
    assert settings["max_files"] == 10
    assert settings["max_file_bytes"] == 2048
    assert settings["embeddings"] == {
        "provider": "openai",
        "model": "embed-model",
        "batch_size": 4,
    }
    stored = config.load_config()["index"]
    assert stored["embeddings"]["provider"] == "openai"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": "yes"},
        {"max_files": 0},
        {"max_files": 10_000_000},
        {"max_files": "many"},
        {"max_file_bytes": 10},
        {"max_file_bytes": "large"},
        {"embed_provider": "ghost"},
        {"embed_batch_size": 0},
        {"embed_batch_size": "many"},
    ],
)
def test_set_index_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        config.set_index(**kwargs)


def test_get_index_normalizes_bad_stored_values():
    config.save_config(
        {
            "index": {
                "enabled": "yes",
                "max_files": "many",
                "max_file_bytes": 10,
                "embeddings": {"provider": 5, "model": None, "batch_size": 0},
            }
        }
    )

    settings = config.get_index()

    assert settings == config.INDEX_DEFAULTS


def test_validate_config_accepts_clean_index_section():
    config.set_index(max_files=100, max_file_bytes=8192)

    issues = [
        issue
        for issue in config.validate_config()
        if issue["key"].startswith("index")
    ]

    assert issues == []


def test_validate_config_catches_bad_index_section():
    config.save_config(
        {
            "index": {
                "enabled": "yes",
                "max_files": 0,
                "max_file_bytes": 5,
                "embeddings": {
                    "provider": "ghost",
                    "model": 5,
                    "batch_size": 0,
                },
            }
        }
    )

    keys = {issue["key"] for issue in config.validate_config()}

    assert {
        "index.enabled",
        "index.max_files",
        "index.max_file_bytes",
        "index.embeddings.provider",
        "index.embeddings.model",
        "index.embeddings.batch_size",
    } <= keys


# ---------------------------------------------------------------------------
# Slash command
# ---------------------------------------------------------------------------


def test_index_command_build_status_clear(session):
    (session.workspace_root / "a.py").write_text("kiwi code\n")

    build_console = _console()
    assert dispatch("/index build", session, build_console) == CommandResult.CONTINUE
    build_output = _output(build_console)
    assert "1 file(s)" in build_output
    assert "1 added" in build_output
    assert store_module.store_path(session.workspace_root).exists()

    status_console = _console()
    dispatch("/index status", session, status_console)
    status_output = _output(status_console)
    assert "Files indexed:" in status_output
    assert "stale" in status_output
    assert "Embeddings:" in status_output

    clear_console = _console()
    dispatch("/index clear", session, clear_console)
    assert "cleared" in _output(clear_console).lower()
    assert not store_module.store_path(session.workspace_root).exists()


def test_index_command_status_empty_and_usage(session):
    empty_console = _console()
    dispatch("/index", session, empty_console)
    assert "Codebase index" in _output(empty_console)

    usage_console = _console()
    dispatch("/index wiggle", session, usage_console)
    assert "Usage:" in _output(usage_console)


def test_index_command_build_when_disabled(session):
    config.set_index(enabled=False)
    console = _console()

    dispatch("/index build", session, console)

    assert "disabled" in _output(console)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_index_show_embed_settings_and_clear(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("kiwi code\n")
    builder.build_index(tmp_path)
    runner = CliRunner()

    show = runner.invoke(main.app, ["config", "index", "show"])
    assert show.exit_code == 0
    assert "Files indexed" in show.output
    assert "Embeddings: off" in show.output

    provider = runner.invoke(
        main.app, ["config", "index", "embed-provider", "openai"]
    )
    assert provider.exit_code == 0
    assert config.get_index()["embeddings"]["provider"] == "openai"

    model = runner.invoke(
        main.app, ["config", "index", "embed-model", "embed-model"]
    )
    assert model.exit_code == 0
    assert config.get_index()["embeddings"]["model"] == "embed-model"

    clear = runner.invoke(main.app, ["config", "index", "clear"])
    assert clear.exit_code == 0
    assert not store_module.store_path(tmp_path).exists()


def test_cli_index_embed_provider_rejects_unknown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main.app, ["config", "index", "embed-provider", "ghost"]
    )

    assert result.exit_code == 1
    assert "Unknown provider" in result.output
