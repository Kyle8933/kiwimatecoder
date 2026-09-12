from __future__ import annotations

import io

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

from kiwimatecoder import catalog, config, main
from kiwimatecoder import client as client_module
from kiwimatecoder import web as web_module
from kiwimatecoder.client import ProviderError, TextDelta, UnifiedClient
from kiwimatecoder.commands import dispatch
from kiwimatecoder.index import embeddings
from kiwimatecoder.providers import REGISTRY, ProviderConfig


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    """Point config storage at a temp dir and clear provider env vars."""
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "LEGACY_CONFIG_FILE", tmp_path / "config")
    monkeypatch.delenv(config.PROJECT_CONFIG_ENV, raising=False)
    for provider in REGISTRY.values():
        monkeypatch.delenv(provider.key_env, raising=False)


def _no_http(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected HTTP request: {request.url}")


def _local_provider(**overrides) -> ProviderConfig:
    fields = {
        "id": "local",
        "name": "Local",
        "base_url": "http://localhost:1234/v1",
        "default_model": "m",
        "key_env": "LOCAL_API_KEY",
    }
    fields.update(overrides)
    return ProviderConfig(**fields)


# ---------------------------------------------------------------------------
# Config CRUD and validation
# ---------------------------------------------------------------------------


def test_network_defaults():
    assert config.get_network() == {"proxy": "", "ca_bundle": "", "offline": False}
    assert config.load_config()["network"] == {
        "proxy": "",
        "ca_bundle": "",
        "offline": False,
    }


def test_set_network_roundtrip(tmp_path):
    ca = tmp_path / "ca.pem"
    ca.write_text("cert")

    settings = config.set_network(
        proxy="http://proxy.local:8080", ca_bundle=str(ca), offline=True
    )

    assert settings == {
        "proxy": "http://proxy.local:8080",
        "ca_bundle": str(ca),
        "offline": True,
    }
    assert config.get_network(config.load_config()) == settings


def test_set_network_validation():
    with pytest.raises(ValueError, match="proxy"):
        config.set_network(proxy="not-a-url")
    with pytest.raises(ValueError, match="ca_bundle"):
        config.set_network(ca_bundle="/definitely/missing/ca.pem")
    with pytest.raises(ValueError, match="offline"):
        config.set_network(offline="yes")  # type: ignore[arg-type]


def test_set_network_clears_values(tmp_path):
    ca = tmp_path / "ca.pem"
    ca.write_text("cert")
    config.set_network(proxy="http://p:1", ca_bundle=str(ca), offline=True)

    cleared = config.set_network(proxy="", ca_bundle="", offline=False)

    assert cleared == {"proxy": "", "ca_bundle": "", "offline": False}


def test_get_network_drops_malformed_values():
    cfg = config.load_config()
    cfg["network"] = {
        "proxy": "not a url",
        "ca_bundle": "/definitely/missing/ca.pem",
        "offline": "yes",
    }
    config.save_config(cfg)

    assert config.get_network() == {"proxy": "", "ca_bundle": "", "offline": False}


def test_get_network_tolerates_non_object_section():
    cfg = config.load_config()
    cfg["network"] = ["nope"]
    config.save_config(cfg)

    assert config.get_network() == {"proxy": "", "ca_bundle": "", "offline": False}


def test_validate_config_flags_network_issues():
    cfg = config.load_config()
    cfg["network"] = {"proxy": 5, "ca_bundle": "nope.pem", "offline": "yes"}

    issues = config.validate_config(cfg)

    error_keys = {issue["key"] for issue in issues if issue["level"] == "error"}
    assert {"network.proxy", "network.ca_bundle", "network.offline"} <= error_keys


def test_validate_config_flags_non_object_network_section():
    cfg = config.load_config()
    cfg["network"] = "nope"

    issues = config.validate_config(cfg)

    assert any(
        issue["level"] == "error" and issue["key"] == "network" for issue in issues
    )


def test_get_network_options_mapping(tmp_path):
    ca = tmp_path / "ca.pem"
    ca.write_text("cert")

    assert config.get_network_options() == {"proxy": None, "verify": True}

    config.set_network(proxy="http://p:1", ca_bundle=str(ca))

    assert config.get_network_options() == {
        "proxy": "http://p:1",
        "verify": str(ca),
    }


# ---------------------------------------------------------------------------
# Offline mode blocks cloud network entry points without a request
# ---------------------------------------------------------------------------


def test_offline_blocks_web_fetch_without_http():
    config.set_network(offline=True)

    with pytest.raises(web_module.WebError, match="offline mode"):
        web_module.fetch_url(
            "https://example.com",
            timeout=1.0,
            max_chars=1_000,
            allow_local=False,
            transport=httpx.MockTransport(_no_http),
        )


def test_offline_blocks_web_search_without_http():
    config.set_network(offline=True)

    with pytest.raises(web_module.WebError, match="offline mode"):
        web_module.search_web(
            "kiwi",
            timeout=1.0,
            provider="duckduckgo",
            transport=httpx.MockTransport(_no_http),
        )


def test_offline_blocks_catalog_fetch_without_http():
    config.set_network(offline=True)

    with pytest.raises(catalog.CatalogFetchError, match="offline mode"):
        catalog.fetch_models(
            REGISTRY["openrouter"],
            "sk-test",
            transport=httpx.MockTransport(_no_http),
        )
    assert (
        catalog.probe(REGISTRY["openrouter"], transport=httpx.MockTransport(_no_http))
        is False
    )


def test_offline_blocks_embeddings_without_http():
    config.set_network(offline=True)
    config.set_key("openai", "sk-test")

    with pytest.raises(embeddings.EmbeddingError, match="offline mode"):
        embeddings.embed_texts(
            ["alpha"],
            "openai",
            "embed-model",
            transport=httpx.MockTransport(_no_http),
        )


def test_offline_allows_local_catalog_fetch():
    config.set_network(offline=True)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"data": [{"id": "llama"}]})
    )

    models = catalog.fetch_models(REGISTRY["ollama"], None, transport=transport)

    assert [model.id for model in models] == ["llama"]
    assert catalog.probe(REGISTRY["ollama"], transport=transport) is True


def test_offline_allows_local_embeddings():
    config.set_network(offline=True)
    config.add_provider("local-embed", "Local Embed", "http://localhost:9999/v1", "e")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]}
        )

    vectors = embeddings.embed_texts(
        ["alpha"],
        "local-embed",
        "e",
        transport=httpx.MockTransport(handler),
    )

    assert vectors == [[1.0, 0.0]]


async def test_offline_client_refuses_cloud_provider():
    config.set_network(offline=True)
    client = UnifiedClient(REGISTRY["openai"], "sk-test")

    with pytest.raises(ProviderError, match="offline mode"):
        await anext(client.stream_chat([], None, "gpt-5.6-sol"))


class _FakeResponse:
    status_code = 200

    async def aread(self) -> bytes:
        return b""

    async def aiter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
        yield "data: [DONE]"


class _FakeStreamContext:
    async def __aenter__(self) -> _FakeResponse:
        return _FakeResponse()

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakeAsyncClient:
    last_kwargs: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = kwargs

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def stream(self, method: str, url: str, **kwargs: object) -> _FakeStreamContext:
        return _FakeStreamContext()


async def test_offline_allows_local_provider_stream(monkeypatch):
    config.set_network(offline=True)
    monkeypatch.setattr(client_module.httpx, "AsyncClient", _FakeAsyncClient)

    events = [
        event
        async for event in UnifiedClient(REGISTRY["ollama"], "").stream_chat(
            [], None, "llama"
        )
    ]

    assert any(isinstance(event, TextDelta) for event in events)


async def test_stream_chat_passes_proxy_and_ca_to_async_client(monkeypatch, tmp_path):
    ca = tmp_path / "ca.pem"
    ca.write_text("cert")
    config.set_network(proxy="http://proxy.local:8080", ca_bundle=str(ca))
    monkeypatch.setattr(client_module.httpx, "AsyncClient", _FakeAsyncClient)

    events = [
        event
        async for event in UnifiedClient(_local_provider(), "").stream_chat([], None, "m")
    ]

    assert events  # the request went through the fake client
    assert _FakeAsyncClient.last_kwargs["proxy"] == "http://proxy.local:8080"
    assert _FakeAsyncClient.last_kwargs["verify"] == str(ca)


# ---------------------------------------------------------------------------
# Proxy / CA options reach the sync httpx client constructors
# ---------------------------------------------------------------------------


class _Stop(Exception):
    """Sentinel raised after a constructor records its kwargs."""


class _RecordingClient:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        type(self).calls.append(kwargs)
        raise _Stop


@pytest.fixture
def proxy_and_ca(tmp_path, monkeypatch):
    ca = tmp_path / "ca.pem"
    ca.write_text("cert")
    config.set_network(proxy="http://proxy.local:8080", ca_bundle=str(ca))
    _RecordingClient.calls = []
    monkeypatch.setattr(httpx, "Client", _RecordingClient)
    return str(ca)


def test_web_fetch_passes_proxy_and_ca(proxy_and_ca):
    with pytest.raises(_Stop):
        web_module.fetch_url(
            "https://example.com", timeout=1.0, max_chars=1_000, allow_local=False
        )

    kwargs = _RecordingClient.calls[0]
    assert kwargs["proxy"] == "http://proxy.local:8080"
    assert kwargs["verify"] == proxy_and_ca


def test_web_search_passes_proxy_and_ca(proxy_and_ca):
    with pytest.raises(_Stop):
        web_module.search_web("kiwi", timeout=1.0, provider="duckduckgo")

    kwargs = _RecordingClient.calls[0]
    assert kwargs["proxy"] == "http://proxy.local:8080"
    assert kwargs["verify"] == proxy_and_ca


def test_catalog_fetch_passes_proxy_and_ca(proxy_and_ca):
    with pytest.raises(_Stop):
        catalog.fetch_models(REGISTRY["openai"], "sk-test")

    kwargs = _RecordingClient.calls[0]
    assert kwargs["proxy"] == "http://proxy.local:8080"
    assert kwargs["verify"] == proxy_and_ca


def test_embeddings_pass_proxy_and_ca(proxy_and_ca):
    config.set_key("openai", "sk-test")

    with pytest.raises(_Stop):
        embeddings.embed_texts(["alpha"], "openai", "embed-model")

    kwargs = _RecordingClient.calls[0]
    assert kwargs["proxy"] == "http://proxy.local:8080"
    assert kwargs["verify"] == proxy_and_ca


# ---------------------------------------------------------------------------
# CLI and slash commands
# ---------------------------------------------------------------------------


def test_cli_network_proxy_ca_offline_roundtrip(tmp_path):
    ca = tmp_path / "ca.pem"
    ca.write_text("cert")
    runner = CliRunner()

    assert runner.invoke(
        main.app, ["config", "network", "proxy", "http://p:1"]
    ).exit_code == 0
    assert runner.invoke(
        main.app, ["config", "network", "ca", str(ca)]
    ).exit_code == 0
    assert runner.invoke(
        main.app, ["config", "network", "offline", "on"]
    ).exit_code == 0

    assert config.get_network() == {
        "proxy": "http://p:1",
        "ca_bundle": str(ca),
        "offline": True,
    }

    show = runner.invoke(main.app, ["config", "network", "show"])
    assert show.exit_code == 0
    assert "Offline mode" in show.output

    assert runner.invoke(
        main.app, ["config", "network", "proxy", "clear"]
    ).exit_code == 0
    assert runner.invoke(
        main.app, ["config", "network", "offline", "off"]
    ).exit_code == 0
    assert config.get_network() == {"proxy": "", "ca_bundle": str(ca), "offline": False}


def test_cli_network_rejects_bad_proxy():
    result = CliRunner().invoke(
        main.app, ["config", "network", "proxy", "not-a-url"]
    )

    assert result.exit_code == 1
    assert "proxy" in result.output


def test_cli_network_rejects_missing_ca():
    result = CliRunner().invoke(
        main.app, ["config", "network", "ca", "/definitely/missing/ca.pem"]
    )

    assert result.exit_code == 1
    assert "ca_bundle" in result.output


def test_slash_network_commands(session, tmp_path):
    ca = tmp_path / "ca.pem"
    ca.write_text("cert")
    console = Console(file=io.StringIO(), force_terminal=False, width=120)

    dispatch("/config network proxy http://p:1", session, console)
    dispatch(f"/config network ca {ca}", session, console)
    dispatch("/config network offline on", session, console)

    assert config.get_network() == {
        "proxy": "http://p:1",
        "ca_bundle": str(ca),
        "offline": True,
    }

    dispatch("/config network show", session, console)
    assert "Offline mode: on" in console.file.getvalue()

    dispatch("/config network offline off", session, console)
    dispatch("/config network proxy clear", session, console)
    dispatch("/config network ca clear", session, console)

    assert config.get_network() == {"proxy": "", "ca_bundle": "", "offline": False}


def test_config_show_lists_network_line(session):
    console = Console(file=io.StringIO(), force_terminal=False, width=120)

    dispatch("/config", session, console)

    assert "Network:" in console.file.getvalue()
