"""Language-server lifecycle: detect, route by extension, start lazily.

Built-in presets cover the common languages and are used when their command is
on ``PATH``; entries under ``lsp.servers`` override or extend them. Servers are
started on first use for the language that owns a file's extension, and a failed
start is recorded so later calls fail fast with a clear reason instead of
re-spawning. ``shutdown`` stops every live client and clears recorded failures.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from kiwimatecoder import config
from kiwimatecoder.lsp.client import LspClient, LspError, LspTimeout

INITIALIZE_TIMEOUT = 10.0

# name -> {command, args, extensions}. The manager also merges user overrides
# from the ``lsp.servers`` config section on top of these.
SERVER_PRESETS: dict[str, dict[str, Any]] = {
    "python": {
        "command": "pyright-langserver",
        "args": ["--stdio"],
        "extensions": [".py", ".pyi"],
    },
    "typescript": {
        "command": "typescript-language-server",
        "args": ["--stdio"],
        "extensions": [".ts", ".tsx", ".js", ".jsx"],
    },
    "go": {"command": "gopls", "args": [], "extensions": [".go"]},
    "rust": {"command": "rust-analyzer", "args": [], "extensions": [".rs"]},
    "c/cpp": {
        "command": "clangd",
        "args": [],
        "extensions": [".c", ".h", ".cpp", ".hpp"],
    },
}

LSP_DISABLED_MESSAGE = (
    "LSP is disabled. Enable it with /lsp on or "
    "`kiwimatecoder config lsp enable on`."
)


@dataclass(frozen=True)
class LspServerSpec:
    """The effective command and file extensions for one language server."""

    name: str
    command: str
    args: tuple[str, ...]
    extensions: tuple[str, ...]


def _spec(name: str, preset: dict[str, Any]) -> LspServerSpec:
    return LspServerSpec(
        name=name,
        command=str(preset["command"]),
        args=tuple(str(arg) for arg in preset.get("args") or ()),
        extensions=tuple(str(ext) for ext in preset.get("extensions") or ()),
    )


def effective_servers() -> dict[str, LspServerSpec]:
    """Return presets merged with ``lsp.servers`` overrides (no PATH check).

    User-configured entries are listed first, so an override or a custom
    server wins extension routing over a built-in preset with the same suffix.
    """
    specs: dict[str, LspServerSpec] = {}
    for name, override in config.get_lsp()["servers"].items():
        preset = SERVER_PRESETS.get(name)
        if preset is None:
            specs[name] = LspServerSpec(
                name=name,
                command=str(override["command"]),
                args=tuple(str(arg) for arg in override["args"]),
                extensions=tuple(str(ext) for ext in override["extensions"]),
            )
        else:
            specs[name] = LspServerSpec(
                name=name,
                command=str(override["command"] or preset["command"]),
                args=tuple(
                    str(arg) for arg in (override["args"] or preset["args"])
                ),
                extensions=tuple(
                    str(ext)
                    for ext in (override["extensions"] or preset["extensions"])
                ),
            )
    for name, preset in SERVER_PRESETS.items():
        if name not in specs:
            specs[name] = _spec(name, preset)
    return specs


def available_servers() -> dict[str, LspServerSpec]:
    """Effective servers whose command resolves on ``PATH``."""
    return {
        name: spec
        for name, spec in effective_servers().items()
        if shutil.which(spec.command)
    }


class LspManager:
    """Owns live LSP clients and routes files to them by extension."""

    def __init__(
        self,
        console: Console | None = None,
        root: str | Path | None = None,
    ) -> None:
        self._console = console if console is not None else Console()
        self._root = (
            Path(root).expanduser().resolve() if root is not None else Path.cwd()
        )
        self._clients: dict[str, LspClient] = {}
        self._failures: dict[str, str] = {}

    @property
    def root(self) -> Path:
        """The workspace root used for relative paths and server handshakes."""
        return self._root

    def set_root(self, root: str | Path) -> None:
        """Point the manager at a new workspace root."""
        self._root = Path(root).expanduser().resolve()

    @property
    def clients(self) -> dict[str, LspClient]:
        """A copy of the language -> live client map."""
        return dict(self._clients)

    def failure_for(self, language: str) -> str | None:
        """The recorded start failure for ``language``, if any."""
        return self._failures.get(language)

    def servers(self) -> dict[str, LspServerSpec]:
        """Effective servers: presets merged with ``lsp.servers`` overrides."""
        return effective_servers()

    def available_servers(self) -> dict[str, LspServerSpec]:
        """Effective servers whose command resolves on ``PATH``."""
        return {
            name: spec
            for name, spec in effective_servers().items()
            if shutil.which(spec.command)
        }

    def language_for(self, path: str | Path) -> str | None:
        """Return the language whose extensions claim ``path``, if any."""
        suffix = Path(path).suffix.lower()
        if not suffix:
            return None
        for name, spec in self.servers().items():
            if suffix in spec.extensions:
                return name
        return None

    def diagnostics_for(
        self, path: str | Path, timeout: float | None = None
    ) -> list[dict[str, Any]]:
        """Return diagnostics for ``path``, starting its server on first use."""
        settings = config.get_lsp()
        if not settings["enabled"]:
            raise LspError(LSP_DISABLED_MESSAGE)
        deadline = self._deadline(settings["timeout"], timeout)
        client, resolved = self._prepare(path, deadline)
        client.sync(resolved, _read_text(resolved))
        remaining = max(0.05, deadline - time.monotonic())
        return client.diagnostics(resolved, remaining)

    def definition(
        self,
        path: str | Path,
        line: int,
        character: int,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return definition locations for a 1-based position."""
        return self._locations(
            "definition", path, line, character, None, timeout
        )

    def references(
        self,
        path: str | Path,
        line: int,
        character: int,
        include_declaration: bool = True,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return reference locations for a 1-based position."""
        return self._locations(
            "references", path, line, character, include_declaration, timeout
        )

    def shutdown(self) -> None:
        """Stop every live client and clear recorded failures."""
        for language in list(self._clients):
            client = self._clients.pop(language)
            try:
                client.shutdown()
            except Exception:
                pass
        self._failures.clear()

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _deadline(configured: float, timeout: float | None) -> float:
        budget = float(configured) if timeout is None else min(
            float(timeout), float(configured)
        )
        return time.monotonic() + max(0.05, budget)

    def _locations(
        self,
        action: str,
        path: str | Path,
        line: int,
        character: int,
        include_declaration: bool | None,
        timeout: float | None,
    ) -> list[dict[str, Any]]:
        settings = config.get_lsp()
        if not settings["enabled"]:
            raise LspError(LSP_DISABLED_MESSAGE)
        deadline = self._deadline(settings["timeout"], timeout)
        client, resolved = self._prepare(path, deadline)
        client.sync(resolved, _read_text(resolved))
        remaining = max(0.05, deadline - time.monotonic())
        if action == "definition":
            return client.definition(
                resolved, int(line) - 1, int(character) - 1, timeout=remaining
            )
        return client.references(
            resolved,
            int(line) - 1,
            int(character) - 1,
            include_declaration=bool(include_declaration),
            timeout=remaining,
        )

    def _prepare(
        self, path: str | Path, deadline: float
    ) -> tuple[LspClient, Path]:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = self._root / resolved
        resolved = resolved.resolve()
        if not resolved.is_file():
            raise LspError(f"File not found: {path}")
        language = self.language_for(resolved)
        if language is None:
            raise LspError(
                f"No language server is configured for {resolved.name}."
            )
        return self._client_for(language, deadline), resolved

    def _client_for(
        self, language: str, deadline: float | None = None
    ) -> LspClient:
        client = self._clients.get(language)
        if client is not None:
            if client.running:
                return client
            client.close()
            self._clients.pop(language, None)
        failure = self._failures.get(language)
        if failure:
            raise LspError(failure)
        spec = self.servers().get(language)
        if spec is None:
            raise LspError(f"Unknown language server '{language}'.")
        command = shutil.which(spec.command)
        if command is None:
            reason = (
                f"language server command '{spec.command}' was not found on PATH."
            )
            self._failures[language] = reason
            raise LspError(reason)

        initialize_timeout = INITIALIZE_TIMEOUT
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LspTimeout(
                    f"no time left to start the '{language}' language server"
                )
            initialize_timeout = min(initialize_timeout, remaining)
        started: LspClient | None = None
        try:
            started = LspClient(
                command,
                spec.args,
                root=self._root,
                initialize_timeout=initialize_timeout,
                request_timeout=initialize_timeout,
            )
            started.initialize()
        except Exception as exc:
            if started is not None:
                started.close()
            reason = f"could not start '{spec.command}': {exc}"
            self._failures[language] = reason
            raise LspError(reason) from exc
        self._clients[language] = started
        return started


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise LspError(f"Could not read {path}: {exc}") from exc


_MANAGER: LspManager | None = None


def get_manager() -> LspManager | None:
    """Return the session's LSP manager, if one has been created."""
    return _MANAGER


def set_manager(manager: LspManager | None) -> None:
    """Install (or clear) the session's LSP manager."""
    global _MANAGER
    _MANAGER = manager


def ensure_manager(
    console: Console | None = None, root: str | Path | None = None
) -> LspManager:
    """Return the session manager, creating it on first use."""
    manager = get_manager()
    if manager is None:
        manager = LspManager(console=console, root=root)
        set_manager(manager)
    elif root is not None:
        manager.set_root(root)
    return manager