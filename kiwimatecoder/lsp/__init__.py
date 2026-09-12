"""Language Server Protocol support: diagnostics and navigation.

Opt-in via the ``lsp`` config section. The manager starts a server subprocess
per language on first use, routes files by extension, and exposes bounded
diagnostics/definition/references calls; failures are reported, never fatal.
"""

from kiwimatecoder.lsp.client import (
    LspClient,
    LspError,
    LspTimeout,
    language_id_for,
    normalize_diagnostics,
    normalize_locations,
    path_to_uri,
    uri_to_path,
)
from kiwimatecoder.lsp.manager import (
    LSP_DISABLED_MESSAGE,
    SERVER_PRESETS,
    LspManager,
    LspServerSpec,
    available_servers,
    effective_servers,
    ensure_manager,
    get_manager,
    set_manager,
)
from kiwimatecoder.lsp.protocol import (
    LspProtocolError,
    decode_messages,
    encode_message,
)

__all__ = [
    "LSP_DISABLED_MESSAGE",
    "SERVER_PRESETS",
    "LspClient",
    "LspError",
    "LspManager",
    "LspProtocolError",
    "LspServerSpec",
    "LspTimeout",
    "available_servers",
    "decode_messages",
    "effective_servers",
    "encode_message",
    "ensure_manager",
    "get_manager",
    "language_id_for",
    "normalize_diagnostics",
    "normalize_locations",
    "path_to_uri",
    "set_manager",
    "uri_to_path",
]
