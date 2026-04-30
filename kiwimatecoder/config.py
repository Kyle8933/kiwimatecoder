import os
from pathlib import Path

from rich.console import Console

console = Console()
CONFIG_DIR = Path.home() / ".kiwimatecoder"
CONFIG_FILE = CONFIG_DIR / "config"


def save_api_key(key: str):
    CONFIG_DIR.mkdir(exist_ok=True)
    CONFIG_FILE.write_text(f"OPENROUTER_API_KEY={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}")


def load_api_key() -> str | None:
    if CONFIG_FILE.exists():
        for line in CONFIG_FILE.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("OPENROUTER_API_KEY")
