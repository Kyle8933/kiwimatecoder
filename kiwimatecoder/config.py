import os
from pathlib import Path

from rich.console import Console

console = Console()
CONFIG_DIR = Path.home() / ".kiwimatecoder"
CONFIG_FILE = CONFIG_DIR / "config"


def save_api_key(key: str):
    CONFIG_DIR.mkdir(exist_ok=True)
    CONFIG_FILE.write_text(f"OPENROUTER_API_KEY={}")


def load_api_key() -> str | None:
    if CONFIG_FILE.exists():
        for line in CONFIG_FILE.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("OPENROUTER_API_KEY")
