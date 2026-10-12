"""Where the project's files live, so no module works out the project root on its own."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
RAW_DIR = ROOT / "data" / "raw"
RUNS_DIR = ROOT / "data" / "runs"
DEV_SPLIT = RAW_DIR / "qasper-dev-v0.3.json"


def load_config(name: str) -> dict:
    """One file from configs/, by name: load_config("answering") reads configs/answering.toml."""
    return tomllib.loads((CONFIGS / f"{name}.toml").read_text())
