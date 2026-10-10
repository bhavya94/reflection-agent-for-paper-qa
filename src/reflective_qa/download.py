"""Download the QASPER archives named in configs/data.toml, verify them, and extract listed files only."""

import hashlib
import os
import shutil
import tarfile
import tempfile
import tomllib
import urllib.request
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "data.toml"


def download(config_path: Path = DEFAULT_CONFIG) -> Path:
    """Paths in the config are relative to the config file."""
    config = tomllib.loads(config_path.read_text())
    raw_dir = (config_path.parent / config["raw_dir"]).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    for archive in config["archives"]:
        if all((raw_dir / name).exists() for name in archive["extract"]):
            continue
        # Work inside raw_dir so the final moves are atomic renames: an interrupted run leaves no partial files.
        with tempfile.TemporaryDirectory(dir=raw_dir) as tmp:
            archive_path = Path(tmp) / "archive.tgz"
            with urllib.request.urlopen(archive["url"], timeout=60) as response, archive_path.open("wb") as out:
                shutil.copyfileobj(response, out)
            digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            if digest != archive["sha256"]:
                raise ValueError(f"Checksum mismatch for {archive['url']}: got {digest}")
            with tarfile.open(archive_path) as tar:
                tar.extractall(tmp, members=[tar.getmember(n) for n in archive["extract"]], filter="data")
            for name in archive["extract"]:
                os.replace(Path(tmp) / name, raw_dir / name)
    return raw_dir


if __name__ == "__main__":
    print(f"QASPER files in {download()}")
