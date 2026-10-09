"""Download the QASPER archives named in configs/data.toml, verify them, and extract listed files only."""

import hashlib
import tarfile
import tempfile
import tomllib
import urllib.request
from pathlib import Path


def download(config_path: Path) -> Path:
    config = tomllib.loads(config_path.read_text())
    raw_dir = Path(config["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    for archive in config["archives"]:
        if all((raw_dir / name).exists() for name in archive["extract"]):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "archive.tgz"
            urllib.request.urlretrieve(archive["url"], path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != archive["sha256"]:
                raise ValueError(f"Checksum mismatch for {archive['url']}: got {digest}")
            with tarfile.open(path) as tar:
                tar.extractall(raw_dir, members=[tar.getmember(n) for n in archive["extract"]], filter="data")
    return raw_dir


if __name__ == "__main__":
    print(f"QASPER files in {download(Path('configs/data.toml'))}")
