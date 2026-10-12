import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from reflective_qa.data.download import download


def _archive(tmp_path: Path, files: dict[str, str]) -> Path:
    path = tmp_path / "archive.tgz"
    with tarfile.open(path, "w:gz") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def _config(tmp_path: Path, archive: Path, sha256: str, extract: list[str]) -> Path:
    config = tmp_path / "configs" / "data.toml"
    config.parent.mkdir()
    config.write_text(
        f'raw_dir = "../raw"\n[[archives]]\nurl = "{archive.as_uri()}"\nsha256 = "{sha256}"\nextract = {extract}\n'
    )
    return config


def test_extracts_only_listed_files(tmp_path):
    archive = _archive(tmp_path, {"qasper_evaluator.py": "evaluator", "qasper-test-v0.3.json": "{}"})
    sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    raw_dir = download(_config(tmp_path, archive, sha256, ["qasper_evaluator.py"]))
    assert raw_dir == (tmp_path / "raw").resolve()
    assert sorted(p.name for p in raw_dir.rglob("*")) == ["qasper_evaluator.py"]


def test_rejects_an_archive_whose_checksum_does_not_match(tmp_path):
    archive = _archive(tmp_path, {"qasper_evaluator.py": "evaluator"})
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download(_config(tmp_path, archive, "0" * 64, ["qasper_evaluator.py"]))
    assert list((tmp_path / "raw").rglob("*")) == []
