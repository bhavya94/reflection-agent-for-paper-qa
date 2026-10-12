"""The run files in one directory. Each writer run and each reflection stage has a JSON lines file of results and,
while a batch is in flight, a file holding its batch ID. Files are named by a fingerprint of everything that shapes
their requests, so a changed prompt or config starts new files instead of mixing results."""

import json
from pathlib import Path
from typing import Protocol

from reflective_qa.paths import RUNS_DIR

STAGES = ("critic", "revise")


class Fingerprinted(Protocol):
    name: str
    fingerprint: str


class RunStore:
    def __init__(self, root: Path = RUNS_DIR):
        self.root = root

    def writer_files(self, writer: Fingerprinted) -> tuple[Path, Path]:
        """(results, batch ID) files for a writer run."""
        stem = f"{writer.name}-{writer.fingerprint}"
        return self.root / f"{stem}.jsonl", self.root / f"{stem}.batch"

    def stage_files(self, agent: Fingerprinted) -> dict[str, tuple[Path, Path]]:
        """(results, batch ID) files for each reflection stage of a setup."""
        stem = f"reflect-{agent.name}-{agent.fingerprint}"
        return {stage: (self.root / f"{stem}.{stage}.jsonl", self.root / f"{stem}.{stage}.batch") for stage in STAGES}

    @staticmethod
    def read(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def answers(self, writer: Fingerprinted) -> list[dict]:
        return self.read(self.writer_files(writer)[0])

    def stage(self, agent: Fingerprinted, stage: str) -> dict[str, dict]:
        """One reflection stage's results, keyed by question ID."""
        return {r["question_id"]: r for r in self.read(self.stage_files(agent)[stage][0])}

    def total_spent(self) -> float:
        return sum(row["cost"] for path in self.root.glob("*.jsonl") for row in self.read(path))
