"""Run requests through the Batch API without paying twice.

- Answered questions are skipped, and a submitted batch's ID is saved before waiting, so an interrupted run resumes
  that batch.
- A run is refused when its estimate is over the budget, unless a batch for it is already paid for and waiting.
"""

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

from reflective_qa.runs.store import RunStore


class OverBudget(Exception):
    pass


class BatchRunner:
    def __init__(self, client: Any, poll_seconds: float = 30):
        self.client, self.poll_seconds = client, poll_seconds

    def run(self, requests: dict[str, dict], files: tuple[Path, Path], record: Callable[[str, Any], dict], *,
            estimate: float = 0.0, max_dollars: float = math.inf, label: str = "") -> Path:
        if estimate > max_dollars and not files[1].exists():
            raise OverBudget(f"{label} estimate ${estimate:.2f} is over --max-dollars {max_dollars}")
        return self.submit_and_collect(requests, files, record)

    def submit_and_collect(self, requests: dict[str, dict], files: tuple[Path, Path],
                           record: Callable[[str, Any], dict]) -> Path:
        """Batch the requests not yet in the results file, wait, and append `record(custom_id, message)` for each
        succeeded result. Only succeeded results are written, so errored or expired requests are retried next time."""
        answers_path, batch_path = files
        answers_path.parent.mkdir(parents=True, exist_ok=True)
        done = {row["question_id"] for row in RunStore.read(answers_path)}
        if batch_path.exists():
            batch_id = batch_path.read_text().strip()
        else:
            pending = [{"custom_id": qid, "params": params} for qid, params in requests.items() if qid not in done]
            if not pending:
                return answers_path
            # No SDK retries: if a retried create's first response were lost, a second paid batch would be submitted.
            # After a failed create, check client.messages.batches.list() before running again.
            batch_id = self.client.with_options(max_retries=0).messages.batches.create(requests=pending).id
            batch_path.write_text(batch_id)
        while self.client.messages.batches.retrieve(batch_id).processing_status != "ended":
            time.sleep(self.poll_seconds)
        with answers_path.open("a") as out:
            for result in self.client.messages.batches.results(batch_id):
                if result.result.type != "succeeded" or result.custom_id in done:
                    continue
                out.write(json.dumps(record(result.custom_id, result.result.message)) + "\n")
                done.add(result.custom_id)
        batch_path.unlink()
        return answers_path
