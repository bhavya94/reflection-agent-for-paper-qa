"""What every experiment shares: the split and its seeded sample, the run files, the prices, and the Batch API for the
steps that call it."""

from typing import Any

from reflective_qa.data.qasper import QasperSplit
from reflective_qa.paths import load_config
from reflective_qa.retrieval.bm25 import Retriever
from reflective_qa.runs.batch import BatchRunner
from reflective_qa.runs.costs import CostEstimator, Costs
from reflective_qa.runs.store import RunStore


class Experiment:
    def __init__(self, split: QasperSplit, store: RunStore, costs: Costs, n: int | None, seed: int,
                 client: Any | None = None):
        """`client` (an Anthropic client) is needed only to estimate and run; scoring reads the run files."""
        self.split, self.store, self.costs, self.n, self.seed = split, store, costs, n, seed
        self.retriever = Retriever(split)
        self._runner = BatchRunner(client) if client else None
        self._estimator = CostEstimator(client, costs) if client else None

    @staticmethod
    def sample_config(n: int | None) -> tuple[int, int]:
        """The seeded sample of configs/answering.toml, shared by every experiment, or its first n questions."""
        sample = load_config("answering")["sample"]
        return n or sample["questions"], sample["seed"]

    def _api(self) -> tuple[BatchRunner, CostEstimator]:
        if self._runner is None or self._estimator is None:
            raise ValueError("estimate and run need an API client; pass client= to the experiment")
        return self._runner, self._estimator
