"""One-shot answers from a writer run on the shared question sample: build the requests, estimate them, run them with
the Batch API, and score them on their own sample."""

import math
from pathlib import Path

from reflective_qa.agent.checks import CodeChecks
from reflective_qa.agent.writer import Writer
from reflective_qa.data.qasper import QasperSplit
from reflective_qa.eval.metrics import answer_f1, is_unanswerable, normalize
from reflective_qa.eval.scorer import Scorer, load_predictions
from reflective_qa.paths import load_config
from reflective_qa.retrieval.bm25 import Retriever
from reflective_qa.runs.batch import BatchRunner
from reflective_qa.runs.costs import CostEstimator, Costs
from reflective_qa.runs.store import RunStore


class BaselineExperiment:
    def __init__(self, split: QasperSplit, store: RunStore, costs: Costs, n: int | None, seed: int,
                 runner: BatchRunner | None = None):
        """`runner` is needed only to estimate and run; scoring reads the run files."""
        self.split, self.store, self.costs, self.n, self.seed, self.runner = split, store, costs, n, seed, runner
        self.retriever = Retriever(split)

    @classmethod
    def from_config(cls, split: QasperSplit, store: RunStore, n: int | None = None,
                    runner: BatchRunner | None = None) -> "BaselineExperiment":
        """The seeded sample of configs/answering.toml, or its first n questions."""
        sample = load_config("answering")["sample"]
        return cls(split, store, Costs.from_config(), n or sample["questions"], sample["seed"], runner)

    def requests(self, writer: Writer) -> dict[str, dict]:
        """Request parameters keyed by question ID, for the sample."""
        return {q.id: writer.request(self.split.paper(q.paper_id), q, self.retriever.retrieve(q, writer.k))
                for q in self.split.sample(self.n, self.seed)}

    def pending(self, writer: Writer, requests: dict[str, dict]) -> dict[str, dict]:
        done = {row["question_id"] for row in self.store.answers(writer)}
        return {qid: params for qid, params in requests.items() if qid not in done}

    def estimate(self, writer: Writer, requests: dict[str, dict]) -> dict:
        """Batch cost of answering `requests`. Output tokens are this run's measured mean if it has answers, else the
        configured assumption."""
        if not requests:
            return {"run": writer.name, "questions": 0, "input_tokens": 0, "output_tokens": 0, "output_measured": False,
                    "dollars": 0.0}
        estimator = CostEstimator(self.runner.client, self.costs)
        answered = self.store.answers(writer)
        output, input_mean = estimator.output_tokens(answered), estimator.input_tokens(requests)
        return {"run": writer.name, "questions": len(requests), "input_tokens": input_mean, "output_tokens": output,
                "output_measured": bool(answered),
                "dollars": self.costs.dollars(writer.model, input_mean * len(requests), output * len(requests))}

    def run(self, writer: Writer, requests: dict[str, dict], *, estimate: float = 0.0,
            max_dollars: float = math.inf) -> Path:
        """Answers go to the run's file. Refused when `estimate` (from `estimate()`) is over max_dollars."""
        return self.runner.run(requests, self.store.writer_files(writer), self.costs.recorder(Writer.parse, writer.model),
                               estimate=estimate, max_dollars=max_dollars, label=f"{writer.name}:")

    def report(self, writer: Writer, requests: dict[str, dict]) -> dict:
        """Scores a run on its own question sample, so questions outside the sample don't count as zeros."""
        answers_path = self.store.writer_files(writer)[0]
        rows = self.store.read(answers_path)
        result = Scorer(self.split).score(load_predictions(answers_path) if rows else {}, set(requests))
        return result | {"run": writer.name, "file": answers_path.name, "cost": sum(r["cost"] for r in rows),
                         "output_tokens": CostEstimator.mean_output(rows, 0)}

    def check_findings(self, writer: Writer, checks: CodeChecks) -> dict:
        """How often each code check fires on a writer's drafts. The F1 comparison covers content answers only (not
        Unanswerable, Yes or No), since too_long can't flag those; it describes the drafts, it doesn't validate the
        checks."""
        rows = {r["question_id"]: r for r in self.store.answers(writer)}
        questions = self.split.sample(self.n, self.seed)
        counts: dict[str, int] = {}
        flagged_f1, clean_f1 = [], []
        for q in questions:
            row = rows[q.id]
            ids = {p.id for p in self.retriever.retrieve(q, writer.k)}
            found = checks.run(q.text, row["predicted_answer"], tuple(row["predicted_evidence"]), ids)
            for check in {f.check for f in found}:
                counts[check] = counts.get(check, 0) + 1
            if not is_unanswerable(row["predicted_answer"]) and normalize(row["predicted_answer"]) not in ("yes", "no"):
                (flagged_f1 if found else clean_f1).append(answer_f1(q, row["predicted_answer"])[0])
        return {"drafts": len(questions), "counts": dict(sorted(counts.items())), "flagged_f1": flagged_f1,
                "clean_f1": clean_f1}
