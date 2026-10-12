"""How revisions moved answer-F1: the mean change per question with a paired-bootstrap 95% interval, and the changed
answers split by kind of revision."""

import random
import statistics
from dataclasses import dataclass
from functools import cached_property

from reflective_qa.data.records import Question
from reflective_qa.eval.metrics import answer_f1, is_unanswerable

KINDS = ("abstain_to_answer", "answer_to_abstain", "reworded")


@dataclass(frozen=True)
class Revision:
    """One question's answer before and after review. An unrevised answer has `after == before`."""
    question: Question
    before: str
    after: str

    @property
    def changed(self) -> bool:
        return self.before != self.after

    @cached_property
    def f1(self) -> tuple[float, float]:
        """Answer-F1 before and after."""
        return answer_f1(self.question, self.before)[0], answer_f1(self.question, self.after)[0]

    @property
    def delta(self) -> float:
        return self.f1[1] - self.f1[0]

    @property
    def kind(self) -> str:
        """An abstention turned into an answer, an answer turned into an abstention, or a rewording."""
        if is_unanswerable(self.before) and not is_unanswerable(self.after):
            return "abstain_to_answer"
        return "answer_to_abstain" if is_unanswerable(self.after) and not is_unanswerable(self.before) else "reworded"


class RevisionAnalysis:
    """A paired bootstrap over the same questions: resampling questions keeps each before and after together."""

    def __init__(self, resamples: int, seed: int):
        self.resamples, self.seed = resamples, seed

    def analyze(self, revisions: list[Revision]) -> dict:
        values = [r.delta for r in revisions]
        rng = random.Random(self.seed)
        means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(self.resamples))
        kinds: dict[str, list[float]] = {kind: [] for kind in KINDS}
        for r in revisions:
            if r.changed:
                kinds[r.kind].append(r.delta)
        return {"delta": statistics.mean(values),
                "ci": (means[int(0.025 * self.resamples)], means[int(0.975 * self.resamples) - 1]),
                "kinds": {k: {"n": len(v), "mean_delta": statistics.mean(v) if v else 0.0} for k, v in kinds.items()}}
