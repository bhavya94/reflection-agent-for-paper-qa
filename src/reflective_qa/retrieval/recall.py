"""Evidence recall@k of BM25 retrieval: how much of the annotators' gold evidence the top-k paragraphs contain."""

import random
import statistics
from dataclasses import dataclass

from reflective_qa.data.qasper import QasperSplit
from reflective_qa.data.records import Question
from reflective_qa.retrieval.bm25 import Retriever, passage_text


@dataclass(frozen=True)
class EvidenceHit:
    recall: float
    """Share of one annotator's evidence paragraphs that were retrieved, best annotator."""
    all_found: bool
    """True if every evidence paragraph of at least one annotator was retrieved."""


def evidence_hit(question: Question, retrieved_ids: set[str]) -> EvidenceHit | None:
    """Scored against each annotator separately, keeping the best, like the official evaluator's max over answers.
    None when no annotator cited a paragraph (unanswerable, or heading-only evidence)."""
    golds = [set(a.evidence_ids) for a in question.answers if a.evidence_ids]
    if not golds:
        return None
    return EvidenceHit(
        recall=max(len(gold & retrieved_ids) / len(gold) for gold in golds),
        all_found=any(gold <= retrieved_ids for gold in golds),
    )


class RecallEvaluator:
    """BM25 vs. a random baseline on a split's text-only questions: k paragraphs sampled uniformly, scored with the
    same `evidence_hit`."""

    def __init__(self, split: QasperSplit):
        self.split = split
        self.retriever = Retriever(split)

    def evaluate(self, ks: list[int], random_samples: int, seed: int) -> list[dict]:
        rng = random.Random(seed)
        rows = {k: {"recall": [], "all_found": [], "random_recall": [], "random_all_found": [], "chars": []} for k in ks}
        for question in self.split.text_only_questions():
            index = self.retriever.index(question.paper_id)
            ids = [p.id for p in index.paragraphs]
            ranked = index.retrieve(question.text, max(ks))
            for k in ks:
                hit = evidence_hit(question, {p.id for p in ranked[:k]})
                if hit is None:
                    continue
                draws = [evidence_hit(question, set(rng.sample(ids, min(k, len(ids))))) for _ in range(random_samples)]
                row = rows[k]
                row["recall"].append(hit.recall)
                row["all_found"].append(hit.all_found)
                row["random_recall"].append(statistics.mean(d.recall for d in draws))
                row["random_all_found"].append(statistics.mean(d.all_found for d in draws))
                row["chars"].append(sum(len(passage_text(p)) for p in ranked[:k]))
        return [
            {"k": k, "questions": len(r["recall"]), "median_chars": statistics.median(r["chars"])}
            | {name: statistics.mean(r[name]) for name in ("recall", "all_found", "random_recall", "random_all_found")}
            for k, r in rows.items()
        ]
