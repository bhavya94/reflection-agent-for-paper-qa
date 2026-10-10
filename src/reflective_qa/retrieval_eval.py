"""Evidence recall@k of BM25 retrieval: how much of the annotators' gold evidence the top-k paragraphs contain."""

import random
import statistics
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from reflective_qa.data import Paper, Question, text_only_questions
from reflective_qa.retrieval import PaperRetriever, passage_text

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "retrieval.toml"


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


def evaluate(papers: list[Paper], ks: list[int], random_samples: int, seed: int) -> list[dict]:
    """BM25 vs. a random baseline: k paragraphs sampled uniformly, scored with the same `evidence_hit`."""
    rng = random.Random(seed)
    retrievers = {paper.id: PaperRetriever(paper) for paper in papers}
    rows = {k: {"recall": [], "all_found": [], "random_recall": [], "random_all_found": [], "chars": []} for k in ks}
    for question in text_only_questions(papers):
        retriever = retrievers[question.paper_id]
        ids = [p.id for p in retriever.paragraphs]
        ranked = retriever.retrieve(question.text, max(ks))
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


if __name__ == "__main__":
    from reflective_qa.data import load_split

    config = tomllib.loads(CONFIG.read_text())
    rows = evaluate(load_split(Path(sys.argv[1])), config["report_ks"], config["random_samples"], config["seed"])
    print(f"{rows[0]['questions']} text-only questions with paragraph evidence")
    print(f"{'k':>3}  {'recall':>6}  {'all found':>9}  {'random recall':>13}  {'random all found':>16}"
          f"  {'context chars (median)':>22}")
    for row in rows:
        print(f"{row['k']:>3}  {row['recall']:>6.2f}  {row['all_found']:>9.2f}  {row['random_recall']:>13.2f}"
              f"  {row['random_all_found']:>16.2f}  {row['median_chars']:>22,.0f}")
