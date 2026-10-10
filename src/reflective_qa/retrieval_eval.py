"""Evidence recall@k of BM25 retrieval: how much of the annotators' gold evidence the top-k paragraphs contain."""

import statistics
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from reflective_qa.data import Paper, Question, text_only_questions
from reflective_qa.retrieval import PaperRetriever

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


def evaluate(papers: list[Paper], ks: list[int]) -> list[dict]:
    by_id = {paper.id: paper for paper in papers}
    retrievers = {paper.id: PaperRetriever(paper) for paper in papers}
    rows = {k: {"recall": [], "all_found": [], "random_recall": [], "chars": []} for k in ks}
    for question in text_only_questions(papers):
        paragraphs = by_id[question.paper_id].paragraphs
        ranked = retrievers[question.paper_id].retrieve(question.text, max(ks))
        for k in ks:
            hit = evidence_hit(question, {p.id for p in ranked[:k]})
            if hit is None:
                continue
            rows[k]["recall"].append(hit.recall)
            rows[k]["all_found"].append(hit.all_found)
            rows[k]["random_recall"].append(min(k / len(paragraphs), 1.0))
            rows[k]["chars"].append(sum(len(p.text) for p in ranked[:k]))
    return [
        {
            "k": k,
            "questions": len(r["recall"]),
            "recall": statistics.mean(r["recall"]),
            "all_found": statistics.mean(r["all_found"]),
            "random_recall": statistics.mean(r["random_recall"]),
            "median_chars": statistics.median(r["chars"]),
        }
        for k, r in rows.items()
    ]


if __name__ == "__main__":
    from reflective_qa.data import load_split

    ks = tomllib.loads(CONFIG.read_text())["report_ks"]
    rows = evaluate(load_split(Path(sys.argv[1])), ks)
    print(f"{rows[0]['questions']} text-only questions with paragraph evidence")
    print(f"{'k':>3}  {'recall':>6}  {'all found':>9}  {'random':>6}  {'context chars (median)':>22}")
    for row in rows:
        print(f"{row['k']:>3}  {row['recall']:>6.2f}  {row['all_found']:>9.2f}  {row['random_recall']:>6.2f}"
              f"  {row['median_chars']:>22,.0f}")
