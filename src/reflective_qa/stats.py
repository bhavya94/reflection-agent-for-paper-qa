"""Summarize a loaded QASPER split: question counts, answer types, and how much evidence maps to paragraphs."""

import statistics
import sys
from collections import Counter
from pathlib import Path

from reflective_qa.data import Paper, text_only_questions


def summarize(papers: list[Paper]) -> dict:
    questions = [q for paper in papers for q in paper.questions]
    text_only = text_only_questions(papers)
    answers = [a for q in text_only for a in q.answers]
    paragraph_chars = [len(p.text) for paper in papers for p in paper.paragraphs]
    return {
        "papers": len(papers),
        "questions": len(questions),
        "text_only_questions": len(text_only),
        "text_only_with_2plus_answers": sum(len(q.answers) >= 2 for q in text_only),
        "answer_types": dict(Counter(a.type.value for a in answers)),
        "evidence_paragraphs_mapped": sum(len(a.evidence_ids) for a in answers),
        "evidence_not_a_paragraph": sum(len(a.other_evidence) for a in answers),
        "paragraphs_per_paper_median": statistics.median(len(paper.paragraphs) for paper in papers),
        "paragraph_chars_median": statistics.median(paragraph_chars),
    }


if __name__ == "__main__":
    from reflective_qa.data import load_split

    for key, value in summarize(load_split(Path(sys.argv[1]))).items():
        print(f"{key:32} {value}")
