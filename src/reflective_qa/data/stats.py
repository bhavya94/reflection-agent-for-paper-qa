"""Summarize a loaded QASPER split: question counts, answer types, and how much evidence maps to paragraphs."""

import statistics
from collections import Counter

from reflective_qa.data.qasper import QasperSplit


def summarize(split: QasperSplit) -> dict:
    papers, questions, text_only = split.papers, split.questions(), split.text_only_questions()
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
