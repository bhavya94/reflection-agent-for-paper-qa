"""Load a QASPER v0.3 split into typed records.

Answer text and answer types follow the official evaluator (`qasper_evaluator.py`), so predictions
scored here and there agree. Gold evidence is mapped to paragraph ids for retrieval metrics.
"""

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

FIGURE_OR_TABLE_MARKER = "FLOAT SELECTED"
UNANSWERABLE = "Unanswerable"


class AnswerType(StrEnum):
    """Values match the official evaluator's type names."""

    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    BOOLEAN = "boolean"
    NONE = "none"


@dataclass(frozen=True)
class Paragraph:
    id: str
    section: str
    text: str


@dataclass(frozen=True)
class Answer:
    text: str
    type: AnswerType
    evidence_ids: tuple[str, ...]
    """Ids of the paragraphs the annotator selected as evidence."""
    other_evidence: tuple[str, ...]
    """Evidence that matches no paragraph. In QASPER v0.3 these are all section headings."""
    uses_figure_or_table: bool


@dataclass(frozen=True)
class Question:
    id: str
    paper_id: str
    text: str
    answers: tuple[Answer, ...]

    @property
    def text_only(self) -> bool:
        """True when no annotator needed a figure or table to answer."""
        return not any(answer.uses_figure_or_table for answer in self.answers)


@dataclass(frozen=True)
class Paper:
    id: str
    title: str
    abstract: str
    paragraphs: tuple[Paragraph, ...]
    questions: tuple[Question, ...]


def load_split(path: Path) -> list[Paper]:
    """Load one released QASPER JSON file (train or dev)."""
    raw = json.loads(path.read_text())
    return [_parse_paper(paper_id, paper) for paper_id, paper in raw.items()]


def text_only_questions(papers: list[Paper]) -> list[Question]:
    return [q for paper in papers for q in paper.questions if q.text_only]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _parse_paper(paper_id: str, raw: dict) -> Paper:
    paragraphs: list[Paragraph] = []
    ids_by_text: dict[str, str] = {}
    for s, section in enumerate(raw["full_text"]):
        for p, text in enumerate(section["paragraphs"]):
            key = _normalize(text)
            # Skip empty and repeated paragraphs, so every evidence text has exactly one id to retrieve.
            if not key or key in ids_by_text:
                continue
            ids_by_text[key] = f"s{s}p{p}"
            paragraphs.append(Paragraph(id=ids_by_text[key], section=section["section_name"] or "", text=text))
    questions = tuple(
        Question(
            id=qa["question_id"],
            paper_id=paper_id,
            text=qa["question"],
            answers=tuple(_parse_answer(a["answer"], ids_by_text) for a in qa["answers"]),
        )
        for qa in raw["qas"]
    )
    return Paper(
        id=paper_id,
        title=raw["title"],
        abstract=raw["abstract"],
        paragraphs=tuple(paragraphs),
        questions=questions,
    )


def _parse_answer(raw: dict, ids_by_text: dict[str, str]) -> Answer:
    text, answer_type = _answer_text_and_type(raw)
    evidence_ids: list[str] = []
    other: list[str] = []
    uses_figure_or_table = False
    for evidence in raw["evidence"]:
        if FIGURE_OR_TABLE_MARKER in evidence:
            uses_figure_or_table = True
        elif (paragraph_id := ids_by_text.get(_normalize(evidence))) is not None:
            evidence_ids.append(paragraph_id)
        else:
            other.append(evidence)
    return Answer(
        text=text,
        type=answer_type,
        evidence_ids=tuple(evidence_ids),
        other_evidence=tuple(other),
        uses_figure_or_table=uses_figure_or_table,
    )


def _answer_text_and_type(raw: dict) -> tuple[str, AnswerType]:
    """Same precedence as the official evaluator's `get_answers_and_evidence`."""
    if raw["unanswerable"]:
        return UNANSWERABLE, AnswerType.NONE
    if raw["extractive_spans"]:
        return ", ".join(raw["extractive_spans"]), AnswerType.EXTRACTIVE
    if raw["free_form_answer"]:
        return raw["free_form_answer"], AnswerType.ABSTRACTIVE
    if raw["yes_no"] is not None:
        return ("Yes" if raw["yes_no"] else "No"), AnswerType.BOOLEAN
    raise ValueError(f"Answer has no content: {raw}")
